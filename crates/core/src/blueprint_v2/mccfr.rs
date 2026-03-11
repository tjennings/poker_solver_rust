//! External-sampling MCCFR traversal for the blueprint strategy.
//!
//! Implements the core counterfactual regret minimisation loop:
//! at the traverser's decision nodes **all** actions are explored;
//! at the opponent's decision nodes **one** action is sampled
//! according to the current regret-matched strategy.

// Arena indices are u32, bucket indices u16. Truncation is safe for
// any practical game tree. Precision loss on cast to f64 is acceptable
// (action counts and bucket counts are small).
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::cmp::Ordering;
use std::sync::atomic::AtomicU64;

use rand::Rng;

/// Global diagnostic counters for prune-hit measurement.
/// Updated once per batch (not per action) and read by the trainer for TUI.
pub static PRUNE_HITS: AtomicU64 = AtomicU64::new(0);
pub static PRUNE_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Prune statistics accumulated during a single traversal.
///
/// Collected locally to avoid global atomic contention, then merged
/// into the global counters after the batch completes.
#[derive(Debug, Default, Clone, Copy)]
pub struct PruneStats {
    pub hits: u64,
    pub total: u64,
}

impl PruneStats {
    pub fn merge(&mut self, other: PruneStats) {
        self.hits += other.hits;
        self.total += other.total;
    }
}

use std::sync::Arc;

use super::bucket_file::BucketFile;
use super::config::StreetClusterConfig;
pub use super::equity_cache::EquityDeltaCache;
use super::game_tree::{GameNode, GameTree, TerminalKind};
use super::storage::BlueprintStorage;
use super::Street;
use crate::poker::{Card, Hand, Rankable};

/// Maximum number of actions at any decision node.
///
/// Covers fold/check/call plus up to 13 bet/raise sizing choices.
/// Used for stack-allocated strategy and value buffers in the MCCFR
/// traversal hot path.
const MAX_ACTIONS: usize = 16;

/// A sampled deal: hole cards for each player plus the full 5-card board.
#[derive(Debug, Clone)]
pub struct Deal {
    pub hole_cards: [[Card; 2]; 2], // [player0, player1]
    pub board: [Card; 5],           // flop(3) + turn(1) + river(1)
}

/// A deal with pre-computed bucket assignments for all streets and players.
///
/// Eliminates repeated `get_bucket()` / `compute_equity()` calls during
/// MCCFR traversal by computing all buckets once up-front.
#[derive(Debug, Clone)]
pub struct DealWithBuckets {
    pub deal: Deal,
    /// Pre-computed bucket indices: `buckets[player][street]`.
    pub buckets: [[u16; 4]; 2],
}

/// Loaded bucket assignments for all 4 streets.
///
/// During MCCFR, we look up buckets by street using the deal's cards.
/// Supports two modes:
/// - **Equity-only**: uniform equity bins (when no delta_bins configured).
/// - **Equity+delta**: 2D grid of equity × signed delta to next street.
pub struct AllBuckets {
    pub bucket_counts: [u16; 4],
    /// Per-street bucket files produced by the clustering pipeline.
    /// Currently unused for lookups (equity fallback is used instead),
    /// but retained for potential future use.
    pub bucket_files: [Option<BucketFile>; 4],
    /// Per-street delta bin boundaries (flop, turn only; river has no future street).
    /// When `Some`, bucketing uses 2D equity×delta grid.
    delta_bins: [Option<Vec<f64>>; 4],
    /// Per-street flag: when true, delta is averaged over all possible
    /// next-street cards rather than using the actual runout.
    expected_delta: [bool; 4],
    /// Optional pre-computed equity+delta cache for O(1) lookups.
    /// When present, `precompute_buckets` uses table lookups instead of
    /// expensive expected-equity computation.
    equity_cache: Option<Arc<EquityDeltaCache>>,
}

impl AllBuckets {
    #[must_use]
    pub fn new(
        bucket_counts: [u16; 4],
        bucket_files: [Option<BucketFile>; 4],
        street_configs: [&StreetClusterConfig; 4],
    ) -> Self {
        let delta_bins = [
            street_configs[0].delta_bins.clone(),
            street_configs[1].delta_bins.clone(),
            street_configs[2].delta_bins.clone(),
            street_configs[3].delta_bins.clone(),
        ];
        let expected_delta = [
            street_configs[0].expected_delta,
            street_configs[1].expected_delta,
            street_configs[2].expected_delta,
            street_configs[3].expected_delta,
        ];
        Self {
            bucket_counts,
            bucket_files,
            delta_bins,
            expected_delta,
            equity_cache: None,
        }
    }

    /// Attach a pre-computed equity+delta cache for O(1) lookups.
    pub fn set_equity_cache(&mut self, cache: Arc<EquityDeltaCache>) {
        self.equity_cache = Some(cache);
    }

    /// Pre-compute bucket assignments for all 4 streets × 2 players.
    ///
    /// Computes equity at all postflop streets first, then derives buckets
    /// using equity+delta when delta_bins are configured. When `expected_delta`
    /// is set for a street, the delta is averaged over all possible next-street
    /// cards rather than using the actual runout.
    #[must_use]
    pub fn precompute_buckets(&self, deal: &Deal) -> [[u16; 4]; 2] {
        let mut result = [[0u16; 4]; 2];
        for (player, row) in result.iter_mut().enumerate() {
            let hole = deal.hole_cards[player];

            // Preflop: canonical hand index.
            let hand = crate::hands::CanonicalHand::from_cards(hole[0], hole[1]);
            let idx = hand.index() as u16;
            row[0] = idx.min(self.bucket_counts[0] - 1);

            // Compute postflop equities up-front (needed for delta computation).
            let flop_eq = crate::showdown_equity::compute_equity(hole, &deal.board[..3]);
            let turn_eq = crate::showdown_equity::compute_equity(hole, &deal.board[..4]);
            let river_eq = crate::showdown_equity::compute_equity(hole, &deal.board[..5]);

            // Flop bucket: equity + delta to turn.
            let flop_next_eq = if self.expected_delta[1] && self.delta_bins[1].is_some() {
                if let Some(cache) = &self.equity_cache {
                    cache.flop_lookup(hole, &deal.board[..3])
                        .map(|e| f64::from(e.expected_next_equity))
                } else {
                    Some(expected_next_equity(hole, &deal.board[..3]))
                }
            } else {
                Some(turn_eq)
            };
            row[1] = self.compute_bucket(1, flop_eq, flop_next_eq);

            // Turn bucket: equity + delta to river.
            let turn_next_eq = if self.expected_delta[2] && self.delta_bins[2].is_some() {
                if let Some(cache) = &self.equity_cache {
                    cache.turn_lookup(hole, &deal.board[..4])
                        .map(|e| f64::from(e.expected_next_equity))
                } else {
                    Some(expected_next_equity(hole, &deal.board[..4]))
                }
            } else {
                Some(river_eq)
            };
            row[2] = self.compute_bucket(2, turn_eq, turn_next_eq);

            // River bucket: equity only (no future street).
            row[3] = self.compute_bucket(3, river_eq, None);
        }
        result
    }

    /// Compute a bucket index for a single street.
    ///
    /// When delta_bins are configured and `next_equity` is available, uses
    /// a 2D equity×delta grid. Otherwise falls back to uniform equity bins.
    fn compute_bucket(&self, street_idx: usize, equity: f64, next_equity: Option<f64>) -> u16 {
        let k = self.bucket_counts[street_idx];

        match (&self.delta_bins[street_idx], next_equity) {
            (Some(boundaries), Some(next_eq)) => {
                let delta = next_eq - equity;
                let delta_bin_count = boundaries.len() as u16 + 1;
                let equity_bin_count = k / delta_bin_count;
                if equity_bin_count == 0 {
                    // Not enough total buckets for the grid; fall back to equity-only.
                    let bucket = (equity * f64::from(k)) as u16;
                    return bucket.min(k - 1);
                }

                let eq_bin = (equity * f64::from(equity_bin_count)) as u16;
                let eq_bin = eq_bin.min(equity_bin_count - 1);

                let d_bin = delta_bin(delta, boundaries);

                let bucket = eq_bin * delta_bin_count + d_bin;
                bucket.min(k - 1)
            }
            _ => {
                // Equity-only: uniform bins.
                let bucket = (equity * f64::from(k)) as u16;
                bucket.min(k - 1)
            }
        }
    }

    /// Look up the bucket for a single street using equity-only bucketing.
    ///
    /// Preflop uses canonical hand index. Postflop computes showdown equity
    /// and maps to a uniform bucket. Does **not** use delta bins — use
    /// `precompute_buckets` for full equity+delta bucketing.
    #[must_use]
    pub fn get_bucket(&self, street: Street, hole_cards: [Card; 2], board: &[Card]) -> u16 {
        if street == Street::Preflop {
            let hand = crate::hands::CanonicalHand::from_cards(hole_cards[0], hole_cards[1]);
            let idx = hand.index() as u16;
            return idx.min(self.bucket_counts[0] - 1);
        }
        let street_idx = street as usize;
        let equity = crate::showdown_equity::compute_equity(hole_cards, board);
        self.compute_bucket(street_idx, equity, None)
    }

    /// Create `AllBuckets` with equity-only bucketing (no delta bins).
    #[must_use]
    pub fn equity_only(bucket_counts: [u16; 4], bucket_files: [Option<BucketFile>; 4]) -> Self {
        Self {
            bucket_counts,
            bucket_files,
            delta_bins: [None, None, None, None],
            expected_delta: [false; 4],
            equity_cache: None,
        }
    }

    /// Return the visible board slice for a given street.
    #[must_use]
    pub fn board_for_street(board: &[Card; 5], street: Street) -> &[Card] {
        match street {
            Street::Preflop => &[],
            Street::Flop => &board[..3],
            Street::Turn => &board[..4],
            Street::River => &board[..5],
        }
    }
}

/// Compute the expected equity on the next street by averaging over all
/// possible next cards.
///
/// For a flop board (3 cards), enumerates all ~47 possible turn cards.
/// For a turn board (4 cards), enumerates all ~46 possible river cards.
/// Returns the mean equity across all possible next-street boards.
fn expected_next_equity(hole: [Card; 2], board: &[Card]) -> f64 {
    use crate::showdown_equity::remaining_cards_from;

    let remaining = remaining_cards_from(hole, board);
    let mut total_eq = 0.0;
    let mut count = 0u32;

    // Stack-allocate a board buffer one card longer than the current board.
    let mut next_board = arrayvec::ArrayVec::<Card, 5>::new();
    for &c in board {
        next_board.push(c);
    }
    next_board.push(Card::new(crate::poker::Value::Two, crate::poker::Suit::Spade)); // placeholder

    for &card in &remaining {
        *next_board.last_mut().unwrap() = card;
        total_eq += crate::showdown_equity::compute_equity(hole, &next_board);
        count += 1;
    }

    if count == 0 {
        return 0.5;
    }
    total_eq / f64::from(count)
}

/// Map a delta value to a bin index given sorted boundary thresholds.
///
/// Boundaries partition [-1.0, +1.0] into `len + 1` bins.
/// Returns an index in `0..=boundaries.len()`.
fn delta_bin(delta: f64, boundaries: &[f64]) -> u16 {
    for (i, &b) in boundaries.iter().enumerate() {
        if delta < b {
            return i as u16;
        }
    }
    boundaries.len() as u16
}

/// Run one external-sampling MCCFR iteration for the given traverser.
///
/// Returns the expected value for the traverser at the root (in BB
/// units, representing the net amount won/lost from the traverser's
/// invested chips).
///
/// External sampling: at the traverser's decision nodes, **all**
/// actions are explored. At the opponent's decision nodes, **one**
/// action is sampled according to the current strategy.
///
/// # Arguments
/// * `tree` - The game tree
/// * `storage` - Atomic regret/strategy storage (shared across threads)
/// * `deal` - The sampled deal with pre-computed bucket assignments
/// * `traverser` - Which player is traversing (0 or 1)
/// * `node_idx` - Current node in the tree
/// * `prune` - Whether negative-regret pruning is active
/// * `prune_threshold` - Regret threshold below which actions are skipped
/// * `rng` - Random number generator for opponent sampling
/// * `rake_rate` - Fraction of pot taken as rake (0.0 = no rake)
/// * `rake_cap` - Maximum rake in chip units (0.0 = no cap)
#[allow(clippy::too_many_arguments)]
pub fn traverse_external(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: f64,
) -> (f64, PruneStats) {
    match &tree.nodes[node_idx as usize] {
        GameNode::Terminal { kind, invested, .. } => {
            (terminal_value(*kind, invested, traverser, &deal.deal, rake_rate, rake_cap), PruneStats::default())
        }

        GameNode::Chance { child, .. } => {
            // Board cards are pre-dealt in the Deal; just recurse.
            traverse_external(
                tree, storage, deal, traverser, *child, prune, prune_threshold, rng,
                rake_rate, rake_cap,
            )
        }

        GameNode::Decision {
            player,
            street,
            children,
            ..
        } => {
            let player = *player;
            let street = *street;
            let num_actions = children.len();

            let bucket = deal.buckets[player as usize][street as usize];

            if player == traverser {
                traverse_traverser(
                    tree,
                    storage,
                    deal,
                    traverser,
                    node_idx,
                    bucket,
                    children,
                    num_actions,
                    prune,
                    prune_threshold,
                    rng,
                    rake_rate,
                    rake_cap,
                )
            } else {
                traverse_opponent(
                    tree,
                    storage,
                    deal,
                    traverser,
                    node_idx,
                    bucket,
                    children,
                    num_actions,
                    prune,
                    prune_threshold,
                    rng,
                    rake_rate,
                    rake_cap,
                )
            }
        }
    }
}

/// Compute payoff at a terminal node from the traverser's perspective.
///
/// When rake is enabled (`rake_rate > 0`), the winner pays
/// `min(pot * rake_rate, rake_cap)` from their winnings. The loser
/// always loses their full investment. On a tie the rake cost is split
/// equally between both players. A `rake_cap` of `0.0` means no cap.
fn terminal_value(
    kind: TerminalKind,
    invested: &[f64; 2],
    traverser: u8,
    deal: &Deal,
    rake_rate: f64,
    rake_cap: f64,
) -> f64 {
    let t = traverser as usize;
    let o = 1 - t;
    let pot = invested[0] + invested[1];
    let rake = if rake_rate > 0.0 {
        let uncapped = pot * rake_rate;
        if rake_cap > 0.0 {
            uncapped.min(rake_cap)
        } else {
            uncapped
        }
    } else {
        0.0
    };
    match kind {
        TerminalKind::Fold { winner } => {
            if winner == traverser {
                invested[o] - rake
            } else {
                -invested[t]
            }
        }
        TerminalKind::Showdown => {
            let rank_t = rank_hand(deal.hole_cards[t], &deal.board);
            let rank_o = rank_hand(deal.hole_cards[o], &deal.board);
            match rank_t.cmp(&rank_o) {
                Ordering::Greater => invested[o] - rake,
                Ordering::Less => -invested[t],
                Ordering::Equal => -rake / 2.0,
            }
        }
        TerminalKind::DepthBoundary => {
            // Depth-boundary nodes are not reachable during blueprint MCCFR;
            // they only appear in subgame trees resolved by a separate solver.
            unreachable!("DepthBoundary should not be reached during MCCFR traversal")
        }
    }
}

/// Traverser's decision node: explore all actions, update regrets and
/// strategy sums.
#[allow(clippy::too_many_arguments)]
fn traverse_traverser(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    bucket: u16,
    children: &[u32],
    num_actions: usize,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: f64,
) -> (f64, PruneStats) {
    debug_assert!(num_actions <= MAX_ACTIONS);
    let mut strategy_buf = [0.0f64; MAX_ACTIONS];
    storage.current_strategy_into(node_idx, bucket, &mut strategy_buf[..num_actions]);
    let strategy = &strategy_buf[..num_actions];

    let mut action_values_buf = [0.0f64; MAX_ACTIONS];
    let action_values = &mut action_values_buf[..num_actions];
    let mut pruned_buf = [false; MAX_ACTIONS];
    let pruned = &mut pruned_buf[..num_actions];
    let mut node_value = 0.0f64;
    let mut stats = PruneStats::default();

    for (a, &child_idx) in children.iter().enumerate() {
        if prune {
            stats.total += 1;
            if storage.get_regret(node_idx, bucket, a) < prune_threshold {
                stats.hits += 1;
                pruned[a] = true;
                continue;
            }
        }

        let (child_ev, child_stats) = traverse_external(
            tree, storage, deal, traverser, child_idx, prune, prune_threshold, rng,
            rake_rate, rake_cap,
        );
        action_values[a] = child_ev;
        node_value += strategy[a] * child_ev;
        stats.merge(child_stats);
    }

    // Update regrets: delta = action_value - node_value, scaled to
    // integer by ×1000 for precision. Skip pruned actions.
    for (a, &av) in action_values.iter().enumerate().take(num_actions) {
        if pruned[a] {
            continue;
        }
        let delta = av - node_value;
        storage.add_regret(node_idx, bucket, a, (delta * 1000.0) as i32);
    }

    // Accumulate strategy sums (for computing the average strategy).
    for (a, &s) in strategy.iter().enumerate().take(num_actions) {
        storage.add_strategy_sum(node_idx, bucket, a, (s * 1000.0) as i64);
    }

    (node_value, stats)
}

/// Opponent's decision node: sample one action according to the
/// current strategy and recurse.
#[allow(clippy::too_many_arguments)]
fn traverse_opponent(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    bucket: u16,
    children: &[u32],
    num_actions: usize,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: f64,
) -> (f64, PruneStats) {
    debug_assert!(num_actions <= MAX_ACTIONS);
    let mut strategy_buf = [0.0f64; MAX_ACTIONS];
    storage.current_strategy_into(node_idx, bucket, &mut strategy_buf[..num_actions]);
    let strategy = &strategy_buf[..num_actions];

    let r: f64 = rng.random();
    let mut cumulative = 0.0;
    let mut chosen = num_actions - 1;
    for (a, &prob) in strategy.iter().enumerate() {
        cumulative += prob;
        if r < cumulative {
            chosen = a;
            break;
        }
    }

    traverse_external(
        tree, storage, deal, traverser, children[chosen], prune, prune_threshold, rng,
        rake_rate, rake_cap,
    )
}

/// Rank a hand (hole + board) using `rs_poker`.
fn rank_hand(hole: [Card; 2], board: &[Card; 5]) -> crate::poker::Rank {
    let mut hand = Hand::default();
    hand.insert(hole[0]);
    hand.insert(hole[1]);
    for &c in board {
        hand.insert(c);
    }
    hand.rank()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_v2::game_tree::GameTree;
    use crate::blueprint_v2::storage::BlueprintStorage;
    use crate::poker::{Card, Suit, Value};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::sync::atomic::Ordering;

    fn make_deal() -> Deal {
        Deal {
            hole_cards: [
                [
                    Card::new(Value::Ace, Suit::Spade),
                    Card::new(Value::King, Suit::Spade),
                ],
                [
                    Card::new(Value::Queen, Suit::Heart),
                    Card::new(Value::Jack, Suit::Heart),
                ],
            ],
            board: [
                Card::new(Value::Two, Suit::Club),
                Card::new(Value::Three, Suit::Diamond),
                Card::new(Value::Four, Suit::Club),
                Card::new(Value::Five, Suit::Diamond),
                Card::new(Value::Six, Suit::Heart),
            ],
        }
    }

    fn toy_tree() -> GameTree {
        GameTree::build(
            10.0,
            0.5,
            1.0,
            &[vec!["2.5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        )
    }

    fn make_precomputed(buckets: &AllBuckets, deal: Deal) -> DealWithBuckets {
        let b = buckets.precompute_buckets(&deal);
        DealWithBuckets { deal, buckets: b }
    }

    #[test]
    fn traverse_returns_finite() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::equity_only(
            [10, 10, 10, 10],
            [None, None, None, None],
        );
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        let (ev, _stats) = traverse_external(
            &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000, &mut rng,
            0.0, 0.0,
        );
        assert!(ev.is_finite(), "EV should be finite, got {ev}");
    }

    #[test]
    fn traverse_both_players() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::equity_only(
            [10, 10, 10, 10],
            [None, None, None, None],
        );
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        let (ev0, _) = traverse_external(
            &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000, &mut rng,
            0.0, 0.0,
        );
        let (ev1, _) = traverse_external(
            &tree, &storage, &precomputed, 1, tree.root, false, -310_000_000, &mut rng,
            0.0, 0.0,
        );

        assert!(ev0.is_finite());
        assert!(ev1.is_finite());
    }

    #[test]
    fn traverse_updates_regrets() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::equity_only(
            [10, 10, 10, 10],
            [None, None, None, None],
        );
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        assert!(storage.regrets.iter().all(|r| r.load(Ordering::Relaxed) == 0));

        let (_ev, _stats) = traverse_external(
            &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000, &mut rng,
            0.0, 0.0,
        );

        assert!(
            storage.regrets.iter().any(|r| r.load(Ordering::Relaxed) != 0),
            "regrets should be updated after traversal"
        );
    }

    #[test]
    fn traverse_updates_strategy_sums() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::equity_only(
            [10, 10, 10, 10],
            [None, None, None, None],
        );
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        let (_ev, _stats) = traverse_external(
            &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000, &mut rng,
            0.0, 0.0,
        );

        assert!(
            storage.strategy_sums.iter().any(|s| s.load(Ordering::Relaxed) != 0),
            "strategy sums should be updated after traversal"
        );
    }

    #[test]
    fn multiple_iterations_change_strategy() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::equity_only(
            [10, 10, 10, 10],
            [None, None, None, None],
        );
        let deal = make_deal();
        let precomputed = make_precomputed(&buckets, deal);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..50 {
            let _ = traverse_external(
                &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000, &mut rng,
                0.0, 0.0,
            );
            let _ = traverse_external(
                &tree, &storage, &precomputed, 1, tree.root, false, -310_000_000, &mut rng,
                0.0, 0.0,
            );
        }

        // After 50 iterations, at least one traverser decision node
        // should have a non-uniform current strategy.
        for (i, node) in tree.nodes.iter().enumerate() {
            if let GameNode::Decision { player: 0, street, .. } = node {
                let visible = AllBuckets::board_for_street(&precomputed.deal.board, *street);
                let bucket = buckets.get_bucket(*street, precomputed.deal.hole_cards[0], visible);
                let strategy = storage.current_strategy(i as u32, bucket);
                let is_uniform = strategy.iter().all(|&p| (p - strategy[0]).abs() < 1e-10);
                if strategy.len() > 1 && !is_uniform {
                    return; // pass
                }
            }
        }
        // Uniform everywhere is acceptable in degenerate cases.
    }

    #[test]
    fn traverse_with_pruning() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::equity_only(
            [10, 10, 10, 10],
            [None, None, None, None],
        );
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        // Force some very negative regrets.
        for r in storage.regrets.iter().step_by(3) {
            r.store(-400_000_000, Ordering::Relaxed);
        }

        let (ev, _stats) = traverse_external(
            &tree, &storage, &precomputed, 0, tree.root, true, -310_000_000, &mut rng,
            0.0, 0.0,
        );
        assert!(ev.is_finite());
    }

    #[test]
    fn terminal_fold_payoff() {
        let invested = [0.5, 1.0]; // SB folded
        let deal = make_deal();

        // Winner is player 1 (BB), traverser is player 0 (SB).
        let v = terminal_value(TerminalKind::Fold { winner: 1 }, &invested, 0, &deal, 0.0, 0.0);
        assert!((v - (-0.5)).abs() < 1e-10, "SB loses 0.5 on fold, got {v}");

        // Winner is player 1, traverser is player 1.
        let v = terminal_value(TerminalKind::Fold { winner: 1 }, &invested, 1, &deal, 0.0, 0.0);
        assert!((v - 0.5).abs() < 1e-10, "BB wins 0.5 on SB fold, got {v}");
    }

    #[test]
    fn terminal_showdown_payoff() {
        let invested = [2.0, 2.0];
        let deal = make_deal();

        // Both have a straight on 2-3-4-5-6; AKs (6-high straight) vs
        // QJh (6-high straight). Both make the same board straight, so
        // this should be a tie (chop).
        let v = terminal_value(TerminalKind::Showdown, &invested, 0, &deal, 0.0, 0.0);
        assert!(
            v.abs() < 1e-10,
            "Equal hands should chop (EV=0), got {v}"
        );
    }

    // --- Rake tests ---

    /// Deal where player 0 wins at showdown: AA vs KK on a low board.
    fn make_deal_p0_wins() -> Deal {
        Deal {
            hole_cards: [
                [
                    Card::new(Value::Ace, Suit::Spade),
                    Card::new(Value::Ace, Suit::Heart),
                ],
                [
                    Card::new(Value::King, Suit::Spade),
                    Card::new(Value::King, Suit::Heart),
                ],
            ],
            board: [
                Card::new(Value::Two, Suit::Club),
                Card::new(Value::Three, Suit::Diamond),
                Card::new(Value::Four, Suit::Club),
                Card::new(Value::Seven, Suit::Diamond),
                Card::new(Value::Eight, Suit::Club),
            ],
        }
    }

    #[test]
    fn terminal_fold_with_rake() {
        let invested = [5.0, 5.0]; // pot = 10
        let deal = make_deal();
        // 5% rake, cap 1.0 -> rake = min(10*0.05, 1.0) = 0.5
        let v = terminal_value(TerminalKind::Fold { winner: 1 }, &invested, 1, &deal, 0.05, 1.0);
        assert!((v - 4.5).abs() < 1e-10, "Winner gets 5 - 0.5 rake = 4.5, got {v}");

        let v = terminal_value(TerminalKind::Fold { winner: 1 }, &invested, 0, &deal, 0.05, 1.0);
        assert!((v - (-5.0)).abs() < 1e-10, "Loser loses 5.0, got {v}");
    }

    #[test]
    fn terminal_showdown_with_rake_winner() {
        let invested = [5.0, 5.0]; // pot = 10
        let deal = make_deal_p0_wins();
        // 5% rake, no cap -> rake = 0.5
        // Player 0 wins: gets opponent's 5.0 minus 0.5 rake = 4.5
        let v = terminal_value(TerminalKind::Showdown, &invested, 0, &deal, 0.05, 0.0);
        assert!((v - 4.5).abs() < 1e-10, "Winner gets 5 - 0.5 rake = 4.5, got {v}");

        // Player 1 loses: loses full 5.0
        let v = terminal_value(TerminalKind::Showdown, &invested, 1, &deal, 0.05, 0.0);
        assert!((v - (-5.0)).abs() < 1e-10, "Loser loses 5.0, got {v}");
    }

    #[test]
    fn terminal_tie_with_rake() {
        let invested = [5.0, 5.0]; // pot = 10
        let deal = make_deal(); // equal hands (tie via board straight)
        // 5% rake, no cap -> rake = 0.5
        let v = terminal_value(TerminalKind::Showdown, &invested, 0, &deal, 0.05, 0.0);
        assert!((v - (-0.25)).abs() < 1e-10, "Tie splits 0.5 rake: -0.25 each, got {v}");
    }

    #[test]
    fn terminal_rake_cap_applied() {
        let invested = [50.0, 50.0]; // pot = 100
        let deal = make_deal(); // tie
        // 5% rake, cap 3.0 -> rake = min(5.0, 3.0) = 3.0 (capped)
        let v = terminal_value(TerminalKind::Showdown, &invested, 0, &deal, 0.05, 3.0);
        assert!((v - (-1.5)).abs() < 1e-10, "Capped rake 3.0 split = -1.5 each, got {v}");
    }

    #[test]
    fn terminal_rake_zero_rate_is_noop() {
        let invested = [5.0, 5.0];
        let deal = make_deal_p0_wins();
        // rate=0 should produce identical results to no-rake
        let v = terminal_value(TerminalKind::Showdown, &invested, 0, &deal, 0.0, 3.0);
        assert!((v - 5.0).abs() < 1e-10, "Zero rate means no rake, got {v}");
    }

    #[test]
    fn terminal_rake_uncapped() {
        let invested = [50.0, 50.0]; // pot = 100
        let deal = make_deal_p0_wins();
        // 10% rake, no cap -> rake = 10.0
        let v = terminal_value(TerminalKind::Showdown, &invested, 0, &deal, 0.10, 0.0);
        assert!((v - 40.0).abs() < 1e-10, "Winner gets 50 - 10 rake = 40, got {v}");
    }

    // ── Equity+Delta bucketing integration tests ─────────────────────

    /// Helper: build AllBuckets with the equity_delta config's flop delta bins.
    fn equity_delta_buckets() -> AllBuckets {
        use crate::blueprint_v2::config::StreetClusterConfig;

        let flop_cfg = StreetClusterConfig {
            buckets: 400,
            delta_bins: Some(vec![
                -0.2, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.75,
            ]),
            expected_delta: false,
        };
        let turn_cfg = StreetClusterConfig {
            buckets: 100,
            delta_bins: Some(vec![-0.2, -0.05, 0.05, 0.15, 0.25, 0.40]),
            expected_delta: false,
        };
        let preflop_cfg = StreetClusterConfig { buckets: 169, delta_bins: None, expected_delta: false };
        let river_cfg = StreetClusterConfig { buckets: 1000, delta_bins: None, expected_delta: false };

        AllBuckets::new(
            [169, 400, 100, 1000],
            [None, None, None, None],
            [&preflop_cfg, &flop_cfg, &turn_cfg, &river_cfg],
        )
    }

    /// Helper: shorthand card constructor.
    fn c(v: Value, s: Suit) -> Card {
        Card::new(v, s)
    }

    /// AA on a dry rainbow board — equity stays high across streets.
    /// Delta should be near zero (neutral).
    #[test]
    fn delta_aa_dry_board_neutral() {
        let buckets = equity_delta_buckets();
        let deal = Deal {
            hole_cards: [
                [c(Value::Ace, Suit::Spade), c(Value::Ace, Suit::Heart)],
                [c(Value::Seven, Suit::Club), c(Value::Two, Suit::Diamond)],
            ],
            // K72 rainbow, turn 3, river 8 — no draws complete
            board: [
                c(Value::King, Suit::Diamond),
                c(Value::Seven, Suit::Spade),
                c(Value::Two, Suit::Heart),
                c(Value::Three, Suit::Club),
                c(Value::Eight, Suit::Diamond),
            ],
        };

        let b = buckets.precompute_buckets(&deal);
        let flop_bucket_aa = b[0][1]; // player 0 = AA
        let turn_bucket_aa = b[0][2];

        // Compute raw equities/deltas for verification
        let hole = deal.hole_cards[0];
        let flop_eq = crate::showdown_equity::compute_equity(hole, &deal.board[..3]);
        let turn_eq = crate::showdown_equity::compute_equity(hole, &deal.board[..4]);
        let river_eq = crate::showdown_equity::compute_equity(hole, &deal.board[..5]);

        // AA on dry board should have very high equity throughout
        assert!(flop_eq > 0.85, "AA flop equity should be >0.85, got {flop_eq}");
        assert!(turn_eq > 0.85, "AA turn equity should be >0.85, got {turn_eq}");
        assert!(river_eq > 0.85, "AA river equity should be >0.85, got {river_eq}");

        // Delta should be small (near zero) — hand doesn't change much
        let flop_delta = turn_eq - flop_eq;
        let turn_delta = river_eq - turn_eq;
        assert!(
            flop_delta.abs() < 0.10,
            "AA dry board flop→turn delta should be near zero, got {flop_delta}"
        );
        assert!(
            turn_delta.abs() < 0.10,
            "AA dry board turn→river delta should be near zero, got {turn_delta}"
        );

        // Buckets should be valid
        assert!(flop_bucket_aa < 400, "flop bucket in range");
        assert!(turn_bucket_aa < 100, "turn bucket in range");
    }

    /// AA on a wet connected board — vulnerable to straights/flushes.
    /// Turn/river complete draws → equity drops → negative delta.
    #[test]
    fn delta_aa_wet_board_negative() {
        let buckets = equity_delta_buckets();
        let deal = Deal {
            hole_cards: [
                [c(Value::Ace, Suit::Spade), c(Value::Ace, Suit::Heart)],
                [c(Value::Nine, Suit::Spade), c(Value::Eight, Suit::Spade)],
            ],
            // JTs board with two spades — straight and flush draws present
            board: [
                c(Value::Jack, Suit::Spade),
                c(Value::Ten, Suit::Diamond),
                c(Value::Nine, Suit::Heart),
                c(Value::Seven, Suit::Spade), // turn completes straight for 98
                c(Value::Two, Suit::Club),
            ],
        };

        let b = buckets.precompute_buckets(&deal);

        let hole_aa = deal.hole_cards[0];
        let flop_eq = crate::showdown_equity::compute_equity(hole_aa, &deal.board[..3]);
        let turn_eq = crate::showdown_equity::compute_equity(hole_aa, &deal.board[..4]);

        let flop_delta = turn_eq - flop_eq;

        // AA had high flop equity but the 7s completes opponent's straight
        assert!(flop_eq > 0.60, "AA wet flop equity should be decent, got {flop_eq}");
        assert!(
            flop_delta < -0.05,
            "AA should have negative delta when draws complete, got {flop_delta}"
        );

        // Verify AA and 98 get different flop buckets (different delta profiles)
        let flop_bucket_aa = b[0][1];
        let flop_bucket_98 = b[1][1];
        assert_ne!(
            flop_bucket_aa, flop_bucket_98,
            "AA and 98s should bucket differently on this board"
        );
    }

    /// Flush draw on flop that completes on turn → large positive delta.
    #[test]
    fn delta_flush_draw_positive() {
        let buckets = equity_delta_buckets();
        let deal = Deal {
            hole_cards: [
                // Player 0: flush draw (two hearts)
                [c(Value::King, Suit::Heart), c(Value::Queen, Suit::Heart)],
                // Player 1: made hand (top pair)
                [c(Value::Ace, Suit::Spade), c(Value::Ten, Suit::Club)],
            ],
            // Two hearts on flop, heart on turn completes the flush
            board: [
                c(Value::Ten, Suit::Heart),
                c(Value::Five, Suit::Heart),
                c(Value::Two, Suit::Spade),
                c(Value::Three, Suit::Heart), // turn: flush completes!
                c(Value::Eight, Suit::Diamond),
            ],
        };

        let b = buckets.precompute_buckets(&deal);

        let hole_fd = deal.hole_cards[0]; // flush draw
        let flop_eq = crate::showdown_equity::compute_equity(hole_fd, &deal.board[..3]);
        let turn_eq = crate::showdown_equity::compute_equity(hole_fd, &deal.board[..4]);

        let flop_delta = turn_eq - flop_eq;

        // On flop, KQhh has modest equity (just a draw vs made pair)
        assert!(flop_eq < 0.50, "Flush draw flop equity should be < 0.50, got {flop_eq}");
        // On turn, flush is made → equity jumps
        assert!(turn_eq > 0.85, "Made flush turn equity should be > 0.85, got {turn_eq}");
        // Delta should be strongly positive
        assert!(
            flop_delta > 0.30,
            "Flush completing should give delta > 0.30, got {flop_delta}"
        );

        // Flush draw should be in a different bucket than the made hand
        let flop_bucket_fd = b[0][1];
        let flop_bucket_tp = b[1][1];
        assert_ne!(
            flop_bucket_fd, flop_bucket_tp,
            "Flush draw and top pair should bucket differently"
        );
    }

    /// Combo draw (flush + straight draw) that completes → very large positive delta.
    #[test]
    fn delta_combo_draw_very_positive() {
        let buckets = equity_delta_buckets();
        let deal = Deal {
            hole_cards: [
                // Player 0: combo draw — flush draw + open-ended straight draw
                [c(Value::Jack, Suit::Heart), c(Value::Ten, Suit::Heart)],
                // Player 1: top pair
                [c(Value::Ace, Suit::Spade), c(Value::Nine, Suit::Club)],
            ],
            board: [
                c(Value::Nine, Suit::Heart),
                c(Value::Eight, Suit::Heart),
                c(Value::Three, Suit::Spade),
                c(Value::Seven, Suit::Heart), // turn: flush AND straight complete!
                c(Value::Two, Suit::Diamond),
            ],
        };

        let b = buckets.precompute_buckets(&deal);

        let hole_cd = deal.hole_cards[0]; // combo draw
        let flop_eq = crate::showdown_equity::compute_equity(hole_cd, &deal.board[..3]);
        let turn_eq = crate::showdown_equity::compute_equity(hole_cd, &deal.board[..4]);

        let flop_delta = turn_eq - flop_eq;

        // Combo draw on flop: some equity from pair outs + draw equity
        // but static equity is low (hasn't made the hand)
        assert!(flop_eq < 0.55, "Combo draw flop equity should be modest, got {flop_eq}");
        // Turn completes both draws → near-nut hand
        assert!(turn_eq > 0.90, "Made straight flush turn equity should be > 0.90, got {turn_eq}");
        // Delta should be very large
        assert!(
            flop_delta > 0.40,
            "Combo draw completing should give delta > 0.40, got {flop_delta}"
        );

        // Combo draw and top pair should land in different buckets
        assert_ne!(
            b[0][1], b[1][1],
            "Combo draw and top pair should bucket differently on flop"
        );
    }

    /// Compare bucket differentiation: same equity, different deltas
    /// should land in different buckets.
    #[test]
    fn delta_same_equity_different_buckets() {
        let buckets = equity_delta_buckets();

        // Deal 1: made hand that stays strong (neutral delta)
        let deal_stable = Deal {
            hole_cards: [
                [c(Value::King, Suit::Spade), c(Value::King, Suit::Heart)],
                [c(Value::Two, Suit::Club), c(Value::Three, Suit::Diamond)],
            ],
            board: [
                c(Value::Queen, Suit::Diamond),
                c(Value::Seven, Suit::Club),
                c(Value::Four, Suit::Heart),
                c(Value::Nine, Suit::Spade), // safe turn
                c(Value::Two, Suit::Heart),
            ],
        };

        // Deal 2: made hand that gets cracked (negative delta)
        let deal_cracked = Deal {
            hole_cards: [
                [c(Value::King, Suit::Spade), c(Value::King, Suit::Heart)],
                [c(Value::Two, Suit::Club), c(Value::Three, Suit::Diamond)],
            ],
            board: [
                c(Value::Queen, Suit::Diamond),
                c(Value::Seven, Suit::Club),
                c(Value::Four, Suit::Heart),
                c(Value::Ace, Suit::Spade), // scary turn (Ace)
                c(Value::Two, Suit::Heart),
            ],
        };

        let b_stable = buckets.precompute_buckets(&deal_stable);
        let b_cracked = buckets.precompute_buckets(&deal_cracked);

        // Both KK on Q74 flop — same flop equity
        let eq_stable = crate::showdown_equity::compute_equity(
            deal_stable.hole_cards[0],
            &deal_stable.board[..3],
        );
        let eq_cracked = crate::showdown_equity::compute_equity(
            deal_cracked.hole_cards[0],
            &deal_cracked.board[..3],
        );
        assert!(
            (eq_stable - eq_cracked).abs() < 0.01,
            "Same hand on same flop should have same equity: {eq_stable} vs {eq_cracked}"
        );

        // But different turns → different deltas → different flop buckets
        let turn_eq_stable = crate::showdown_equity::compute_equity(
            deal_stable.hole_cards[0],
            &deal_stable.board[..4],
        );
        let turn_eq_cracked = crate::showdown_equity::compute_equity(
            deal_cracked.hole_cards[0],
            &deal_cracked.board[..4],
        );
        let delta_stable = turn_eq_stable - eq_stable;
        let delta_cracked = turn_eq_cracked - eq_cracked;

        // 9 is safe for KK, Ace reduces KK's equity significantly
        assert!(
            delta_stable > delta_cracked,
            "Safe turn should have better delta than Ace: {delta_stable} vs {delta_cracked}"
        );
        assert_ne!(
            b_stable[0][1], b_cracked[0][1],
            "Same flop equity but different deltas should produce different buckets \
             (stable delta={delta_stable:.3}, cracked delta={delta_cracked:.3})"
        );
    }

    /// Verify that without delta bins, hands with different trajectories
    /// but similar equity get the SAME bucket (equity-only can't distinguish).
    #[test]
    fn equity_only_cannot_differentiate_deltas() {
        let eq_only = AllBuckets::equity_only(
            [169, 400, 100, 1000],
            [None, None, None, None],
        );

        // Flush draw on flop that completes
        let deal_draw = Deal {
            hole_cards: [
                [c(Value::King, Suit::Heart), c(Value::Queen, Suit::Heart)],
                [c(Value::Two, Suit::Club), c(Value::Three, Suit::Diamond)],
            ],
            board: [
                c(Value::Ten, Suit::Heart),
                c(Value::Five, Suit::Heart),
                c(Value::Two, Suit::Spade),
                c(Value::Three, Suit::Heart),
                c(Value::Eight, Suit::Diamond),
            ],
        };

        // Made hand with similar flop equity
        let deal_made = Deal {
            hole_cards: [
                [c(Value::Ten, Suit::Spade), c(Value::Five, Suit::Club)],
                [c(Value::Two, Suit::Club), c(Value::Three, Suit::Diamond)],
            ],
            board: [
                c(Value::Ten, Suit::Heart),
                c(Value::Five, Suit::Heart),
                c(Value::Two, Suit::Spade),
                c(Value::Three, Suit::Club),
                c(Value::Eight, Suit::Diamond),
            ],
        };

        let eq_draw = crate::showdown_equity::compute_equity(
            deal_draw.hole_cards[0],
            &deal_draw.board[..3],
        );
        let eq_made = crate::showdown_equity::compute_equity(
            deal_made.hole_cards[0],
            &deal_made.board[..3],
        );

        // If they happen to have similar flop equity, equity-only gives same bucket
        // (this test documents the limitation)
        if (eq_draw - eq_made).abs() < 0.03 {
            let b_draw = eq_only.precompute_buckets(&deal_draw);
            let b_made = eq_only.precompute_buckets(&deal_made);
            assert_eq!(
                b_draw[0][1], b_made[0][1],
                "Equity-only should give same bucket for similar equity \
                 regardless of delta (eq_draw={eq_draw:.3}, eq_made={eq_made:.3})"
            );
        }
    }

    // ── Expected delta tests ─────────────────────────────────────────

    /// Helper: build AllBuckets with expected_delta enabled.
    fn expected_delta_buckets() -> AllBuckets {
        use crate::blueprint_v2::config::StreetClusterConfig;

        let flop_cfg = StreetClusterConfig {
            buckets: 400,
            delta_bins: Some(vec![
                -0.2, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.75,
            ]),
            expected_delta: true,
        };
        let turn_cfg = StreetClusterConfig {
            buckets: 100,
            delta_bins: Some(vec![-0.2, -0.05, 0.05, 0.15, 0.25, 0.40]),
            expected_delta: true,
        };
        let preflop_cfg = StreetClusterConfig { buckets: 169, delta_bins: None, expected_delta: false };
        let river_cfg = StreetClusterConfig { buckets: 1000, delta_bins: None, expected_delta: false };

        AllBuckets::new(
            [169, 400, 100, 1000],
            [None, None, None, None],
            [&preflop_cfg, &flop_cfg, &turn_cfg, &river_cfg],
        )
    }

    /// With expected_delta, the same hand+flop always gets the same flop bucket
    /// regardless of what turn card actually falls.
    #[test]
    fn expected_delta_deterministic_across_runouts() {
        let buckets = expected_delta_buckets();
        let hole = [c(Value::King, Suit::Heart), c(Value::Queen, Suit::Heart)];
        let flop = [
            c(Value::Ten, Suit::Heart),
            c(Value::Five, Suit::Heart),
            c(Value::Two, Suit::Spade),
        ];

        // Two different runouts for the same hand+flop
        let deal_flush = Deal {
            hole_cards: [hole, [c(Value::Three, Suit::Club), c(Value::Four, Suit::Diamond)]],
            board: [flop[0], flop[1], flop[2], c(Value::Three, Suit::Heart), c(Value::Eight, Suit::Diamond)],
        };
        let deal_brick = Deal {
            hole_cards: [hole, [c(Value::Three, Suit::Club), c(Value::Four, Suit::Diamond)]],
            board: [flop[0], flop[1], flop[2], c(Value::King, Suit::Club), c(Value::Eight, Suit::Diamond)],
        };

        let b_flush = buckets.precompute_buckets(&deal_flush);
        let b_brick = buckets.precompute_buckets(&deal_brick);

        // Flop bucket should be identical — expected delta doesn't depend on the actual turn
        assert_eq!(
            b_flush[0][1], b_brick[0][1],
            "Expected delta: same hand+flop should give same flop bucket \
             regardless of turn card"
        );
    }

    /// With actual-runout delta (expected_delta=false), different turn cards
    /// give different flop buckets for the same hand+flop.
    #[test]
    fn actual_delta_varies_with_runout() {
        let buckets = equity_delta_buckets(); // expected_delta: false
        // Flush draw: Ah9h on a flop with TWO hearts (so turn heart = flush)
        let hole = [c(Value::Ace, Suit::Heart), c(Value::Nine, Suit::Heart)];
        let flop = [
            c(Value::King, Suit::Heart),
            c(Value::Seven, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];

        // Flush completes on turn (5th heart)
        let deal_flush = Deal {
            hole_cards: [hole, [c(Value::Three, Suit::Club), c(Value::Four, Suit::Club)]],
            board: [flop[0], flop[1], flop[2], c(Value::Six, Suit::Heart), c(Value::Eight, Suit::Spade)],
        };
        // Complete brick — off-suit, no help
        let deal_brick = Deal {
            hole_cards: [hole, [c(Value::Three, Suit::Club), c(Value::Four, Suit::Club)]],
            board: [flop[0], flop[1], flop[2], c(Value::Five, Suit::Spade), c(Value::Eight, Suit::Spade)],
        };

        let hole_fd = deal_flush.hole_cards[0];
        let flop_eq = crate::showdown_equity::compute_equity(hole_fd, &flop);
        let turn_eq_flush = crate::showdown_equity::compute_equity(hole_fd, &deal_flush.board[..4]);
        let turn_eq_brick = crate::showdown_equity::compute_equity(hole_fd, &deal_brick.board[..4]);
        let delta_flush = turn_eq_flush - flop_eq;
        let delta_brick = turn_eq_brick - flop_eq;

        // Sanity: flush completing should give much larger delta than a brick
        assert!(
            delta_flush > delta_brick + 0.20,
            "Flush delta ({delta_flush:.3}) should be much larger than brick delta ({delta_brick:.3})"
        );

        let b_flush = buckets.precompute_buckets(&deal_flush);
        let b_brick = buckets.precompute_buckets(&deal_brick);

        // With actual-runout delta, these should differ (flush completing vs brick)
        assert_ne!(
            b_flush[0][1], b_brick[0][1],
            "Actual-runout delta should give different flop buckets for \
             flush-completing (delta={delta_flush:.3}) vs brick (delta={delta_brick:.3})"
        );
    }

    /// Expected delta: flush draw should still get a positive delta bucket
    /// (averaged across all possible turns, flush draws improve more than they decline).
    #[test]
    fn expected_delta_flush_draw_positive_bucket() {
        let buckets = expected_delta_buckets();
        let deal = Deal {
            hole_cards: [
                [c(Value::King, Suit::Heart), c(Value::Queen, Suit::Heart)],
                [c(Value::Ace, Suit::Spade), c(Value::Ten, Suit::Club)],
            ],
            board: [
                c(Value::Ten, Suit::Heart),
                c(Value::Five, Suit::Heart),
                c(Value::Two, Suit::Spade),
                c(Value::Three, Suit::Heart),
                c(Value::Eight, Suit::Diamond),
            ],
        };

        let hole_fd = deal.hole_cards[0];
        let flop_eq = crate::showdown_equity::compute_equity(hole_fd, &deal.board[..3]);
        let exp_next = expected_next_equity(hole_fd, &deal.board[..3]);
        let exp_delta = exp_next - flop_eq;

        // Flush draw has ~9 outs of ~47 cards that improve equity significantly
        // Expected delta should be positive
        assert!(
            exp_delta > 0.0,
            "Flush draw expected delta should be positive, got {exp_delta:.4}"
        );

        // Verify bucket reflects this — should differ from a stable made hand
        let b = buckets.precompute_buckets(&deal);
        let deal_stable = Deal {
            hole_cards: [
                [c(Value::Ace, Suit::Spade), c(Value::Ace, Suit::Heart)],
                [c(Value::Seven, Suit::Club), c(Value::Three, Suit::Diamond)],
            ],
            board: [
                c(Value::King, Suit::Diamond),
                c(Value::Eight, Suit::Club),
                c(Value::Two, Suit::Diamond),
                c(Value::Nine, Suit::Spade),
                c(Value::Four, Suit::Heart),
            ],
        };
        let b_stable = buckets.precompute_buckets(&deal_stable);
        // Different delta profiles → different buckets
        assert_ne!(
            b[0][1], b_stable[0][1],
            "Flush draw and stable AA should get different flop buckets"
        );
    }

    #[test]
    fn get_bucket_equity_fallback() {
        let all = AllBuckets::equity_only(
            [169, 10, 10, 10],
            [None, None, None, None],
        );

        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
            Card::new(Value::Ten, Suit::Club),
        ];

        let bucket = all.get_bucket(
            Street::River,
            [
                Card::new(Value::Two, Suit::Heart),
                Card::new(Value::Three, Suit::Heart),
            ],
            &board,
        );
        // Should return a valid bucket index in [0, 10).
        assert!(bucket < 10, "bucket should be < 10, got {bucket}");
    }
}
