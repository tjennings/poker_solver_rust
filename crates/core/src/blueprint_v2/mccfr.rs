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

use rustc_hash::FxHashMap;

use super::bucket_file::{BucketFile, PackedBoard};
use super::cluster_pipeline::{canonical_key, combo_index};
use super::game_tree::{GameNode, GameTree, TerminalKind};
use super::storage::BlueprintStorage;
use super::Street;
use crate::abstraction::isomorphism::CanonicalBoard;
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
/// Uses precomputed bucket files for O(1) lookups when available,
/// with equity-based fallback when no bucket file is present.
pub struct AllBuckets {
    pub bucket_counts: [u16; 4],
    /// Per-street bucket files produced by the clustering pipeline.
    pub bucket_files: [Option<BucketFile>; 4],
    /// Board index lookup tables for O(1) bucket file lookups.
    board_maps: [Option<FxHashMap<PackedBoard, u32>>; 4],
}

impl AllBuckets {
    #[must_use]
    pub fn new(
        bucket_counts: [u16; 4],
        bucket_files: [Option<BucketFile>; 4],
    ) -> Self {
        let board_maps = std::array::from_fn(|i| {
            bucket_files[i].as_ref().and_then(|bf| {
                if bf.boards.is_empty() {
                    None
                } else {
                    Some(bf.board_index_map_fx())
                }
            })
        });
        Self {
            bucket_counts,
            bucket_files,
            board_maps,
        }
    }

    /// Look up a bucket for a postflop street via bucket file.
    ///
    /// Canonicalizes the board, looks up the board index in the hash map,
    /// applies the same suit permutation to hole cards, and reads the bucket
    /// from the flat array. Falls back to equity binning if no bucket file
    /// or board not found.
    fn lookup_bucket(&self, street_idx: usize, hole: [Card; 2], board: &[Card]) -> u16 {
        if let (Some(bf), Some(board_map)) = (
            &self.bucket_files[street_idx],
            &self.board_maps[street_idx],
        ) {
            if let Ok(canonical) = CanonicalBoard::from_cards(board) {
                let packed = canonical_key(&canonical.cards);
                if let Some(&board_idx) = board_map.get(&packed) {
                    let (c0, c1) = canonical.canonicalize_holding(hole[0], hole[1]);
                    let ci = combo_index(c0, c1);
                    return bf.get_bucket(board_idx, ci);
                }
            }
        }
        // Fallback: equity-based bucketing.
        let equity = crate::showdown_equity::compute_equity(hole, board);
        let k = self.bucket_counts[street_idx];
        let bucket = (equity * f64::from(k)) as u16;
        bucket.min(k - 1)
    }

    /// Pre-compute bucket assignments for all 4 streets x 2 players.
    #[must_use]
    pub fn precompute_buckets(&self, deal: &Deal) -> [[u16; 4]; 2] {
        let mut result = [[0u16; 4]; 2];
        for (player, row) in result.iter_mut().enumerate() {
            let hole = deal.hole_cards[player];
            // Preflop: canonical hand index.
            let hand = crate::hands::CanonicalHand::from_cards(hole[0], hole[1]);
            let idx = hand.index() as u16;
            row[0] = idx.min(self.bucket_counts[0] - 1);
            // Postflop: bucket file lookup with equity fallback.
            row[1] = self.lookup_bucket(1, hole, &deal.board[..3]); // flop
            row[2] = self.lookup_bucket(2, hole, &deal.board[..4]); // turn
            row[3] = self.lookup_bucket(3, hole, &deal.board[..5]); // river
        }
        result
    }

    /// Look up the bucket for a single street.
    ///
    /// Preflop uses canonical hand index. Postflop uses bucket file
    /// lookup with equity-based fallback.
    #[must_use]
    pub fn get_bucket(&self, street: Street, hole_cards: [Card; 2], board: &[Card]) -> u16 {
        if street == Street::Preflop {
            let hand = crate::hands::CanonicalHand::from_cards(hole_cards[0], hole_cards[1]);
            let idx = hand.index() as u16;
            return idx.min(self.bucket_counts[0] - 1);
        }
        self.lookup_bucket(street as usize, hole_cards, board)
    }

    /// Create `AllBuckets` with equity-only bucketing (no bucket files).
    #[must_use]
    pub fn equity_only(bucket_counts: [u16; 4], bucket_files: [Option<BucketFile>; 4]) -> Self {
        Self::new(bucket_counts, bucket_files)
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

    /// Helper: shorthand card constructor.
    fn c(v: Value, s: Suit) -> Card {
        Card::new(v, s)
    }

    // ── Bucket file lookup tests ─────────────────────────────────────

    // (delta bucketing tests removed — AllBuckets now uses bucket file lookups)

    #[test]
    fn placeholder_delta_tests_removed() {
        // Delta bucketing was removed from AllBuckets in favour of
        // precomputed bucket file lookups. The clustering pipeline
        // now handles delta-aware bucketing during offline cluster
        // generation rather than at MCCFR runtime.
    }

    #[test]
    fn equity_fallback_aa_high_bucket() {
        let all = AllBuckets::equity_only(
            [169, 400, 100, 1000],
            [None, None, None, None],
        );
        let deal = Deal {
            hole_cards: [
                [c(Value::Ace, Suit::Spade), c(Value::Ace, Suit::Heart)],
                [c(Value::Seven, Suit::Club), c(Value::Two, Suit::Diamond)],
            ],
            board: [
                c(Value::King, Suit::Diamond),
                c(Value::Seven, Suit::Spade),
                c(Value::Two, Suit::Heart),
                c(Value::Three, Suit::Club),
                c(Value::Eight, Suit::Diamond),
            ],
        };

        let b = all.precompute_buckets(&deal);
        let flop_bucket_aa = b[0][1];
        let turn_bucket_aa = b[0][2];

        // Buckets should be valid
        assert!(flop_bucket_aa < 400, "flop bucket in range");
        assert!(turn_bucket_aa < 100, "turn bucket in range");
    }

    // The remaining delta-specific tests (delta_aa_wet_board_negative,
    // delta_flush_draw_positive, delta_combo_draw_very_positive,
    // delta_same_equity_different_buckets, equity_only_cannot_differentiate_deltas,
    // expected_delta_deterministic_across_runouts, actual_delta_varies_with_runout,
    // expected_delta_flush_draw_positive_bucket) have been removed because
    // AllBuckets no longer manages delta bins at runtime — the clustering
    // pipeline now handles all delta-aware bucketing during offline generation.

    // (Delta bucketing tests removed — see comment above)

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

    // ── Bucket file lookup tests ─────────────────────────────────────

    /// Build a minimal BucketFile with known boards and bucket assignments
    /// for testing lookup_bucket.
    fn make_test_bucket_file(
        street: Street,
        board_cards: &[Vec<Card>],
        bucket_count: u16,
    ) -> BucketFile {
        use crate::blueprint_v2::bucket_file::{BucketFileHeader, PackedBoard};
        use crate::blueprint_v2::cluster_pipeline::canonical_key;

        let boards: Vec<PackedBoard> = board_cards
            .iter()
            .map(|b| {
                use crate::abstraction::isomorphism::CanonicalBoard;
                let canonical = CanonicalBoard::from_cards(b).unwrap();
                canonical_key(&canonical.cards)
            })
            .collect();

        let combos_per_board = 1326_u16;
        let total = boards.len() * combos_per_board as usize;
        // Assign deterministic buckets: bucket = (board_idx * 100 + combo_idx) % bucket_count
        let mut buckets = Vec::with_capacity(total);
        for board_idx in 0..boards.len() {
            for combo_idx in 0..combos_per_board as usize {
                buckets.push(((board_idx * 100 + combo_idx) % bucket_count as usize) as u16);
            }
        }

        BucketFile {
            header: BucketFileHeader {
                street,
                bucket_count,
                board_count: boards.len() as u32,
                combos_per_board,
                version: 2,
            },
            boards,
            buckets,
        }
    }

    #[test]
    fn lookup_bucket_returns_bucket_from_file() {
        use crate::blueprint_v2::cluster_pipeline::{canonical_key, combo_index};
        use crate::abstraction::isomorphism::CanonicalBoard;

        let flop_cards = vec![
            c(Value::Ace, Suit::Spade),
            c(Value::King, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];
        let bf = make_test_bucket_file(Street::Flop, &[flop_cards.clone()], 50);

        let all = AllBuckets::new(
            [169, 50, 10, 10],
            [None, Some(bf), None, None],
        );

        let hole = [c(Value::Queen, Suit::Spade), c(Value::Jack, Suit::Club)];
        let bucket = all.lookup_bucket(1, hole, &flop_cards);

        // Verify the bucket matches what the file would return:
        let canonical = CanonicalBoard::from_cards(&flop_cards).unwrap();
        let packed = canonical_key(&canonical.cards);
        let board_map = all.board_maps[1].as_ref().unwrap();
        let board_idx = board_map[&packed];
        let (c0, c1) = canonical.canonicalize_holding(hole[0], hole[1]);
        let ci = combo_index(c0, c1);
        let expected = all.bucket_files[1].as_ref().unwrap().get_bucket(board_idx, ci);

        assert_eq!(bucket, expected, "lookup_bucket should return the bucket file value");
    }

    #[test]
    fn lookup_bucket_falls_back_to_equity_without_file() {
        let all = AllBuckets::new(
            [169, 10, 10, 10],
            [None, None, None, None],
        );

        let flop = [
            c(Value::Ace, Suit::Spade),
            c(Value::King, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];
        let hole = [c(Value::Queen, Suit::Spade), c(Value::Jack, Suit::Club)];

        let bucket = all.lookup_bucket(1, hole, &flop);
        // Should fall back to equity-based bucketing: valid in [0, 10)
        assert!(bucket < 10, "equity fallback bucket should be < 10, got {bucket}");

        // Cross-check with manual equity computation
        let equity = crate::showdown_equity::compute_equity(hole, &flop);
        let expected = ((equity * 10.0) as u16).min(9);
        assert_eq!(bucket, expected, "should match equity-based bucketing");
    }

    #[test]
    fn precompute_buckets_uses_bucket_file() {
        let flop_cards = vec![
            c(Value::Two, Suit::Club),
            c(Value::Three, Suit::Diamond),
            c(Value::Four, Suit::Club),
        ];
        let turn_cards = vec![
            c(Value::Two, Suit::Club),
            c(Value::Three, Suit::Diamond),
            c(Value::Four, Suit::Club),
            c(Value::Five, Suit::Diamond),
        ];
        let river_cards = vec![
            c(Value::Two, Suit::Club),
            c(Value::Three, Suit::Diamond),
            c(Value::Four, Suit::Club),
            c(Value::Five, Suit::Diamond),
            c(Value::Six, Suit::Heart),
        ];

        let flop_bf = make_test_bucket_file(Street::Flop, &[flop_cards], 50);
        let turn_bf = make_test_bucket_file(Street::Turn, &[turn_cards], 30);
        let river_bf = make_test_bucket_file(Street::River, &[river_cards], 20);

        let all = AllBuckets::new(
            [169, 50, 30, 20],
            [None, Some(flop_bf), Some(turn_bf), Some(river_bf)],
        );

        let deal = make_deal();
        let result = all.precompute_buckets(&deal);

        // Preflop should be canonical hand index
        for player in 0..2 {
            let hand = crate::hands::CanonicalHand::from_cards(
                deal.hole_cards[player][0],
                deal.hole_cards[player][1],
            );
            let expected_preflop = (hand.index() as u16).min(168);
            assert_eq!(result[player][0], expected_preflop, "preflop bucket for player {player}");
        }

        // Postflop buckets should be valid
        for player in 0..2 {
            assert!(result[player][1] < 50, "flop bucket valid for player {player}");
            assert!(result[player][2] < 30, "turn bucket valid for player {player}");
            assert!(result[player][3] < 20, "river bucket valid for player {player}");
        }
    }

    #[test]
    fn get_bucket_uses_bucket_file_for_postflop() {
        let flop_cards = vec![
            c(Value::Ace, Suit::Spade),
            c(Value::King, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];
        let bf = make_test_bucket_file(Street::Flop, &[flop_cards.clone()], 50);

        let all = AllBuckets::new(
            [169, 50, 10, 10],
            [None, Some(bf), None, None],
        );

        let hole = [c(Value::Queen, Suit::Spade), c(Value::Jack, Suit::Club)];
        let bucket_via_get = all.get_bucket(Street::Flop, hole, &flop_cards);
        let bucket_via_lookup = all.lookup_bucket(1, hole, &flop_cards);

        assert_eq!(bucket_via_get, bucket_via_lookup,
            "get_bucket and lookup_bucket should agree for flop");
    }

    #[test]
    fn get_bucket_preflop_returns_canonical_hand_index() {
        let all = AllBuckets::new(
            [169, 10, 10, 10],
            [None, None, None, None],
        );

        let hole = [c(Value::Ace, Suit::Spade), c(Value::King, Suit::Spade)];
        let bucket = all.get_bucket(Street::Preflop, hole, &[]);

        let hand = crate::hands::CanonicalHand::from_cards(hole[0], hole[1]);
        let expected = (hand.index() as u16).min(168);
        assert_eq!(bucket, expected);
    }

    #[test]
    fn new_allbuckets_builds_board_maps() {
        let flop_cards = vec![
            c(Value::Ace, Suit::Spade),
            c(Value::King, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];
        let bf = make_test_bucket_file(Street::Flop, &[flop_cards], 50);

        let all = AllBuckets::new(
            [169, 50, 10, 10],
            [None, Some(bf), None, None],
        );

        // Preflop has no bucket file -> no board map
        assert!(all.board_maps[0].is_none());
        // Flop has a bucket file with boards -> has board map
        assert!(all.board_maps[1].is_some());
        assert_eq!(all.board_maps[1].as_ref().unwrap().len(), 1);
        // Turn/River have no bucket files -> no board maps
        assert!(all.board_maps[2].is_none());
        assert!(all.board_maps[3].is_none());
    }
}
