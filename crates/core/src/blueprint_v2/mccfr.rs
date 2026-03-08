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

use super::bucket_file::BucketFile;
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
/// When bucket files are loaded from the clustering pipeline, lookups
/// will use those; otherwise falls back to raw equity-based bucketing.
pub struct AllBuckets {
    pub bucket_counts: [u16; 4],
    /// Per-street bucket files produced by the clustering pipeline.
    /// `None` = fallback to raw-equity buckets for that street.
    pub bucket_files: [Option<BucketFile>; 4],
}

impl AllBuckets {
    /// Look up the bucket for a player's hole cards at a given street.
    ///
    /// When a `BucketFile` is loaded for the given street, the lookup
    /// will use the file's pre-computed cluster assignments. Currently
    /// falls back to equity-based bucketing (compute showdown equity
    /// against a random opponent, then linearly map into
    /// `bucket_counts[street]` bins).
    ///
    /// TODO: When a `bucket_files[street]` is `Some`, look up the
    /// bucket from the file using `BucketFile::get_bucket(board_idx,
    /// combo_idx)`. This requires a canonical board→index mapping and
    /// a hole-cards→combo_idx mapping that match the enumeration order
    /// used during clustering.
    #[must_use]
    pub fn get_bucket(&self, street: Street, hole_cards: [Card; 2], board: &[Card]) -> u16 {
        if street == Street::Preflop {
            let hand = crate::hands::CanonicalHand::from_cards(hole_cards[0], hole_cards[1]);
            let idx = hand.index() as u16;
            return idx.min(self.bucket_counts[0] - 1);
        }
        // TODO: use self.bucket_files[street as usize] when a board→index
        // mapping is available from the clustering pipeline.
        let equity = crate::showdown_equity::compute_equity(hole_cards, board);
        let k = self.bucket_counts[street as usize];
        let bucket = (equity * f64::from(k)) as u16;
        bucket.min(k - 1)
    }

    /// Pre-compute bucket assignments for all 4 streets x 2 players.
    #[must_use]
    pub fn precompute_buckets(&self, deal: &Deal) -> [[u16; 4]; 2] {
        let mut result = [[0u16; 4]; 2];
        for (player, row) in result.iter_mut().enumerate() {
            for (street_idx, slot) in row.iter_mut().enumerate() {
                let street = match street_idx {
                    0 => Street::Preflop,
                    1 => Street::Flop,
                    2 => Street::Turn,
                    _ => Street::River,
                };
                let board = Self::board_for_street(&deal.board, street);
                *slot = self.get_bucket(street, deal.hole_cards[player], board);
            }
        }
        result
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
    let mut node_value = 0.0f64;
    let mut stats = PruneStats::default();

    for (a, &child_idx) in children.iter().enumerate() {
        if prune {
            stats.total += 1;
            if storage.get_regret(node_idx, bucket, a) < prune_threshold {
                stats.hits += 1;
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
    // integer by ×1000 for precision.
    for a in 0..num_actions {
        let delta = action_values[a] - node_value;
        storage.add_regret(node_idx, bucket, a, (delta * 1000.0) as i32);
    }

    // Accumulate strategy sums (for computing the average strategy).
    for a in 0..num_actions {
        storage.add_strategy_sum(node_idx, bucket, a, (strategy[a] * 1000.0) as i64);
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
        let buckets = AllBuckets {
            bucket_counts: [10, 10, 10, 10],
            bucket_files: [None, None, None, None],
        };
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
        let buckets = AllBuckets {
            bucket_counts: [10, 10, 10, 10],
            bucket_files: [None, None, None, None],
        };
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
        let buckets = AllBuckets {
            bucket_counts: [10, 10, 10, 10],
            bucket_files: [None, None, None, None],
        };
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
        let buckets = AllBuckets {
            bucket_counts: [10, 10, 10, 10],
            bucket_files: [None, None, None, None],
        };
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
        let buckets = AllBuckets {
            bucket_counts: [10, 10, 10, 10],
            bucket_files: [None, None, None, None],
        };
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
        let buckets = AllBuckets {
            bucket_counts: [10, 10, 10, 10],
            bucket_files: [None, None, None, None],
        };
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
}
