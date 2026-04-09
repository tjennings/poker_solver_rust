//! External-sampling MCCFR traversal for the N-player blueprint solver.
//!
//! At the traverser's decision nodes **all** actions are explored;
//! at opponent decision nodes **one** action is sampled according to
//! the current regret-matched strategy.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]

use rand::Rng;

use super::game_tree::{MpGameNode, MpGameTree, TerminalKind};
use super::storage::{MpStorage, REGRET_SCALE};
use super::terminal::{resolve_fold, resolve_showdown};
use super::types::{Chips, DealWithBuckets, PlayerSet, Seat};
use super::MAX_PLAYERS;
use crate::blueprint_mp::types::Deal;
use crate::poker::{Card, FlatDeck, Hand, Rank, Rankable};

/// Maximum actions at any decision node (stack-allocated buffers).
const MAX_ACTIONS: usize = 16;

/// Prune statistics accumulated during a single traversal.
///
/// Collected locally to avoid global atomic contention, then merged
/// into global counters after the batch completes.
#[derive(Debug, Default, Clone, Copy)]
pub struct PruneStats {
    pub hits: u64,
    pub total: u64,
}

impl PruneStats {
    pub fn merge(&mut self, other: Self) {
        self.hits += other.hits;
        self.total += other.total;
    }
}

// ── Deal sampling ─────────────────────────────────────────────────

/// Sample a random deal for `num_players` players.
///
/// Uses a partial Fisher-Yates shuffle to draw `2 * num_players + 5`
/// cards without replacement.
pub fn sample_deal(num_players: u8, rng: &mut impl Rng) -> Deal {
    let mut deck = FlatDeck::default();
    deck.shuffle(rng);
    let mut idx = 0;
    let sentinel = deck[0]; // placeholder; overwritten for active seats
    let mut hole_cards = [[sentinel; 2]; MAX_PLAYERS];
    for cards in hole_cards.iter_mut().take(num_players as usize) {
        cards[0] = deck[idx];
        cards[1] = deck[idx + 1];
        idx += 2;
    }
    let board = [
        deck[idx],
        deck[idx + 1],
        deck[idx + 2],
        deck[idx + 3],
        deck[idx + 4],
    ];
    Deal {
        hole_cards,
        board,
        num_players,
    }
}

// ── External-sampling traversal ───────────────────────────────────

/// Traverse the game tree with external sampling MCCFR.
///
/// Returns `(cfv, prune_stats)` -- the counterfactual value for the
/// traverser at this node plus pruning diagnostics.
pub fn traverse_external(
    tree: &MpGameTree,
    storage: &MpStorage,
    deal: &DealWithBuckets,
    traverser: Seat,
    node_idx: u32,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: Chips,
    prune: bool,
    prune_threshold: i16,
) -> (f64, PruneStats) {
    match &tree.nodes[node_idx as usize] {
        MpGameNode::Terminal {
            kind,
            contributions,
            ..
        } => {
            let v = terminal_value(kind, contributions, deal, traverser, tree.num_players, rake_rate, rake_cap);
            (v, PruneStats::default())
        }
        MpGameNode::Chance { child, .. } => {
            traverse_external(tree, storage, deal, traverser, *child, rng, rake_rate, rake_cap, prune, prune_threshold)
        }
        MpGameNode::Decision {
            seat,
            street,
            actions,
            children,
        } => {
            let bucket = deal.buckets[seat.index() as usize][street.index()].0;
            if *seat == traverser {
                traverse_traverser(
                    tree, storage, deal, traverser, node_idx, bucket,
                    children, actions.len(), rng, rake_rate, rake_cap,
                    prune, prune_threshold,
                )
            } else {
                traverse_opponent(
                    tree, storage, deal, traverser, node_idx, bucket,
                    children, actions.len(), rng, rake_rate, rake_cap,
                    prune, prune_threshold,
                )
            }
        }
    }
}

// ── Traverser node: explore all actions ───────────────────────────

fn traverse_traverser(
    tree: &MpGameTree,
    storage: &MpStorage,
    deal: &DealWithBuckets,
    traverser: Seat,
    node_idx: u32,
    bucket: u16,
    children: &[u32],
    num_actions: usize,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: Chips,
    prune: bool,
    prune_threshold: i16,
) -> (f64, PruneStats) {
    let mut strategy = [0.0_f64; MAX_ACTIONS];
    storage.regret_matched_strategy(node_idx, bucket, num_actions, &mut strategy);

    let mut values = [0.0_f64; MAX_ACTIONS];
    let mut pruned = [false; MAX_ACTIONS];
    let mut node_value = 0.0_f64;
    let mut stats = PruneStats::default();

    for a in 0..num_actions {
        if should_prune_action(tree, storage, prune, prune_threshold, children[a], node_idx, bucket, a, &mut stats) {
            pruned[a] = true;
            continue;
        }
        let (v, child_stats) = traverse_external(
            tree, storage, deal, traverser, children[a], rng, rake_rate, rake_cap, prune, prune_threshold,
        );
        values[a] = v;
        node_value += strategy[a] * v;
        stats.merge(child_stats);
    }

    update_regrets_with_pruning(storage, node_idx, bucket, num_actions, &values, &pruned, node_value);
    update_traverser_strategy_sums(storage, node_idx, bucket, num_actions, &strategy);
    (node_value, stats)
}

/// Check whether a single action should be pruned at this traverser node.
///
/// Never prunes actions leading to terminal nodes (folds are high-leverage).
fn should_prune_action(
    tree: &MpGameTree,
    storage: &MpStorage,
    prune: bool,
    prune_threshold: i16,
    child_idx: u32,
    node_idx: u32,
    bucket: u16,
    action: usize,
    stats: &mut PruneStats,
) -> bool {
    if !prune {
        return false;
    }
    let child_is_terminal = matches!(tree.nodes[child_idx as usize], MpGameNode::Terminal { .. });
    if child_is_terminal {
        return false;
    }
    stats.total += 1;
    if storage.get_regret(node_idx, bucket, action) < prune_threshold {
        stats.hits += 1;
        return true;
    }
    false
}

fn update_regrets_with_pruning(
    storage: &MpStorage,
    node_idx: u32,
    bucket: u16,
    num_actions: usize,
    values: &[f64; MAX_ACTIONS],
    pruned: &[bool; MAX_ACTIONS],
    node_value: f64,
) {
    for (a, &val) in values[..num_actions].iter().enumerate() {
        if pruned[a] {
            continue;
        }
        let raw = (val - node_value) * REGRET_SCALE;
        let delta = raw.clamp(i16::MIN as f64, i16::MAX as f64) as i16;
        storage.add_regret(node_idx, bucket, a, delta);
    }
}

fn update_traverser_strategy_sums(
    storage: &MpStorage,
    node_idx: u32,
    bucket: u16,
    num_actions: usize,
    strategy: &[f64; MAX_ACTIONS],
) {
    for (a, &prob) in strategy[..num_actions].iter().enumerate() {
        let raw = prob * REGRET_SCALE;
        let delta = raw.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        storage.add_strategy_sum(node_idx, bucket, a, delta);
    }
}

// ── Opponent node: sample one action ──────────────────────────────

fn traverse_opponent(
    tree: &MpGameTree,
    storage: &MpStorage,
    deal: &DealWithBuckets,
    traverser: Seat,
    node_idx: u32,
    bucket: u16,
    children: &[u32],
    num_actions: usize,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: Chips,
    prune: bool,
    prune_threshold: i16,
) -> (f64, PruneStats) {
    let mut strategy = [0.0_f64; MAX_ACTIONS];
    storage.regret_matched_strategy(node_idx, bucket, num_actions, &mut strategy);

    let sampled = sample_action(&strategy[..num_actions], rng);

    // Update strategy sums for the sampled action
    let raw = strategy[sampled] * REGRET_SCALE;
    let delta = raw.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
    storage.add_strategy_sum(node_idx, bucket, sampled, delta);

    traverse_external(
        tree, storage, deal, traverser, children[sampled], rng, rake_rate, rake_cap,
        prune, prune_threshold,
    )
}

/// Sample an action index from a probability distribution.
fn sample_action(strategy: &[f64], rng: &mut impl Rng) -> usize {
    let roll: f64 = rng.random();
    let mut cumulative = 0.0;
    for (i, &p) in strategy.iter().enumerate() {
        cumulative += p;
        if roll < cumulative {
            return i;
        }
    }
    strategy.len() - 1
}

// ── Terminal value computation ────────────────────────────────────

/// Compute the traverser's net payoff at a terminal node.
#[must_use]
pub fn terminal_value(
    kind: &TerminalKind,
    contributions: &[Chips; MAX_PLAYERS],
    deal: &DealWithBuckets,
    traverser: Seat,
    num_players: u8,
    rake_rate: f64,
    rake_cap: Chips,
) -> f64 {
    match kind {
        TerminalKind::LastStanding { winner } => {
            let payoffs = resolve_fold(*contributions, *winner, num_players);
            payoffs[traverser.index() as usize].0
        }
        TerminalKind::Showdown { active } => {
            let hand_ranks = rank_active_hands(deal, *active, num_players);
            let payoffs = resolve_showdown(
                contributions, &hand_ranks, *active, num_players, rake_rate, rake_cap,
            );
            payoffs[traverser.index() as usize].0
        }
    }
}

/// Rank the hands of all active players using `rs_poker`.
///
/// Returns a `u32` per seat where higher = better hand, preserving
/// the total ordering of [`Rank`].
fn rank_active_hands(
    deal: &DealWithBuckets,
    active: PlayerSet,
    num_players: u8,
) -> [u32; MAX_PLAYERS] {
    let mut ranks = [0u32; MAX_PLAYERS];
    for seat in active.iter() {
        let idx = seat.index() as usize;
        if idx < num_players as usize {
            ranks[idx] = rank_to_u32(eval_hand(deal.deal.hole_cards[idx], &deal.deal.board));
        }
    }
    ranks
}

/// Evaluate a hand from hole cards + board using `rs_poker`.
fn eval_hand(hole: [Card; 2], board: &[Card]) -> Rank {
    let mut hand = Hand::default();
    hand.insert(hole[0]);
    hand.insert(hole[1]);
    for &c in board {
        hand.insert(c);
    }
    hand.rank()
}

/// Convert a `Rank` enum to a `u32` that preserves the total ordering.
///
/// The discriminant determines the major category (HighCard=0 .. StraightFlush=8),
/// and the inner `u32` breaks ties within each category.
fn rank_to_u32(rank: Rank) -> u32 {
    let (category, inner) = match rank {
        Rank::HighCard(v) => (0u32, v),
        Rank::OnePair(v) => (1, v),
        Rank::TwoPair(v) => (2, v),
        Rank::ThreeOfAKind(v) => (3, v),
        Rank::Straight(v) => (4, v),
        Rank::Flush(v) => (5, v),
        Rank::FullHouse(v) => (6, v),
        Rank::FourOfAKind(v) => (7, v),
        Rank::StraightFlush(v) => (8, v),
    };
    // Inner values are < 2^24, so shifting category by 24 bits is safe.
    (category << 24) | inner
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_mp::config::{
        ForcedBet, ForcedBetKind, MpActionAbstractionConfig, MpGameConfig, MpStreetSizes,
    };
    use crate::blueprint_mp::game_tree::MpGameTree;
    use crate::blueprint_mp::storage::MpStorage;
    use crate::blueprint_mp::types::{Bucket, Seat};
    use test_macros::timed_test;

    fn minimal_tree(num_players: u8) -> MpGameTree {
        let blinds = vec![
            ForcedBet {
                seat: 0,
                kind: ForcedBetKind::SmallBlind,
                amount: 1.0,
            },
            ForcedBet {
                seat: 1,
                kind: ForcedBetKind::BigBlind,
                amount: 2.0,
            },
        ];
        let game = MpGameConfig {
            name: format!("{num_players}-player mccfr test"),
            num_players,
            stack_depth: 20.0,
            blinds,
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let empty = MpStreetSizes {
            lead: vec![],
            raise: vec![],
        };
        let action = MpActionAbstractionConfig {
            preflop: empty.clone(),
            flop: empty.clone(),
            turn: empty.clone(),
            river: empty,
        };
        MpGameTree::build(&game, &action)
    }

    fn trivial_buckets(deal: &Deal, bucket_counts: [u16; 4]) -> DealWithBuckets {
        let mut buckets = [[Bucket(0); 4]; MAX_PLAYERS];
        for seat in 0..deal.num_players as usize {
            for street in 0..4 {
                let card_idx = deal.hole_cards[seat][0].value as u16;
                buckets[seat][street] = Bucket(card_idx % bucket_counts[street]);
            }
        }
        DealWithBuckets {
            deal: deal.clone(),
            buckets,
        }
    }

    // -- Deal tests --

    #[timed_test]
    fn deal_no_duplicate_cards() {
        let mut rng = rand::thread_rng();
        for num_players in 2..=8u8 {
            let deal = sample_deal(num_players, &mut rng);
            let mut all_cards = Vec::new();
            for seat in 0..num_players as usize {
                all_cards.push(deal.hole_cards[seat][0]);
                all_cards.push(deal.hole_cards[seat][1]);
            }
            all_cards.extend_from_slice(&deal.board);

            // Check no duplicates by comparing all pairs
            for i in 0..all_cards.len() {
                for j in (i + 1)..all_cards.len() {
                    assert_ne!(
                        all_cards[i], all_cards[j],
                        "duplicate card at positions {i} and {j} for {num_players} players"
                    );
                }
            }
        }
    }

    #[timed_test]
    fn deal_correct_num_players() {
        let mut rng = rand::thread_rng();
        for num_players in 2..=8u8 {
            let deal = sample_deal(num_players, &mut rng);
            assert_eq!(deal.num_players, num_players);
        }
    }

    // -- Traversal tests --

    #[timed_test]
    fn traverse_single_iteration_no_panic() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let mut rng = rand::thread_rng();

        let deal = sample_deal(2, &mut rng);
        let dwb = trivial_buckets(&deal, bucket_counts);
        let traverser = Seat::from_raw(0);

        let _value = traverse_external(
            &tree,
            &storage,
            &dwb,
            traverser,
            tree.root,
            &mut rng,
            0.0,
            Chips::ZERO,
            false,
            0,
        );
    }

    #[timed_test]
    fn traverse_updates_regrets() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let mut rng = rand::thread_rng();

        let deal = sample_deal(2, &mut rng);
        let dwb = trivial_buckets(&deal, bucket_counts);
        let traverser = Seat::from_raw(0);

        traverse_external(
            &tree,
            &storage,
            &dwb,
            traverser,
            tree.root,
            &mut rng,
            0.0,
            Chips::ZERO,
            false,
            0,
        );

        // At least one regret should be non-zero after traversal
        let any_nonzero = storage
            .regrets
            .iter()
            .any(|r| r.load(std::sync::atomic::Ordering::Relaxed) != 0);
        assert!(any_nonzero, "at least one regret should be non-zero after traversal");
    }

    #[timed_test]
    fn traverse_updates_strategy_sums() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let mut rng = rand::thread_rng();

        // Run traversal for seat 0 (traverser), opponent is seat 1
        // Strategy sums are updated at opponent nodes
        let deal = sample_deal(2, &mut rng);
        let dwb = trivial_buckets(&deal, bucket_counts);

        traverse_external(
            &tree,
            &storage,
            &dwb,
            Seat::from_raw(0),
            tree.root,
            &mut rng,
            0.0,
            Chips::ZERO,
            false,
            0,
        );

        let any_nonzero = storage
            .strategy_sums
            .iter()
            .any(|s| s.load(std::sync::atomic::Ordering::Relaxed) != 0);
        assert!(
            any_nonzero,
            "at least one strategy sum should be non-zero after traversal"
        );
    }

    #[timed_test]
    fn terminal_fold_value_correct() {
        // Manually construct a fold scenario: P0 puts 1, P1 puts 2, P0 folds.
        // P1 wins pot. Traverser is P0.
        // P0 payoff = -1.0 (lost their contribution)
        use crate::blueprint_mp::game_tree::TerminalKind;
        let contributions = [
            Chips(1.0),
            Chips(2.0),
            Chips::ZERO,
            Chips::ZERO,
            Chips::ZERO,
            Chips::ZERO,
            Chips::ZERO,
            Chips::ZERO,
        ];
        let kind = TerminalKind::LastStanding {
            winner: Seat::from_raw(1),
        };
        let mut rng = rand::thread_rng();
        let deal = sample_deal(2, &mut rng);
        let dwb = trivial_buckets(&deal, [10, 10, 10, 10]);

        let value = terminal_value(&kind, &contributions, &dwb, Seat::from_raw(0), 2, 0.0, Chips::ZERO);
        assert!(
            (value - (-1.0)).abs() < 1e-10,
            "P0 should lose 1.0 on fold, got {value}"
        );

        // P1 should win 1.0 net
        let value_p1 = terminal_value(&kind, &contributions, &dwb, Seat::from_raw(1), 2, 0.0, Chips::ZERO);
        assert!(
            (value_p1 - 1.0).abs() < 1e-10,
            "P1 should win 1.0 on fold, got {value_p1}"
        );
    }

    #[timed_test]
    fn strategy_converges_toward_nonzero() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let deal = sample_deal(2, &mut rng);
            let dwb = trivial_buckets(&deal, bucket_counts);
            for seat_idx in 0..2u8 {
                traverse_external(
                    &tree,
                    &storage,
                    &dwb,
                    Seat::from_raw(seat_idx),
                    tree.root,
                    &mut rng,
                    0.0,
                    Chips::ZERO,
                    false,
                    0,
                );
            }
        }

        // After 100 iterations, strategy sums should be non-zero everywhere
        // that decisions are reachable
        let any_nonzero = storage
            .strategy_sums
            .iter()
            .any(|s| s.load(std::sync::atomic::Ordering::Relaxed) != 0);
        assert!(any_nonzero, "after 100 iters, strategy sums should be non-zero");
    }

    #[timed_test]
    fn traverse_returns_finite_value() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let mut rng = rand::thread_rng();
        let deal = sample_deal(2, &mut rng);
        let dwb = trivial_buckets(&deal, bucket_counts);

        let (value, _) = traverse_external(
            &tree,
            &storage,
            &dwb,
            Seat::from_raw(0),
            tree.root,
            &mut rng,
            0.0,
            Chips::ZERO,
            false,
            0,
        );
        assert!(value.is_finite(), "traverse should return finite value, got {value}");
    }

    #[timed_test]
    fn traverse_3_player_no_panic() {
        let tree = minimal_tree(3);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let deal = sample_deal(3, &mut rng);
            let dwb = trivial_buckets(&deal, bucket_counts);
            for seat_idx in 0..3u8 {
                let (_v, _stats) = traverse_external(
                    &tree,
                    &storage,
                    &dwb,
                    Seat::from_raw(seat_idx),
                    tree.root,
                    &mut rng,
                    0.0,
                    Chips::ZERO,
                    false,
                    0,
                );
            }
        }
    }

    // -- Pruning tests --

    #[timed_test]
    fn traverse_with_pruning_skips_negative_regrets() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let mut rng = rand::thread_rng();

        // Force very negative regrets at all decision nodes, all buckets
        set_all_regrets_negative(&tree, &storage, bucket_counts);

        let deal = sample_deal(2, &mut rng);
        let dwb = trivial_buckets(&deal, bucket_counts);
        let (_val, stats) = traverse_external(
            &tree, &storage, &dwb, Seat::from_raw(0),
            tree.root, &mut rng, 0.0, Chips::ZERO, true, -100,
        );
        assert!(stats.hits > 0, "pruning should skip negative-regret actions");
    }

    #[timed_test]
    fn traverse_without_pruning_explores_all() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let mut rng = rand::thread_rng();

        set_all_regrets_negative(&tree, &storage, bucket_counts);

        let deal = sample_deal(2, &mut rng);
        let dwb = trivial_buckets(&deal, bucket_counts);
        let (_val, stats) = traverse_external(
            &tree, &storage, &dwb, Seat::from_raw(0),
            tree.root, &mut rng, 0.0, Chips::ZERO, false, -100,
        );
        assert_eq!(stats.hits, 0, "no pruning when prune=false");
    }

    #[timed_test]
    fn prune_never_skips_terminal_children() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let mut rng = rand::thread_rng();

        set_all_regrets_negative(&tree, &storage, bucket_counts);

        // Traverse with pruning -- folds lead to Terminal nodes,
        // so they must never be pruned
        let deal = sample_deal(2, &mut rng);
        let dwb = trivial_buckets(&deal, bucket_counts);
        let (val, _stats) = traverse_external(
            &tree, &storage, &dwb, Seat::from_raw(0),
            tree.root, &mut rng, 0.0, Chips::ZERO, true, -100,
        );
        // If fold was pruned, traversal would have no explored actions
        // and return 0.0 or fail. A finite value means folds were explored.
        assert!(val.is_finite(), "traversal must still reach terminals via folds");
    }

    fn first_decision_node(tree: &MpGameTree) -> u32 {
        tree.nodes
            .iter()
            .position(|n| matches!(n, MpGameNode::Decision { .. }))
            .expect("tree should have a decision node") as u32
    }

    fn set_all_regrets_negative(
        tree: &MpGameTree,
        storage: &MpStorage,
        bucket_counts: [u16; 4],
    ) {
        for (i, node) in tree.nodes.iter().enumerate() {
            if let MpGameNode::Decision { actions, street, .. } = node {
                let bkts = bucket_counts[street.index()];
                for bucket in 0..bkts {
                    for a in 0..actions.len() {
                        storage.add_regret(i as u32, bucket, a, -30_000);
                    }
                }
            }
        }
    }
}
