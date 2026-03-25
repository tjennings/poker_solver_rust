//! MCCFR solver adapter for the convergence harness.
//!
//! Wraps `BlueprintTrainer` from poker-solver-core, implementing the
//! `ConvergenceSolver` trait with fixed-flop deals and canonical preflop
//! hand bucketing.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use poker_solver_core::blueprint_v2::config::{
    ActionAbstractionConfig, BlueprintV2Config, ClusteringAlgorithm, ClusteringConfig, GameConfig,
    SnapshotConfig, StreetClusterConfig, TrainingConfig,
};
use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree};
use poker_solver_core::blueprint_v2::mccfr::{Deal, DealWithBuckets, traverse_external};
use poker_solver_core::blueprint_v2::storage::BlueprintStorage;
use poker_solver_core::blueprint_v2::trainer::BlueprintTrainer;
use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::hands::CanonicalHand;
use poker_solver_core::poker::{Card, Suit, Value, ALL_SUITS, ALL_VALUES};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::game::FlopPokerConfig;
use crate::solver_trait::{ComboEvMap, ConvergenceSolver, SolverMetrics, StrategyMap};

/// Number of MCCFR iterations per `solve_step()` call.
const BATCH_SIZE: u64 = 1000;

/// Number of canonical preflop hand types.
const NUM_BUCKETS: u16 = 169;

// ---------------------------------------------------------------------------
// Card type conversion
// ---------------------------------------------------------------------------

/// Convert a range-solver card (u8: `4*rank + suit`) to a core `Card`.
///
/// Range-solver encoding: rank 2=0..A=12, suit c=0,d=1,h=2,s=3.
/// Core encoding: Value Two=0..Ace=12, Suit Spade=0,Club=1,Heart=2,Diamond=3.
fn rs_card_to_core_card(card_id: u8) -> Card {
    let rank = card_id >> 2;
    let rs_suit = card_id & 3;
    let value = Value::from_u8(rank);
    let suit = match rs_suit {
        0 => Suit::Club,
        1 => Suit::Diamond,
        2 => Suit::Heart,
        3 => Suit::Spade,
        _ => unreachable!(),
    };
    Card::new(value, suit)
}

// ---------------------------------------------------------------------------
// Fixed-flop deal sampling
// ---------------------------------------------------------------------------

/// The fixed flop cards: Qh Jd Th.
fn flop_cards() -> [Card; 3] {
    [
        Card::new(Value::Queen, Suit::Heart),
        Card::new(Value::Jack, Suit::Diamond),
        Card::new(Value::Ten, Suit::Heart),
    ]
}

/// Build the full 52-card deck in value-major order (matching core's canonical deck).
fn build_deck() -> [Card; 52] {
    let mut deck = [Card::new(ALL_VALUES[0], ALL_SUITS[0]); 52];
    let mut idx = 0;
    for &v in &ALL_VALUES {
        for &s in &ALL_SUITS {
            deck[idx] = Card::new(v, s);
            idx += 1;
        }
    }
    deck
}

/// Sample a deal with a fixed flop (QhJdTh).
/// Hole cards, turn, and river are random from the remaining 49 cards.
fn sample_fixed_flop_deal(rng: &mut impl Rng) -> Deal {
    let blocked = flop_cards();
    let full_deck = build_deck();

    let mut deck: Vec<Card> = full_deck
        .iter()
        .copied()
        .filter(|c| !blocked.contains(c))
        .collect();

    // Partial Fisher-Yates: shuffle first 6 positions (2+2 hole + turn + river)
    for i in 0..6 {
        let j = rng.random_range(i..deck.len());
        deck.swap(i, j);
    }

    Deal {
        hole_cards: [[deck[0], deck[1]], [deck[2], deck[3]]],
        board: [blocked[0], blocked[1], blocked[2], deck[4], deck[5]],
    }
}

/// Assign canonical preflop hand index as bucket for ALL streets.
/// This ignores board interaction -- pipeline validation only.
fn canonical_buckets(deal: &Deal) -> [[u16; 4]; 2] {
    let mut result = [[0u16; 4]; 2];
    for (player, row) in result.iter_mut().enumerate() {
        let hole = deal.hole_cards[player];
        let hand = CanonicalHand::from_cards(hole[0], hole[1]);
        let idx = hand.index() as u16;
        *row = [idx; 4];
    }
    result
}

// ---------------------------------------------------------------------------
// Config translation
// ---------------------------------------------------------------------------

/// Parse a percentage-based bet size string (e.g. "33%,67%") into pot fractions
/// for the blueprint tree builder.
///
/// Returns a `Vec<Vec<f64>>` where each inner vec is a raise depth.
/// - `"67%"` => `[[0.67]]` — one raise depth with 67% pot
/// - `"33%,67%"` => `[[0.33, 0.67]]` — one depth with two sizes
/// - `"a"` => `[]` — all-in only (no sized bets; all-in added automatically)
fn parse_bet_sizes(s: &str) -> Vec<Vec<f64>> {
    // "a" means all-in only — no pot-fraction sizes needed since the
    // blueprint tree builder always adds all-in.
    if s.trim().eq_ignore_ascii_case("a") {
        return vec![];
    }

    let sizes: Vec<f64> = s
        .split(',')
        .filter_map(|part| {
            let trimmed = part.trim().trim_end_matches('%');
            trimmed.parse::<f64>().ok().map(|v| v / 100.0)
        })
        .collect();
    if sizes.is_empty() {
        vec![vec![0.67]]
    } else {
        vec![sizes]
    }
}

/// Build a `BlueprintV2Config` from a `FlopPokerConfig`.
fn build_mccfr_config(config: &FlopPokerConfig) -> BlueprintV2Config {
    let street_cluster = |buckets| StreetClusterConfig {
        buckets,
        delta_bins: None,
        expected_delta: false,
        sample_boards: None,
    };

    let postflop_sizes = parse_bet_sizes(&config.bet_sizes);

    BlueprintV2Config {
        game: GameConfig {
            name: "Flop Poker convergence test".into(),
            players: 2,
            stack_depth: config.effective_stack as f64,
            small_blind: 0.5,
            big_blind: 1.0,
            rake_rate: 0.0,
            rake_cap: 0.0,
        },
        clustering: ClusteringConfig {
            algorithm: ClusteringAlgorithm::PotentialAwareEmd,
            preflop: street_cluster(NUM_BUCKETS),
            flop: street_cluster(NUM_BUCKETS),
            turn: street_cluster(NUM_BUCKETS),
            river: street_cluster(NUM_BUCKETS),
            seed: 42,
            kmeans_iterations: 0,
            cfvnet_river_data: None,
            per_flop: None,
        },
        action_abstraction: ActionAbstractionConfig {
            preflop: vec![vec!["2bb".into()]],
            flop: postflop_sizes.clone(),
            turn: postflop_sizes.clone(),
            river: postflop_sizes,
        },
        training: TrainingConfig {
            cluster_path: None,
            iterations: Some(1_000_000),
            time_limit_minutes: None,
            lcfr_warmup_iterations: 10_000,
            lcfr_discount_interval: 10_000,
            prune_after_iterations: 50_000,
            prune_threshold: -300,
            prune_explore_pct: 0.05,
            print_every_minutes: 60,
            batch_size: BATCH_SIZE,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
            dcfr_epoch_cap: None,
            target_strategy_delta: None,
            purify_threshold: 0.0,
            equity_cache_path: None,
        },
        snapshots: SnapshotConfig {
            warmup_minutes: 9999,
            snapshot_every_minutes: 9999,
            output_dir: "/tmp/convergence-harness-mccfr".into(),
            resume: false,
            max_snapshots: None,
        },
    }
}

// ---------------------------------------------------------------------------
// Strategy lifting
// ---------------------------------------------------------------------------

/// Lift a bucket-level strategy to combo-level.
///
/// For each combo (represented as a range-solver card pair), looks up the
/// canonical hand bucket and copies the bucket's strategy probabilities.
/// Returns a flat Vec in layout: `action_idx * num_combos + combo_idx`.
fn lift_bucket_strategy_for_node(
    storage: &BlueprintStorage,
    node_idx: u32,
    num_actions: usize,
    combos: &[(u8, u8)],
) -> Vec<f32> {
    let num_combos = combos.len();
    let mut result = vec![0.0f32; num_actions * num_combos];

    for (combo_idx, &(c1, c2)) in combos.iter().enumerate() {
        let core_c1 = rs_card_to_core_card(c1);
        let core_c2 = rs_card_to_core_card(c2);
        let hand = CanonicalHand::from_cards(core_c1, core_c2);
        let bucket = hand.index() as u16;

        let strat = storage.average_strategy(node_idx, bucket);
        for (action_idx, &prob) in strat.iter().enumerate() {
            if action_idx < num_actions {
                result[action_idx * num_combos + combo_idx] = prob as f32;
            }
        }
    }

    result
}

/// Walk the blueprint game tree and extract lifted strategies at every
/// decision node. Uses arena index as node key.
fn extract_lifted_strategies(
    tree: &GameTree,
    storage: &BlueprintStorage,
    combos: &[(u8, u8)],
) -> StrategyMap {
    let mut result = StrategyMap::new();

    for (idx, node) in tree.nodes.iter().enumerate() {
        if let GameNode::Decision { actions, .. } = node {
            let num_actions = actions.len();
            let strat = lift_bucket_strategy_for_node(
                storage,
                idx as u32,
                num_actions,
                combos,
            );
            result.insert(idx as u64, strat);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Exploitability computation
// ---------------------------------------------------------------------------

/// Find the first Flop decision node in the blueprint tree.
///
/// The blueprint tree starts with preflop decisions and chance nodes. This
/// walks from the root, following through preflop decisions and the
/// preflop-to-flop chance node, until it reaches the first Flop Decision.
fn find_flop_root(tree: &GameTree) -> u32 {
    let mut idx = tree.root;
    loop {
        match &tree.nodes[idx as usize] {
            GameNode::Decision { street, children, .. } => {
                if *street != Street::Preflop {
                    return idx;
                }
                // Follow the first child through preflop (trivial preflop
                // has one meaningful path). We follow the LAST child which
                // is typically the action that continues the hand (call/check)
                // rather than fold. But with trivial preflop (SB limps, BB
                // checks), every path eventually reaches the flop. The
                // structure is: SB acts -> [fold, call, ...]. We want the
                // call path, which is the last non-fold child. In our
                // trivial preflop tree, the SB has [Fold, Call] or similar.
                // The call leads to BB acting, BB checks, then flop chance.
                //
                // Walk ALL children via DFS to find the flop.
                for &child in children {
                    if let Some(flop_idx) = find_flop_root_dfs(tree, child) {
                        return flop_idx;
                    }
                }
                panic!("No flop decision found from preflop root");
            }
            GameNode::Chance { child, .. } => {
                idx = *child;
            }
            GameNode::Terminal { .. } => {
                panic!("Reached terminal before finding flop decision");
            }
        }
    }
}

/// DFS helper to find the first Flop decision node reachable from `idx`.
fn find_flop_root_dfs(tree: &GameTree, idx: u32) -> Option<u32> {
    match &tree.nodes[idx as usize] {
        GameNode::Decision { street, children, .. } => {
            if *street != Street::Preflop {
                return Some(idx);
            }
            for &child in children {
                if let Some(found) = find_flop_root_dfs(tree, child) {
                    return Some(found);
                }
            }
            None
        }
        GameNode::Chance { child, .. } => find_flop_root_dfs(tree, *child),
        GameNode::Terminal { .. } => None,
    }
}

/// Compute exploitability of the MCCFR strategy in the full (unabstracted) game.
///
/// Builds a range-solver game with the same parameters, walks the tree in
/// parallel with the blueprint tree, locks the lifted MCCFR strategy at
/// every decision node, and then runs best-response exploitability.
///
/// The config's `add_allin_threshold` is overridden to a large value so the
/// range-solver always adds all-in, matching the blueprint tree which always
/// includes all-in at every decision node.
pub fn compute_mccfr_exploitability(
    solver: &MccfrSolver,
    config: &FlopPokerConfig,
) -> Result<f64, String> {
    // 1. Build range-solver game with matching tree structure.
    //    The blueprint tree always adds all-in at every node, so we must
    //    ensure the range-solver does the same by using a high threshold.
    let rs_config = FlopPokerConfig {
        add_allin_threshold: 100.0,
        force_allin_threshold: 0.0,
        ..config.clone()
    };
    let mut game = crate::game::build_flop_poker_game_with_config(&rs_config)?;
    game.allocate_memory(false);

    // 2. Find the flop root in the blueprint tree
    let bp_flop_root = find_flop_root(solver.tree());

    // 3. Walk both trees in parallel, locking MCCFR strategy at each decision node
    let mut history: Vec<usize> = Vec::new();
    lock_strategy_recursive(
        &mut game,
        solver.tree(),
        solver.storage(),
        bp_flop_root,
        &mut history,
    );

    // 4. Compute exploitability
    Ok(range_solver::compute_exploitability(&game) as f64)
}

/// Recursively walk the range-solver game tree and blueprint tree in parallel,
/// locking the MCCFR lifted strategy at every decision node.
fn lock_strategy_recursive(
    game: &mut range_solver::PostFlopGame,
    tree: &GameTree,
    storage: &BlueprintStorage,
    bp_node_idx: u32,
    history: &mut Vec<usize>,
) {
    if game.is_terminal_node() {
        return;
    }

    if game.is_chance_node() {
        // At a chance node (turn/river deal), the range-solver enumerates
        // each possible card. The blueprint tree has a single Chance child
        // (it doesn't enumerate individual cards).
        let bp_child = match &tree.nodes[bp_node_idx as usize] {
            GameNode::Chance { child, .. } => *child,
            other => panic!(
                "Expected blueprint Chance node at idx {}, got {:?}",
                bp_node_idx,
                std::mem::discriminant(other)
            ),
        };

        // Iterate over available chance actions directly (handles isomorphism correctly)
        let num_chance_actions = game.available_actions().len();
        for i in 0..num_chance_actions {
            // Re-fetch actions each time since game state changes after navigate_back
            let actions = game.available_actions();
            if let range_solver::action_tree::Action::Chance(card) = actions[i] {
                history.push(card as usize);
                game.play(card as usize);
                lock_strategy_recursive(game, tree, storage, bp_child, history);
                history.pop();
                crate::evaluator::navigate_back(game, history);
            }
        }
        return;
    }

    // Decision node -- extract blueprint info
    let bp_node = &tree.nodes[bp_node_idx as usize];
    let (bp_actions, bp_children) = match bp_node {
        GameNode::Decision {
            actions, children, ..
        } => (actions, children),
        other => panic!(
            "Expected blueprint Decision node at idx {}, got {:?}",
            bp_node_idx,
            std::mem::discriminant(other)
        ),
    };

    let rs_actions = game.available_actions();
    let num_actions = rs_actions.len();
    assert_eq!(
        num_actions,
        bp_actions.len(),
        "Action count mismatch at history {:?}: range-solver has {}, blueprint has {}",
        history,
        num_actions,
        bp_actions.len()
    );

    // Build combo-level strategy and lock it
    let player = game.current_player();
    let combos = game.private_cards(player);
    let strategy =
        lift_bucket_strategy_for_node(storage, bp_node_idx, num_actions, combos);
    game.lock_current_strategy(&strategy);

    // Recurse into children
    for (action_idx, &bp_child_idx) in bp_children.iter().enumerate() {
        history.push(action_idx);
        game.play(action_idx);
        lock_strategy_recursive(game, tree, storage, bp_child_idx, history);
        history.pop();
        crate::evaluator::navigate_back(game, history);
    }
}

// ---------------------------------------------------------------------------
// MccfrSolver
// ---------------------------------------------------------------------------

/// MCCFR solver adapter: runs blueprint MCCFR with canonical preflop bucketing.
pub struct MccfrSolver {
    tree: GameTree,
    storage: BlueprintStorage,
    /// Combo list from range-solver (card pairs using range-solver u8 encoding).
    combos: Vec<(u8, u8)>,
    iteration: u64,
    rng: SmallRng,
}

impl MccfrSolver {
    /// Create a new MCCFR solver from a `FlopPokerConfig`.
    pub fn new(config: FlopPokerConfig) -> Self {
        // Build a range-solver game to get the combo list.
        let rs_game = crate::game::build_flop_poker_game_with_config(&config)
            .expect("Failed to build range-solver game for combo list");
        let combos: Vec<(u8, u8)> = rs_game.private_cards(0).to_vec();

        let mccfr_config = build_mccfr_config(&config);
        let mut trainer = BlueprintTrainer::new(mccfr_config);
        trainer.skip_bucket_validation = true;

        // Extract the tree and storage from the trainer so we can call
        // traverse_external directly without the full training loop.
        let tree = std::mem::replace(
            &mut trainer.tree,
            GameTree {
                nodes: Vec::new(),
                root: 0,
                dealer: 0,
                starting_stack: 0.0,
            },
        );
        let storage = std::mem::replace(
            &mut trainer.storage,
            BlueprintStorage::new(
                &GameTree {
                    nodes: Vec::new(),
                    root: 0,
                    dealer: 0,
                    starting_stack: 0.0,
                },
                [1, 1, 1, 1],
            ),
        );

        Self {
            tree,
            storage,
            combos,
            iteration: 0,
            rng: SmallRng::seed_from_u64(42),
        }
    }

    /// Access the blueprint game tree.
    pub fn tree(&self) -> &GameTree {
        &self.tree
    }

    /// Access the blueprint storage.
    pub fn storage(&self) -> &BlueprintStorage {
        &self.storage
    }
}

impl ConvergenceSolver for MccfrSolver {
    fn name(&self) -> &str {
        "MCCFR (169 canonical buckets)"
    }

    fn solve_step(&mut self) {
        for _ in 0..BATCH_SIZE {
            let seed: u64 = self.rng.random();
            let mut deal_rng = SmallRng::seed_from_u64(seed);
            let deal = sample_fixed_flop_deal(&mut deal_rng);
            let buckets = canonical_buckets(&deal);
            let dwb = DealWithBuckets { deal, buckets };

            // Traverse for both players (external sampling MCCFR)
            traverse_external(
                &self.tree,
                &self.storage,
                &dwb,
                0,
                self.tree.root,
                false,
                0,
                &mut deal_rng,
                0.0,
                0.0,
                None,
                None,
            );
            traverse_external(
                &self.tree,
                &self.storage,
                &dwb,
                1,
                self.tree.root,
                false,
                0,
                &mut deal_rng,
                0.0,
                0.0,
                None,
                None,
            );
        }
        self.iteration += BATCH_SIZE;
    }

    fn iterations(&self) -> u64 {
        self.iteration
    }

    fn average_strategy(&self) -> StrategyMap {
        extract_lifted_strategies(&self.tree, &self.storage, &self.combos)
    }

    fn combo_evs(&self) -> ComboEvMap {
        ComboEvMap::new()
    }

    fn self_reported_metrics(&self) -> SolverMetrics {
        SolverMetrics::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;

    #[test]
    fn test_fixed_flop_deal_has_correct_board() {
        let mut rng = SmallRng::seed_from_u64(42);
        let deal = sample_fixed_flop_deal(&mut rng);
        let expected = flop_cards();
        assert_eq!(deal.board[0], expected[0], "board[0] should be Qh");
        assert_eq!(deal.board[1], expected[1], "board[1] should be Jd");
        assert_eq!(deal.board[2], expected[2], "board[2] should be Th");
    }

    #[test]
    fn test_fixed_flop_deal_no_duplicate_cards() {
        let mut rng = SmallRng::seed_from_u64(42);
        let deal = sample_fixed_flop_deal(&mut rng);
        let mut all_cards: Vec<Card> = vec![
            deal.hole_cards[0][0],
            deal.hole_cards[0][1],
            deal.hole_cards[1][0],
            deal.hole_cards[1][1],
            deal.board[0],
            deal.board[1],
            deal.board[2],
            deal.board[3],
            deal.board[4],
        ];
        all_cards.sort();
        all_cards.dedup();
        assert_eq!(all_cards.len(), 9, "All 9 dealt cards must be unique");
    }

    #[test]
    fn test_canonical_buckets_same_for_all_streets() {
        let mut rng = SmallRng::seed_from_u64(42);
        let deal = sample_fixed_flop_deal(&mut rng);
        let buckets = canonical_buckets(&deal);
        for player in 0..2 {
            assert_eq!(buckets[player][0], buckets[player][1]);
            assert_eq!(buckets[player][1], buckets[player][2]);
            assert_eq!(buckets[player][2], buckets[player][3]);
            assert!(
                buckets[player][0] < 169,
                "Bucket must be valid canonical index"
            );
        }
    }

    #[test]
    fn test_canonical_buckets_different_hands_get_different_buckets() {
        let mut rng = SmallRng::seed_from_u64(99);
        let deal = sample_fixed_flop_deal(&mut rng);
        let buckets = canonical_buckets(&deal);
        assert!(buckets[0][0] < 169);
        assert!(buckets[1][0] < 169);
    }

    #[test]
    fn test_range_solver_card_to_core_card_round_trip() {
        // As = 4*12 + 3 = 51
        let core_card = rs_card_to_core_card(51);
        assert_eq!(core_card.value, Value::Ace);
        assert_eq!(core_card.suit, Suit::Spade);

        // 2c = 4*0 + 0 = 0
        let core_card = rs_card_to_core_card(0);
        assert_eq!(core_card.value, Value::Two);
        assert_eq!(core_card.suit, Suit::Club);

        // Qh = 4*10 + 2 = 42
        let core_card = rs_card_to_core_card(42);
        assert_eq!(core_card.value, Value::Queen);
        assert_eq!(core_card.suit, Suit::Heart);

        // Jd = 4*9 + 1 = 37
        let core_card = rs_card_to_core_card(37);
        assert_eq!(core_card.value, Value::Jack);
        assert_eq!(core_card.suit, Suit::Diamond);
    }

    #[test]
    fn test_range_solver_card_to_core_card_all_52() {
        // Every card should convert without panic and have valid value/suit
        for card_id in 0..52u8 {
            let core_card = rs_card_to_core_card(card_id);
            let rank = card_id >> 2;
            assert_eq!(core_card.value as u8, rank);
        }
    }

    #[test]
    fn test_mccfr_solver_name() {
        let config = FlopPokerConfig::default();
        let solver = MccfrSolver::new(config);
        assert_eq!(solver.name(), "MCCFR (169 canonical buckets)");
    }

    #[test]
    fn test_mccfr_solver_starts_at_zero_iterations() {
        let config = FlopPokerConfig::default();
        let solver = MccfrSolver::new(config);
        assert_eq!(solver.iterations(), 0);
    }

    #[test]
    fn test_mccfr_solver_runs_iterations() {
        let config = FlopPokerConfig::default();
        let mut solver = MccfrSolver::new(config);
        assert_eq!(solver.iterations(), 0);
        solver.solve_step();
        assert_eq!(solver.iterations(), BATCH_SIZE);
    }

    #[test]
    fn test_mccfr_strategy_after_iterations() {
        let config = FlopPokerConfig::default();
        let mut solver = MccfrSolver::new(config);
        for _ in 0..10 {
            solver.solve_step();
        }
        let strategy = solver.average_strategy();
        assert!(
            !strategy.is_empty(),
            "Should have at least one node's strategy"
        );
    }

    #[test]
    fn test_mccfr_strategy_has_valid_probabilities() {
        let config = FlopPokerConfig::default();
        let mut solver = MccfrSolver::new(config);
        for _ in 0..5 {
            solver.solve_step();
        }
        let strategy = solver.average_strategy();
        for (_nid, strat) in &strategy {
            assert!(!strat.is_empty(), "Strategy vector should not be empty");
            for &val in strat.iter() {
                assert!(
                    val >= -1e-6,
                    "Strategy probability should be >= 0, got {}",
                    val
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // parse_bet_sizes tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_bet_sizes_percentage() {
        assert_eq!(parse_bet_sizes("67%"), vec![vec![0.67]]);
    }

    #[test]
    fn test_parse_bet_sizes_multiple() {
        assert_eq!(parse_bet_sizes("33%,67%"), vec![vec![0.33, 0.67]]);
    }

    #[test]
    fn test_parse_bet_sizes_allin_returns_empty() {
        let result = parse_bet_sizes("a");
        assert!(result.is_empty(), "All-in should return empty sizes");
    }

    #[test]
    fn test_parse_bet_sizes_allin_case_insensitive() {
        assert!(parse_bet_sizes("A").is_empty());
    }

    #[test]
    fn test_parse_bet_sizes_fallback_to_default() {
        // Non-numeric, non-"a" input falls back to 67%
        assert_eq!(parse_bet_sizes("invalid"), vec![vec![0.67]]);
    }

    // -----------------------------------------------------------------------
    // find_flop_root tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_flop_root_returns_flop_decision_node() {
        let config = FlopPokerConfig {
            effective_stack: 10,
            bet_sizes: "a".into(),
            raise_sizes: "a".into(),
            ..Default::default()
        };
        let solver = MccfrSolver::new(config);
        let flop_root = find_flop_root(&solver.tree);
        match &solver.tree.nodes[flop_root as usize] {
            GameNode::Decision { street, .. } => {
                assert_eq!(
                    *street,
                    poker_solver_core::blueprint_v2::Street::Flop,
                    "Flop root should be a Flop decision node"
                );
            }
            other => panic!(
                "Expected a Decision node at flop root, got {:?}",
                std::mem::discriminant(other)
            ),
        }
    }

    #[test]
    fn test_find_flop_root_is_not_tree_root() {
        let config = FlopPokerConfig {
            effective_stack: 10,
            bet_sizes: "a".into(),
            raise_sizes: "a".into(),
            ..Default::default()
        };
        let solver = MccfrSolver::new(config);
        let flop_root = find_flop_root(&solver.tree);
        assert_ne!(
            flop_root, solver.tree.root,
            "Flop root should differ from tree root (preflop)"
        );
    }

    // -----------------------------------------------------------------------
    // Exploitability computation tests
    // -----------------------------------------------------------------------

    /// Config that produces matching tree structures in both the blueprint
    /// and range-solver: all-in only (no pot-relative sizes that differ
    /// between float and integer rounding).
    fn allin_only_config() -> FlopPokerConfig {
        FlopPokerConfig {
            effective_stack: 10,
            bet_sizes: "a".into(),
            raise_sizes: "a".into(),
            add_allin_threshold: 100.0,
            force_allin_threshold: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_tree_correspondence_allin_only() {
        let config = allin_only_config();
        let solver = MccfrSolver::new(config.clone());
        let flop_root = find_flop_root(&solver.tree);

        let rs_config = FlopPokerConfig {
            add_allin_threshold: 100.0,
            force_allin_threshold: 0.0,
            ..config.clone()
        };
        let game = crate::game::build_flop_poker_game_with_config(&rs_config).unwrap();
        let rs_actions = game.available_actions();

        // Both should have Check + AllIn = 2 actions at the root
        if let GameNode::Decision { actions, .. } = &solver.tree.nodes[flop_root as usize] {
            assert_eq!(
                actions.len(),
                rs_actions.len(),
                "Blueprint and range-solver should have same action count at flop root"
            );
        }
    }

    #[test]
    fn test_compute_mccfr_exploitability_is_nonnegative_after_training() {
        let config = allin_only_config();
        let mut solver = MccfrSolver::new(config.clone());
        for _ in 0..20 {
            solver.solve_step();
        }
        let expl = compute_mccfr_exploitability(&solver, &config).unwrap();
        // Exploitability should be >= 0 (within floating point tolerance)
        assert!(
            expl > -1e-6,
            "Exploitability should be non-negative (within tolerance), got {}",
            expl
        );
    }

    #[test]
    fn test_compute_mccfr_exploitability_is_finite() {
        let config = allin_only_config();
        let mut solver = MccfrSolver::new(config.clone());
        for _ in 0..20 {
            solver.solve_step();
        }
        let expl = compute_mccfr_exploitability(&solver, &config).unwrap();
        assert!(
            expl.is_finite(),
            "Exploitability should be finite, got {}",
            expl
        );
    }

    #[test]
    fn test_compute_mccfr_exploitability_decreases_with_more_training() {
        let config = allin_only_config();
        let mut solver = MccfrSolver::new(config.clone());

        // After few iterations
        for _ in 0..2 {
            solver.solve_step();
        }
        let expl_early = compute_mccfr_exploitability(&solver, &config).unwrap();

        // After many more iterations
        for _ in 0..50 {
            solver.solve_step();
        }
        let expl_late = compute_mccfr_exploitability(&solver, &config).unwrap();

        // Later exploitability should be less than or equal to early
        // (MCCFR converges, so exploitability should decrease)
        assert!(
            expl_late <= expl_early + 1e-6,
            "Exploitability should decrease with training: early={}, late={}",
            expl_early,
            expl_late
        );
    }
}
