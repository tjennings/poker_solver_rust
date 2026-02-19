//! Full pipeline integration test: preflop -> blueprint -> subgame.
//!
//! Verifies all three solving layers work together end-to-end:
//! 1. Solve preflop (HU, 20BB, 200 iterations)
//! 2. Verify preflop strategy validity
//! 3. Build and solve a river subgame with uniform opponent reach
//! 4. Verify subgame strategy validity

use poker_solver_core::blueprint::{
    BlueprintStrategy, SubgameCfrSolver, SubgameHands, SubgameStrategy, SubgameTreeBuilder,
};
use poker_solver_core::poker::{Card, Suit, Value};
use poker_solver_core::preflop::{PreflopConfig, PreflopSolver, PreflopStrategy};

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Verify that a probability vector sums to ~1.0 (within tolerance).
fn assert_valid_distribution(probs: &[f64], label: &str) {
    assert!(!probs.is_empty(), "{label}: empty probability vector");
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.02,
        "{label}: probability sum = {sum}, expected ~1.0"
    );
    for (i, &p) in probs.iter().enumerate() {
        assert!(
            p >= -1e-9,
            "{label}: negative probability at index {i}: {p}"
        );
    }
}

/// A 5-card river board for subgame testing.
fn river_board() -> Vec<Card> {
    vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Heart),
        Card::new(Value::Seven, Suit::Diamond),
        Card::new(Value::Four, Suit::Club),
        Card::new(Value::Ten, Suit::Club),
    ]
}

/// Build a small hand set (first `n` combos) for fast testing.
fn small_hands(board: &[Card], n: usize) -> SubgameHands {
    let full = SubgameHands::enumerate(board);
    SubgameHands {
        combos: full.combos.into_iter().take(n).collect(),
    }
}

/// Number of combos to use in subgame tests (small for speed).
const SUBGAME_COMBO_COUNT: usize = 60;

/// Preflop iteration count (enough for valid distributions, fast enough for CI).
const PREFLOP_ITERATIONS: u64 = 200;

/// Subgame iteration count.
const SUBGAME_ITERATIONS: u32 = 200;

// -----------------------------------------------------------------------
// Step 1 & 2: Preflop
// -----------------------------------------------------------------------

#[test]
fn step1_preflop_solver_produces_valid_strategy() {
    let config = PreflopConfig::heads_up(20);
    let mut solver = PreflopSolver::new(&config);
    solver.train(PREFLOP_ITERATIONS);

    let strategy: PreflopStrategy = solver.strategy();

    // Strategy should be non-empty (169 hands x multiple decision nodes).
    assert!(
        !strategy.is_empty(),
        "preflop strategy should have entries after training"
    );
    assert!(
        strategy.len() > 169,
        "preflop strategy should cover multiple nodes, got {}",
        strategy.len()
    );

    // Check root action probabilities for every canonical hand.
    for hand_idx in 0..169 {
        let probs = strategy.get_root_probs(hand_idx);
        assert_valid_distribution(&probs, &format!("hand {hand_idx} root"));
    }

    // Spot-check a few non-root nodes.
    for node_idx in 1..5_u32 {
        for hand_idx in [0_usize, 84, 168] {
            let probs = strategy.get_probs(node_idx, hand_idx);
            if !probs.is_empty() {
                assert_valid_distribution(
                    &probs,
                    &format!("node {node_idx} hand {hand_idx}"),
                );
            }
        }
    }
}

// -----------------------------------------------------------------------
// Step 3 & 4: Subgame
// -----------------------------------------------------------------------

#[test]
fn step3_subgame_solver_produces_valid_strategy() {
    let board = river_board();

    let tree = SubgameTreeBuilder::new()
        .board(&board)
        .bet_sizes(&[0.5, 1.0])
        .pot(40)
        .stacks(&[40, 40])
        .build();

    let hands = small_hands(&board, SUBGAME_COMBO_COUNT);
    let n = hands.combos.len();

    let opponent_reach = vec![1.0; n];
    let leaf_values = vec![0.0; n];

    let mut solver = SubgameCfrSolver::new(tree, hands, opponent_reach, leaf_values);
    solver.train(SUBGAME_ITERATIONS);

    assert_eq!(solver.iteration, SUBGAME_ITERATIONS);

    let strategy: SubgameStrategy = solver.strategy();
    assert_eq!(strategy.num_combos(), n);

    // Every combo should have a valid root distribution.
    let mut non_empty = 0;
    for combo_idx in 0..n {
        let probs = strategy.root_probs(combo_idx);
        if !probs.is_empty() {
            assert_valid_distribution(&probs, &format!("combo {combo_idx}"));
            non_empty += 1;
        }
    }

    assert!(
        non_empty > n / 2,
        "expected at least half of combos to have strategy, got {non_empty}/{n}"
    );

    // Check a few interior nodes.
    for node_idx in 1..5_u32 {
        for combo_idx in [0_usize, n / 2, n - 1] {
            let probs = strategy.get_probs(node_idx, combo_idx);
            if !probs.is_empty() {
                let sum: f64 = probs.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 0.02,
                    "node {node_idx} combo {combo_idx}: sum = {sum}"
                );
            }
        }
    }
}

#[test]
fn step3_subgame_full_enumeration_creates_correct_combo_count() {
    let board = river_board();
    let hands = SubgameHands::enumerate(&board);
    // C(47, 2) = 1081 combos for a 5-card board.
    assert_eq!(hands.combos.len(), 1081, "river should have C(47,2) = 1081 combos");
}

// -----------------------------------------------------------------------
// Step 5: Blueprint strategy (structural test, no training needed)
// -----------------------------------------------------------------------

#[test]
fn step5_blueprint_strategy_stores_and_retrieves() {
    let mut blueprint = BlueprintStrategy::new();
    assert!(blueprint.is_empty());

    blueprint.insert(100, vec![0.3, 0.5, 0.2]);
    blueprint.insert(200, vec![0.6, 0.4]);
    blueprint.set_iterations(1000);

    assert_eq!(blueprint.len(), 2);
    assert_eq!(blueprint.iterations_trained(), 1000);

    let probs = blueprint.lookup(100).expect("should find key 100");
    assert_eq!(probs.len(), 3);
    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "blueprint probs should sum to ~1.0, got {sum}"
    );
}

// -----------------------------------------------------------------------
// Full end-to-end: preflop + subgame in one test
// -----------------------------------------------------------------------

#[test]
fn full_pipeline_preflop_then_subgame() {
    // -- Preflop phase --
    let config = PreflopConfig::heads_up(20);
    let mut preflop_solver = PreflopSolver::new(&config);
    preflop_solver.train(PREFLOP_ITERATIONS);
    let preflop_strategy = preflop_solver.strategy();

    assert!(!preflop_strategy.is_empty());

    // Verify AA (hand index 0) has a valid strategy at the root.
    let aa_probs = preflop_strategy.get_root_probs(0);
    assert_valid_distribution(&aa_probs, "AA root");

    // -- Subgame phase --
    let board = river_board();
    let tree = SubgameTreeBuilder::new()
        .board(&board)
        .bet_sizes(&[0.5, 1.0])
        .pot(40)
        .stacks(&[40, 40])
        .build();

    let hands = small_hands(&board, SUBGAME_COMBO_COUNT);
    let n = hands.combos.len();

    // In a real pipeline, opponent_reach would come from the blueprint.
    // Here we use uniform reach to test the subgame solver in isolation.
    let opponent_reach = vec![1.0; n];
    let leaf_values = vec![0.0; n];

    let mut subgame_solver = SubgameCfrSolver::new(tree, hands, opponent_reach, leaf_values);
    subgame_solver.train(SUBGAME_ITERATIONS);
    let subgame_strategy = subgame_solver.strategy();

    assert_eq!(subgame_strategy.num_combos(), n);

    // Verify at least some combos have valid strategies.
    let valid_count = (0..n)
        .filter(|&i| {
            let p = subgame_strategy.root_probs(i);
            if p.is_empty() {
                return false;
            }
            let s: f64 = p.iter().sum();
            (s - 1.0).abs() < 0.02
        })
        .count();

    assert!(
        valid_count > n / 2,
        "expected most combos to have valid strategy, got {valid_count}/{n}"
    );

    // -- Blueprint structure check --
    let mut blueprint = BlueprintStrategy::new();
    blueprint.insert(42, vec![0.25, 0.50, 0.25]);
    assert_eq!(blueprint.len(), 1);
    assert!(blueprint.lookup(42).is_some());
}
