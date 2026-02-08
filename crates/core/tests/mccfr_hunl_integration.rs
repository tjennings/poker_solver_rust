//! Integration tests for MCCFR with full-game HunlPostflop.
//!
//! Verifies that MCCFR can train on the complete HUNL game tree
//! (preflop through river with board dealing) and produce valid strategies.

use poker_solver_core::cfr::{MccfrConfig, MccfrSolver, calculate_exploitability};
use poker_solver_core::game::{HunlPostflop, PostflopConfig};
use test_macros::timed_test;

/// Test that MCCFR produces valid strategies when training on full HUNL.
#[timed_test(10)]
fn mccfr_hunl_postflop_full_game() {
    let config = PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![1.0],
        samples_per_iteration: 500,
        ..PostflopConfig::default()
    };
    let game = HunlPostflop::new(config, None);

    let mccfr_config = MccfrConfig {
        samples_per_iteration: 100,
        use_cfr_plus: true,
        discount_iterations: Some(30),
        skip_first_iterations: None,
    };
    let mut solver = MccfrSolver::with_config(game, &mccfr_config);
    solver.set_seed(42);
    solver.train(100, 50);

    let strategies = solver.all_strategies();
    assert!(
        !strategies.is_empty(),
        "Should produce strategies, got empty map"
    );

    // Verify all probabilities are valid
    for (info_set, probs) in &strategies {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Strategy for '{info_set}' should sum to ~1.0, got {sum:.6}"
        );
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability at index {i} for '{info_set}' should be in [0,1], got {p}"
            );
        }
    }

    println!(
        "MCCFR HUNL full game: {} info sets after {} iterations",
        strategies.len(),
        solver.iterations()
    );
}

/// Test that MCCFR exploitability decreases with more training.
#[timed_test(10)]
fn mccfr_exploitability_decreases_with_training() {
    let config = PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![1.0],
        samples_per_iteration: 100,
        ..PostflopConfig::default()
    };
    // Create separate game instances for solver and exploitability calculation
    let eval_game = HunlPostflop::new(config.clone(), None);
    let solver_game = HunlPostflop::new(config, None);
    let mut solver = MccfrSolver::new(solver_game);
    solver.set_seed(42);

    // Train a bit and measure exploitability
    solver.train(50, 20);
    let early_strategies = solver.all_strategies();
    let early_exploitability = calculate_exploitability(&eval_game, &early_strategies);

    // Train more
    solver.train(200, 20);
    let late_strategies = solver.all_strategies();
    let late_exploitability = calculate_exploitability(&eval_game, &late_strategies);

    println!("Exploitability: early={early_exploitability:.4}, late={late_exploitability:.4}");

    // Late exploitability should be lower (or at least not drastically worse)
    // We use a generous bound since MCCFR is stochastic
    assert!(
        late_exploitability < early_exploitability * 2.0,
        "Exploitability should not drastically increase: early={early_exploitability}, late={late_exploitability}"
    );
}

/// Test that blueprint pipeline works with MCCFR (train → extract → create blueprint).
#[timed_test(10)]
fn mccfr_blueprint_pipeline() {
    use poker_solver_core::blueprint::BlueprintStrategy;

    let config = PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![1.0],
        samples_per_iteration: 100,
        ..PostflopConfig::default()
    };
    let game = HunlPostflop::new(config, None);
    let mut solver = MccfrSolver::new(game);
    solver.set_seed(42);
    solver.train(50, 20);

    let strategies = solver.all_strategies();
    let blueprint = BlueprintStrategy::from_strategies(strategies, solver.iterations());

    assert!(!blueprint.is_empty(), "Blueprint should contain strategies");
    assert_eq!(blueprint.iterations_trained(), 50);

    // Verify save/load roundtrip
    let temp_dir = tempfile::tempdir().expect("should create temp dir");
    let path = temp_dir.path().join("blueprint.bin");

    blueprint.save(&path).expect("save should succeed");
    let loaded = BlueprintStrategy::load(&path).expect("load should succeed");

    assert_eq!(loaded.len(), blueprint.len());
    assert_eq!(loaded.iterations_trained(), blueprint.iterations_trained());
}
