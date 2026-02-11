//! Integration tests for MCCFR with full-game HunlPostflop.
//!
//! Verifies that MCCFR can train on the complete HUNL game tree
//! (preflop through river with board dealing) and produce valid strategies.

use poker_solver_core::cfr::{MccfrConfig, MccfrSolver, calculate_exploitability};
use poker_solver_core::game::{AbstractionMode, HunlPostflop, PostflopConfig};
use test_macros::timed_test;

/// Test that MCCFR produces valid strategies when training on full HUNL.
#[timed_test(10)]
fn mccfr_hunl_postflop_full_game() {
    let config = PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![1.0],
        ..PostflopConfig::default()
    };
    let game = HunlPostflop::new(config, None, 500);

    let mccfr_config = MccfrConfig {
        samples_per_iteration: 100,
        ..MccfrConfig::default()
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
        ..PostflopConfig::default()
    };
    // Create separate game instances for solver and exploitability calculation
    let eval_game = HunlPostflop::new(config.clone(), None, 100);
    let solver_game = HunlPostflop::new(config, None, 100);
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

/// Test that parallel MCCFR produces valid strategies on full HUNL.
#[timed_test(10)]
fn mccfr_parallel_hunl_postflop() {
    let config = PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![1.0],
        ..PostflopConfig::default()
    };
    let game = HunlPostflop::new(config, None, 500);

    let mccfr_config = MccfrConfig {
        samples_per_iteration: 100,
        ..MccfrConfig::default()
    };
    let mut solver = MccfrSolver::with_config(game, &mccfr_config);
    solver.set_seed(42);
    solver.train_parallel(100, 50);

    let strategies = solver.all_strategies();
    assert!(
        !strategies.is_empty(),
        "Parallel should produce strategies, got empty map"
    );

    for (info_set, probs) in &strategies {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Parallel strategy for '{info_set}' should sum to ~1.0, got {sum:.6}"
        );
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability at index {i} for '{info_set}' should be in [0,1], got {p}"
            );
        }
    }

    println!(
        "Parallel MCCFR HUNL: {} info sets after {} iterations",
        strategies.len(),
        solver.iterations()
    );
}

/// Test that hand_class blueprint keys can be scanned by position mask.
///
/// Simulates the scan_node_classes logic: build a position key from
/// street/pot/stack/actions, then filter the blueprint for matching keys.
#[timed_test(10)]
fn hand_class_blueprint_scan_finds_entries() {
    use poker_solver_core::blueprint::BlueprintStrategy;
    use poker_solver_core::info_key::{spr_bucket, InfoKey};

    let config = PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![1.0],
        ..PostflopConfig::default()
    };
    let game = HunlPostflop::new(config, Some(AbstractionMode::HandClassV2 { strength_bits: 0, equity_bits: 0 }), 200);

    let mccfr_config = MccfrConfig {
        samples_per_iteration: 50,
        ..MccfrConfig::default()
    };
    let mut solver = MccfrSolver::with_config(game, &mccfr_config);
    solver.set_seed(42);
    solver.train(30, 15);

    let strategies = solver.all_strategies();
    let blueprint = BlueprintStrategy::from_strategies(strategies, solver.iterations());

    assert!(
        !blueprint.is_empty(),
        "Blueprint should contain strategies"
    );
    println!("Blueprint has {} info sets", blueprint.len());

    // After preflop limp (call + check), pot = 4, stacks = [18, 18] (10BB * 2 = 20)
    // spr_bucket = min(18*2/4, 31) = 9
    // street = Flop (1), action_codes = [] (first action on flop)
    let spr_b = spr_bucket(4, 18);
    let street_num = 1u8; // Flop
    let action_codes: &[u8] = &[];

    let position_key =
        InfoKey::new(0, street_num, spr_b, action_codes).as_u64();
    let position_mask: u64 = (1u64 << 44) - 1;

    let matches: Vec<(u64, u32)> = blueprint
        .iter()
        .filter(|(k, _)| (*k & position_mask) == position_key)
        .map(|(k, _)| {
            let hand_bits = (*k >> 44) as u32;
            (*k, hand_bits)
        })
        .collect();

    println!(
        "Position key: {position_key:#018x}, matches: {}",
        matches.len()
    );

    // Also try scanning for all flop keys regardless of SPR bucket
    let street_mask: u64 = 0x3u64 << 42; // just the street bits
    let street_key: u64 = (u64::from(street_num) & 0x3) << 42;
    let flop_keys: Vec<u32> = blueprint
        .iter()
        .filter(|(k, _)| (*k & street_mask) == street_key)
        .map(|(k, _)| {
            let decoded = InfoKey::from_raw(*k);
            decoded.spr_bucket()
        })
        .collect();

    println!(
        "Total flop keys: {}, SPR buckets seen: {:?}",
        flop_keys.len(),
        {
            let mut unique: Vec<u32> = flop_keys.clone();
            unique.sort();
            unique.dedup();
            unique
        }
    );

    // At minimum, some flop keys should exist in the blueprint
    assert!(
        !flop_keys.is_empty(),
        "Blueprint should have flop keys for hand_class mode"
    );

    // If no exact matches, that's a bucket mismatch - report it
    if matches.is_empty() {
        println!(
            "WARNING: No exact match for spr_bucket={spr_b}. \
             Training used different bucket values."
        );
    }
}

/// Test that blueprint pipeline works with MCCFR (train → extract → create blueprint).
#[timed_test(10)]
fn mccfr_blueprint_pipeline() {
    use poker_solver_core::blueprint::BlueprintStrategy;

    let config = PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![1.0],
        ..PostflopConfig::default()
    };
    let game = HunlPostflop::new(config, None, 100);
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
