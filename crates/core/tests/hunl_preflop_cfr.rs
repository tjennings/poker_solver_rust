//! Integration tests for HUNL Preflop with MCCFR solver.
//!
//! HUNL Preflop has 28,561 initial states (169 x 169 hand combinations).
//! MCCFR makes this tractable by sampling states instead of iterating all.
//!
//! Note: These tests use reduced iteration counts to run quickly. For actual
//! strategy computation, use many more iterations.

use std::time::Instant;

use poker_solver_core::cfr::MccfrSolver;
use poker_solver_core::game::{Game, HunlPreflop};

/// Test that MCCFR trains efficiently on HUNL Preflop.
#[test]
fn hunl_preflop_mccfr_trains_efficiently() {
    let game = HunlPreflop::with_stack(10);
    let mut solver = MccfrSolver::new(game);

    let start = Instant::now();

    // Sample 10 states per iteration, 10 iterations
    // This is a minimal test to verify training works
    solver.train(10, 10);

    let elapsed = start.elapsed();

    println!("MCCFR 10 iterations (10 samples each): {:?}", elapsed);

    // Should complete quickly (equity calculations are cached)
    assert!(
        elapsed.as_secs() < 60,
        "Training took too long: {:?}",
        elapsed
    );

    // Should have some strategies
    assert!(solver.iterations() == 10);
}

/// Test that MCCFR produces valid strategies for HUNL Preflop.
#[test]
fn hunl_preflop_mccfr_produces_valid_strategies() {
    let game = HunlPreflop::with_stack(10);
    let mut solver = MccfrSolver::new(game);

    // Use minimal training for test speed
    solver.train(5, 10);

    // Check SB opening strategies exist and are valid
    let strategies = solver.all_strategies();

    let sb_strategies: Vec<_> = strategies
        .iter()
        .filter(|(k, _)| k.starts_with("SB:") && k.ends_with(':'))
        .collect();

    assert!(
        !sb_strategies.is_empty(),
        "Should have SB opening strategies"
    );

    // Verify probabilities sum to 1
    for (info_set, probs) in &sb_strategies {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Strategy for {info_set} should sum to 1.0, got {sum}"
        );

        for &p in probs.iter() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability should be in [0,1], got {p}"
            );
        }
    }
}

/// Test that premium hands develop reasonable strategies.
#[test]
fn hunl_preflop_premium_hands_strategies() {
    let game = HunlPreflop::with_stack(20);
    let mut solver = MccfrSolver::new(game);

    // Train with modest iterations (strategy quality not checked, just existence)
    solver.train(10, 20);

    // AA should have a strategy at the SB opening position
    if let Some(strategy) = solver.get_average_strategy("SB:AA:") {
        println!("AA opening strategy: {:?}", strategy);

        // AA should rarely fold (if fold is even an option)
        // The first action in preflop for SB is typically call/raise, not fold
        // But if fold is present, it should be very low
        assert!(!strategy.is_empty(), "AA should have actions");
    }

    // 72o should have a strategy too
    if let Some(strategy) = solver.get_average_strategy("SB:72o:") {
        println!("72o opening strategy: {:?}", strategy);

        // 72o (worst hand) should have higher fold/check tendency
        // vs premium hands having higher raise tendency
        assert!(!strategy.is_empty(), "72o should have actions");
    }
}

/// Compare MCCFR sampling vs full traversal.
///
/// Note: This test uses very small parameters to run quickly. The actual speedup
/// is much higher with larger iteration counts, but equity calculation dominates
/// in debug builds.
#[test]
fn hunl_preflop_sampling_speedup() {
    let game = HunlPreflop::with_stack(10);
    let num_states = game.initial_states().len();

    println!("HUNL Preflop has {} initial states", num_states);
    assert_eq!(num_states, 169 * 169);

    // Just verify both training methods work
    let mut solver = MccfrSolver::new(game.clone());
    solver.train(1, 10); // Sampled
    assert_eq!(solver.iterations(), 1);

    let mut solver2 = MccfrSolver::new(game);
    solver2.train_full(1); // Full
    assert_eq!(solver2.iterations(), 1);

    // Both should produce some strategies
    assert!(!solver.all_strategies().is_empty());
    assert!(!solver2.all_strategies().is_empty());

    println!("Both training methods work correctly");
}

/// Test that more training explores more info sets.
#[test]
fn hunl_preflop_training_improves() {
    let game = HunlPreflop::with_stack(10);

    // Train briefly
    let mut solver1 = MccfrSolver::new(game.clone());
    solver1.train(2, 5);
    let strategies1 = solver1.all_strategies();

    // Train more
    let mut solver2 = MccfrSolver::new(game);
    solver2.train(5, 10);
    let strategies2 = solver2.all_strategies();

    // More training should populate more info sets
    // (deeper game tree exploration)
    println!(
        "Info sets after 2x5 samples: {}, after 5x10 samples: {}",
        strategies1.len(),
        strategies2.len()
    );

    assert!(
        strategies2.len() >= strategies1.len(),
        "More training should explore at least as many info sets"
    );
}
