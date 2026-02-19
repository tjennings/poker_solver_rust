//! Integration test: verify the preflop solver converges to valid strategies.
//!
//! The solver uses a uniform equity table (all matchups = 0.5), so all 169
//! canonical hands are strategically equivalent. We verify structural properties:
//! strategies sum to 1.0 and the solver produces entries for every hand.

use poker_solver_core::preflop::{PreflopConfig, PreflopSolver};

#[test]
fn hu_preflop_converges_in_500_iterations() {
    // Use a small stack depth and restricted raise sizes for a compact tree.
    let mut config = PreflopConfig::heads_up(10);
    config.raise_sizes = vec![vec![3.0]];
    config.raise_cap = 2;

    let mut solver = PreflopSolver::new(&config);
    solver.train(500);
    let strategy = solver.strategy();

    // Strategy should not be empty.
    assert!(
        !strategy.is_empty(),
        "strategy should have entries after training"
    );

    // Every hand should have a strategy at the root.
    for hand_idx in 0..169 {
        let probs = strategy.get_root_probs(hand_idx);
        assert!(
            !probs.is_empty(),
            "hand {hand_idx} should have a strategy at root"
        );
    }

    // Strategy probabilities should sum to ~1.0 for every hand at root.
    for hand_idx in 0..169 {
        let probs = strategy.get_root_probs(hand_idx);
        if probs.is_empty() {
            continue;
        }
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.02,
            "hand {hand_idx}: strategy sum = {sum}, expected ~1.0"
        );
        // All probabilities should be non-negative.
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p >= -1e-9,
                "hand {hand_idx} action {i}: negative probability {p}"
            );
        }
    }
}

#[test]
fn solver_iteration_count_tracks_correctly() {
    let mut config = PreflopConfig::heads_up(5);
    config.raise_sizes = vec![vec![3.0]];
    config.raise_cap = 1;

    let mut solver = PreflopSolver::new(&config);
    assert_eq!(solver.iteration(), 0);

    solver.train(10);
    assert_eq!(solver.iteration(), 10);

    solver.train(5);
    assert_eq!(solver.iteration(), 15);
}
