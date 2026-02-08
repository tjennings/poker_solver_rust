//! Integration tests for HunlPostflop with GPU solver.
//!
//! These tests verify that HunlPostflop integrates correctly with the
//! GpuCfrSolver. Uses NdArray backend (CPU) for CI compatibility.
//!
//! Note: Uses very small config values to keep tests fast. The game tree
//! grows exponentially with samples and bet sizes, so we use minimal values.

#![cfg(feature = "gpu")]

use poker_solver_core::cfr::GpuCfrSolver;
use poker_solver_core::game::{HunlPostflop, PostflopConfig};
use test_macros::timed_test;

use burn::backend::ndarray::NdArray;

type TestBackend = NdArray;

/// Test that HunlPostflop compiles and trains with the GPU solver.
///
/// Uses minimal config values to keep the test fast - we just need to
/// verify integration, not convergence.
#[timed_test]
fn hunl_postflop_compiles_and_trains() {
    let config = PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![1.0],
        samples_per_iteration: 2,
        ..PostflopConfig::default()
    };

    let game = HunlPostflop::new(config, None);
    let device = Default::default();
    let mut solver = GpuCfrSolver::<TestBackend>::new(&game, device);

    solver.train(5);

    assert_eq!(
        solver.iterations(),
        5,
        "Solver should have completed 5 iterations"
    );
}

/// Test that HunlPostflop produces valid strategies after training.
///
/// Verifies that strategies are non-empty and probabilities sum to 1.0.
#[timed_test]
fn hunl_postflop_produces_valid_strategies() {
    let config = PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![1.0],
        samples_per_iteration: 2,
        ..PostflopConfig::default()
    };

    let game = HunlPostflop::new(config, None);
    let device = Default::default();
    let mut solver = GpuCfrSolver::<TestBackend>::new(&game, device);

    solver.train(10);

    let strategies = solver.all_strategies();

    assert!(
        !strategies.is_empty(),
        "Solver should produce at least one strategy"
    );

    for (info_set, probs) in &strategies {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Strategy for info set '{}' should sum to ~1.0, got {:.6}",
            info_set,
            sum
        );

        for (i, &p) in probs.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability at index {} for info set '{}' should be in [0,1], got {}",
                i,
                info_set,
                p
            );
        }
    }

    println!(
        "Validated {} strategies from HunlPostflop solver",
        strategies.len()
    );
}
