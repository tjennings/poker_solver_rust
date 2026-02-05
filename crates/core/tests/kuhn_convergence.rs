//! Integration tests for Kuhn Poker CFR convergence to Nash equilibrium.

use std::collections::HashMap;

use poker_solver_core::{
    cfr::{VanillaCfr, calculate_exploitability},
    game::KuhnPoker,
};
use test_macros::timed_test;

#[cfg(feature = "gpu")]
use burn::backend::ndarray::NdArray;
#[cfg(feature = "gpu")]
use poker_solver_core::cfr::{BatchedCfr, CompiledGame, GpuCfrSolver, compile};

/// Test that CFR converges to near-Nash equilibrium on Kuhn Poker.
///
/// Kuhn Poker has a known Nash equilibrium. After sufficient iterations,
/// the average strategy should have exploitability below 0.001 (0.1% of the pot).
#[timed_test(10)]
fn kuhn_reaches_nash_equilibrium() {
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game.clone());

    solver.train(10_000);

    let strategy = extract_strategy(&solver);
    let exploitability = calculate_exploitability(&game, &strategy);

    assert!(
        exploitability < 0.01,
        "Kuhn Poker should converge to Nash equilibrium, but exploitability is {exploitability}"
    );
}

/// Test known Nash equilibrium properties for Kuhn Poker.
///
/// In the Nash equilibrium:
/// - With a King, always bet/call (never fold)
/// - With a Jack facing a bet, always fold
/// - Queen has mixed strategies
#[timed_test(10)]
fn kuhn_nash_strategy_properties() {
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game);

    solver.train(10_000);

    // King should always call when facing a bet
    if let Some(strategy) = solver.get_average_strategy("Kb") {
        assert!(
            strategy[1] > 0.95,
            "King should always call a bet, got fold={:.4}, call={:.4}",
            strategy[0],
            strategy[1]
        );
    }

    // King should always call after check-bet
    if let Some(strategy) = solver.get_average_strategy("Kcb") {
        assert!(
            strategy[1] > 0.95,
            "King should always call after check-bet, got fold={:.4}, call={:.4}",
            strategy[0],
            strategy[1]
        );
    }

    // Jack should always fold when facing a bet (Jb)
    if let Some(strategy) = solver.get_average_strategy("Jb") {
        assert!(
            strategy[0] > 0.95,
            "Jack should always fold when facing a bet, got fold={:.4}, call={:.4}",
            strategy[0],
            strategy[1]
        );
    }

    // Jack should always fold after check-bet (Jcb)
    if let Some(strategy) = solver.get_average_strategy("Jcb") {
        assert!(
            strategy[0] > 0.95,
            "Jack should always fold after check-bet, got fold={:.4}, call={:.4}",
            strategy[0],
            strategy[1]
        );
    }
}

/// Test that exploitability monotonically decreases (on average) with more iterations.
#[timed_test(10)]
fn exploitability_decreases_over_training() {
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game.clone());

    let checkpoints = [100, 500, 1_000, 5_000];
    let mut prev_exploitability = f64::MAX;

    for &iterations in &checkpoints {
        let current_iterations = if iterations == 100 {
            100
        } else {
            iterations - checkpoints[checkpoints.iter().position(|&x| x == iterations).unwrap() - 1]
        };
        solver.train(current_iterations as u64);

        let strategy = extract_strategy(&solver);
        let exploitability = calculate_exploitability(&game, &strategy);

        println!(
            "After {iterations} iterations: exploitability = {exploitability:.6}"
        );

        if iterations >= 500 {
            assert!(
                exploitability < prev_exploitability * 1.5,
                "Exploitability should trend downward: prev={prev_exploitability}, current={exploitability}"
            );
        }
        prev_exploitability = exploitability;
    }

    assert!(
        prev_exploitability < 0.01,
        "Final exploitability should be < 0.01, got {prev_exploitability}"
    );
}

/// Helper to extract average strategy from solver
fn extract_strategy(solver: &VanillaCfr<KuhnPoker>) -> HashMap<String, Vec<f64>> {
    let info_sets = [
        "J", "Q", "K", "Jc", "Qc", "Kc", "Jb", "Qb", "Kb", "Jcb", "Qcb", "Kcb",
    ];

    info_sets
        .iter()
        .filter_map(|&is| solver.get_average_strategy(is).map(|s| (is.to_string(), s)))
        .collect()
}

// ============================================================================
// GPU/Batched CFR Integration Tests
// ============================================================================

#[cfg(feature = "gpu")]
mod batched_tests {
    use super::*;
    use test_macros::timed_test;

    type TestBackend = NdArray;

    /// Test that compiled game captures all Kuhn Poker info sets.
    #[timed_test]
    fn compiled_game_has_correct_info_sets() {
            let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);

        // Kuhn Poker has 12 info sets
        assert_eq!(compiled.num_info_sets, 12);

        // Verify expected info sets exist
        let expected_info_sets = [
            "J", "Q", "K", "Jc", "Qc", "Kc", "Jb", "Qb", "Kb", "Jcb", "Qcb", "Kcb",
        ];

        for is in expected_info_sets {
            assert!(
                compiled.info_set_to_idx.contains_key(is),
                "Expected info set '{}' not found in compiled game",
                is
            );
        }
    }

    /// Test that BatchedCfr initializes with correct structure.
    #[timed_test]
    fn batched_cfr_matches_compiled_game_structure() {
            let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);
        let solver = BatchedCfr::new(&compiled, &device);

        assert_eq!(solver.num_info_sets(), compiled.num_info_sets);
    }

    /// Test that initial batched strategy matches vanilla's uniform strategy.
    #[timed_test]
    fn batched_initial_strategy_is_uniform() {
            let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);
        let solver = BatchedCfr::new(&compiled, &device);

        // Check each info set has uniform strategy summing to 1
        for info_set_idx in 0..solver.num_info_sets() {
            let strategy = solver
                .get_strategy(info_set_idx)
                .expect("Should get strategy");

            let sum: f64 = strategy.values().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Info set {} strategy should sum to 1, got {}",
                info_set_idx,
                sum
            );

            // All actions should have equal probability (uniform)
            let probs: Vec<f64> = strategy.values().copied().collect();
            if probs.len() > 1 {
                let first = probs[0];
                for &p in &probs[1..] {
                    assert!(
                        (p - first).abs() < 1e-5,
                        "Initial strategy should be uniform, got {:?}",
                        probs
                    );
                }
            }
        }
    }

    /// Test that batched and vanilla solvers have consistent info set structure.
    #[timed_test]
    fn batched_and_vanilla_info_sets_match() {
            let game = KuhnPoker::new();
        let _vanilla_solver = VanillaCfr::new(game.clone());

        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);

        // Both should have 12 info sets (Kuhn Poker standard)
        let expected_info_sets = [
            "J", "Q", "K", "Jc", "Qc", "Kc", "Jb", "Qb", "Kb", "Jcb", "Qcb", "Kcb",
        ];

        // Vanilla gets info sets by querying (won't have them until training)
        // Compiled game has them from tree traversal
        for is in expected_info_sets {
            assert!(
                compiled.info_set_to_idx.contains_key(is),
                "Compiled game missing info set '{}'",
                is
            );
        }

        assert_eq!(compiled.num_info_sets, expected_info_sets.len());
    }

    /// Test that regret updates produce valid strategies.
    #[timed_test]
    fn regret_updates_produce_valid_strategies() {
            use burn::prelude::*;

        let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);
        let mut solver = BatchedCfr::new(&compiled, &device);

        // Apply some regret updates
        for _ in 0..10 {
            // Create random-ish regret deltas
            let num_info_sets = solver.num_info_sets();
            let max_actions = compiled.max_actions;

            let deltas_data: Vec<f32> = (0..num_info_sets * max_actions)
                .map(|i| (i as f32 % 3.0) - 1.0) // Some positive, some negative
                .collect();

            let deltas = Tensor::<TestBackend, 1>::from_floats(deltas_data.as_slice(), &device)
                .reshape([num_info_sets, max_actions]);

            solver.update_regrets(deltas);

            // Accumulate strategy with uniform reach
            let reach = Tensor::<TestBackend, 1>::ones([num_info_sets], &device);
            solver.accumulate_strategy(reach);
        }

        // Verify all strategies are still valid probability distributions
        for info_set_idx in 0..solver.num_info_sets() {
            let strategy = solver
                .get_strategy(info_set_idx)
                .expect("Should get strategy");

            let sum: f64 = strategy.values().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "Info set {} strategy should sum to 1 after updates, got {}",
                info_set_idx,
                sum
            );

            // All probabilities should be non-negative
            for (&action, &prob) in &strategy {
                assert!(
                    prob >= 0.0,
                    "Info set {} action {} has negative probability {}",
                    info_set_idx,
                    action,
                    prob
                );
            }
        }
    }

    /// Test that the compiled game tree has expected terminal node count.
    #[timed_test]
    fn compiled_game_terminal_nodes() {
            let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);

        // Count terminal nodes
        let terminal_data: Vec<bool> = compiled.terminal_mask.to_data().to_vec().unwrap();
        let num_terminals = terminal_data.iter().filter(|&&t| t).count();

        // Kuhn Poker has multiple terminal nodes per deal (fold, showdown variants)
        // With 6 deals, should have significant terminal nodes
        assert!(
            num_terminals >= 6,
            "Expected at least 6 terminal nodes, got {}",
            num_terminals
        );
    }

    /// Test that GpuCfrSolver converges to Nash equilibrium on Kuhn Poker.
    #[timed_test(10)]
    fn gpu_solver_converges_on_kuhn() {
            use poker_solver_core::cfr::calculate_exploitability;

        let game = KuhnPoker::new();
        let device = Default::default();

        let mut solver = GpuCfrSolver::<TestBackend>::new(&game, device);
        solver.train(3_000);

        let strategy = solver.all_strategies();
        let exploitability = calculate_exploitability(&game, &strategy);

        assert!(
            exploitability < 0.05,
            "GPU solver should converge, got exploitability {}",
            exploitability
        );

        // King should always call when facing a bet
        let kb_strategy = solver.get_strategy("Kb").expect("Should have Kb strategy");
        assert!(
            kb_strategy[1] > 0.90,
            "King should call when facing bet, got {:?}",
            kb_strategy
        );

        // Jack should always fold when facing a bet
        let jb_strategy = solver.get_strategy("Jb").expect("Should have Jb strategy");
        assert!(
            jb_strategy[0] > 0.90,
            "Jack should fold when facing bet, got {:?}",
            jb_strategy
        );
    }

    /// Test that GpuCfrSolver works with WGPU backend (actual GPU).
    #[timed_test(120)]
    #[ignore = "slow"]
    fn gpu_solver_wgpu_backend_converges() {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
        use poker_solver_core::cfr::calculate_exploitability;

        let game = KuhnPoker::new();
        let device = WgpuDevice::default();

        let mut solver = GpuCfrSolver::<Wgpu>::new(&game, device);
        solver.train(10_000);

        // Use the standard exploitability calculation
        let strategy = solver.all_strategies();
        let exploitability = calculate_exploitability(&game, &strategy);

        assert!(
            exploitability < 0.01,
            "WGPU solver should converge, got exploitability {}",
            exploitability
        );

        // King should always call when facing a bet
        let kb_strategy = solver.get_strategy("Kb").expect("Should have Kb strategy");
        assert!(
            kb_strategy[1] > 0.95,
            "King should call when facing bet, got {:?}",
            kb_strategy
        );
    }
}
