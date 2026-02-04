//! Integration tests for Kuhn Poker CFR convergence to Nash equilibrium.

use std::collections::HashMap;

use poker_solver_core::{
    cfr::{VanillaCfr, calculate_exploitability},
    game::KuhnPoker,
};

/// Test that CFR converges to near-Nash equilibrium on Kuhn Poker.
///
/// Kuhn Poker has a known Nash equilibrium. After sufficient iterations,
/// the average strategy should have exploitability below 0.001 (0.1% of the pot).
#[test]
fn kuhn_reaches_nash_equilibrium() {
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game.clone());

    // Train for many iterations
    solver.train(100_000);

    // Extract the average strategy
    let strategy = extract_strategy(&solver);

    // Compute exploitability
    let exploitability = calculate_exploitability(&game, &strategy);

    // Nash equilibrium should have exploitability < 0.001
    assert!(
        exploitability < 0.001,
        "Kuhn Poker should converge to Nash equilibrium, but exploitability is {exploitability}"
    );
}

/// Test known Nash equilibrium properties for Kuhn Poker.
///
/// In the Nash equilibrium:
/// - With a King, always bet/call (never fold)
/// - With a Jack facing a bet, always fold
/// - Queen has mixed strategies
#[test]
fn kuhn_nash_strategy_properties() {
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game);

    solver.train(100_000);

    // King should always call when facing a bet
    if let Some(strategy) = solver.get_average_strategy("Kb") {
        // strategy[0] = fold, strategy[1] = call
        assert!(
            strategy[1] > 0.99,
            "King should always call a bet, got fold={:.4}, call={:.4}",
            strategy[0],
            strategy[1]
        );
    }

    // King should always call after check-bet
    if let Some(strategy) = solver.get_average_strategy("Kcb") {
        assert!(
            strategy[1] > 0.99,
            "King should always call after check-bet, got fold={:.4}, call={:.4}",
            strategy[0],
            strategy[1]
        );
    }

    // Jack should always fold when facing a bet (Jb)
    if let Some(strategy) = solver.get_average_strategy("Jb") {
        // strategy[0] = fold, strategy[1] = call
        assert!(
            strategy[0] > 0.99,
            "Jack should always fold when facing a bet, got fold={:.4}, call={:.4}",
            strategy[0],
            strategy[1]
        );
    }

    // Jack should always fold after check-bet (Jcb)
    if let Some(strategy) = solver.get_average_strategy("Jcb") {
        assert!(
            strategy[0] > 0.99,
            "Jack should always fold after check-bet, got fold={:.4}, call={:.4}",
            strategy[0],
            strategy[1]
        );
    }
}

/// Test that exploitability monotonically decreases (on average) with more iterations.
#[test]
fn exploitability_decreases_over_training() {
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game.clone());

    let checkpoints = [100, 1_000, 10_000, 100_000];
    let mut prev_exploitability = f64::MAX;

    for &iterations in &checkpoints {
        // Train to this checkpoint
        let current_iterations = if iterations == 100 {
            100
        } else {
            iterations - checkpoints[checkpoints.iter().position(|&x| x == iterations).unwrap() - 1]
        };
        solver.train(current_iterations as u64);

        let strategy = extract_strategy(&solver);
        let exploitability = calculate_exploitability(&game, &strategy);

        println!(
            "After {} iterations: exploitability = {:.6}",
            iterations, exploitability
        );

        // Exploitability should generally decrease
        // (small fluctuations possible, but trend should be down)
        if iterations >= 1_000 {
            assert!(
                exploitability < prev_exploitability * 1.5, // Allow some fluctuation
                "Exploitability should trend downward: prev={prev_exploitability}, current={exploitability}"
            );
        }
        prev_exploitability = exploitability;
    }

    // Final exploitability should be very low
    assert!(
        prev_exploitability < 0.001,
        "Final exploitability should be < 0.001, got {prev_exploitability}"
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
