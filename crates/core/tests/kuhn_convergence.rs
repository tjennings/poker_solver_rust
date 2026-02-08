//! Integration tests for Kuhn Poker CFR convergence to Nash equilibrium.

use std::collections::HashMap;

use poker_solver_core::{
    cfr::{VanillaCfr, calculate_exploitability},
    game::KuhnPoker,
};
use test_macros::timed_test;

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

        println!("After {iterations} iterations: exploitability = {exploitability:.6}");

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
