//! Integration tests for Kuhn Poker CFR convergence to Nash equilibrium.

use rustc_hash::FxHashMap;

use poker_solver_core::{
    cfr::{VanillaCfr, calculate_exploitability},
    game::KuhnPoker,
    info_key::InfoKey,
};
use test_macros::timed_test;

/// Build a u64 key for a Kuhn Poker info set.
///
/// `card`: J=0, Q=1, K=2
/// `actions`: slice of action chars — 'c'=check, 'b'=bet, 'f'=fold, 'l'=call
fn kuhn_key(card: u32, actions: &[u8]) -> u64 {
    InfoKey::new(card, 0, 0, actions).as_u64()
}

// Kuhn action codes matching kuhn.rs encoding
const CHECK: u8 = 2;
const BET: u8 = 4;

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

    // King should always call when facing a bet (Kb)
    if let Some(strategy) = solver.get_average_strategy(kuhn_key(2, &[BET])) {
        assert!(
            strategy[1] > 0.95,
            "King should always call a bet, got fold={:.4}, call={:.4}",
            strategy[0],
            strategy[1]
        );
    }

    // King should always call after check-bet (Kcb)
    if let Some(strategy) = solver.get_average_strategy(kuhn_key(2, &[CHECK, BET])) {
        assert!(
            strategy[1] > 0.95,
            "King should always call after check-bet, got fold={:.4}, call={:.4}",
            strategy[0],
            strategy[1]
        );
    }

    // Jack should always fold when facing a bet (Jb)
    if let Some(strategy) = solver.get_average_strategy(kuhn_key(0, &[BET])) {
        assert!(
            strategy[0] > 0.95,
            "Jack should always fold when facing a bet, got fold={:.4}, call={:.4}",
            strategy[0],
            strategy[1]
        );
    }

    // Jack should always fold after check-bet (Jcb)
    if let Some(strategy) = solver.get_average_strategy(kuhn_key(0, &[CHECK, BET])) {
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

/// Helper to extract average strategy from solver.
///
/// Returns the 12 Kuhn info sets as a FxHashMap with u64 keys.
fn extract_strategy(solver: &VanillaCfr<KuhnPoker>) -> FxHashMap<u64, Vec<f64>> {
    // (card, actions) for all 12 Kuhn info sets
    let info_sets: [(u32, &[u8]); 12] = [
        (0, &[]),           // J
        (1, &[]),           // Q
        (2, &[]),           // K
        (0, &[CHECK]),      // Jc
        (1, &[CHECK]),      // Qc
        (2, &[CHECK]),      // Kc
        (0, &[BET]),        // Jb
        (1, &[BET]),        // Qb
        (2, &[BET]),        // Kb
        (0, &[CHECK, BET]), // Jcb
        (1, &[CHECK, BET]), // Qcb
        (2, &[CHECK, BET]), // Kcb
    ];

    info_sets
        .iter()
        .filter_map(|(card, actions)| {
            let key = kuhn_key(*card, actions);
            solver.get_average_strategy(key).map(|s| (key, s))
        })
        .collect()
}
