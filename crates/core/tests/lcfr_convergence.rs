//! Integration test: verify LCFR (Linear CFR) converges on Kuhn Poker.

use poker_solver_core::{
    cfr::{SequenceCfrConfig, SequenceCfrSolver, materialize},
    game::{Game, KuhnPoker},
    info_key::InfoKey,
};

/// Build per-deal trees and DealInfo for all 6 Kuhn deals.
fn kuhn_trees_and_deals() -> (
    Vec<poker_solver_core::cfr::GameTree>,
    Vec<poker_solver_core::cfr::DealInfo>,
) {
    let game = KuhnPoker::new();
    let states = game.initial_states();
    let mut trees = Vec::new();
    let mut deals = Vec::new();

    for state in &states {
        let tree = materialize(&game, state);
        let key_p1 = game.info_set_key(state);
        let hand_bits_p1 = InfoKey::from_raw(key_p1).hand_bits();

        let next = game.next_state(state, game.actions(state)[0]);
        let key_p2 = game.info_set_key(&next);
        let hand_bits_p2 = InfoKey::from_raw(key_p2).hand_bits();

        let p1_equity = if hand_bits_p1 > hand_bits_p2 {
            1.0
        } else {
            0.0
        };

        trees.push(tree);
        deals.push(poker_solver_core::cfr::DealInfo {
            hand_bits_p1: [hand_bits_p1; 4],
            hand_bits_p2: [hand_bits_p2; 4],
            p1_equity,
            weight: 1.0,
        });
    }

    (trees, deals)
}

#[test]
fn lcfr_converges_on_kuhn_poker() {
    let (trees, deals) = kuhn_trees_and_deals();
    let config = SequenceCfrConfig::linear_cfr();

    // Verify LCFR parameters
    assert!((config.dcfr_alpha - 1.0).abs() < f64::EPSILON);
    assert!((config.dcfr_beta - 1.0).abs() < f64::EPSILON);
    assert!((config.dcfr_gamma - 1.0).abs() < f64::EPSILON);

    let mut solver = SequenceCfrSolver::from_per_deal_trees(trees, deals, config);
    solver.train(10_000);

    let strategies = solver.all_strategies();

    // Verify all strategies are valid distributions (sum to ~1.0)
    for (key, probs) in &strategies {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Strategy for key {key:#x} doesn't sum to 1.0: {sum}"
        );
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p >= 0.0,
                "Strategy for key {key:#x} action {i} is negative: {p}"
            );
        }
    }

    // Check known Nash equilibrium properties:
    // King should always call when facing a bet
    let king_bet_key = InfoKey::new(2, 0, 0, &[4]).as_u64();
    if let Some(probs) = strategies.get(&king_bet_key) {
        assert!(
            probs[1] > 0.90,
            "King should always call a bet, got fold={:.4}, call={:.4}",
            probs[0],
            probs[1]
        );
    }

    // Jack should always fold when facing a bet
    let jack_bet_key = InfoKey::new(0, 0, 0, &[4]).as_u64();
    if let Some(probs) = strategies.get(&jack_bet_key) {
        assert!(
            probs[0] > 0.90,
            "Jack should always fold facing a bet, got fold={:.4}, call={:.4}",
            probs[0],
            probs[1]
        );
    }
}
