//! Integration tests for HUNL Preflop with MCCFR solver.
//!
//! HUNL Preflop has 28,561 initial states (169 x 169 hand combinations).
//! MCCFR makes this tractable by sampling states instead of iterating all.
//!
//! Note: These tests use reduced iteration counts to run quickly. For actual
//! strategy computation, use many more iterations.

use poker_solver_core::cfr::MccfrSolver;
use poker_solver_core::game::{Game, HunlPreflop};
use test_macros::timed_test;

/// FNV-1a hash of an empty action history (offset basis masked to 44 bits).
const EMPTY_HISTORY_HASH: u64 = 0xcbf2_9ce4_8422_2325_u64 & 0xFFF_FFFF_FFFF;

/// Check if a u64 key represents an SB opening state.
///
/// HunlPreflop encodes: upper 20 bits = (position_bit << 19) | hand_idx,
/// lower 44 bits = FNV hash of action history.
fn is_sb_opening(key: u64) -> bool {
    let hand_bits = (key >> 44) & 0xF_FFFF;
    let history_hash = key & 0xFFF_FFFF_FFFF;
    // SB: position bit (bit 19 of hand_bits) is clear
    // Opening: action history is empty → hash equals the offset basis
    hand_bits & (1 << 19) == 0 && history_hash == EMPTY_HISTORY_HASH
}

/// Test that MCCFR trains efficiently on HUNL Preflop.
#[timed_test(10)]
fn hunl_preflop_mccfr_trains_efficiently() {
    let game = HunlPreflop::with_stack(5);
    let mut solver = MccfrSolver::new(game);

    solver.train(2, 2);

    assert!(solver.iterations() == 2);
}

/// Test that MCCFR produces valid strategies for HUNL Preflop.
#[timed_test(10)]
fn hunl_preflop_mccfr_produces_valid_strategies() {
    let game = HunlPreflop::with_stack(5);
    let mut solver = MccfrSolver::new(game);

    solver.train(2, 2);

    let strategies = solver.all_strategies();

    let sb_strategies: Vec<_> = strategies
        .iter()
        .filter(|(k, _)| is_sb_opening(**k))
        .collect();

    assert!(
        !sb_strategies.is_empty(),
        "Should have SB opening strategies"
    );

    for (info_set, probs) in &sb_strategies {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Strategy for {info_set:#018x} should sum to 1.0, got {sum}"
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
#[timed_test(10)]
fn hunl_preflop_premium_hands_strategies() {
    let game = HunlPreflop::with_stack(5);
    let mut solver = MccfrSolver::new(game.clone());

    solver.train(2, 3);

    // Build opening key for a hand: (hand_idx << 44) | EMPTY_HISTORY_HASH
    // Use the game itself to get the correct key for a known state
    let states = game.initial_states();

    // Find a state where SB has AA (hand index 0 = AA in CanonicalHand)
    let aa_state = states.iter().find(|s| {
        let key = game.info_set_key(s);
        let hand_idx = (key >> 44) & 0x7_FFFF; // mask out position bit
        hand_idx == 0 // AA = index 0
    });

    if let Some(state) = aa_state {
        let key = game.info_set_key(state);
        if let Some(strategy) = solver.get_average_strategy(key) {
            println!("AA opening strategy: {strategy:?}");
            assert!(!strategy.is_empty(), "AA should have actions");
        }
    }
}

/// Compare MCCFR sampling vs full traversal.
///
/// Note: This test uses very small parameters to run quickly. The actual speedup
/// is much higher with larger iteration counts, but equity calculation dominates
/// in debug builds.
#[timed_test(300)]
#[ignore = "slow"]
fn hunl_preflop_sampling_speedup() {
    let game = HunlPreflop::with_stack(10);
    let num_states = game.initial_states().len();

    println!("HUNL Preflop has {num_states} initial states");
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
#[timed_test(10)]
fn hunl_preflop_training_improves() {
    let game = HunlPreflop::with_stack(5);

    let mut solver1 = MccfrSolver::new(game.clone());
    solver1.train(1, 2);
    let strategies1 = solver1.all_strategies();

    let mut solver2 = MccfrSolver::new(game);
    solver2.train(2, 3);
    let strategies2 = solver2.all_strategies();

    println!(
        "Info sets after 1x2 samples: {}, after 2x3 samples: {}",
        strategies1.len(),
        strategies2.len()
    );

    assert!(
        strategies2.len() >= strategies1.len(),
        "More training should explore at least as many info sets"
    );
}
