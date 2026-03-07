//! Phase-level diagnostic tests for the postflop abstraction pipeline.
//!
//! Each test validates one phase independently with small inputs and runs in seconds.
//! When the smoke test fails, run these to pinpoint the broken phase.
//!
//! Run: `cargo test -p poker-solver-core --test postflop_diagnostics --release -- --nocapture`

use poker_solver_core::preflop::postflop_abstraction::PostflopAbstraction;
use poker_solver_core::preflop::postflop_model::PostflopModelConfig;
use test_macros::timed_test;


/// After building the full postflop abstraction (169-hand MCCFR + values),
/// the value table should show strong hands winning against weak hands.
#[timed_test(10)]
#[ignore = "slow: builds full postflop abstraction pipeline"]
fn diag_value_table_strong_beats_weak() {
    let config = PostflopModelConfig {
        postflop_sprs: vec![5.0],
        postflop_solve_iterations: 100,
        bet_sizes: vec![1.0],
        max_raises_per_street: 0,
        max_flop_boards: 3,
        ..PostflopModelConfig::fast()
    };

    let abstraction = PostflopAbstraction::build(
        &config, None, |phase| eprintln!("  [build] {phase}"),
    ).expect("build should succeed");

    let n = 169;
    // Determine strongest/weakest hands by average EV as hero (pos 0) across opponents
    let mut best_hand = 0;
    let mut worst_hand = 0;
    let mut best_ev = f64::NEG_INFINITY;
    let mut worst_ev = f64::INFINITY;
    for b in 0..n {
        let avg_ev: f64 = (0..n)
            .map(|o| abstraction.values.get_by_flop(0, 0, b as u16, o as u16))
            .sum::<f64>() / n as f64;
        if avg_ev > best_ev { best_ev = avg_ev; best_hand = b; }
        if avg_ev < worst_ev { worst_ev = avg_ev; worst_hand = b; }
    }

    eprintln!("strongest hand: {best_hand} (avg EV {best_ev:.4})");
    eprintln!("weakest hand: {worst_hand} (avg EV {worst_ev:.4})");

    let strong_ev = abstraction.values.get_by_flop(
        0, 0, best_hand as u16, worst_hand as u16,
    );
    let weak_ev = abstraction.values.get_by_flop(
        0, 0, worst_hand as u16, best_hand as u16,
    );

    eprintln!("strong_hero EV={strong_ev:.4}, weak_hero EV={weak_ev:.4}");
    assert!(
        strong_ev > weak_ev,
        "strong hand should have higher EV: {strong_ev} vs {weak_ev}"
    );
}
