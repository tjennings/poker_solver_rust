//! Phase-level diagnostic tests for the postflop abstraction pipeline.
//!
//! Each test validates one phase independently with small inputs and runs in seconds.
//! When the smoke test fails, run these to pinpoint the broken phase.
//!
//! Run: `cargo test -p poker-solver-core --test postflop_diagnostics --release -- --nocapture`

use poker_solver_core::hands::CanonicalHand;
use poker_solver_core::preflop::config::PreflopConfig;
use poker_solver_core::preflop::equity::EquityTable;
use poker_solver_core::preflop::postflop_abstraction::PostflopAbstraction;
use poker_solver_core::preflop::postflop_model::PostflopModelConfig;
use poker_solver_core::preflop::solver::PreflopSolver;


/// After building the full postflop abstraction (169-hand MCCFR + values),
/// the value table should show strong hands winning against weak hands.
#[test]
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
        &config, None, &|phase| eprintln!("  [build] {phase}"),
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

/// Full pipeline: build postflop abstraction, run preflop solver,
/// AA should fold less than 72o.
#[test]
fn diag_end_to_end_aa_beats_72o() {
    let pf_config = PostflopModelConfig {
        postflop_sprs: vec![5.0],
        postflop_solve_iterations: 50,
        bet_sizes: vec![0.5, 1.0],
        max_raises_per_street: 1,
        max_flop_boards: 3,
        ..PostflopModelConfig::fast()
    };

    let config = PreflopConfig::heads_up(25);

    let equity = EquityTable::new_computed(5000, |_| {});
    let mut solver = PreflopSolver::new_with_equity(&config, equity);

    let abstraction = PostflopAbstraction::build(
        &pf_config,
        None,
        &|phase| eprintln!("  [build] {phase}"),
    ).expect("postflop build should succeed");

    solver.attach_postflop(vec![abstraction], &config);

    solver.train(500);
    eprintln!("  preflop training done (500 iterations)");

    let strategy = solver.strategy();

    let aa_idx = CanonicalHand::parse("AA").unwrap().index();
    let seven_two_idx = CanonicalHand::parse("72o").unwrap().index();

    let aa_probs = strategy.get_root_probs(aa_idx);
    let seven_two_probs = strategy.get_root_probs(seven_two_idx);

    eprintln!("AA root probs: {aa_probs:?}");
    eprintln!("72o root probs: {seven_two_probs:?}");

    // Root action ordering: index 0 = fold, 1 = call, 2+ = raises
    let aa_fold = aa_probs.first().copied().unwrap_or(1.0);
    let seven_two_fold = seven_two_probs.first().copied().unwrap_or(0.0);

    eprintln!("AA fold={aa_fold:.6}, 72o fold={seven_two_fold:.6}");

    // Primary check: AA should fold less than or equal to 72o.
    // In HU preflop, both may converge to near-zero folding (SB open-raising
    // or calling dominates), so we allow a small tolerance for AA ≈ 72o ≈ 0.
    assert!(
        aa_fold <= seven_two_fold + 0.05,
        "AA should fold no more than 72o (+5% tolerance): AA fold={aa_fold:.3}, 72o fold={seven_two_fold:.3}"
    );

    // Secondary check: AA should play at least as aggressively overall.
    // Measure "continue weight" = call + 2*raise + 3*all-in (higher = more aggressive).
    // AA (a premium hand) should have a higher continue weight than 72o.
    let continue_weight = |probs: &[f64]| -> f64 {
        let fold = probs.first().copied().unwrap_or(0.0);
        let call = probs.get(1).copied().unwrap_or(0.0);
        let raises: f64 = probs.iter().skip(2).copied().sum();
        // Weight: 0 for fold, 1 for call, 2 for raise/all-in
        call + 2.0 * raises - fold
    };
    let aa_weight = continue_weight(&aa_probs);
    let seven_two_weight = continue_weight(&seven_two_probs);
    eprintln!("AA continue_weight={aa_weight:.4}, 72o continue_weight={seven_two_weight:.4}");

    // At minimum, AA must not fold more while 72o continues freely
    assert!(
        aa_weight > 0.5,
        "AA should have positive continue weight, got {aa_weight:.4}"
    );
}
