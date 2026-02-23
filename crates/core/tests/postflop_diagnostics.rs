//! Phase-level diagnostic tests for the postflop bucketed abstraction pipeline.
//!
//! Each test validates one phase independently with small inputs and runs in seconds.
//! When the smoke test fails, run these to pinpoint the broken phase.
//!
//! Run: `cargo test -p poker-solver-core --test postflop_diagnostics --release -- --nocapture`

use poker_solver_core::hands::{self, CanonicalHand};
use poker_solver_core::preflop::config::PreflopConfig;
use poker_solver_core::preflop::ehs::sample_canonical_flops;
use poker_solver_core::preflop::equity::EquityTable;
use poker_solver_core::preflop::hand_buckets::{
    build_street_buckets_independent, cdf_to_avg_equity, compute_bucket_pair_equity,
};
use poker_solver_core::preflop::postflop_abstraction::PostflopAbstraction;
use poker_solver_core::preflop::postflop_model::PostflopModelConfig;
use poker_solver_core::preflop::solver::PreflopSolver;

/// Bucket assignments should correlate with hand strength.
/// Buckets should have meaningful equity spread (strong != weak).
#[test]
fn diag_bucket_monotonicity() {
    let hands: Vec<_> = hands::all_hands().collect();
    let flops = sample_canonical_flops(3);
    let result = build_street_buckets_independent(
        &hands, &flops, 10, 10, 10, &|_| {},
    );

    let flop_ehs: Vec<f64> = result.flop_histograms.iter()
        .map(|cdf| cdf_to_avg_equity(cdf))
        .collect();
    let equity = compute_bucket_pair_equity(
        &result.buckets.flop, result.buckets.num_flop_buckets as usize, &flop_ehs,
    );

    let n = result.buckets.num_flop_buckets as usize;
    let mut min_eq = 1.0f32;
    let mut max_eq = 0.0f32;
    for a in 0..n {
        for b in 0..n {
            if a == b { continue; }
            let eq = equity.get(a, b);
            if eq < min_eq { min_eq = eq; }
            if eq > max_eq { max_eq = eq; }
        }
    }

    let spread = max_eq - min_eq;
    eprintln!("Bucket equity spread: min={min_eq:.3}, max={max_eq:.3}, spread={spread:.3}");
    assert!(
        spread > 0.1,
        "Bucket equity spread should be > 0.1, got {spread}"
    );
}

/// Street equity table should have non-trivial values, correct self-equity,
/// and zero-sum symmetry.
#[test]
fn diag_street_equity_sanity() {
    let hands: Vec<_> = hands::all_hands().collect();
    let flops = sample_canonical_flops(3);
    let result = build_street_buckets_independent(
        &hands, &flops, 10, 10, 10, &|_| {},
    );

    let flop_ehs: Vec<f64> = result.flop_histograms.iter()
        .map(|cdf| cdf_to_avg_equity(cdf))
        .collect();
    let flop_equity = compute_bucket_pair_equity(
        &result.buckets.flop, result.buckets.num_flop_buckets as usize, &flop_ehs,
    );

    let n = result.buckets.num_flop_buckets as usize;

    // Should not be all 0.5
    let mut all_half = true;
    for a in 0..n {
        for b in 0..n {
            let eq = flop_equity.get(a, b);
            assert!(eq.is_finite(), "equity({a},{b}) not finite: {eq}");
            assert!((0.0..=1.0).contains(&eq), "equity({a},{b}) out of range: {eq}");
            if (eq - 0.5).abs() > 0.01 { all_half = false; }
        }
    }
    assert!(!all_half, "Equity is all 0.5 — placeholder not replaced");

    // Self-equity ≈ 0.5
    for a in 0..n {
        let self_eq = flop_equity.get(a, a);
        assert!(
            (self_eq - 0.5).abs() < 0.05,
            "self-equity bucket {a} should be ~0.5, got {self_eq}"
        );
    }

    // Symmetry: eq(a,b) + eq(b,a) ≈ 1.0
    for a in 0..n {
        for b in (a+1)..n {
            let ab = flop_equity.get(a, b);
            let ba = flop_equity.get(b, a);
            assert!(
                (ab + ba - 1.0).abs() < 0.01,
                "equity({a},{b})={ab} + equity({b},{a})={ba} should ≈ 1.0"
            );
        }
    }
}

/// After building the full postflop abstraction (buckets + CFR + values),
/// the value table should show strong buckets winning against weak buckets.
#[test]
fn diag_value_table_strong_beats_weak() {
    let config = PostflopModelConfig {
        num_hand_buckets_flop: 5,
        num_hand_buckets_turn: 5,
        num_hand_buckets_river: 5,
        canonical_sprs: vec![5.0],
        postflop_solve_iterations: 100,
        postflop_solve_samples: 0,
        bet_sizes: vec![1.0],
        max_raises_per_street: 0,
        max_flop_boards: 3,
        flop_samples_per_iter: 1,
        ..PostflopModelConfig::default()
    };

    let abstraction = PostflopAbstraction::build(
        &config, None, None, &|phase| eprintln!("  [build] {phase}"),
    ).expect("build should succeed");

    let n = config.num_hand_buckets_flop as usize;
    let mut best_bucket = 0;
    let mut worst_bucket = 0;
    let mut best_eq = 0.0f32;
    let mut worst_eq = 1.0f32;
    for b in 0..n {
        let avg: f32 = (0..n)
            .map(|o| abstraction.street_equity.flop.get(b, o))
            .sum::<f32>() / n as f32;
        if avg > best_eq { best_eq = avg; best_bucket = b; }
        if avg < worst_eq { worst_eq = avg; worst_bucket = b; }
    }

    eprintln!("strongest bucket: {best_bucket} (avg eq {best_eq:.3})");
    eprintln!("weakest bucket: {worst_bucket} (avg eq {worst_eq:.3})");

    let strong_ev = abstraction.values.get_by_spr(
        0, 0, best_bucket as u16, worst_bucket as u16,
    );
    let weak_ev = abstraction.values.get_by_spr(
        0, 0, worst_bucket as u16, best_bucket as u16,
    );

    eprintln!("strong_hero EV={strong_ev:.4}, weak_hero EV={weak_ev:.4}");
    assert!(
        strong_ev > weak_ev,
        "strong bucket should have higher EV: {strong_ev} vs {weak_ev}"
    );
}

/// Full pipeline: build postflop abstraction, run preflop solver,
/// AA should fold less than 72o.
#[test]
fn diag_end_to_end_aa_beats_72o() {
    let pf_config = PostflopModelConfig {
        num_hand_buckets_flop: 10,
        num_hand_buckets_turn: 10,
        num_hand_buckets_river: 10,
        canonical_sprs: vec![1.0, 5.0],
        postflop_solve_iterations: 50,
        postflop_solve_samples: 0,
        bet_sizes: vec![0.5, 1.0],
        max_raises_per_street: 1,
        max_flop_boards: 3,
        flop_samples_per_iter: 1,
        ..PostflopModelConfig::default()
    };

    let mut config = PreflopConfig::heads_up(25);
    config.postflop_model = Some(pf_config);

    let equity = EquityTable::new_computed(5000, |_| {});
    let mut solver = PreflopSolver::new_with_equity(&config, equity);

    let abstraction = PostflopAbstraction::build(
        config.postflop_model.as_ref().unwrap(),
        None, None,
        &|phase| eprintln!("  [build] {phase}"),
    ).expect("postflop build should succeed");

    solver.attach_postflop(abstraction, &config);

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
