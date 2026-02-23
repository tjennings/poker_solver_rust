//! Phase-level diagnostic tests for the postflop bucketed abstraction pipeline.
//!
//! Each test validates one phase independently with small inputs and runs in seconds.
//! When the smoke test fails, run these to pinpoint the broken phase.
//!
//! Run: `cargo test -p poker-solver-core --test postflop_diagnostics --release -- --nocapture`

use poker_solver_core::hands;
use poker_solver_core::preflop::ehs::sample_canonical_flops;
use poker_solver_core::preflop::hand_buckets::{
    build_street_buckets_independent, cdf_to_avg_equity, compute_bucket_pair_equity,
};

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
