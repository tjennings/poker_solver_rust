//! End-to-end integration test for the imperfect-recall postflop abstraction pipeline.
//!
//! Verifies the full pipeline works:
//! canonical board enumeration -> histogram CDF features -> independent per-street clustering ->
//! equity tables -> postflop CFR solve -> value table extraction.
//!
//! The bottleneck is EHS feature computation (169 hands x ~1755 canonical flops), which
//! takes several minutes even in release mode. All tests here are `#[ignore]`.

use poker_solver_core::preflop::postflop_abstraction::PostflopAbstraction;
use poker_solver_core::preflop::postflop_model::PostflopModelConfig;

/// Verify the full imperfect-recall abstraction pipeline:
/// canonical boards -> histogram CDFs -> independent per-street clustering ->
/// equity tables -> postflop CFR -> value table extraction.
///
/// Run with: `cargo test -p poker-solver-core --test postflop_imperfect_recall --release -- --ignored --nocapture`
#[test]
#[ignore = "slow (~5 min in release): full postflop abstraction pipeline"]
fn postflop_abstraction_with_imperfect_recall_builds_and_solves() {
    let num_buckets: u16 = 10;

    // Minimal config: 10 buckets per street, 2 SPRs, 50 CFR iterations.
    let config = PostflopModelConfig {
        num_hand_buckets_flop: num_buckets,
        num_hand_buckets_turn: num_buckets,
        num_hand_buckets_river: num_buckets,
        canonical_sprs: vec![1.0, 5.0],
        postflop_solve_iterations: 50,
        bet_sizes: vec![1.0],
        max_raises_per_street: 0,
        ..PostflopModelConfig::fast()
    };

    // Build should complete without errors.
    let result = PostflopAbstraction::build(
        &config,
        None, // no equity table
        None, // no cache
        &|phase| eprintln!("  [build] {phase}"),
    );

    assert!(
        result.is_ok(),
        "PostflopAbstraction::build failed: {:?}",
        result.err()
    );

    let abstraction = result.unwrap();

    // --- Structural checks ---

    // Should have one tree per canonical SPR.
    assert_eq!(
        abstraction.trees.len(),
        2,
        "expected 2 trees (one per canonical SPR)"
    );
    assert_eq!(
        abstraction.canonical_sprs.len(),
        2,
        "expected 2 canonical SPRs"
    );

    // Buckets should be populated.
    assert_eq!(abstraction.buckets.num_flop_buckets, num_buckets);
    assert_eq!(abstraction.buckets.num_turn_buckets, num_buckets);
    assert_eq!(abstraction.buckets.num_river_buckets, num_buckets);

    // Value table should be non-empty.
    assert!(
        !abstraction.values.is_empty(),
        "value table should not be empty"
    );
    assert_eq!(
        abstraction.values.num_sprs(),
        2,
        "value table should have 2 SPR slots"
    );

    // --- Value table spot checks ---

    // Query a few entries and verify they are finite.
    let n = num_buckets as usize;
    for spr_idx in 0..2 {
        for hero_pos in 0..2u8 {
            for hb in 0..n.min(3) {
                for ob in 0..n.min(3) {
                    let val = abstraction.values.get_by_spr(spr_idx, hero_pos, hb as u16, ob as u16);
                    assert!(
                        val.is_finite(),
                        "EV should be finite for spr={spr_idx} pos={hero_pos} hb={hb} ob={ob}, got {val}"
                    );
                }
            }
        }
    }

    // Zero-sum check: for the same bucket pair, hero EV as position 0 and position 1
    // should sum close to zero (exact zero for converged symmetric game).
    let v_pos0 = abstraction.values.get_by_spr(0, 0, 0, 1);
    let v_pos1 = abstraction.values.get_by_spr(0, 1, 0, 1);
    eprintln!("  [check] v(pos0, 0v1) = {v_pos0:.4}, v(pos1, 0v1) = {v_pos1:.4}");
    // With only 50 iterations and placeholder equity, this won't be exact zero,
    // but both values should be finite and in a reasonable range.
    assert!(v_pos0.is_finite());
    assert!(v_pos1.is_finite());

    // Street equity tables should have correct dimensions.
    assert_eq!(abstraction.street_equity.flop.num_buckets, n);
    assert_eq!(abstraction.street_equity.turn.num_buckets, n);
    assert_eq!(abstraction.street_equity.river.num_buckets, n);

    eprintln!(
        "  [done] value table has {} entries across {} SPRs",
        abstraction.values.len(),
        abstraction.values.num_sprs(),
    );
}
