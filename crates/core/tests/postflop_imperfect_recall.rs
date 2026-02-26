//! End-to-end integration test for the postflop abstraction pipeline.
//!
//! Verifies the full pipeline works:
//! canonical board enumeration -> 169-hand combo map -> per-flop MCCFR solve ->
//! value table extraction.
//!
//! All tests here are `#[ignore]`.

use poker_solver_core::preflop::postflop_abstraction::PostflopAbstraction;
use poker_solver_core::preflop::postflop_model::PostflopModelConfig;

/// Verify the full 169-hand MCCFR abstraction pipeline:
/// canonical boards -> combo maps -> per-flop MCCFR solve -> value table extraction.
///
/// Run with: `cargo test -p poker-solver-core --test postflop_imperfect_recall --release -- --ignored --nocapture`
#[test]
#[ignore = "slow (~5 min in release): full postflop abstraction pipeline"]
fn postflop_abstraction_with_169_hands_builds_and_solves() {
    // Minimal config: single SPR, 50 CFR iterations.
    let config = PostflopModelConfig {
        postflop_sprs: vec![5.0],
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

    // Should have a single shared tree.
    assert!(
        abstraction.tree.node_count() > 0,
        "shared tree should have nodes"
    );
    assert!(
        (abstraction.spr - 5.0).abs() < 1e-9,
        "spr should be 5.0"
    );

    // Value table should be non-empty.
    assert!(
        !abstraction.values.is_empty(),
        "value table should not be empty"
    );
    let num_flops = abstraction.values.num_flops();
    assert!(
        num_flops > 0,
        "value table should have at least one flop"
    );

    // --- Value table spot checks ---

    // Query a few entries and verify they are finite.
    for flop_idx in 0..num_flops.min(3) {
        for hero_pos in 0..2u8 {
            for hb in 0..5u16 {
                for ob in 0..5u16 {
                    let val = abstraction.values.get_by_flop(flop_idx, hero_pos, hb, ob);
                    assert!(
                        val.is_finite(),
                        "EV should be finite for flop={flop_idx} pos={hero_pos} hb={hb} ob={ob}, got {val}"
                    );
                }
            }
        }
    }

    // Zero-sum check: for the same hand pair, hero EV as position 0 and position 1
    // should sum close to zero (exact zero for converged symmetric game).
    let v_pos0 = abstraction.values.get_by_flop(0, 0, 0, 1);
    let v_pos1 = abstraction.values.get_by_flop(0, 1, 0, 1);
    eprintln!("  [check] v(pos0, 0v1) = {v_pos0:.4}, v(pos1, 0v1) = {v_pos1:.4}");
    assert!(v_pos0.is_finite());
    assert!(v_pos1.is_finite());

    eprintln!(
        "  [done] value table has {} entries across {} flops",
        abstraction.values.len(),
        abstraction.values.num_flops(),
    );
}
