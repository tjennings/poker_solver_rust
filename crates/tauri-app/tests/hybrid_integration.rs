//! Integration tests for the hybrid MCCFR solver mode.
//!
//! These tests require a trained blueprint + snapshot + spot to be available.
//! Real validation is done via the `compare-solve` CLI (Phase 10).
//!
//! Run with:
//! ```
//! BLUEPRINT_TEST_DIR=/path/to/blueprint cargo test -p poker-solver-tauri -- --ignored hybrid
//! ```

/// When depth_limit exceeds the tree depth, hybrid mode should produce
/// strategies identical to exact mode (no boundaries are reached, sampling
/// is never invoked).
///
/// TODO(Phase 4B): This test needs full fixture loading infrastructure
/// (bundle + snapshot + spot) which does not exist as reusable helpers.
/// Manual validation via `compare-solve` on the izod repro is the real
/// acceptance gate for Phase 10.
#[test]
#[ignore = "requires blueprint fixtures — validate via compare-solve"]
fn hybrid_equals_exact_when_depth_exceeds_tree() {
    // Placeholder: when fixture infrastructure is available, this test should:
    // 1. Load a bundle + snapshot + spot
    // 2. Run exact solve (mode="exact") for N iterations
    // 3. Run hybrid solve (mode="hybrid", depth_limit=99) for N iterations
    // 4. Compare resulting strategies byte-for-byte
    //
    // Since depth_limit=99 exceeds any tree depth, no boundaries are reached
    // and hybrid degenerates to exact.
}

/// Smoke test: hybrid mode with depth_limit=0 should complete without panicking.
///
/// TODO(Phase 4B): Same fixture dependency as above.
#[test]
#[ignore = "requires blueprint fixtures — validate via compare-solve"]
fn hybrid_depth_zero_completes_without_panic() {
    // Placeholder: when fixture infrastructure is available, this test should:
    // 1. Load a bundle + snapshot + spot
    // 2. Run hybrid solve (mode="hybrid", depth_limit=0, iters=10)
    // 3. Assert it completes without panicking
    // 4. Assert exploitability is finite
}
