---
# poker_solver_rust-7wfv
title: Align EMD k-means training metric with weighted assignment
status: completed
type: bug
priority: normal
created_at: 2026-05-06T16:39:11Z
updated_at: 2026-05-06T20:24:46Z
parent: poker_solver_rust-hfnv
---

cluster_histogram_exhaustive trains centroids with unweighted EMD, then exhaustive assignment can use weighted EMD gaps derived from child bucket EVs. The sampled centroid objective and final assignment metric should match.

Acceptance:
- Decide whether weighted child EV gaps should apply during both k-means and assignment or neither
- Implement weighted EMD centroid assignment/update path if needed
- Add tests showing training labels and exhaustive assignment use the same distance semantics
- Re-run/record cluster diagnostics comparing before/after separation and intra-bucket EMD

## Work Start

Started after a clean worktree check and a passing full pre-change test suite. Plan: inspect the histogram EMD implementation, make sampled k-means use the same child-EV-gap weighted distance semantics as exhaustive assignment when gaps are available, and add focused tests for the metric selection.

## Summary of Changes

Completed the metric alignment decision and implementation: child centroid EV gaps now apply to both sampled Elkan EMD k-means training and exhaustive assignment whenever gaps are available; otherwise the existing unweighted EMD path is preserved. Added weighted f64 EMD, gap-aware Elkan helper paths for seeding, bounds, reassignment, reseeding, drift, and final labels, and focused tests proving uniform gaps match unweighted behavior and final labels match weighted nearest-centroid semantics.

Verification:
- `cargo test -p poker-solver-core blueprint_v2::clustering` passed.
- `cargo test` passed after implementation.
- Warm-cache `time cargo test` passed in 56.727s.
- `cargo fmt -p poker-solver-core --check` still reports pre-existing formatting drift in unrelated files (`blueprint_mp/game_tree.rs`, `blueprint_mp/trainer.rs`, `nut_features.rs`); the new clustering edit was manually adjusted to match rustfmt.

Artifact diagnostics note: this code change does not regenerate bucket artifacts, so there is no before/after bucket directory pair to compare yet. The new metric-level tests cover the training/assignment semantic mismatch; once buckets are regenerated, run `diag-clusters --audit` and `diff-clusters` against the old and new directories to capture separation/intra-bucket diagnostics.
