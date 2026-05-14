---
# poker_solver_rust-tqwo
title: Bucket CFVNet boundary errors by runtime class
status: completed
type: task
priority: high
created_at: 2026-05-14T04:46:17Z
updated_at: 2026-05-14T05:03:59Z
parent: poker_solver_rust-lnpl
---

Extend the diagnostic path to break candidate-vs-oracle errors down by player, remaining stack, pot, SPR, all-in state, reach sparsity, target magnitude, and boundary action class. The goal is to identify the dominant failure class instead of averaging it away.

## Summary of Changes

Extended `compare-solve --dump-boundary-cfvs` with aggregate bucket summaries by player/all-in state, pot, SPR, reach density, and oracle magnitude. The existing per-boundary and aggregate candidate-vs-oracle output is unchanged; the bucket section prints immediately before the boundary CFV gate check.

Added focused unit tests for bucket classification and summary averaging, and updated `docs/training.md` to describe the new bucket output.

Verification passed:
- `cargo fmt`
- `cargo test -p poker-solver-trainer boundary_cfv`
- `git diff --check`

Ran the corrected v5 diagnostic with `model.onnx` and `direct_normalized_legacy`. The gate still fails at OOP/IP mean_abs `0.262599/0.545222`, but the bucket output makes the dominant failure class clear:
- IP all-in boundaries: mean_abs `0.603152`
- IP non-all-in boundaries: mean_abs `0.479016`
- IP reach `15-30%`: mean_abs `1.506680` versus IP reach `>30%`: mean_abs `0.397305`
- IP oracle magnitude `>=0.75`: mean_abs `1.437352`; IP oracle magnitude `0.25-0.75`: mean_abs `0.891660`

Conclusion: the remaining deployment failure is concentrated in IP high-magnitude targets, especially the sparser-reach boundary subset. This points toward solver-distribution validation and targeted sampling/loss weighting before another broad 10m-record training run.
