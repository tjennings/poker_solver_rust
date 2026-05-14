---
# poker_solver_rust-o9bc
title: Fix CFVNet neural boundary raw-CFV contract
status: completed
type: bug
priority: high
created_at: 2026-05-14T04:17:18Z
updated_at: 2026-05-14T04:26:20Z
parent: poker_solver_rust-8e9f
---

Long-term contract fix for CFVNet boundary magnitude mismatch: neural evaluators returned conditional chip EVs through `compute_raw_cfvs_both`, but range-solver raw boundaries require opponent-reach-integrated chip CFVs. Disable that raw path and align tests/docs so CFVNet boundaries use one explicit conditional-value contract.

- [x] Add failing coverage for neural raw path contract decision
- [x] Change neural evaluator to avoid incorrect raw-CFV handoff
- [x] Update docs to describe conditional model output vs solver reach integration
- [x] Run focused boundary evaluator and compare-solve tests
- [x] Summarize result and follow-up target-unit cleanup

## Summary of Changes

Changed CFVNet neural boundary evaluators so `compute_raw_cfvs_both` returns `None` instead of handing conditional model EVs to the range-solver raw-CFV path. This forces neural boundaries through the conditional BCFV path, where range-solver applies blocker-aware opponent reach integration. Exact/oracle evaluators can continue using raw CFVs because they really do return reach-integrated chip CFVs.

Added regression coverage that neural evaluators do not advertise the raw-CFV path and updated architecture/training docs to describe conditional model outputs versus raw reach-integrated evaluator outputs.

Verification passed:
- `cargo test -p cfvnet boundary_evaluator`
- `cargo test -p poker-solver-trainer boundary_cfv_gate`
- `git diff --check`

Re-ran the v5 diagnostic compare. Canonical direct aggregate improved from OOP/IP mean_abs `147.780289/154.519104` to `0.383081/0.832239`. The gate still fails at max_mean_abs `0.25`, but the remaining error is now model/target quality, not runtime reach-integration blow-up.

Follow-up: align Rust writer docs, Rust BoundaryNet dataset encoder, Python loader, and manifest normalization around one explicit stored target unit before the next training run.
