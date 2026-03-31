---
# poker_solver_rust-gvmf
title: Add CBV-based boundary evaluator as configurable alternative to rollouts
status: completed
type: feature
priority: normal
created_at: 2026-03-31T01:52:21Z
updated_at: 2026-03-31T02:05:55Z
---

Replace rollout-based boundary evaluation in depth-limited solver with an option to use CBV table lookups. CBV lookups are microseconds vs ~40s for rollouts.

## Acceptance Criteria
- [x] Add `BoundaryMode` enum (Rollout/Cbv) to game_session.rs
- [x] Add `boundary_index` parameter to `BoundaryEvaluator::compute_cfvs` trait
- [x] Implement CBV lookup path in SolveBoundaryEvaluator
- [x] Wire boundary_mode config through game_solve_core
- [x] Default to Cbv mode when CBV data is available
- [x] All existing tests pass
- [x] cargo build succeeds for full workspace


## Summary of Changes

Added CBV-based boundary evaluator as a configurable alternative to rollouts. The `BoundaryMode` enum (Cbv/Rollout) controls which evaluation strategy is used at depth-limited boundary nodes. CBV mode uses pre-computed continuation values from the blueprint for microsecond lookups instead of 40s Monte Carlo rollouts. The `BoundaryEvaluator::compute_cfvs` trait now receives the `boundary_index` parameter so evaluators can look up the correct boundary node in the CBV table.
