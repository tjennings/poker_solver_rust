---
# poker_solver_rust-hm1p
title: Add avg_positive_regret() to MccfrSolver
status: completed
type: task
priority: normal
created_at: 2026-02-25T16:48:52Z
updated_at: 2026-02-25T16:54:08Z
---

Add pub fn avg_positive_regret() to MccfrSolver in crates/core/src/cfr/mccfr.rs. Add tests in convergence_metrics_test.rs.

## Todo
- [ ] Add avg_positive_regret_decreases_over_training test
- [ ] Add avg_positive_regret_zero_on_empty_solver test
- [ ] Implement avg_positive_regret() on MccfrSolver
- [ ] Verify tests pass

## Summary of Changes

Added `avg_positive_regret()` method to `MccfrSolver` in `crates/core/src/cfr/mccfr.rs` (after `regret_sum()`). The method sums `max(R[I][a], 0)` across all info sets and actions, divides by `count * iterations`, providing a convergence metric that directly bounds exploitability.

Added two integration tests in `crates/core/tests/convergence_metrics_test.rs`:
- `avg_positive_regret_decreases_over_training`: trains 200 then 19800 more iterations, asserts metric decreases
- `avg_positive_regret_zero_on_empty_solver`: asserts 0.0 on fresh solver
