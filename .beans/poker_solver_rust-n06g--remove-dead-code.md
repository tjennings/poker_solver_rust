---
# poker_solver_rust-n06g
title: Remove dead code
status: completed
type: task
priority: normal
created_at: 2026-02-25T16:48:53Z
updated_at: 2026-02-25T17:51:11Z
---

Remove strategy_delta from convergence.rs, update convergence_metrics_test.rs, remove PrevMatrices type alias.

## Todo
- [ ] Remove strategy_delta function and its tests
- [ ] Update convergence_metrics_test.rs to use avg_positive_regret
- [ ] Remove PrevMatrices type alias
- [ ] Run full test suite

## Summary of Changes\nRemoved matrix_delta from lhe_viz.rs, removed delta tracking from convergence integration test. Kept strategy_delta (still used by gpu-cfr bench). PrevMatrices already removed in prior commit. Commit: ed54ce0
