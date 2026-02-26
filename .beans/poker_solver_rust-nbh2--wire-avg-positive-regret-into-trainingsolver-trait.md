---
# poker_solver_rust-nbh2
title: Wire avg_positive_regret into TrainingSolver trait and reporting
status: completed
type: task
priority: normal
created_at: 2026-02-25T16:48:53Z
updated_at: 2026-02-25T17:48:53Z
---

Add avg_positive_regret to TrainingSolver trait. Remove previous_strategies from solver wrappers. Replace strategy delta in checkpoint reporting with avg positive regret.

## Todo
- [ ] Add avg_positive_regret() to TrainingSolver trait
- [ ] Implement in MccfrTrainingSolver and SimpleTrainingSolver
- [ ] Remove previous_strategies fields from solver wrappers
- [ ] Update StrategyReportCtx (remove previous, add regret metric)
- [ ] Update compute_and_print_convergence to print regret
- [ ] Verify compilation

## Summary of Changes\nWired avg_positive_regret() into TrainingSolver trait and all implementations. Removed previous_strategies tracking. Commit: 171cdd5
