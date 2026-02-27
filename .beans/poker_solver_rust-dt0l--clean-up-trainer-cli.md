---
# poker_solver_rust-dt0l
title: Clean up trainer CLI
status: todo
type: task
created_at: 2026-02-27T23:18:14Z
updated_at: 2026-02-27T23:18:14Z
---

Delete trainer/src/tree.rs. Remove train/generate-deals/merge-deals/inspect-deals/tree-stats/tree commands from main.rs. Remove SolverMode, DealSortOrder, TrainingConfig, TrainingParams and handler fns. Keep solve-preflop, solve-postflop, flops, diag-buckets, trace-hand.
