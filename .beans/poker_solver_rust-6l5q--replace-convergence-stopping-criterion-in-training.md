---
# poker_solver_rust-6l5q
title: Replace convergence stopping criterion in training loops
status: completed
type: task
priority: normal
created_at: 2026-02-25T16:48:53Z
updated_at: 2026-02-25T17:48:53Z
---

Update run_convergence_loop and preflop training to stop on avg positive regret. Simplify print_preflop_matrices.

## Todo
- [ ] Update checkpoint_report return type
- [ ] Update run_convergence_loop to check avg_positive_regret
- [ ] Update preflop training loop early-stopping
- [ ] Simplify print_preflop_matrices (remove delta)
- [ ] Verify compilation and tests

## Summary of Changes\nReplaced strategy delta stopping criterion with regret_threshold. Simplified print_preflop_matrices to remove delta computation. Commit: 171cdd5
