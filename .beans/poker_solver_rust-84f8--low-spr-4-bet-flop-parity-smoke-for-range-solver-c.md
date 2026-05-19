---
# poker_solver_rust-84f8
title: Low-SPR 4-bet flop parity smoke for range-solver-compare
status: in-progress
type: task
priority: high
tags:
    - range-solver
    - compare
created_at: 2026-05-19T15:46:05Z
updated_at: 2026-05-19T15:47:30Z
---

Add deterministic low-SPR 4-bet flop parity smoke coverage to the range-solver-compare harness.

Checklist:
- [x] Commit this bean tracking state before implementation.
- [ ] Run the full test suite baseline and confirm it passes in under one minute. Blocked: baseline `cargo test` failed in 27.41s due existing `blueprint_mp::trainer` 1-second timed-test overruns (`dcfr_discount_parallel_path_runs_above_threshold`, `lazy_train_2_player_toy_completes`, `run_batch_updates_regrets`, `run_batch_updates_strategy_sums`, `train_batch_aligns_to_iteration_limit`).
- [ ] Dispatch research/brainstorming on low-SPR 4-bet flop parity dimensions and scope.
- [ ] Dispatch Rust implementation in a separate worktree.
- [ ] Dispatch review before integration.
- [ ] Integrate accepted changes into the feature branch.
- [ ] Run flop smoke compare tests, existing turn/river compare smoke tests, and the full suite under one minute.
- [ ] Complete the bean and commit the final tracking update.
