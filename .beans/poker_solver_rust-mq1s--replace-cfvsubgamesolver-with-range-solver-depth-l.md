---
# poker_solver_rust-mq1s
title: Replace CfvSubgameSolver with range-solver depth-limited solving
status: todo
type: task
priority: normal
created_at: 2026-03-15T03:38:36Z
updated_at: 2026-03-15T03:38:36Z
---

Now that range-solver supports depth boundaries, CfvSubgameSolver is redundant (~50x slower). Plan and execute replacement across all call sites (turn datagen, compare_turn, tests). Remove CfvSubgameSolver and related dead code (CfvLayout, propagate_ranges, etc). Update docs/architecture.md.
