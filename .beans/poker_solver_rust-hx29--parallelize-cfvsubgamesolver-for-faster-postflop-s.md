---
# poker_solver_rust-hx29
title: Parallelize CfvSubgameSolver for faster postflop solving
status: todo
type: task
created_at: 2026-03-21T13:58:18Z
updated_at: 2026-03-21T13:58:18Z
---

CfvSubgameSolver runs single-threaded vectorized CFR. The O(n^2) conditional showdown computation and the per-combo regret updates are the main bottlenecks. Need rayon parallelization for the inner loops. Current: ~1.5s per iteration with 1000 combos. Target: <0.1s per iteration.
