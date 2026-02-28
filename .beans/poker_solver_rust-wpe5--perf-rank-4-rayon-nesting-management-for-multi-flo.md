---
# poker_solver_rust-wpe5
title: 'Perf Rank 4: Rayon nesting management for multi-flop builds'
status: completed
type: task
priority: normal
created_at: 2026-02-28T22:40:21Z
updated_at: 2026-02-28T22:40:21Z
---

Disable inner hand-pair parallelism when outer flop-level parallelism is active. Prevents rayon oversubscription with 3-level nesting (flops x equity x cfr). ~20-40% speedup for multi-flop builds. Uses dedicated single-thread pool via rayon::ThreadPoolBuilder. Implemented in commit 222cdb7.
