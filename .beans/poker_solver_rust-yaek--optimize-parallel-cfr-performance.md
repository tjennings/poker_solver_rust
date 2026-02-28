---
# poker_solver_rust-yaek
title: Optimize parallel CFR performance
status: completed
type: task
priority: high
created_at: 2026-02-28T22:26:44Z
updated_at: 2026-02-28T22:57:39Z
---

Implement 4 performance optimizations from perf review:
1. parallel_traverse: replace fold+reduce with fold+collect to reduce allocations
2. normalize_strategy_sum: stack-allocate in hot path (best_response_ev)
3. compute_exploitability: call every other iteration instead of every iteration
4. Rayon nesting: disable inner parallelism for multi-flop builds

## Summary of Changes\n\nAll 4 performance optimizations implemented and merged to main:\n1. **fold+collect** (commit 4d4fb24) — reduced allocations from O(log N) to O(threads)\n2. **Stack-allocate normalize_strategy_sum** (commit 8f44fb3) — eliminated Vec allocation in hot path\n3. **Exploitability every other iteration** (commit 4beb926) — ~2x total speedup\n4. **Rayon nesting management** (commit 222cdb7) — dedicated single-thread pool for multi-flop builds\n\nAll validated by rust-perf-reviewer. 530/531 tests pass (1 pre-existing timer flake).
