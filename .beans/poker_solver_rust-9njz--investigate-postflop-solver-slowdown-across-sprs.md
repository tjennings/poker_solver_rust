---
# poker_solver_rust-9njz
title: Investigate postflop solver slowdown across SPRs
status: completed
type: bug
priority: high
created_at: 2026-03-03T02:05:39Z
updated_at: 2026-03-03T02:15:10Z
---

Performance degrades as more SPRs are solved sequentially. Large gaps where traversals/sec drops to zero between iterations. Config: exhaustive solve, SPRs [0.5, 1, 3, 6], 50 iterations, 3 bet sizes.

## Symptoms
- Each subsequent SPR runs slower than previous
- Traversals/sec drops to zero for extended periods between iterations
- Was not happening on earlier (smaller) SPRs
- Sparkline shows intermittent activity with growing gaps

## Investigation (Phase 1: Root Cause)
- [x] Read postflop solving pipeline to understand multi-SPR flow
- [x] Check if previous SPR data accumulates in memory (not the issue — 68GB RAM)
- [x] Look for serialization/IO between iterations (none found)
- [x] Check for lock contention or GC-like pauses
- [x] Identify what happens during zero-traversal gaps

## Summary of Changes

Root cause: `exhaustive_solve_one_flop` allocated `rayon::current_num_threads()` buffer partitions per flop, but since it runs inside `build_exhaustive`s `into_par_iter()` over 1755 flops, the inner parallelism was unused. This wasted O(n_threads × buf_size) zeroing and merging per iteration, growing worse with larger SPR trees.

Fix: Set `n_partitions = 1` in `postflop_exhaustive.rs:647`. One-line change, all tests pass.
