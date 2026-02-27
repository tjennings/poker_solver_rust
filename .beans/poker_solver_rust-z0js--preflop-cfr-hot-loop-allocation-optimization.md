---
# poker_solver_rust-z0js
title: Preflop CFR hot-loop allocation optimization
status: completed
type: feature
priority: normal
created_at: 2026-02-27T04:51:03Z
updated_at: 2026-02-27T05:17:24Z
---

Eliminate per-iteration heap allocations in PreflopSolver's training loop.

## Tasks
- [x] Task 1: Add Criterion benchmark for preflop solver (baseline)
- [x] Task 2: Snapshot buffer reuse (clone_from instead of clone)
- [x] Task 3: fold_with + reduce_with — REVERTED (see summary)
- [x] Task 4: Run benchmark and verify improvement

## Summary of Changes

### What was done
- Added Criterion benchmark harness for PreflopSolver (tiny_config, 50/200/500 iterations)
- Added `snapshot_buf` field to PreflopSolver; uses `clone_from()` instead of `clone()` to reuse allocation across iterations

### What was tried and reverted
- `fold_with` + `reduce_with` for rayon scratch buffers: slightly WORSE because `vec![0.0; n]` uses calloc (zero-page OS mapping) while `fold_with` clones via memcpy. Reverted.

### Benchmark results (tiny tree)
- clone_from: within noise (~0% change). Buffer is too small for allocation to matter.
- The real benefit of clone_from scales with tree size — production trees have much larger buffers.

### Key learning
The preflop solver already has excellent allocation patterns in the hot path (stack-allocated `[f64; MAX_ACTIONS]`, flat Vec buffers). The rayon fold/reduce allocations use `vec![0.0; n]` which benefits from OS zero-page optimization, making them effectively free for reasonable buffer sizes.
