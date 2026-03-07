---
# poker_solver_rust-w0al
title: Parallelize Blueprint V2 MCCFR training
status: completed
type: feature
priority: normal
created_at: 2026-03-06T23:17:22Z
updated_at: 2026-03-07T00:33:07Z
---

Replace single-threaded MCCFR training with Rayon-based parallel batched execution. Atomic shared buffers (AtomicI32/AtomicI64) for lock-free concurrent traversal. Batch size 200, thread-local RNG per worker.


## Summary of Changes

Parallelized Blueprint V2 MCCFR training to use all CPU cores:

- **Atomic storage**: `Vec<i32>` → `Vec<AtomicI32>`, `Vec<i64>` → `Vec<AtomicI64>` with `Relaxed` ordering
- **Shared traversal**: `traverse_external` takes `&BlueprintStorage` (shared ref) instead of `&mut`
- **Batched parallel execution**: `rayon::par_iter` over batches of 200 deals, thread-local `SmallRng` per worker
- **LCFR discount**: runs between batches (no concurrent traversals)
- **Config**: `batch_size` field added to `TrainingConfig` (default 200)
- **Backward-compatible serialization**: snapshot format unchanged (collects atomics to plain vecs)

All 92 blueprint_v2 tests + 60 trainer tests pass.
