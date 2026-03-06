---
# poker_solver_rust-w0al
title: Parallelize Blueprint V2 MCCFR training
status: in-progress
type: feature
created_at: 2026-03-06T23:17:22Z
updated_at: 2026-03-06T23:17:22Z
---

Replace single-threaded MCCFR training with Rayon-based parallel batched execution. Atomic shared buffers (AtomicI32/AtomicI64) for lock-free concurrent traversal. Batch size 200, thread-local RNG per worker.
