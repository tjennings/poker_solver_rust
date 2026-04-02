---
# poker_solver_rust-kse7
title: Parallelize Stage 1 deal gen in turn datagen pipeline
status: completed
type: task
priority: high
created_at: 2026-04-02T04:43:38Z
updated_at: 2026-04-02T04:54:13Z
---

Stage 1 is single-threaded, building PostFlopGame trees one at a time. This gates the entire pipeline at ~16/s. Parallelize tree building with rayon to saturate the GPU and solve stages.


## Summary of Changes

Replaced sequential Stage 1 loop with batched approach:
1. Sample situations sequentially (cheap, needs single RNG)
2. Build trees in parallel via dedicated rayon pool (expensive)
3. Send to channel sequentially

Batch size = 64. Uses its own rayon pool (same thread count as Stage 3).
