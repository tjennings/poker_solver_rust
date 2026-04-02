---
# poker_solver_rust-kse7
title: Parallelize Stage 1 deal gen in turn datagen pipeline
status: in-progress
type: task
priority: high
created_at: 2026-04-02T04:43:38Z
updated_at: 2026-04-02T04:43:38Z
---

Stage 1 is single-threaded, building PostFlopGame trees one at a time. This gates the entire pipeline at ~16/s. Parallelize tree building with rayon to saturate the GPU and solve stages.
