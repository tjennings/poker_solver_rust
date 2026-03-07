---
# poker_solver_rust-hsgo
title: Parallelize k-means EMD and add progress output
status: completed
type: task
created_at: 2026-03-07T02:17:51Z
updated_at: 2026-03-07T02:17:51Z
---

K-means clustering assignment step was single-threaded, causing the flop clustering phase to hang on a single core. Added Rayon parallelization to the assignment step and wired progress callbacks through the pipeline so users see feature extraction (0-80%) and k-means (80-100%) phases.
