---
# poker_solver_rust-prbu
title: 'Phase A: Bucketed River Solver + Training'
status: in-progress
type: feature
priority: critical
tags:
    - gpu
    - bucket
created_at: 2026-03-16T17:47:25Z
updated_at: 2026-03-16T23:10:10Z
parent: poker_solver_rust-618h
---

Bucketed river solver + datagen + training + eval.

Completed:
- [x] A0: Supremus DCFR+ update_regrets kernel
- [x] A1: Bucket equity table precomputation
- [x] A2: BucketedTree builder
- [x] A3: Bucketed showdown kernel (matrix-vector multiply)
- [x] A4: Bucketed fold kernel
- [x] A5: BucketedGpuSolver
- [x] A6: cfvnet-quality sampler with bucket mapping
- [x] A7: Combo-to-bucket mapping utilities
- [x] A8: BatchBucketedSolver + datagen pipeline
- [x] A9: eval command + CLI
- [x] River CFV net training (Supremus-exact: dual-player output, zero-sum)

In progress:
- [ ] Fix bucketed solver correctness (0% check, heavy all-in — payoff/equity scale mismatch)
- [ ] Validation: MAE < 0.05, >90% dominant action agreement vs CPU

Performance target: 50M samples in 1 hour
