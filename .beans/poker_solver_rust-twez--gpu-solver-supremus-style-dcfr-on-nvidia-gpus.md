---
# poker_solver_rust-twez
title: 'GPU Solver: Supremus-style DCFR+ on NVIDIA GPUs'
status: completed
type: epic
priority: high
tags:
    - gpu
    - cuda
    - solver
created_at: 2026-03-15T04:14:02Z
updated_at: 2026-03-15T23:48:38Z
---

Build a GPU DCFR+ solver using cudarc for custom CUDA kernels and burn-cuda for neural net inference. Five phases: Phase 1 (GPU DCFR+ solver), Phase 2 (River CFVNet training), Phase 3 (Turn CFVNet with leaf eval), Phase 4 (Flop + Preflop models), Phase 5 (Explorer integration). See docs/plans/2026-03-14-gpu-solver-impl.md for full plan.
