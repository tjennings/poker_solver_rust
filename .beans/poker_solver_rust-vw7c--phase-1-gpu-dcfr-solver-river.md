---
# poker_solver_rust-vw7c
title: 'Phase 1: GPU DCFR+ Solver (River)'
status: completed
type: feature
priority: high
tags:
    - gpu
created_at: 2026-03-15T04:14:14Z
updated_at: 2026-03-15T04:14:14Z
parent: poker_solver_rust-twez
---

Solve postflop river subtrees on GPU with CLI identical to range-solve.
Validated correctness against CPU range-solver across 30 test positions.

Completed tasks:
- [x] Crate scaffolding (cudarc behind cuda feature flag)
- [x] Flat tree data structures (BFS-ordered FlatTree)
- [x] Tree builder from PostFlopGame
- [x] CUDA device wrapper (GpuContext)
- [x] 7 CUDA kernels: regret_match, forward_reach, forward_pass, terminal_fold_eval, terminal_showdown_eval, backward_cfv, update_regrets, extract_strategy
- [x] GPU solver orchestration (GpuSolver with full DCFR+ loop)
- [x] gpu-solve CLI command
- [x] Integration tests (30 positions, max diff < 0.01)
- [x] Card blocking in terminal evaluation
- [x] Kernel caching (RefCell<HashMap>)

Known limitations:
- GPU is slower than CPU for small river trees (15-39 nodes)
- No chance node support (river-only)
- init_reach allocates new buffers each iteration
