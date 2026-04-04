---
# poker_solver_rust-a22j
title: 'cudarc GPU solver: replace burn with custom CUDA kernels'
status: completed
type: task
priority: high
created_at: 2026-04-04T03:34:44Z
updated_at: 2026-04-04T04:06:55Z
parent: poker_solver_rust-xkx0
---

Replace burn with cudarc + custom CUDA kernels for gpu-range-solver.

## Tasks
- [x] Task 1: Drop burn, add cudarc dependency
- [x] Task 2: CUDA kernel source (kernels.rs)
- [x] Task 3: GPU device management (gpu.rs)
- [x] Task 4: Solver iteration loop (solver.rs rewrite)
- [x] Task 5: Terminal eval wrappers (terminal.rs rewrite)
- [x] Task 6: Wire up lib.rs, delete tensors.rs
- [x] Task 7: Benchmark + verify

Plan: docs/plans/2026-04-03-cudarc-gpu-solver-impl.md
