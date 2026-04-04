---
# poker_solver_rust-nvie
title: Persistent batched CUDA mega-kernel
status: completed
type: task
priority: high
created_at: 2026-04-04T04:43:30Z
updated_at: 2026-04-04T12:00:26Z
parent: poker_solver_rust-xkx0
---

Replace 11 separate CUDA kernels with single cooperative mega-kernel.

## Tasks
- [ ] Task 1: Persistent mega-kernel with batch dimension
- [ ] Task 2: Batch-dimensioned GPU state + cooperative launch
- [ ] Task 3: Single-launch solver, delete terminal.rs
- [ ] Task 4: Multi-street turn support (B=44)
- [ ] Task 5: Benchmark + verify

Plan: docs/plans/2026-04-03-persistent-batched-kernel-impl.md


## Results Summary

Persistent mega-kernel with cooperative groups implemented. Key findings:

- **River (19 hands, 500 iter):** 0.68s GPU vs 0.00s CPU. GPU overhead is CUDA init (280ms) + grid.sync() barriers.
- **Turn (135 hands, 200 iter):** 19s GPU vs 0.08s CPU. 237x slower. Too many depth levels = too many grid.sync() barriers.
- **Critical insight:** Two-pass decomposition is incompatible with DCFR (per-iteration interleaving required). Full-tree approach means turn depth = ~50 levels × ~25 syncs/level = massive barrier overhead.
- **Per-iteration cost:** River 0.8ms/iter (vs CPU 0.001ms), Turn ~95ms/iter (vs CPU 0.4ms).
- The CPU's recursive DFS with rayon has zero synchronization overhead — fundamentally superior for tree-structured problems at this scale.

## Conclusion
GPU CFR with level-synchronous architecture cannot compete with CPU recursive DFS for NLHE tree sizes (50-2200 nodes). The GPU approach requires either: (1) much larger trees (>50K nodes) where per-level parallelism amortizes sync overhead, or (2) a fundamentally different architecture (depth-limited solving with neural leaf evaluation where the GPU accelerates the NN inference, not the tree traversal).
