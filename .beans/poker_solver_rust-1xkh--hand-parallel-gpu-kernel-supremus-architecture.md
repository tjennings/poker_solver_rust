---
# poker_solver_rust-1xkh
title: Hand-parallel GPU kernel (Supremus architecture)
status: completed
type: task
priority: high
created_at: 2026-04-04T12:44:05Z
updated_at: 2026-04-04T13:41:11Z
parent: poker_solver_rust-xkx0
---

Rewrite from node-parallel (grid.sync) to hand-parallel (one block = one subgame).

## Tasks
- [ ] Task 1: Hand-parallel CUDA kernel
- [ ] Task 2: Simplified GPU state
- [ ] Task 3: Single-launch solver, delete terminal.rs
- [x] Task 4: Benchmark + verify — HP 3x faster than mega-kernel, 55x faster than burn

Plan: docs/plans/2026-04-04-hand-parallel-kernel-impl.md
