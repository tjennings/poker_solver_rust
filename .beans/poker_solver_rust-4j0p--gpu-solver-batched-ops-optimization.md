---
# poker_solver_rust-4j0p
title: GPU solver batched ops optimization
status: in-progress
type: task
priority: high
created_at: 2026-04-03T16:09:19Z
updated_at: 2026-04-03T16:09:19Z
parent: poker_solver_rust-xkx0
---

Rewrite GPU solver from per-node loops to batched gather/scatter ops.

## Tasks
- [ ] Task 1: Pre-computed level index tensors
- [ ] Task 2: Batched regret matching
- [ ] Task 3: Batched forward pass
- [ ] Task 4: Batched backward pass
- [ ] Task 5: Wire into solve loop
- [ ] Task 6: GPU-only fold eval
- [ ] Task 7: Benchmark + cleanup

Plan: docs/plans/2026-04-03-gpu-solver-batched-ops-impl.md
