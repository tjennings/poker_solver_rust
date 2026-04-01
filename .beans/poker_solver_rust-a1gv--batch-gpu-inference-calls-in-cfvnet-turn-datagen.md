---
# poker_solver_rust-a1gv
title: Batch GPU inference calls in cfvnet turn datagen
status: todo
type: task
priority: high
created_at: 2026-04-01T06:14:48Z
updated_at: 2026-04-01T06:14:48Z
---

Turn datagen currently calls the river model one boundary node at a time. Each call incurs CPU→GPU transfer, inference, and GPU→CPU transfer overhead. Batching all leaf evaluations from a tree traversal into a single GPU call would amortize the transfer overhead and utilize GPU parallelism.

Likely 10-50x speedup for turn datagen. Current rate is ~2.3 samples/sec on M-series Mac (wgpu/Metal). Each sample solves a turn subgame where the river model evaluates leaf nodes.

Approach: collect all boundary evaluation requests during a solver iteration, batch them into one tensor, run inference once, scatter results back to the requesting nodes.

Key files:
- crates/cfvnet/src/datagen/turn_generate.rs — the datagen loop and river model calls
- crates/cfvnet/src/eval/ — the neural net evaluator interface
- crates/range-solver/src/game/evaluation.rs — BoundaryEvaluator trait (may need batched variant)
