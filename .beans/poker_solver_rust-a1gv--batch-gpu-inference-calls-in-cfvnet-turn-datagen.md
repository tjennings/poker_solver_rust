---
# poker_solver_rust-a1gv
title: Batch GPU inference calls in cfvnet turn datagen
status: completed
type: task
priority: high
created_at: 2026-04-01T06:14:48Z
updated_at: 2026-04-02T02:19:01Z
---

Turn datagen currently calls the river model one boundary node at a time. Each call incurs CPU→GPU transfer, inference, and GPU→CPU transfer overhead. Batching all leaf evaluations from a tree traversal into a single GPU call would amortize the transfer overhead and utilize GPU parallelism.

Likely 10-50x speedup for turn datagen. Current rate is ~2.3 samples/sec on M-series Mac (wgpu/Metal). Each sample solves a turn subgame where the river model evaluates leaf nodes.

Approach: collect all boundary evaluation requests during a solver iteration, batch them into one tensor, run inference once, scatter results back to the requesting nodes.

Key files:
- crates/cfvnet/src/datagen/turn_generate.rs — the datagen loop and river model calls
- crates/cfvnet/src/eval/ — the neural net evaluator interface
- crates/range-solver/src/game/evaluation.rs — BoundaryEvaluator trait (may need batched variant)


## Summary of Changes

Extracted CUDA path's batching functions (`BoundaryRequest`, `build_game_inputs`, `decode_boundary_cfvs`) to module level and replaced wgpu Stage 2's per-game `evaluate_game_boundaries()` loop with a single batched `evaluate_batch()` forward pass per batch.

Key commits on branch `worktree-wgpu-batch-inference`:
1. Extract BoundaryRequest + PREFIX_LEN to module level
2. Extract build_game_inputs to module level  
3. Extract decode_boundary_cfvs for reuse across GPU backends
4. Add batched_evaluation_matches_per_game test
5. Batch wgpu Stage 2 GPU inference across games (the core change)
6. Fix per_file storage rotation + pre-existing test errors

Reduces GPU round-trips from ~32 per batch to 1 in the wgpu turn datagen path.
