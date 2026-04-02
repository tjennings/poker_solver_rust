# Batch wgpu Stage 2 GPU Inference

**Date**: 2026-04-01
**Status**: Approved
**Bean**: poker_solver_rust-a1gv

## Problem

The wgpu turn datagen pipeline's Stage 2 calls `evaluate_game_boundaries()` per game — each game triggers a separate GPU forward pass (~48 rows). With a batch of 32 games, that's 32 separate GPU round-trips instead of 1. On Metal/wgpu where CPU↔GPU transfer overhead dominates, this is the primary throughput bottleneck (~2.3 samples/sec on M-series Mac).

The CUDA path already solves this by collecting all boundary inputs across 128 games into a single tensor and doing one forward pass, but the batching functions (`build_game_inputs`, `scatter_gpu_results`) are trapped inside the `#[cfg(feature = "cuda")]` block.

## Solution

Extract the CUDA path's batching functions to module level and reuse them in the wgpu Stage 2.

### Step 1: Extract to module level

Move these items out of `generate_turn_training_data_cuda` to module-level:

- `BoundaryRequest` struct
- `PREFIX_LEN` const
- `build_game_inputs()` fn
- `scatter_gpu_results()` fn

No signature changes needed — they operate on `PostFlopGame`/`Situation` directly and produce/consume raw `f32` buffers.

### Step 2: Modify wgpu Stage 2

Replace the per-game evaluation loop (lines 1477-1487) with:

1. Collect all inputs across all games in the batch via `build_game_inputs()`
2. Single `model.forward(combined_tensor)` call
3. Scatter results back via `scatter_gpu_results()`

The Stage 2 thread already owns the model directly (not behind a trait), so it can call `model.forward()` on raw tensors.

### Step 3: Update CUDA path

CUDA path calls the same extracted functions (mechanical — they're already the right shape).

## Expected Impact

- ~32x fewer GPU round-trips per batch (32 games × ~48 rivers → 1 forward pass of ~1536 rows)
- Primary improvement on Metal/wgpu where transfer overhead dominates
- No changes to: `LeafEvaluator` trait, `RiverNetEvaluator`, Stage 1, Stage 3, tests, configs

## Files Changed

- `crates/cfvnet/src/datagen/turn_generate.rs` — extract + modify Stage 2
