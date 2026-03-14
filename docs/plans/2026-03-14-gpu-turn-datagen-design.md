# GPU-Accelerated Turn Datagen

**Date**: 2026-03-14
**Status**: Approved

## Problem

Turn datagen uses a river CFV network as the leaf evaluator. Currently the model runs on CPU (NdArray backend), doing 48 sequential forward passes per `evaluate()` call (one per possible river card). With 5000 DCFR iterations per situation and 2 traversers, that's ~480,000 CPU forward passes per situation. The model is 7x768 MLP (~18M params) — significant compute per pass.

## Design

### Shared GPU Model with Mutex

- Load river model on `CudaJit` backend, wrap in `Arc<Mutex<CfvNet<CudaJit>>>`
- Rayon worker threads each own their `CfvSubgameSolver` (CPU tree traversal)
- At depth boundaries, `GpuRiverNetEvaluator` acquires the mutex for batched GPU inference

### Batched River Card Inference

Current `RiverNetEvaluator::evaluate()` loops over 48 river cards, calling `model.forward()` once per card. New `GpuRiverNetEvaluator` builds all valid river card inputs CPU-side, stacks them into a `[N, 3224]` tensor (N up to 48), runs a single batched `model.forward()`, and splits the output — all within one mutex acquisition.

### Contention Tracking

Each `evaluate()` records mutex wait time via `Instant::now()` before/after `mutex.lock()`, accumulated into a shared `Arc<AtomicU64>` (nanoseconds). Printed at completion:
```
GPU mutex wait: 12.3s total (4.2% of wall time)
```

### CLI

Add `--backend` flag to `generate` command. Default: `ndarray` (current behavior). `cuda`: loads river model on GPU.

### Components

1. **`GpuRiverNetEvaluator`** in `crates/cfvnet/src/eval/river_net_evaluator.rs`
   - `Arc<Mutex<CfvNet<CudaJit>>>` + `CudaDevice` + `Arc<AtomicU64>` wait counter
   - Implements `LeafEvaluator` with batched forward pass

2. **`generate_turn_training_data` updates** in `crates/cfvnet/src/datagen/turn_generate.rs`
   - New CUDA-aware variant or generic over backend
   - Wraps model in Arc<Mutex>, creates shared wait counter
   - Prints contention stats at end

3. **CLI `--backend` for generate** in `crates/cfvnet/src/main.rs`
   - Plumbs backend choice to turn datagen

### What stays the same

- Rayon parallelism for DCFR solve (CPU)
- `LeafEvaluator` trait unchanged
- NdArray path preserved as default
- Per-file chunking, progress bar, seed handling
