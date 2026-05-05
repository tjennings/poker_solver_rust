# GPU Turn Datagen with BoundaryNet — Design Brief

## Goal

Generate turn training data entirely on GPU. The GPU solver runs turn DCFR,
uses the trained river BoundaryNet (ONNX) for leaf evaluation at river
boundaries, and dumps solved turn CFVs to disk as TrainingRecords.

## Architecture

```
Config (CPU) → GPU
  ↓
Sample random turn situation (board, pot, stack, ranges)
  ↓
Build turn action tree on GPU
  ↓
DCFR iteration loop:
  - Traverse turn tree
  - At river boundary nodes: collect (board, pot, stack, ranges, player)
  - Batch all boundaries → BoundaryNet forward pass (ONNX, on GPU)
  - Denormalize EVs → feed back as boundary CFVs
  - Continue DCFR iteration
  - Check convergence
  ↓
Extract turn-level CFVs per combo
  ↓
Write TrainingRecord to disk (same binary format as river datagen)
```

## Key Design Points

- **Nothing on CPU** except config loading and disk I/O for final records
- **BoundaryNet inference is a GPU kernel** in the solve loop, not a round-trip
- **Batched boundary evaluation**: each DCFR iteration may encounter multiple
  boundary nodes across the tree — collect all, run one batched forward pass
- **Same output format**: TrainingRecord binary format, same as river datagen,
  so the existing PyTorch training pipeline works unchanged
- **ONNX on GPU**: use onnxruntime with CUDA execution provider, or export
  to TensorRT for fused inference

## Inputs

- YAML config: bet sizes, stack distribution, pot intervals, SPR intervals,
  solver iterations, target exploitability, num_samples
- River BoundaryNet ONNX model path
- Output directory for binary training records

## Questions for Design Session

1. Does the existing GPU solver (`crates/gpu-range-solver`) support turn trees,
   or only river? What modifications are needed?
2. How does the GPU solver currently handle boundary nodes? Is there a hook
   for injecting BoundaryNet evaluation?
3. Should the ONNX model be loaded via onnxruntime CUDA EP, or converted to
   TensorRT for lower latency?
4. Batching strategy: collect boundaries across one DCFR iteration, or across
   multiple concurrent game solves?
5. How to handle the 48 possible river cards at each turn boundary — does each
   river card produce a separate boundary evaluation, or is there a way to
   batch all 48?

## Prerequisites

- [x] River BoundaryNet trained and exported to ONNX
- [x] ONNX inference working in Rust (`ort` crate, `--features onnx`)
- [ ] GPU solver supports turn trees with boundary nodes
- [ ] ONNX runtime with CUDA execution provider
