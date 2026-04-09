# GPU Turn Datagen with BoundaryNet — Design

## Goal

Generate turn training data entirely on GPU. A `TurnDatagenOrchestrator` in the
`cfvnet` crate runs turn DCFR via `GpuBatchSolver`, uses a TensorRT-compiled
river BoundaryNet for leaf evaluation at river boundaries, and writes solved
turn CFVs to disk as TrainingRecords.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              TurnDatagenOrchestrator                 │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ GpuBatch     │  │ TensorRT     │  │ Situation  │ │
│  │ Solver       │  │ Boundary     │  │ Sampler    │ │
│  │              │  │ Evaluator    │  │            │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘ │
│         │    GPU           │   GPU          │ CPU   │
└─────────┼──────────────────┼────────────────┼───────┘
          │                  │                │
          ▼                  ▼                ▼
   ┌─────────────────────────────┐    ┌─────────────┐
   │        GPU Memory           │    │   Disk      │
   │  - DCFR state (regrets,     │    │  Training   │
   │    strategy, reach, cfv)    │    │  Records    │
   │  - Leaf CFV buffers         │    │  (binary)   │
   │  - TensorRT engine weights  │    └─────────────┘
   └─────────────────────────────┘
```

### Approach: Orchestrator in `cfvnet` (Approach A)

The orchestrator owns both a `GpuBatchSolver` and a TensorRT engine. It manages
the interleaving of DCFR iterations and boundary re-evaluation. The GPU solver
stays generic — BoundaryNet coupling is datagen-specific.

**Why this over alternatives:**
- Approach B (extend GpuBatchSolver directly) couples TensorRT into the solver,
  making it less reusable for range solving and the explorer.
- Approach C (fused CUDA pipeline) requires reimplementing the neural net in CUDA.
  TensorRT already does kernel fusion internally.

## Main Loop

```
for each batch of K situations:
  1. Sample K turn situations on CPU (board, pot, stack, ranges)
  2. Build PostflopGame trees, extract topology, upload to GpuBatchSolver
  3. Initial boundary evaluation:
     - Extract reach at boundary nodes from GPU
     - Build BoundaryNet inputs (×48 rivers × 2 players)
     - TensorRT batched forward pass
     - Reach-weighted average over river cards
     - Upload leaf CFVs to solver
  4. Solve loop (total max_iterations, re-eval every leaf_eval_interval):
     - Run leaf_eval_interval DCFR iterations on GPU
     - Re-evaluate boundaries (same as step 3 with updated reach)
     - Repeat until max_iterations
  5. Extract final turn CFVs from GPU
  6. Write TrainingRecords to disk (same binary format as river)
```

**CPU-only work:** config loading, situation sampling, tree building, disk I/O.
Everything else runs on GPU.

## Key Design Decisions

### Boundary re-evaluation: fixed interval

Action frequency determines range probability at boundary nodes. As DCFR
strategy evolves, reach at boundaries changes, so BoundaryNet inputs change.
We re-evaluate at a fixed interval (e.g., every 50 iterations) controlled by
the existing `leaf_eval_interval` config parameter.

- Not every iteration (too expensive)
- Not once only (ranges at boundaries shift as strategy evolves)
- Fixed interval is simple, predictable, already supported by config

### Reach-weighted river averaging

Each turn boundary has 48 possible river cards. For each, we evaluate the
BoundaryNet and weight the output by the opponent's total reach for hands not
conflicting with that river card. This correctly implements the chance-node
expectation accounting for card removal.

### TensorRT (not ONNX CUDA EP)

TensorRT provides kernel fusion and FP16 optimization for maximum inference
throughput. The ONNX model is compiled to a TensorRT engine at startup and
cached to disk for reuse.

### Same output format

TrainingRecord binary format with variable `board_size` prefix byte. Turn
records use 4-card boards. The existing PyTorch training pipeline works
unchanged.

### Cold-start every subgame

No warm-starting for datagen. Each situation is independent with different
board/pot/stack/ranges.

## New Files

```
cfvnet/src/datagen/
  ├── generate.rs          # existing — add dispatch to GPU path
  ├── gpu_orchestrator.rs  # NEW — TurnDatagenOrchestrator
  ├── boundary_eval.rs     # NEW — TensorRT + reach-weighted averaging kernels
  ├── sampler.rs           # existing — reuse as-is
  ├── storage.rs           # existing — reuse as-is
  └── solver.rs            # existing CPU solver (unchanged)
```

## TensorRT Boundary Evaluator

### Boundary re-evaluation pipeline (5 steps)

**Step 1 — Extract reach at boundary nodes** (CUDA kernel: `extract_boundary_reach`)

```
Inputs:  reach[batch × nodes × hands], boundary_node_ids[N_boundary]
Outputs: boundary_reach_oop[batch × N_boundary × 1326]
         boundary_reach_ip[batch × N_boundary × 1326]

One thread per (subgame, boundary, hand).
Copies reach at the boundary node into a contiguous staging buffer.
```

**Step 2 — Build BoundaryNet inputs** (CUDA kernel: `build_boundary_inputs`)

```
Inputs:  boundary_reach[batch × N_boundary × 2 × 1326],
         board_cards[batch × 4], pot[batch], stack[batch]
Outputs: trt_input[total_rows × 3078]

total_rows = batch × N_boundary × 2 × 48

One thread per (subgame, boundary, player, river_card, feature_chunk).
For each of 48 non-conflicting river cards:
  - Copy reach into range slots, zero out conflicting hands
  - Write board one-hot (4 turn cards + 1 river card = 5)
  - Write rank presence (13), pot_frac (1), stack_frac (1), player (1)
```

BoundaryNet input layout (3078 floats):
- oop_range: 1326
- ip_range: 1326
- board_onehot: 52
- rank_presence: 13
- pot_frac: 1
- stack_frac: 1
- player: 1

**Step 3 — TensorRT forward pass**

Single batched inference. Input: `[total_rows, 3078]` → Output: `[total_rows, 1326]`
normalized EVs. Runs on same CUDA stream as solver.

**Step 4 — Reach-weighted average + denormalization** (CUDA kernel: `reduce_boundary_outputs`)

```
Inputs:  trt_output[total_rows × 1326], boundary_reach (opponent weights),
         pot[batch], stack[batch]
Outputs: leaf_cfv_p0[batch × N_boundary × hands]
         leaf_cfv_p1[batch × N_boundary × hands]

One thread per (subgame, boundary, player, hand).
For each of 48 river cards:
  weight = sum of opponent reach for non-conflicting hands
  cfv[h] += weight × trt_output[row][h] × (pot + stack)
Normalize by total weight.
```

**Step 5 — Upload leaf CFVs**

Device-to-device copy into `GpuBatchSolver`'s `leaf_cfv_p0` / `leaf_cfv_p1`
buffers. No CPU round-trip.

### TensorRT engine lifecycle

```rust
struct TensorRTEngine {
    engine: tensorrt::Engine,      // serialized + cached to disk
    context: tensorrt::Context,    // execution context
    d_input: CudaSlice<f32>,       // pre-allocated GPU buffer
    d_output: CudaSlice<f32>,      // pre-allocated GPU buffer
}
```

- Load ONNX → build engine with FP16, dynamic batch axis
- Cache serialized engine at `{onnx_path}.trt_cache`
- Rebuild only when ONNX mtime changes
- Pre-allocate input/output buffers for max expected batch size

## GPU Memory Budget (RTX 4090, 24GB)

| Component | Memory |
|-----------|--------|
| DCFR state (256 subgames × ~4 boundary nodes) | ~2-4 GB |
| TensorRT engine weights | ~50-100 MB |
| Boundary eval input buffers (98K rows × 3078 f32) | ~1.2 GB |
| Boundary eval output buffers (98K rows × 1326 f32) | ~0.5 GB |
| Headroom | ~16+ GB |

Batch size of 256 concurrent subgames adjustable via `gpu_batch_size` config.

## Reused Components

| Component | Source | Changes |
|-----------|--------|---------|
| `GpuBatchSolver` | `gpu-range-solver` | None — already supports turn trees + leaf injection |
| Situation sampling | `cfvnet/src/datagen/sampler.rs` | None — pass `board_size: 4` |
| TrainingRecord writing | `cfvnet/src/datagen/storage.rs` | None — variable `board_size` already supported |
| Config | `cfvnet/src/config.rs` | None — all fields exist (`street`, `mode`, `backend`, `leaf_eval_interval`, etc.) |
| Topology extraction | `gpu-range-solver/src/extract.rs` | None — `decompose_at_chance()` identifies boundary nodes |

## Configuration

Uses existing `CfvnetConfig` — no new config format. Example:

```yaml
game:
  river_model_path: "models/river_boundary.onnx"
  bet_sizes: [0.33, 0.67, 1.0, 1.5]
  raise_sizes: [0.5, 1.0]

datagen:
  street: "turn"
  mode: "model"
  backend: "gpu"
  num_samples: 10_000_000
  solver_iterations: 300
  leaf_eval_interval: 50
  gpu_batch_size: 256
  pot_intervals: [[10, 50], [50, 200], [200, 1000]]
  spr_intervals: [[0.5, 2], [2, 5], [5, 15]]
  turn_output: "local_data/cfvnet/turn/v1"
  per_file: 10000
```

## CLI Invocation

No CLI changes needed:

```bash
cargo run -p cfvnet --release --features onnx -- generate \
  --config sample_configurations/turn_cfvnet.yaml \
  --backend cuda
```

Dispatches to `TurnDatagenOrchestrator` when `backend=gpu/cuda` and `street=turn`.

## Convergence Notes

- Fixed iteration count (300) preferred over exploitability target — all
  subgames in a batch finish simultaneously, GPU-friendly.
- Error does not compound multiplicatively across streets (DeepStack, Moravcik
  et al. 2017). BoundaryNet error enters additively into turn CFVs.
- After training TurnNet, validate against exactly-solved turn situations to
  measure error propagation.
