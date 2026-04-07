# PyTorch BoundaryNet Trainer — Design

## Problem

The burn-based training pipeline for BoundaryNet has GPU data pipeline limitations: manual prefetch threading, sync channel sizing, memory leaks, and idle GPU/CPU gaps. These are solved infrastructure in PyTorch. The model is a simple MLP that translates trivially to PyTorch. ONNX export enables Rust inference without burn.

## Decisions

- **Training framework**: PyTorch (replaces burn for training, burn kept as fallback)
- **Inference**: ONNX Runtime via `ort` Rust crate (replaces burn inference entirely)
- **Data format**: Read Rust binary `TrainingRecord` directly in Python (no conversion step)
- **Config**: Reuse existing `CfvnetConfig` YAML format
- **ONNX export**: Direct `torch.onnx.export()` — no manual BatchNorm fusion (ort optimizes internally)
- **Location**: `crates/cfvnet/python/` — colocated with model code

## Project Structure

```
crates/cfvnet/python/
├── pyproject.toml          # deps: torch, numpy, pyyaml, onnx, onnxruntime
├── cfvnet/
│   ├── __init__.py
│   ├── data.py             # Binary TrainingRecord reader + PyTorch Dataset
│   ├── model.py            # BoundaryNet (mirrors Rust architecture exactly)
│   ├── loss.py             # Weighted Huber + aux loss with component breakdown
│   ├── train.py            # Training loop with DataLoader, AMP, checkpointing
│   ├── export.py           # ONNX export with verification
│   └── config.py           # Read CfvnetConfig YAML
├── tests/
│   ├── test_data.py        # Binary reader round-trip, encoding correctness
│   ├── test_model.py       # Output shape, forward pass
│   ├── test_loss.py        # Known inputs → known outputs
│   ├── test_export.py      # ONNX export/import equivalence
│   └── test_e2e.py         # Tiny dataset → train → export → onnxruntime predict
└── scripts/
    ├── train_boundary.py   # CLI entry point for training
    └── eval_boundary.py    # CLI entry point for evaluation
```

## Data Pipeline

### Binary Reader (`data.py`)

Reads `TrainingRecord` from Rust binary format using `struct` module:
- Format per record: `[board_size: u8] [board: N*u8] [pot: f32] [stack: f32] [player: u8] [game_value: f32] [oop_range: 1326*f32] [ip_range: 1326*f32] [cfvs: 1326*f32] [valid_mask: 1326*u8]`
- All multi-byte values are little-endian (native x86)
- `encode_boundary_record()` reimplemented in numpy — same normalization as Rust:
  - `total_stake = pot + effective_stack`
  - Input: ranges + board one-hot (52) + rank presence (13) + `pot/total_stake` + `stack/total_stake` + player
  - Target: `cfv * pot / total_stake`
  - Game value: `sum(range[i] * target[i])` (recomputed from normalized targets)
  - Sample weight: `1 / max(SPR, 0.1)`, capped at 10

### PyTorch Dataset

- `BoundaryDataset` loads all records from a file or directory into numpy arrays at init
- Returns `(input, target, mask, range, game_value, sample_weight)` tensors
- Validation split: random sample across all records (not positional)

### DataLoader

Standard `torch.utils.data.DataLoader`:
- `num_workers=4`, `pin_memory=True`, `prefetch_factor=2`
- Shuffle via DataLoader (no custom shuffle buffer)
- This replaces the entire Rust streaming reader → encoder threads → GPU upload thread pipeline

## Model (`model.py`)

```
Input(2720) → [Linear → BatchNorm1d → PReLU] × N → Linear(1326)
```

Constants: `INPUT_SIZE = 2720`, `OUTPUT_SIZE = 1326`, `POT_INDEX = 2717`

Mirrors the Rust `BoundaryNet` exactly:

```python
class HiddenBlock(nn.Module):
    # Linear → BatchNorm1d → PReLU(num_parameters=out_features)

class BoundaryNet(nn.Module):
    # __init__(num_layers: int, hidden_size: int)
    # forward(x: Tensor[batch, 2720]) → Tensor[batch, 1326]
```

## Loss (`loss.py`)

Three functions:

- `weighted_huber_loss(pred, target, mask, sample_weight, delta)` — per-sample masked Huber with SPR weighting
- `weighted_aux_loss(pred, range, game_value, sample_weight)` — weighted game-value sum constraint (squared residual)
- `boundary_loss(pred, target, mask, range, game_value, sample_weight, delta, aux_weight)` — combined, returns `(combined, huber, aux)` for logging

## Training Loop (`train.py`)

- **Optimizer**: Adam with gradient norm clipping (`torch.nn.utils.clip_grad_norm_`)
- **LR schedule**: Cosine annealing (`CosineAnnealingLR`)
- **Mixed precision**: `torch.cuda.amp.autocast` + `GradScaler` for ~2x throughput
- **Checkpointing**: Save model state_dict + optimizer state + scaler state + epoch + config every N epochs
- **Resume**: Load checkpoint and continue from saved epoch
- **Logging**: Per-epoch: lr, train loss (combined/huber/aux), val loss (combined/huber/aux), epoch time
- **Val**: Computed once per epoch on held-out random sample, reports combined + components

## ONNX Export (`export.py`)

```python
model.eval()
dummy = torch.zeros(1, INPUT_SIZE)
torch.onnx.export(
    model, dummy, path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}},  # variable batch size
    opset_version=17,
)
```

- Dynamic batch axis for flexible Rust inference
- Post-export verification: run same input through PyTorch and ONNX, assert outputs match within tolerance

## Rust Inference Changes

### Add `ort` dependency

`crates/cfvnet/Cargo.toml`: add `ort` crate for ONNX Runtime inference.

### Replace burn inference in `NeuralBoundaryEvaluator`

Current:
```rust
struct NeuralBoundaryEvaluator {
    model: Mutex<BoundaryNet<NdArray>>,
    ...
}
```

New:
```rust
struct NeuralBoundaryEvaluator {
    session: ort::Session,
    ...
}
```

- `load_neural_boundary_evaluator()` loads `.onnx` file via `ort::Session::builder()`
- Forward pass: `session.run(ort::inputs!["input" => input_array]?)` → extract output → denormalize
- No burn dependency on the inference path
- Thread safety: `ort::Session` is `Send + Sync` natively — no Mutex needed

### Keep burn training as fallback

The burn-based `train-boundary` CLI command and `BoundaryNet` model struct remain in the codebase. The only change is the inference path switching from burn to ort.

## Project Standards

- **Dependency management**: uv
- **Python version**: 3.10+
- **Type hints**: All function signatures fully annotated
- **Docstrings**: All public functions, Google style
- **Max function length**: 60 lines — extract helpers aggressively
- **Formatting/linting**: ruff
- **No notebooks** — all code in importable modules, CLI via scripts
- **Testing**: pytest
  - Binary record reader: round-trip with known data
  - Encoding correctness: verify Python produces same outputs as Rust for identical inputs
  - Model: output shape, forward pass with known weights
  - Loss: known inputs → known outputs, matches Rust loss values
  - ONNX: export → reload → predictions match PyTorch within 1e-5
  - End-to-end: generate tiny dataset (via Rust CLI) → train 10 epochs → export → load in onnxruntime → predict
- **Cross-validation**: Generate reference test vectors from Rust (encode a known record, run forward pass) and verify Python matches exactly

## Out of Scope

- Turn/flop BoundaryNet training (river only for now)
- TensorBoard or wandb integration (can add later)
- Distributed training / multi-GPU
- Changes to datagen (stays in Rust)
- Removing burn training code
