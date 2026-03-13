# GPU Reservoir Training Pipeline

**Date**: 2026-03-13
**Status**: Approved
**Replaces**: Chunked streaming pipeline with `epochs_per_chunk` replay

## Problem

The current training pipeline loads chunks of `gpu_chunk_size` records to GPU, trains on them `epochs_per_chunk` times, then discards and loads the next chunk. This causes the model to see identical data repeatedly in succession, which can harm training dynamics (overfitting to chunk ordering, correlated gradient updates).

The `epochs_per_chunk` parameter was introduced to keep the GPU busy while the next chunk loads from disk, but it's a workaround, not a solution.

## Design

### GPU-Resident Reservoir

Replace the chunk-based pipeline with a **GPU-resident reservoir**: a set of large tensors that hold `reservoir_size` pre-encoded training records directly in GPU memory. The training loop samples random batches from the reservoir via `index_select` — a pure GPU operation with zero PCIe transfer during training.

A background **refresh thread** continuously reads new records from disk, encodes them, and scatters them into random positions in the reservoir. This gradually rotates the full dataset through the reservoir without interrupting training.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     DISK (10-50M records)                 │
└─────────────────────────┬────────────────────────────────┘
                          │ StreamingReader (sequential)
                          ▼
┌──────────────────────────────────────────────────────────┐
│               REFRESH THREAD (background)                │
│                                                          │
│  Read records → encode → upload small batches to GPU     │
│  Scatter into random positions in reservoir tensors      │
│  Continuous trickle: ~few hundred records/step via PCIe  │
└─────────────────────────┬────────────────────────────────┘
                          │ scatter updates
                          ▼
┌──────────────────────────────────────────────────────────┐
│              GPU RESERVOIR (resident tensors)             │
│                                                          │
│  input:      Tensor [reservoir_size, input_size]         │
│  target:     Tensor [reservoir_size, 1326]               │
│  mask:       Tensor [reservoir_size, 1326]               │
│  range:      Tensor [reservoir_size, 1326]               │
│  game_value: Tensor [reservoir_size]                     │
│                                                          │
│  ~26KB per record                                        │
│  48GB GPU → ~1.5M records (after model+grads headroom)   │
│  96GB Mac → ~3.5M records                                │
└─────────────────────────┬────────────────────────────────┘
                          │ index_select(random_indices)
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                          │
│                                                          │
│  for step in 0..total_steps:                             │
│    indices = random_sample(reservoir_size, batch_size)    │
│    batch = reservoir.index_select(indices)                │
│    loss = forward_backward(model, batch)                  │
│    optimizer.step(lr)                                     │
│                                                          │
│  Epoch = total_records / batch_size steps                 │
│  Validation + checkpoint at epoch boundaries              │
└──────────────────────────────────────────────────────────┘
```

### Epoch Definition

An "epoch" is defined as `total_records / batch_size` training steps, matching the classical definition statistically (same number of samples trained on as records in the dataset). The reservoir's random sampling means coverage is statistical, not exact — over 2-3 epochs, virtually every record is seen.

### Reservoir Turnover

The refresh rate is derived from a `reservoir_turnover` config parameter (default: 1.0), meaning the entire reservoir rotates once per epoch:

```
steps_per_epoch = total_records / batch_size
refresh_per_step = ceil(reservoir_size * reservoir_turnover / steps_per_epoch)
```

- `turnover = 1.0`: reservoir fully rotates once per epoch (default)
- `turnover = 2.0`: faster rotation, more diversity, each record trained on less before eviction
- `turnover = 0.5`: slower rotation, more stable reservoir, records trained on longer

### Reservoir Lifecycle

1. **Initial fill**: Read sequentially from disk until reservoir is full. Training blocks during this phase. Progress bar shows fill status.
2. **Steady-state**: Training loop runs. Refresh thread continuously reads from disk and scatters new records into random reservoir positions.
3. **Epoch boundary**: After `steps_per_epoch` steps, run validation, checkpoint, update LR schedule.
4. **Disk exhaustion**: When all files are read, reset the StreamingReader and continue from the top. The reservoir keeps churning — the training loop is unaware of disk passes.

### Config Changes

```yaml
training:
  reservoir_size: 1_500_000     # NEW — records held on GPU
  reservoir_turnover: 1.0       # NEW — full rotations per epoch
  batch_size: 2048              # unchanged
  epochs: 10                    # unchanged
  learning_rate: 0.001          # unchanged
  lr_min: 0.00001               # unchanged
  huber_delta: 1.0              # unchanged
  aux_loss_weight: 1.0          # unchanged
  validation_split: 0.05        # unchanged
  checkpoint_every_n_epochs: 1  # unchanged
  # REMOVED: gpu_chunk_size, epochs_per_chunk, prefetch_chunks
```

### What Gets Removed

- `gpu_chunk_size` config param
- `epochs_per_chunk` config param
- `prefetch_chunks` config param
- `ChunkMsg` enum
- `ChunkTensors` struct
- `MiniBatch` struct
- Producer/consumer channel pipeline
- Chunk-level shuffle (index_select permutation of entire chunk)
- `PreEncoded::into_tensors`, `to_tensors`, `chunk_tensors` methods

### What Gets Added

- `GpuReservoir<B>` struct: holds the reservoir tensors + metadata
  - `fill()`: initial bulk load from disk
  - `sample_batch()`: random index_select for training
  - `scatter_refresh()`: update random positions with new records
- `RefreshThread`: background thread that reads/encodes and sends refresh batches
- `reservoir_size` and `reservoir_turnover` config params
- Auto-sizing logic: if `reservoir_size` is omitted, compute from available GPU memory

### Memory Budget

| Component | 48GB GPU | 96GB Mac |
|-----------|----------|----------|
| Model + gradients + Adam state | ~500MB | ~500MB |
| Activations (batch 2048) | ~200MB | ~200MB |
| Available for reservoir | ~47GB | ~95GB |
| Records @ 26KB each | ~1.8M | ~3.6M |
| Conservative default | 1.5M | 3.0M |

### Synchronization

The refresh thread writes to reservoir positions that the training loop may be reading from. Two options:

1. **Lock-free with atomic index**: refresh thread writes to a staging buffer; training thread swaps it in between steps. Simple, no contention.
2. **No synchronization needed**: if we accept that a training step might see a partially-updated record (one field from old, another from new), this is statistically harmless for SGD training. The next step will see the fully updated record.

Recommended: option 1 (staging buffer swap between steps) for correctness without measurable overhead.
