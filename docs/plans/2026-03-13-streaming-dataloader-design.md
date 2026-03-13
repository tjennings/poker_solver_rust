# Streaming Dataloader Design

**Date**: 2026-03-13
**Status**: Approved

## Problem

The GPU reservoir holds 1.5M records in GPU memory and samples random batches via `select` (zero PCIe). A background thread scatter-refreshes records into random positions. This works but has a core inefficiency: our CFVNet is a small 7-layer MLP with 500-wide hidden layers — the GPU finishes compute almost instantly, but each sample is ~26KB of data. The GPU is data-bound, not compute-bound.

With async prefetch and a bounded channel, a standard streaming dataloader can keep the GPU fed by having the next batch ready before the current one finishes. This is simpler than the reservoir and gives uniform dataset coverage for free (every record is seen exactly once per epoch).

## Design

### Architecture

```
[Disk] → background thread → [sync_channel(prefetch_depth)] → training loop
              │
              ├─ read shuffle_buffer_size records sequentially
              ├─ shuffle in-place (ChaCha8Rng)
              ├─ split into batch_size chunks
              ├─ encode each chunk (rayon parallel)
              └─ send PreEncoded batches through channel

         On EOF: reset StreamingReader, skip val records, start next epoch pass
```

### Approach: Channel-based streaming pipeline

A background thread reads sequential chunks from disk, shuffles within each chunk, encodes into `PreEncoded` batches, and sends them through a bounded `mpsc::sync_channel`. The training loop simply receives the next batch. Channel capacity provides prefetch depth.

**Shuffling strategy**: Chunked shuffle — read `shuffle_buffer_size` records sequentially (fast I/O), shuffle within the chunk, yield batches. Coverage is local within chunks but cycles through the full dataset each epoch.

### Background thread behavior

1. Open `StreamingReader` over all files, skip `val_count` records
2. Read up to `shuffle_buffer_size` records into a `Vec<TrainingRecord>`
3. Shuffle the vec with `rng.shuffle()`
4. Split into `batch_size`-sized slices, encode each with `PreEncoded::from_records()`, send through channel
5. Repeat from step 2 until EOF
6. On EOF: reset reader, skip val records, continue (infinite loop until channel closes)

### Training loop

```
for each epoch:
    for each step:
        batch = channel.recv()    // blocks until prefetch ready
        create device tensors (autodiff)
        forward → loss → backward → optimizer step
```

No reservoir, no scatter_refresh, no sample_batch. The loop just consumes pre-made batches.

### Config changes

Remove `reservoir_size` and `reservoir_turnover`. Add:

```rust
pub shuffle_buffer_size: usize,  // default 262_144 (256K records)
pub prefetch_depth: usize,       // default 4 (channel capacity in batches)
```

### What stays the same

- `StreamingReader` — unchanged
- `PreEncoded` / `DeviceTensors` — unchanged
- `compute_val_loss` — unchanged
- `load_validation_set` — unchanged
- Cosine LR, loss reading interval, checkpoint logic — unchanged
- Model save/load, resume from checkpoint — unchanged

### CUDA considerations

Burn doesn't expose pinned memory through its public API. The channel-based prefetch with depth 4 provides overlap: the background thread encodes the next batch on CPU while the GPU trains on the current one. If profiling later shows PCIe transfer is still a bottleneck, pinned memory can be added via cudarc interop as a drop-in enhancement to `PreEncoded` buffer allocation.

### Benefits over reservoir

1. **Uniform coverage** — every record seen exactly once per epoch (no duplicate sampling, no missed records)
2. **Simpler training loop** — just recv + train, no reservoir management
3. **Lower GPU memory** — no large reservoir tensors resident on GPU
4. **Deterministic epochs** — step count is exact, not approximated from reservoir turnover
5. **Simpler config** — two intuitive params instead of reservoir_size + turnover
