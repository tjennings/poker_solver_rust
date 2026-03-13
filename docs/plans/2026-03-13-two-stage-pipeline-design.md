# Two-Stage Pipeline Dataloader Design

**Date**: 2026-03-13
**Status**: Approved

## Problem

The single-thread streaming dataloader stalls every ~25% of an epoch at chunk boundaries. The reader thread does read → shuffle → split → encode → send sequentially, so the GPU starves while the thread reads the next chunk from disk. Reading and encoding need to overlap.

## Design

### Architecture

```
Reader Thread (1, I/O-bound)           Encoder Thread (1 + rayon fanout)       Training Loop
  read shuffle_buffer_size records       recv Vec<TrainingRecord>                recv PreEncoded
  shuffle in-place                       PreEncoded::from_records (par_iter)     create tensors
  split into batch_size chunks           send PreEncoded                         forward/backward
  send Vec<TrainingRecord>
       ↓                                      ↓                                      ↓
  [record_channel]                →      [batch_channel]                  →     data_rx.recv()
  capacity: prefetch_depth * 2           capacity: prefetch_depth
```

### Two threads, not N

- **Reading** is I/O-bound (one disk), so more readers don't help.
- **Encoding** already parallelizes via rayon's `par_iter` in `PreEncoded::from_records()`, so one dispatcher thread fans across all available cores automatically.

### Changes from single-thread dataloader

`spawn_dataloader_thread` splits into a reader thread and an encoder thread:

1. **Reader thread**: read chunk → shuffle → split into `batch_size` slices → send `Vec<TrainingRecord>` through `record_channel`. On EOF, reset + skip val records. On send error, return.

2. **Encoder thread**: recv `Vec<TrainingRecord>` → `PreEncoded::from_records()` (rayon `par_iter`) → send `PreEncoded` through `batch_channel`. On recv error, return.

### Shutdown

Training loop drops `batch_rx` → encoder's send fails → encoder drops `record_rx` → reader's send fails → both exit. Clean cascade.

### Config

No new params. `prefetch_depth` controls `batch_channel` capacity. `record_channel` gets `prefetch_depth * 2`.

### What stays the same

- `PreEncoded::from_records()`, `StreamingReader`, training loop, all config params
- Return type becomes `(Receiver<PreEncoded>, Vec<JoinHandle<()>>)` — train() joins both handles on shutdown
