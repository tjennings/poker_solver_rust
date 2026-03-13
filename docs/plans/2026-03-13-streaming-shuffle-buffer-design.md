# Streaming Shuffle Buffer Design

## Problem

The current chunked-shuffle dataloader creates pipeline stalls that starve the GPU.
The reader thread reads 262K records, shuffles all of them, then yields batches.
During the read phase, the encoder and GPU sit idle. With ~10M records and 256K
chunks, this produces ~38 stalls per epoch. Observed symptoms: bursty disk reads,
low CPU utilization, bursty GPU I/O, and low overall GPU utilization.

## Solution

Replace the chunked read-shuffle-drain loop with an **eviction-based shuffle buffer**
(the same algorithm as `tf.data.Dataset.shuffle`). After an initial fill, the reader
continuously reads one record at a time from disk. Each new record replaces a random
slot in the buffer; the evicted record flows into the batch pipeline. This keeps disk
reads, CPU encoding, and GPU training all active simultaneously.

## Key Property: Exact Once-Per-Epoch Coverage

Every record enters the buffer exactly once and exits exactly once per epoch:

1. Records loaded during initial fill get evicted later by incoming records
2. Records entering during streaming get evicted by later records
3. Records still in the buffer at EOF are drained in the final flush

This is WITHOUT-replacement sampling — the same coverage guarantee as the current
chunked shuffle, and fundamentally different from the old GPU reservoir (which used
WITH-replacement sampling and broke model quality).

## Algorithm

```
reader_thread(files, val_count, shuffle_buffer_size, batch_size, record_tx):
    rng = ChaCha8Rng::seed_from_u64(42)
    reader = StreamingReader::new(files)
    skip(reader, val_count)
    epoch = 0

    loop:
        // Phase 1: Fill buffer (one-time per epoch)
        buffer = reader.read_chunk(shuffle_buffer_size)
        buffer.shuffle(&mut rng)

        // Phase 2: Stream with eviction (continuous)
        batch_buf = Vec::new()
        loop:
            match reader.read_one():
                Some(record):
                    idx = rng.gen_range(0..buffer.len())
                    evicted = mem::replace(&mut buffer[idx], record)
                    batch_buf.push(evicted)
                None:  // EOF
                    break

            if batch_buf.len() == batch_size:
                send(batch_buf) or return
                batch_buf = Vec::new()

        // Phase 3: Drain remaining buffer + partial batch
        buffer.shuffle(&mut rng)
        for record in buffer.drain(..).chain(batch_buf.drain(..)):
            drain_buf.push(record)
            if drain_buf.len() == batch_size:
                send(drain_buf) or return
                drain_buf = Vec::new()
        // send any partial final batch

        // Phase 4: Reset for next epoch
        epoch += 1
        rng = ChaCha8Rng::seed_from_u64(42 + epoch as u64)
        files.shuffle(&mut rng)
        reader = StreamingReader::new(files)
        skip(reader, val_count)
```

## Changes

| Component | Change |
|-----------|--------|
| `StreamingReader` | Add `read_one() -> Option<TrainingRecord>` method |
| Reader thread | Replace chunk-shuffle-drain loop with fill-stream-drain-reset |
| `count_total_records` | Replace full file scan with `file_size / record_size` |
| Config | None — `shuffle_buffer_size` reused as buffer capacity |
| Encoder thread | No change |
| Training loop | No change |
| Epoch semantics | Preserved — same step count, same LR schedule |

## Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Pipeline stalls per epoch | ~38 (one per 262K chunk) | 1 (initial fill only) |
| Disk read pattern | Bursty (read 4GB, pause, repeat) | Continuous |
| GPU idle time | High (starved between chunks) | Minimal (continuous batch flow) |
| Shuffle quality | Within-chunk only | Within-buffer (same window, continuous) |
| Coverage per epoch | Exact (once per record) | Exact (once per record) |
| Startup scan | Full dataset read-and-discard | file_size / record_size |

## Epoch Boundary Improvements

- **Shuffle file list** at each epoch reset (currently always sorted)
- **Re-seed RNG per epoch** (seed + epoch_counter) for different shuffle orders

## Config Parameters (unchanged)

- `shuffle_buffer_size`: Buffer capacity in records (default 262,144 = ~4 GB)
- `prefetch_depth`: Channel depths (default 4)
- `batch_size`: Training batch size (default 8192)
