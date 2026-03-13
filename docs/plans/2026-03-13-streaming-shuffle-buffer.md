# Streaming Shuffle Buffer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace the chunked read-shuffle-drain reader thread with an eviction-based streaming shuffle buffer that keeps disk reads continuous and eliminates GPU starvation between chunks.

**Architecture:** Only the reader thread inside `spawn_dataloader_thread` changes. A new `read_one()` method on `StreamingReader` enables record-at-a-time reads. The reader fills a buffer once, then continuously swaps in new records and evicts old ones into batches. The encoder thread and training loop are untouched. A bonus fix replaces the `count_total_records` full-file scan with `count_records` (file_size / record_size).

**Tech Stack:** Rust, `rand`/`rand_chacha` (existing deps), `std::sync::mpsc` channels (existing)

---

### Task 1: Add `read_one()` to `StreamingReader`

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs:201-266` (StreamingReader impl)

**Step 1: Write the failing test**

Add to the existing `streaming_reader_spans_files` test area in `training.rs`:

```rust
#[test]
fn streaming_reader_read_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.bin");
    {
        let mut file = std::fs::File::create(&path).unwrap();
        for i in 0..3 {
            let rec = TrainingRecord {
                board: vec![0, 4, 8, 12, 16],
                pot: (i as f32) * 10.0,
                effective_stack: 50.0,
                player: 0,
                game_value: 0.0,
                oop_range: [0.0; 1326],
                ip_range: [0.0; 1326],
                cfvs: [0.0; 1326],
                valid_mask: [1; 1326],
            };
            write_record(&mut file, &rec).unwrap();
        }
    }

    let mut reader = StreamingReader::new(vec![path]);

    // Read all 3 records one at a time.
    let r0 = reader.read_one().unwrap();
    assert!((r0.pot - 0.0).abs() < 1e-6);
    let r1 = reader.read_one().unwrap();
    assert!((r1.pot - 10.0).abs() < 1e-6);
    let r2 = reader.read_one().unwrap();
    assert!((r2.pot - 20.0).abs() < 1e-6);

    // Exhausted.
    assert!(reader.read_one().is_none());

    // Reset and read again.
    reader.reset();
    assert!(reader.read_one().is_some());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet streaming_reader_read_one`
Expected: FAIL — `read_one` method does not exist.

**Step 3: Write minimal implementation**

Add this method to `StreamingReader` (after `read_chunk`):

```rust
/// Read a single record, advancing across file boundaries.
/// Returns `None` when all files are exhausted.
fn read_one(&mut self) -> Option<TrainingRecord> {
    while !self.exhausted {
        // Open next file if no reader active.
        if self.reader.is_none() {
            if self.current_file_idx >= self.files.len() {
                self.exhausted = true;
                return None;
            }
            let path = &self.files[self.current_file_idx];
            match std::fs::File::open(path) {
                Ok(f) => {
                    self.reader = Some(BufReader::new(f));
                }
                Err(e) => {
                    eprintln!("Warning: skipping {}: {e}", path.display());
                    self.current_file_idx += 1;
                    continue;
                }
            }
        }

        let reader = self.reader.as_mut().unwrap();
        match read_record(reader) {
            Ok(rec) => return Some(rec),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.reader = None;
                self.current_file_idx += 1;
            }
            Err(e) => {
                eprintln!(
                    "Warning: read error in {}: {e}",
                    self.files[self.current_file_idx].display()
                );
                self.reader = None;
                self.current_file_idx += 1;
            }
        }
    }
    None
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p cfvnet streaming_reader_read_one`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "feat(cfvnet): add read_one() to StreamingReader"
```

---

### Task 2: Replace `count_total_records` with file-size arithmetic

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs:294-310` (`count_total_records`)
- Modify: `crates/cfvnet/src/model/training.rs:479-481` (call site in `train`)
- Reference: `crates/cfvnet/src/datagen/storage.rs:10` (`record_size` const fn)

**Step 1: Write the failing test**

```rust
#[test]
fn count_total_records_uses_file_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.bin");
    {
        let mut file = std::fs::File::create(&path).unwrap();
        for i in 0..5 {
            let rec = TrainingRecord {
                board: vec![0, 4, 8, 12, 16],
                pot: i as f32,
                effective_stack: 50.0,
                player: 0,
                game_value: 0.0,
                oop_range: [0.0; 1326],
                ip_range: [0.0; 1326],
                cfvs: [0.0; 1326],
                valid_mask: [1; 1326],
            };
            write_record(&mut file, &rec).unwrap();
        }
    }
    let count = count_total_records(&[path], 5);
    assert_eq!(count, 5);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet count_total_records_uses_file_size`
Expected: FAIL — `count_total_records` does not take a `board_size` parameter.

**Step 3: Rewrite `count_total_records`**

Replace the existing `count_total_records` function (lines 294-310) with:

```rust
/// Count total records across all files using file size arithmetic.
///
/// All records must have the same `board_size`. Files whose size is not
/// an exact multiple of `record_size` are still counted (truncated).
fn count_total_records(files: &[PathBuf], board_size: usize) -> u64 {
    let rec_size = crate::datagen::storage::record_size(board_size) as u64;
    let mut total = 0u64;
    for path in files {
        match std::fs::metadata(path) {
            Ok(meta) => total += meta.len() / rec_size,
            Err(e) => eprintln!("Warning: cannot stat {}: {e}", path.display()),
        }
    }
    total
}
```

**Step 4: Update call site in `train()`**

Change line 480 from:
```rust
let total_records = count_total_records(&files) as usize;
```
to:
```rust
let total_records = count_total_records(&files, board_cards) as usize;
```

**Step 5: Run test to verify it passes**

Run: `cargo test -p cfvnet count_total_records_uses_file_size`
Expected: PASS

**Step 6: Run all cfvnet tests**

Run: `cargo test -p cfvnet`
Expected: All pass (the existing tests create 5-card board data, so `board_cards=5` is correct throughout).

**Step 7: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "perf(cfvnet): replace count_total_records with file-size arithmetic

Eliminates a full sequential scan of every data file at startup.
For 10M records at ~15KB each, this saves reading ~150GB before
training begins."
```

---

### Task 3: Rewrite reader thread with streaming shuffle buffer

This is the core change. Replace the chunked read-shuffle-drain loop with the eviction-based streaming shuffle buffer.

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs:394-454` (`spawn_dataloader_thread`)

**Step 1: Write the failing test**

This test verifies that the streaming shuffle buffer produces exactly `N` records per epoch (once-per-record coverage) and that batches flow continuously.

```rust
#[test]
fn streaming_shuffle_buffer_coverage() {
    // 64 records, shuffle buffer of 16, batch size 8.
    // Expect: all 64 records seen per epoch, delivered in 8 batches of 8.
    let file = write_test_data(64);
    let config = TrainConfig {
        batch_size: 8,
        shuffle_buffer_size: 16,
        prefetch_depth: 2,
        epochs: 2,
        ..default_test_config()
    };
    let val_count = 0;

    let (rx, handles) = spawn_dataloader_thread(
        &[file.path().to_path_buf()],
        &config,
        val_count,
        5,
    );

    // One epoch = 64 / 8 = 8 batches.
    let mut total_batches = 0;
    // Receive two epochs worth of batches (16 batches).
    for _ in 0..16 {
        let batch = rx.recv().expect("should receive batch");
        assert_eq!(batch.len, 8, "each batch should have batch_size records");
        total_batches += 1;
    }
    assert_eq!(total_batches, 16);

    drop(rx);
    for handle in handles {
        handle.join().expect("thread should not panic");
    }
}
```

**Step 2: Run test to verify it fails or establishes baseline**

Run: `cargo test -p cfvnet streaming_shuffle_buffer_coverage`
Expected: May pass with current impl (since the test checks batch count, not continuity). This test establishes a correctness baseline to ensure the rewrite doesn't break batch delivery.

**Step 3: Rewrite the reader thread**

Replace the reader thread closure in `spawn_dataloader_thread` (lines 410-441) with:

```rust
// Stage 1: Reader thread — streaming shuffle buffer with eviction.
let reader_files = files.to_vec();
let reader_thread = std::thread::spawn(move || {
    use rand::seq::SliceRandom;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut files = reader_files;
    let mut epoch = 0u64;

    loop {
        let mut reader = StreamingReader::new(files.clone());
        if val_count > 0 {
            let _ = reader.read_chunk(val_count);
        }

        // Phase 1: Fill buffer.
        let mut buffer = reader.read_chunk(shuffle_buffer_size);
        if buffer.is_empty() {
            eprintln!("Warning: no training records found, stopping dataloader");
            return;
        }
        buffer.shuffle(&mut rng);

        // Phase 2: Stream with eviction — read one record, evict one from buffer.
        let mut batch_buf = Vec::with_capacity(batch_size);
        while let Some(record) = reader.read_one() {
            let idx = rng.gen_range(0..buffer.len());
            let evicted = std::mem::replace(&mut buffer[idx], record);
            batch_buf.push(evicted);

            if batch_buf.len() == batch_size {
                if record_tx.send(batch_buf).is_err() {
                    return;
                }
                batch_buf = Vec::with_capacity(batch_size);
            }
        }

        // Phase 3: Drain remaining buffer + partial batch_buf.
        buffer.shuffle(&mut rng);
        for record in buffer.drain(..) {
            batch_buf.push(record);
            if batch_buf.len() == batch_size {
                if record_tx.send(batch_buf).is_err() {
                    return;
                }
                batch_buf = Vec::with_capacity(batch_size);
            }
        }
        // Send any partial final batch.
        if !batch_buf.is_empty() {
            if record_tx.send(batch_buf).is_err() {
                return;
            }
        }

        // Phase 4: Prepare next epoch.
        epoch += 1;
        rng = ChaCha8Rng::seed_from_u64(42 + epoch);
        files.shuffle(&mut rng);
    }
});
```

Note: `files` is now a mutable `Vec<PathBuf>` owned by the thread (cloned from the input). Each epoch creates a fresh `StreamingReader` from the (possibly shuffled) file list.

**Step 4: Run test to verify it passes**

Run: `cargo test -p cfvnet streaming_shuffle_buffer_coverage`
Expected: PASS

**Step 5: Run all cfvnet tests**

Run: `cargo test -p cfvnet`
Expected: All pass. Key tests to watch:
- `training_with_streaming` — exercises data > shuffle_buffer_size
- `overfit_single_batch` — ensures training still converges
- `training_reduces_loss` — ensures loss decreases
- `dataloader_stops_on_empty_files` — empty file handling still works

**Step 6: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "perf(cfvnet): replace chunked shuffle with streaming shuffle buffer

Eviction-based shuffle buffer keeps disk reads continuous instead of
the previous read-262K-shuffle-drain cycle that stalled the pipeline
~38 times per epoch. Now stalls only once per epoch (initial fill).

Each record enters the buffer once and exits once, preserving exact
once-per-epoch coverage. File list and RNG re-shuffled each epoch."
```

---

### Task 4: Update `spawn_dataloader_thread` signature and doc comment

The function's doc comment (lines 388-393) references the old chunked design. Update it.

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs:388-393`

**Step 1: Update the doc comment**

Replace lines 388-393 with:

```rust
/// Spawn a two-stage dataloader pipeline: a reader thread with a streaming
/// shuffle buffer, and an encoder thread that encodes batches in parallel
/// via rayon. The reader continuously reads records one at a time, evicting
/// shuffled records into batches, eliminating pipeline stalls between chunks.
///
/// Returns `(batch_receiver, thread_handles)`. The threads exit cleanly when
/// the receiver is dropped (cascading channel close).
```

**Step 2: Run all cfvnet tests**

Run: `cargo test -p cfvnet`
Expected: All pass.

**Step 3: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "docs(cfvnet): update spawn_dataloader_thread doc comment"
```

---

### Task 5: Update design docs and sample config comments

**Files:**
- Modify: `docs/training.md` (if it references shuffle_buffer_size semantics)
- Modify: `sample_configurations/river_cfvnet.yaml` (add comment explaining buffer is streaming)

**Step 1: Check docs/training.md for shuffle_buffer_size references**

Read `docs/training.md` and search for `shuffle_buffer`. If found, update the description from "chunk size" to "streaming shuffle buffer capacity".

**Step 2: Add inline comment to sample config**

In `sample_configurations/river_cfvnet.yaml`, if `shuffle_buffer_size` is present, add/update comment:

```yaml
  # Streaming shuffle buffer capacity (records). Larger = better shuffle quality,
  # more RAM. Records flow continuously via eviction — no pipeline stalls.
  shuffle_buffer_size: 262144
```

**Step 3: Commit**

```bash
git add docs/training.md sample_configurations/river_cfvnet.yaml
git commit -m "docs: update shuffle_buffer_size description for streaming buffer"
```

---

### Task 6: Final verification

**Step 1: Run full test suite**

Run: `cargo test`
Expected: All pass, completes in < 1 minute.

**Step 2: Run clippy**

Run: `cargo clippy -p cfvnet`
Expected: No warnings.

**Step 3: Verify no dead code from old chunked approach**

Check that the old chunked reader logic (the `consecutive_empty` counter, the chunk-based `for chunk in records.chunks(batch_size)` pattern) has been fully replaced and no dead code remains.
