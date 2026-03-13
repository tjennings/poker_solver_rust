# GPU Reservoir Training Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace the chunk-replay training pipeline with a GPU-resident reservoir that samples random batches from GPU memory and continuously refreshes from disk.

**Architecture:** A `GpuReservoir<B>` struct holds 5 large tensors on GPU (input, target, mask, range, game_value) sized to `reservoir_size`. The training loop samples random indices per step — pure GPU, zero PCIe transfer. A background thread reads records from disk, encodes them, and sends small refresh batches through a channel. Between training steps, the main thread scatters these into random reservoir positions. Epoch = `total_records / batch_size` steps.

**Tech Stack:** Rust, burn 0.16 (Tensor, Backend, AutodiffBackend), rayon, rand/rand_chacha, indicatif, std::sync::mpsc

**Design doc:** `docs/plans/2026-03-13-gpu-reservoir-training-design.md`

---

### Task 1: Update TrainingConfig — remove old params, add reservoir params

**Files:**
- Modify: `crates/cfvnet/src/config.rs:135-223`
- Modify: `crates/cfvnet/src/config.rs:239-294` (tests)

**Step 1: Write the failing test**

Add to `config.rs` tests:

```rust
#[test]
fn parse_config_with_reservoir_params() {
    let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
training:
  reservoir_size: 1500000
  reservoir_turnover: 1.0
"#;
    let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(config.training.reservoir_size, 1_500_000);
    assert!((config.training.reservoir_turnover - 1.0).abs() < 1e-9);
}

#[test]
fn config_defaults_for_reservoir() {
    let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
"#;
    let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(config.training.reservoir_size, 1_500_000);
    assert!((config.training.reservoir_turnover - 1.0).abs() < 1e-9);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet parse_config_with_reservoir_params -- --nocapture`
Expected: FAIL — `reservoir_size` field doesn't exist yet

**Step 3: Write minimal implementation**

In `TrainingConfig` struct, replace these three fields:

```rust
// REMOVE:
pub gpu_chunk_size: usize,
pub epochs_per_chunk: usize,
pub prefetch_chunks: usize,
```

With:

```rust
#[serde(default = "default_reservoir_size")]
pub reservoir_size: usize,
#[serde(default = "default_reservoir_turnover")]
pub reservoir_turnover: f64,
```

Add default functions:

```rust
fn default_reservoir_size() -> usize {
    1_500_000
}
fn default_reservoir_turnover() -> f64 {
    1.0
}
```

Remove the three old default functions: `default_gpu_chunk_size`, `default_epochs_per_chunk`, `default_prefetch_chunks`.

Update `Default for TrainingConfig` impl: replace the three old fields with `reservoir_size: 1_500_000` and `reservoir_turnover: 1.0`.

Update existing tests `parse_full_config` if it references old fields — remove `gpu_chunk_size`/`epochs_per_chunk`/`prefetch_chunks` from the YAML in that test.

**Step 4: Run test to verify it passes**

Run: `cargo test -p cfvnet -- config --nocapture`
Expected: ALL config tests PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/src/config.rs
git commit -m "refactor(cfvnet): replace chunk params with reservoir_size and reservoir_turnover in config"
```

---

### Task 2: Update TrainConfig struct and cmd_train bridge

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs:22-37` (TrainConfig struct)
- Modify: `crates/cfvnet/src/main.rs:223-237` (cmd_train config bridge)

**Step 1: Update `TrainConfig` in training.rs**

Replace:

```rust
pub gpu_chunk_size: usize,
pub epochs_per_chunk: usize,
pub prefetch_chunks: usize,
```

With:

```rust
pub reservoir_size: usize,
pub reservoir_turnover: f64,
```

**Step 2: Update `cmd_train` in main.rs**

In the `train_config` construction (around line 223-237), replace:

```rust
gpu_chunk_size: cfg.training.gpu_chunk_size,
epochs_per_chunk: cfg.training.epochs_per_chunk,
prefetch_chunks: cfg.training.prefetch_chunks,
```

With:

```rust
reservoir_size: cfg.training.reservoir_size,
reservoir_turnover: cfg.training.reservoir_turnover,
```

**Step 3: Update all test TrainConfig instantiations in training.rs**

Every test that constructs a `TrainConfig` needs the old fields replaced. For all tests, replace:

```rust
gpu_chunk_size: ...,
epochs_per_chunk: ...,
prefetch_chunks: ...,
```

With:

```rust
reservoir_size: 100,  // small for tests
reservoir_turnover: 1.0,
```

**Step 4: Verify compilation**

Run: `cargo check -p cfvnet`
Expected: Compiles (training loop will have errors but check should show what remains)

Note: The training loop itself still references old types — that's expected, we'll replace it in Tasks 3-5.

**Step 5: Commit**

```bash
git add crates/cfvnet/src/model/training.rs crates/cfvnet/src/main.rs
git commit -m "refactor(cfvnet): update TrainConfig and cmd_train bridge for reservoir params"
```

---

### Task 3: Implement GpuReservoir struct

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs` — add `GpuReservoir<B>` after the existing structs

This is the core new type. It holds the GPU-resident tensors and provides `fill`, `sample_batch`, and `scatter_refresh` operations.

**Step 1: Write the failing test**

Add to training.rs tests:

```rust
#[test]
fn reservoir_fill_and_sample() {
    let file = write_test_data(20);
    let device = Default::default();
    let files = collect_data_files(file.path()).unwrap();
    let total = count_total_records(&files) as usize;

    let mut reservoir = GpuReservoir::<B>::new(&device, 20, 5);
    let mut reader = StreamingReader::new(files);
    let filled = reservoir.fill(&mut reader, 5);
    assert_eq!(filled, 20);

    // Sample a batch — should not panic, should return correct shapes.
    let batch = reservoir.sample_batch(4, &device);
    assert_eq!(batch.input.dims(), [4, input_size(5)]);
    assert_eq!(batch.target.dims(), [4, OUTPUT_SIZE]);
}

#[test]
fn reservoir_scatter_refresh() {
    let file = write_test_data(20);
    let device = Default::default();
    let files = collect_data_files(file.path()).unwrap();

    let mut reservoir = GpuReservoir::<B>::new(&device, 10, 5);
    let mut reader = StreamingReader::new(files);
    reservoir.fill(&mut reader, 5);

    // Read some more records and scatter them in.
    let mut reader2 = StreamingReader::new(vec![file.path().to_path_buf()]);
    let records = reader2.read_chunk(3);
    let encoded = PreEncoded::from_records(&records, 5);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    reservoir.scatter_refresh(&encoded, &mut rng, &device);
    // No panic = success. Reservoir still works.
    let batch = reservoir.sample_batch(4, &device);
    assert_eq!(batch.input.dims(), [4, input_size(5)]);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p cfvnet reservoir_fill_and_sample -- --nocapture`
Expected: FAIL — `GpuReservoir` doesn't exist

**Step 3: Implement GpuReservoir**

Add to training.rs, replacing `ChunkTensors`, `MiniBatch`, and `ChunkMsg`:

```rust
/// GPU-resident reservoir of training data.
///
/// Holds pre-encoded training records as large tensors in GPU memory.
/// The training loop samples random batches via `index_select` (pure GPU).
/// A background thread continuously refreshes records via `scatter_refresh`.
struct GpuReservoir<B: Backend> {
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    range: Tensor<B, 2>,
    game_value: Tensor<B, 1>,
    capacity: usize,
    in_size: usize,
}

/// A single mini-batch of tensors sampled from the reservoir.
struct SampledBatch<B: Backend> {
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    range: Tensor<B, 2>,
    game_value: Tensor<B, 1>,
}

impl<B: Backend> GpuReservoir<B> {
    /// Create an empty reservoir with `capacity` slots on `device`.
    fn new(device: &B::Device, capacity: usize, board_cards: usize) -> Self {
        let in_size = input_size(board_cards);
        Self {
            input: Tensor::zeros([capacity, in_size], device),
            target: Tensor::zeros([capacity, OUTPUT_SIZE], device),
            mask: Tensor::zeros([capacity, OUTPUT_SIZE], device),
            range: Tensor::zeros([capacity, OUTPUT_SIZE], device),
            game_value: Tensor::zeros([capacity], device),
            capacity,
            in_size,
        }
    }

    /// Fill the reservoir from disk. Returns number of records loaded.
    fn fill(&mut self, reader: &mut StreamingReader, board_cards: usize) -> usize {
        let records = reader.read_chunk(self.capacity);
        let n = records.len();
        if n == 0 {
            return 0;
        }
        let encoded = PreEncoded::from_records(&records, board_cards);
        let device = self.input.device();

        // If we read fewer records than capacity, only fill the first n slots.
        // For simplicity, we require the reservoir to be fully filled.
        // If dataset is smaller than reservoir, we fill what we have and
        // the remaining slots stay as zeros (training will sample from 0..n).
        let n_actual = n.min(self.capacity);
        let pe = if n_actual < self.capacity {
            // Partial fill: create tensors of size n_actual, then pad
            encoded
        } else {
            encoded
        };

        self.input = Tensor::from_data(
            TensorData::new(pe.input, [n_actual, self.in_size]),
            &device,
        );
        self.target = Tensor::from_data(
            TensorData::new(pe.target, [n_actual, OUTPUT_SIZE]),
            &device,
        );
        self.mask = Tensor::from_data(
            TensorData::new(pe.mask, [n_actual, OUTPUT_SIZE]),
            &device,
        );
        self.range = Tensor::from_data(
            TensorData::new(pe.range, [n_actual, OUTPUT_SIZE]),
            &device,
        );
        self.game_value = Tensor::from_data(
            TensorData::new(pe.game_value, [n_actual]),
            &device,
        );
        self.capacity = n_actual;
        n_actual
    }

    /// Sample a random mini-batch from the reservoir. Pure GPU operation.
    fn sample_batch(&self, batch_size: usize, device: &B::Device) -> SampledBatch<B> {
        let indices = Tensor::<B, 1, Int>::random(
            [batch_size],
            burn::tensor::Distribution::Uniform(0.0, self.capacity as f64),
            device,
        );
        SampledBatch {
            input: self.input.clone().select(0, indices.clone()),
            target: self.target.clone().select(0, indices.clone()),
            mask: self.mask.clone().select(0, indices.clone()),
            range: self.range.clone().select(0, indices.clone()),
            game_value: self.game_value.clone().select(0, indices),
        }
    }

    /// Scatter a batch of new pre-encoded records into random reservoir positions.
    fn scatter_refresh(
        &mut self,
        encoded: &PreEncoded,
        rng: &mut ChaCha8Rng,
        device: &B::Device,
    ) {
        let n = encoded.len;
        if n == 0 {
            return;
        }

        // Pick n random target positions in the reservoir.
        let positions: Vec<i64> = (0..n)
            .map(|_| rng.gen_range(0..self.capacity) as i64)
            .collect();

        // Create new-value tensors from the encoded data.
        let new_input = Tensor::<B, 2>::from_data(
            TensorData::new(encoded.input.clone(), [n, self.in_size]),
            device,
        );
        let new_target = Tensor::<B, 2>::from_data(
            TensorData::new(encoded.target.clone(), [n, OUTPUT_SIZE]),
            device,
        );
        let new_mask = Tensor::<B, 2>::from_data(
            TensorData::new(encoded.mask.clone(), [n, OUTPUT_SIZE]),
            device,
        );
        let new_range = Tensor::<B, 2>::from_data(
            TensorData::new(encoded.range.clone(), [n, OUTPUT_SIZE]),
            device,
        );
        let new_gv = Tensor::<B, 1>::from_data(
            TensorData::new(encoded.game_value.clone(), [n]),
            device,
        );

        // Use select_assign to scatter new values into the reservoir.
        let idx = Tensor::<B, 1, Int>::from_data(
            TensorData::new(positions, [n]),
            device,
        );
        self.input = self.input.clone().select_assign(0, idx.clone(), new_input);
        self.target = self.target.clone().select_assign(0, idx.clone(), new_target);
        self.mask = self.mask.clone().select_assign(0, idx.clone(), new_mask);
        self.range = self.range.clone().select_assign(0, idx.clone(), new_range);
        self.game_value = self.game_value.clone().select_assign(0, idx, new_gv);
    }
}
```

**Important burn API notes for the implementer:**
- `Tensor::random` with `Distribution::Uniform` generates floats. For integer indices, you may need to cast: `.int()` or use `Tensor::<B, 1, Int>::from_data(...)` with random indices generated on CPU via `rng.gen_range()`.
- `select_assign` is the scatter operation: `tensor.select_assign(dim, indices, values)`. Verify this exists in burn 0.16. If not, fall back to building the full tensor on CPU and uploading.
- The `sample_batch` approach of generating random indices on GPU is ideal. If `Tensor::random` for Int isn't available, generate indices on CPU with `rng` and upload as a small 1D tensor.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p cfvnet reservoir_ -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "feat(cfvnet): add GpuReservoir struct with fill, sample_batch, scatter_refresh"
```

---

### Task 4: Implement the refresh channel and new training loop

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs` — replace the `train<B>()` function body

This is the biggest task. Replace the entire producer/consumer chunk pipeline with the reservoir-based loop.

**Step 1: Write the failing test (reservoir training end-to-end)**

Add to training.rs tests:

```rust
#[test]
fn reservoir_training_reduces_loss() {
    let file = write_test_data(32);
    let device = Default::default();
    let config = TrainConfig {
        hidden_layers: 2,
        hidden_size: 64,
        batch_size: 8,
        epochs: 50,
        learning_rate: 0.001,
        lr_min: 0.001,
        huber_delta: 1.0,
        aux_loss_weight: 0.0,
        validation_split: 0.0,
        checkpoint_every_n_epochs: 0,
        reservoir_size: 32,
        reservoir_turnover: 1.0,
    };
    let result = train::<B>(&device, file.path(), 5, &config, None);
    assert!(
        result.final_train_loss < 0.1,
        "reservoir training should reduce loss, got {}",
        result.final_train_loss
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet reservoir_training_reduces_loss -- --nocapture`
Expected: FAIL (old train function still uses chunks, won't compile with new config)

**Step 3: Rewrite the `train<B>()` function**

Replace the entire function body. Keep the signature identical. The new structure:

```rust
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    data_path: &Path,
    board_cards: usize,
    config: &TrainConfig,
    output_dir: Option<&std::path::Path>,
) -> TrainResult {
    let in_size = input_size(board_cards);
    let mut model = CfvNet::<B>::new(device, config.hidden_layers, config.hidden_size, in_size);

    // Resume from checkpoint (same as before).
    if let Some(dir) = output_dir {
        let model_path = dir.join("model");
        if model_path.with_extension("mpk.gz").exists() {
            let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
            match model.clone().load_file(model_path, &recorder, device) {
                Ok(loaded) => {
                    eprintln!("Resuming from checkpoint");
                    model = loaded;
                }
                Err(e) => {
                    eprintln!("Warning: failed to load checkpoint, starting fresh: {e}");
                }
            }
        }
    }

    let mut optim = AdamConfig::new().init::<B, CfvNet<B>>();
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let files = collect_data_files(data_path).unwrap_or_else(|e| {
        eprintln!("failed to collect data files: {e}");
        std::process::exit(1);
    });

    eprintln!("Counting records across {} file(s)...", files.len());
    let total_records = count_total_records(&files) as usize;
    eprintln!("Total records: {total_records}");

    if total_records == 0 {
        eprintln!("No training records found.");
        return TrainResult { final_train_loss: f32::MAX };
    }

    // Load validation set.
    let val_count = (total_records as f64 * config.validation_split) as usize;
    let val_encoded = if val_count > 0 {
        eprintln!("Loading {val_count} validation records...");
        let mut val_reader = StreamingReader::new(files.clone());
        let val_records = val_reader.read_chunk(val_count);
        let actual_val = val_records.len();
        eprintln!("Loaded {actual_val} validation records");
        Some(PreEncoded::from_records(&val_records, board_cards))
    } else {
        None
    };

    // Fill reservoir from disk.
    let reservoir_cap = config.reservoir_size.min(total_records);
    eprintln!("Filling reservoir ({reservoir_cap} records)...");
    let mut reservoir = GpuReservoir::<B>::new(device, reservoir_cap, board_cards);
    {
        let mut fill_reader = StreamingReader::new(files.clone());
        let filled = reservoir.fill(&mut fill_reader, board_cards);
        eprintln!("Reservoir filled: {filled} records");
    }

    // Compute training schedule.
    let batch_size = config.batch_size;
    let steps_per_epoch = total_records.div_ceil(batch_size);
    let total_steps = steps_per_epoch * config.epochs;
    let refresh_per_step = ((reservoir_cap as f64 * config.reservoir_turnover)
        / steps_per_epoch as f64)
        .ceil() as usize;

    eprintln!(
        "Training: {} epochs x {} steps/epoch = {} total steps, refresh {}/step",
        config.epochs, steps_per_epoch, total_steps, refresh_per_step
    );

    // Spawn refresh thread: reads from disk, encodes, sends PreEncoded batches.
    let (refresh_tx, refresh_rx) = mpsc::sync_channel::<PreEncoded>(4);
    let refresh_files = files.clone();
    let refresh_thread = std::thread::spawn(move || {
        let mut reader = StreamingReader::new(refresh_files);
        loop {
            let records = reader.read_chunk(refresh_per_step.max(1));
            if records.is_empty() {
                // Exhausted all files — loop back to start.
                reader.reset();
                continue;
            }
            let encoded = PreEncoded::from_records(&records, board_cards);
            if refresh_tx.send(encoded).is_err() {
                return; // Training done, channel closed.
            }
        }
    });

    let mut final_loss = f32::MAX;
    let mut global_step: usize = 0;

    for epoch in 0..config.epochs {
        let pb = ProgressBar::new(steps_per_epoch as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Epoch {msg} {wide_bar} {pos}/{len} [{elapsed}] ETA {eta}")
                .unwrap(),
        );

        let mut epoch_loss = 0.0_f64;
        let mut epoch_loss_count = 0_u64;

        for step_in_epoch in 0..steps_per_epoch {
            // Apply any pending refresh batch (non-blocking).
            if let Ok(encoded) = refresh_rx.try_recv() {
                reservoir.scatter_refresh(&encoded, &mut rng, device);
            }

            // Sample batch from reservoir and train.
            let batch = reservoir.sample_batch(batch_size, device);
            let pred = model.forward(batch.input);
            let loss = cfvnet_loss(
                pred,
                batch.target,
                batch.mask,
                batch.range,
                batch.game_value,
                config.huber_delta,
                config.aux_loss_weight,
            );

            // Read loss periodically to avoid GPU sync stalls.
            let is_last = step_in_epoch + 1 == steps_per_epoch;
            if step_in_epoch % LOSS_READ_INTERVAL == 0 || is_last {
                final_loss = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
                epoch_loss += final_loss as f64;
                epoch_loss_count += 1;
            }

            let lr = cosine_lr(config.learning_rate, config.lr_min, global_step, total_steps);
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads_params);

            global_step += 1;
            pb.inc(1);
        }

        let avg_loss = if epoch_loss_count > 0 {
            epoch_loss / epoch_loss_count as f64
        } else {
            0.0
        };
        let lr_now = cosine_lr(
            config.learning_rate,
            config.lr_min,
            global_step.saturating_sub(1),
            total_steps,
        );

        let mut summary = format!(
            "{}/{} lr={lr_now:.2e} train={avg_loss:.6}",
            epoch + 1, config.epochs,
        );

        // Validation loss at epoch boundary.
        if let Some(ref val_enc) = val_encoded {
            let val_loss = compute_val_loss(&model, val_enc, config, device);
            summary.push_str(&format!(" val={val_loss:.6}"));
        }

        pb.finish_with_message(summary);

        // Checkpoint.
        if config.checkpoint_every_n_epochs > 0
            && (epoch + 1) % config.checkpoint_every_n_epochs == 0
            && let Some(dir) = output_dir
        {
            save_model(&model, dir, &format!("checkpoint_epoch{}", epoch + 1));
        }
    }

    // Drop refresh channel to signal thread to exit, then join.
    drop(refresh_rx);
    let _ = refresh_thread.join();

    // Save final model.
    if let Some(dir) = output_dir {
        save_model(&model, dir, "model");
    }

    TrainResult { final_train_loss: final_loss }
}
```

**Step 4: Update `compute_val_loss` to work with new types**

The existing `compute_val_loss` uses `PreEncoded::to_tensors` which returns `ChunkTensors`. We need to adapt it. The simplest approach: keep `PreEncoded` and `PreEncoded::from_records` (they're still useful for encoding), but change `compute_val_loss` to work with the PreEncoded directly by creating tensors inline:

```rust
fn compute_val_loss<B: AutodiffBackend>(
    model: &CfvNet<B>,
    encoded: &PreEncoded,
    config: &TrainConfig,
    device: &B::Device,
) -> f64 {
    let valid_model = model.valid();
    let n = encoded.len;
    let in_size = encoded.in_size;
    let batch_size = config.batch_size;

    // Create full validation tensors on the inner (non-autodiff) backend.
    let input: Tensor<B::InnerBackend, 2> = Tensor::from_data(
        TensorData::new(encoded.input.clone(), [n, in_size]), device);
    let target: Tensor<B::InnerBackend, 2> = Tensor::from_data(
        TensorData::new(encoded.target.clone(), [n, OUTPUT_SIZE]), device);
    let mask: Tensor<B::InnerBackend, 2> = Tensor::from_data(
        TensorData::new(encoded.mask.clone(), [n, OUTPUT_SIZE]), device);
    let range: Tensor<B::InnerBackend, 2> = Tensor::from_data(
        TensorData::new(encoded.range.clone(), [n, OUTPUT_SIZE]), device);
    let game_value: Tensor<B::InnerBackend, 1> = Tensor::from_data(
        TensorData::new(encoded.game_value.clone(), [n]), device);

    let mut total_loss = 0.0_f64;
    let mut batch_count = 0_u64;

    for batch_start in (0..n).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n);
        let len = batch_end - batch_start;

        let b_input = input.clone().narrow(0, batch_start, len);
        let b_target = target.clone().narrow(0, batch_start, len);
        let b_mask = mask.clone().narrow(0, batch_start, len);
        let b_range = range.clone().narrow(0, batch_start, len);
        let b_gv = game_value.clone().narrow(0, batch_start, len);

        let pred = valid_model.forward(b_input);
        let loss = cfvnet_loss(pred, b_target, b_mask, b_range, b_gv, config.huber_delta, config.aux_loss_weight);
        let val: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];
        total_loss += val as f64;
        batch_count += 1;
    }

    if batch_count == 0 { 0.0 } else { total_loss / batch_count as f64 }
}
```

**Step 5: Remove dead code**

Delete:
- `ChunkTensors<B>` struct and its impl block
- `MiniBatch<B>` struct
- `ChunkMsg` enum
- `PreEncoded::into_tensors`, `PreEncoded::to_tensors`, `PreEncoded::chunk_tensors` methods

Keep:
- `PreEncoded` struct and `PreEncoded::from_records` (used by fill, refresh, and validation)
- `StreamingReader` (used by fill and refresh threads)
- `collect_data_files`, `count_total_records` (unchanged)
- `cosine_lr`, `save_model`, `LOSS_READ_INTERVAL` (unchanged)

**Step 6: Run test to verify it passes**

Run: `cargo test -p cfvnet reservoir_training_reduces_loss -- --nocapture`
Expected: PASS

**Step 7: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "feat(cfvnet): replace chunk pipeline with GPU reservoir training loop"
```

---

### Task 5: Update and verify all existing tests

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs` (test module)

All existing tests need to work with the new reservoir-based `train()`. Most already had their `TrainConfig` updated in Task 2 — now verify they pass.

**Step 1: Run the full test suite**

Run: `cargo test -p cfvnet -- --nocapture`

**Step 2: Fix any failing tests**

Expected issues:
- `multi_epoch_per_chunk` test: DELETE this test entirely — the concept no longer exists.
- `overfit_single_batch`: Should still work — reservoir with 16 records, 200 epochs. May need `reservoir_size: 16` to match the data size.
- `streaming_reader_spans_files`: Pure unit test, should pass unchanged.
- Other training tests: Should pass with the new reservoir params.

**Step 3: Add a test for reservoir turnover behavior**

```rust
#[test]
fn training_with_refresh() {
    // Use more data than reservoir_size to exercise the refresh path.
    let file = write_test_data(64);
    let device = Default::default();
    let config = TrainConfig {
        hidden_layers: 2,
        hidden_size: 64,
        batch_size: 8,
        epochs: 5,
        learning_rate: 0.001,
        lr_min: 0.001,
        huber_delta: 1.0,
        aux_loss_weight: 0.0,
        validation_split: 0.0,
        checkpoint_every_n_epochs: 0,
        reservoir_size: 16,  // smaller than data → refresh thread runs
        reservoir_turnover: 2.0,
    };
    let result = train::<B>(&device, file.path(), 5, &config, None);
    assert!(result.final_train_loss < 10.0);
}
```

**Step 4: Run full suite again**

Run: `cargo test -p cfvnet -- --nocapture`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "test(cfvnet): update tests for reservoir training, add refresh coverage"
```

---

### Task 6: Update sample config and documentation

**Files:**
- Modify: `sample_configurations/river_cfvnet.yaml`
- Modify: `docs/training.md` (if it references `gpu_chunk_size`, `epochs_per_chunk`, `prefetch_chunks`)

**Step 1: Update sample config**

In `sample_configurations/river_cfvnet.yaml`, replace:

```yaml
  gpu_chunk_size: 200000
  epochs_per_chunk: 15
```

With:

```yaml
  reservoir_size: 1500000
  reservoir_turnover: 1.0
```

**Step 2: Update docs/training.md if it references old params**

Search for `gpu_chunk_size`, `epochs_per_chunk`, `prefetch_chunks` in docs/training.md and replace with `reservoir_size` / `reservoir_turnover` descriptions.

**Step 3: Verify the full build**

Run: `cargo build -p cfvnet --release`
Expected: Compiles cleanly

**Step 4: Run entire test suite**

Run: `cargo test`
Expected: ALL PASS, completes in < 1 minute

**Step 5: Commit**

```bash
git add sample_configurations/river_cfvnet.yaml docs/training.md
git commit -m "docs: update cfvnet config and training docs for GPU reservoir pipeline"
```

---

## Implementation Notes for the Developer

### burn API caveats

1. **`Tensor::random` for integers**: burn may not support `Tensor::<B, 1, Int>::random(...)` directly. If not, generate random indices on CPU with `rng.gen_range(0..capacity)` and upload as `Tensor::from_data(TensorData::new(indices_vec, [batch_size]), device)`. This is a tiny tensor (batch_size i64s ≈ 16KB) — negligible PCIe cost.

2. **`select_assign`**: This is the scatter operation. Check if burn 0.16 supports `tensor.select_assign(dim, indices, source)`. If not, the fallback is to read the full reservoir to CPU, scatter on CPU, and re-upload — acceptable for the refresh path since it's off the training hot path. But try `select_assign` first.

3. **`Tensor::select`**: This IS supported in burn — it's `tensor.select(dim, indices)` and returns a new tensor with rows picked by `indices`. This is used for `sample_batch`.

4. **`clone()` on tensors**: In burn, `Tensor::clone()` is cheap (reference-counted). The `select`, `select_assign`, and `narrow` operations may or may not copy. This is fine — the framework handles memory.

### Thread safety

The refresh thread sends `PreEncoded` (plain `Vec<f32>`) through a `sync_channel`. The main thread calls `scatter_refresh` between training steps. No tensor sharing across threads — all GPU ops happen on the main thread. This is correct for burn which is not thread-safe for GPU tensors.

### Reservoir smaller than dataset

When `reservoir_size < total_records` (the normal case), the reservoir holds a sliding window. The refresh thread continuously reads the next `refresh_per_step` records and the main thread scatters them into random positions. Over one epoch, the entire reservoir rotates once (with `turnover=1.0`).

### Reservoir larger than or equal to dataset

When `reservoir_size >= total_records`, the entire dataset fits in the reservoir. The refresh thread still runs (reading from the start when files exhaust), but since it writes to random positions of already-seen data, it effectively just re-shuffles. This is correct behavior — the training loop samples randomly from the full dataset each step.
