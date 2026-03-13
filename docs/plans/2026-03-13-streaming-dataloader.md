# Streaming Dataloader Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace the GPU-resident reservoir with a channel-based streaming dataloader that reads sequentially from disk, shuffles within chunks, and sends pre-encoded batches through a bounded channel.

**Architecture:** A background thread reads `shuffle_buffer_size` records at a time from disk via `StreamingReader`, shuffles them, splits into `batch_size`-sized chunks, encodes each with `PreEncoded::from_records()`, and sends through `mpsc::sync_channel(prefetch_depth)`. The training loop just calls `recv()` and trains. On EOF, the reader resets and loops. The thread exits when the receiver is dropped.

**Tech Stack:** Rust, burn (0.16), mpsc::sync_channel, rayon, rand (ChaCha8Rng)

---

### Task 1: Update config — replace reservoir params with dataloader params

**Files:**
- Modify: `crates/cfvnet/src/config.rs:135-217` (TrainingConfig, defaults, serde)

**Step 1: Update TrainingConfig struct and defaults**

Replace the two reservoir fields with two new dataloader fields:

```rust
// In TrainingConfig struct (line 135), REPLACE:
//   pub reservoir_size: usize,
//   pub reservoir_turnover: f64,
// WITH:
    #[serde(default = "default_shuffle_buffer_size")]
    pub shuffle_buffer_size: usize,
    #[serde(default = "default_prefetch_depth")]
    pub prefetch_depth: usize,
```

In the `Default` impl (line 163), replace:
```rust
// REPLACE:
//   reservoir_size: 1_500_000,
//   reservoir_turnover: 1.0,
// WITH:
    shuffle_buffer_size: 262_144,
    prefetch_depth: 4,
```

Replace the default functions (lines 212-217):
```rust
// REPLACE default_reservoir_size and default_reservoir_turnover WITH:
fn default_shuffle_buffer_size() -> usize {
    262_144
}
fn default_prefetch_depth() -> usize {
    4
}
```

**Step 2: Update config tests**

In the test module at bottom of config.rs:

- Replace `parse_config_with_reservoir_params` test to parse the new fields:
```rust
#[test]
fn parse_config_with_dataloader_params() {
    let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
training:
  shuffle_buffer_size: 500000
  prefetch_depth: 8
"#;
    let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(config.training.shuffle_buffer_size, 500_000);
    assert_eq!(config.training.prefetch_depth, 8);
}
```

- Replace `config_defaults_for_reservoir` test:
```rust
#[test]
fn config_defaults_for_dataloader() {
    let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
"#;
    let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(config.training.shuffle_buffer_size, 262_144);
    assert_eq!(config.training.prefetch_depth, 4);
}
```

**Step 3: Run tests to verify**

Run: `cargo test -p cfvnet config`
Expected: All config tests pass, including the two new ones.

**Step 4: Commit**

```bash
git add crates/cfvnet/src/config.rs
git commit -m "feat(cfvnet): replace reservoir config with shuffle_buffer_size and prefetch_depth"
```

---

### Task 2: Update TrainConfig struct in training.rs and all call sites

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs:23-36` (TrainConfig struct)
- Modify: `crates/cfvnet/src/main.rs:223-236` (cmd_train TrainConfig construction)
- Modify: `crates/cfvnet/tests/integration_test.rs:45-58` (integration test TrainConfig)

**Step 1: Update TrainConfig struct**

In `training.rs` lines 23-36, replace `reservoir_size` and `reservoir_turnover` with:
```rust
pub struct TrainConfig {
    pub hidden_layers: usize,
    pub hidden_size: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub lr_min: f64,
    pub huber_delta: f64,
    pub aux_loss_weight: f64,
    pub validation_split: f64,
    pub checkpoint_every_n_epochs: usize,
    pub shuffle_buffer_size: usize,
    pub prefetch_depth: usize,
}
```

**Step 2: Update cmd_train in main.rs**

In `main.rs` lines 223-236, replace the TrainConfig construction:
```rust
    let train_config = cfvnet::model::training::TrainConfig {
        hidden_layers: cfg.training.hidden_layers,
        hidden_size: cfg.training.hidden_size,
        batch_size: cfg.training.batch_size,
        epochs: cfg.training.epochs,
        learning_rate: cfg.training.learning_rate,
        lr_min: cfg.training.lr_min,
        huber_delta: cfg.training.huber_delta,
        aux_loss_weight: cfg.training.aux_loss_weight,
        validation_split: cfg.training.validation_split,
        checkpoint_every_n_epochs: cfg.training.checkpoint_every_n_epochs,
        shuffle_buffer_size: cfg.training.shuffle_buffer_size,
        prefetch_depth: cfg.training.prefetch_depth,
    };
```

**Step 3: Update integration test**

In `crates/cfvnet/tests/integration_test.rs` lines 45-58:
```rust
    let train_config = cfvnet::model::training::TrainConfig {
        hidden_layers: 2,
        hidden_size: 32,
        batch_size: 8,
        epochs: 10,
        learning_rate: 0.001,
        lr_min: 0.001,
        huber_delta: 1.0,
        aux_loss_weight: 0.0,
        validation_split: 0.0,
        checkpoint_every_n_epochs: 0,
        shuffle_buffer_size: 100,
        prefetch_depth: 2,
    };
```

**Step 4: Update all test helpers in training.rs**

In `training.rs`, the `default_test_config()` function (line ~699):
```rust
    fn default_test_config() -> TrainConfig {
        TrainConfig {
            hidden_layers: 2,
            hidden_size: 64,
            batch_size: 16,
            epochs: 2,
            learning_rate: 0.001,
            lr_min: 0.001,
            huber_delta: 1.0,
            aux_loss_weight: 0.0,
            validation_split: 0.0,
            checkpoint_every_n_epochs: 0,
            shuffle_buffer_size: 100,
            prefetch_depth: 2,
        }
    }
```

**Step 5: Verify it compiles (tests will fail until training loop is updated)**

Run: `cargo check -p cfvnet`
Expected: Compiles (training.rs will have unused field warnings, but no errors — the struct is used but the fields aren't consumed yet).

**Step 6: Commit**

```bash
git add crates/cfvnet/src/model/training.rs crates/cfvnet/src/main.rs crates/cfvnet/tests/integration_test.rs
git commit -m "refactor(cfvnet): update TrainConfig to use shuffle_buffer_size and prefetch_depth"
```

---

### Task 3: Replace GpuReservoir and spawn_refresh_thread with streaming dataloader

This is the core change. Remove `GpuReservoir`, `SampledBatch`, and `spawn_refresh_thread`. Replace with `spawn_dataloader_thread` that sends complete `PreEncoded` batches.

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs:123-235` (remove GpuReservoir, SampledBatch)
- Modify: `crates/cfvnet/src/model/training.rs:468-516` (replace spawn_refresh_thread)

**Step 1: Remove GpuReservoir and SampledBatch**

Delete lines 123-235 (the `GpuReservoir` struct, `SampledBatch` struct, and all their `impl` blocks).

**Step 2: Replace spawn_refresh_thread with spawn_dataloader_thread**

Replace the `spawn_refresh_thread` function (lines 468-516) with:

```rust
/// Spawn the background dataloader thread that reads records from disk,
/// shuffles within chunks, encodes into batches, and sends them through
/// a bounded channel.
///
/// The thread loops infinitely over the dataset (resetting the reader on EOF)
/// until the channel receiver is dropped.
fn spawn_dataloader_thread(
    files: &[PathBuf],
    batch_size: usize,
    shuffle_buffer_size: usize,
    prefetch_depth: usize,
    val_count: usize,
    board_cards: usize,
) -> (mpsc::Receiver<PreEncoded>, std::thread::JoinHandle<()>) {
    let (tx, rx) = mpsc::sync_channel::<PreEncoded>(prefetch_depth);
    let files = files.to_vec();
    let handle = std::thread::spawn(move || {
        let mut reader = StreamingReader::new(files);
        let mut rng = ChaCha8Rng::seed_from_u64(0xDA7A);

        // Skip past validation records.
        if val_count > 0 {
            let _ = reader.read_chunk(val_count);
        }

        loop {
            // Read a chunk, shuffle it, split into batches.
            let mut records = reader.read_chunk(shuffle_buffer_size);

            if records.is_empty() {
                // EOF — reset and start a new pass.
                reader.reset();
                if val_count > 0 {
                    let _ = reader.read_chunk(val_count);
                }
                continue;
            }

            // Shuffle within the chunk.
            use rand::seq::SliceRandom;
            records.shuffle(&mut rng);

            // Split into batch-sized pieces and send.
            for chunk in records.chunks(batch_size) {
                let encoded = PreEncoded::from_records(chunk, board_cards);
                if tx.send(encoded).is_err() {
                    return; // Receiver dropped, training done.
                }
            }
        }
    });
    (rx, handle)
}
```

**Step 3: Verify it compiles**

Run: `cargo check -p cfvnet`
Expected: Warnings about unused imports/code in `train()` function (which still references removed types), but the new function should compile. If `train()` errors block compilation, proceed to Task 4 immediately.

**Step 4: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "feat(cfvnet): add spawn_dataloader_thread, remove GpuReservoir"
```

---

### Task 4: Rewrite the training loop to consume from the dataloader channel

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs:518-686` (the `train()` function)

**Step 1: Rewrite train() function**

Replace the body of `train()` (everything after the function signature, lines 530-686) with:

```rust
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    data_path: &Path,
    board_cards: usize,
    config: &TrainConfig,
    output_dir: Option<&std::path::Path>,
) -> TrainResult {
    let in_size = input_size(board_cards);
    let model = CfvNet::<B>::new(device, config.hidden_layers, config.hidden_size, in_size);
    let mut model = load_or_create_model(model, output_dir, device);

    let mut optim = AdamConfig::new().init::<B, CfvNet<B>>();

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

    let val_count = (total_records as f64 * config.validation_split) as usize;
    let val_encoded = load_validation_set(&files, val_count, board_cards);

    let train_records = total_records - val_count;
    let batch_size = config.batch_size;
    let steps_per_epoch = train_records.div_ceil(batch_size);
    let total_steps = steps_per_epoch * config.epochs;

    eprintln!(
        "Training: {} epochs x {} steps/epoch = {} total steps (shuffle_buffer={})",
        config.epochs, steps_per_epoch, total_steps, config.shuffle_buffer_size
    );

    let (data_rx, loader_thread) = spawn_dataloader_thread(
        &files,
        batch_size,
        config.shuffle_buffer_size,
        config.prefetch_depth,
        val_count,
        board_cards,
    );

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
            // Receive next pre-encoded batch from dataloader (blocks until ready).
            let encoded = match data_rx.recv() {
                Ok(batch) => batch,
                Err(_) => {
                    eprintln!("Warning: dataloader channel closed unexpectedly at step {global_step}");
                    break;
                }
            };

            // Create tensors on device and lift to autodiff.
            let tensors = encoded.into_device_tensors::<B::InnerBackend>(device);
            let input = Tensor::<B, 2>::from_inner(tensors.input);
            let target = Tensor::<B, 2>::from_inner(tensors.target);
            let mask = Tensor::<B, 2>::from_inner(tensors.mask);
            let range = Tensor::<B, 2>::from_inner(tensors.range);
            let game_value = Tensor::<B, 1>::from_inner(tensors.game_value);

            let pred = model.forward(input);
            let loss = cfvnet_loss(
                pred,
                target,
                mask,
                range,
                game_value,
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

    // Drop the receiver to signal dataloader thread to exit, then join.
    drop(data_rx);
    let _ = loader_thread.join();

    // Save final model.
    if let Some(dir) = output_dir {
        save_model(&model, dir, "model");
    }

    TrainResult { final_train_loss: final_loss }
}
```

**Step 2: Remove unused imports**

Remove the `use rand::Rng;` import (line 8) — it was only needed for `scatter_refresh`'s `rng.gen_range()`. Keep `SeedableRng` and `ChaCha8Rng` — but actually check: `ChaCha8Rng` and `SeedableRng` are now only used in the dataloader thread (which constructs its own rng). The `train()` function no longer needs its own `rng`. Remove the `let mut rng = ChaCha8Rng::seed_from_u64(42);` line from `train()`.

Check remaining imports — `rand::seq::index` was only used in `sample_batch`, which is gone. The `rand::Rng` import is no longer needed. Keep `rand::SeedableRng` and `ChaCha8Rng` since they're used in `spawn_dataloader_thread`.

**Step 3: Verify compilation**

Run: `cargo check -p cfvnet`
Expected: Compiles with no errors. May have warnings about unused things in tests (which still reference old patterns) — that's fine, tests are updated in the next task.

**Step 4: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "feat(cfvnet): rewrite training loop to consume from streaming dataloader channel"
```

---

### Task 5: Update tests in training.rs

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs:688+` (test module)

**Step 1: Remove reservoir-specific tests**

Delete these tests entirely (they test removed code):
- `reservoir_fill_and_sample`
- `reservoir_scatter_refresh`
- `reservoir_sample_converts_to_autodiff`

**Step 2: Update training_with_refresh test**

Rename to `training_with_streaming` and update config:
```rust
    #[test]
    fn training_with_streaming() {
        // Use more data than shuffle_buffer_size to exercise multi-chunk reads.
        let file = write_test_data(64);
        let device = Default::default();
        let config = TrainConfig {
            batch_size: 8,
            epochs: 5,
            shuffle_buffer_size: 16, // smaller than data -> multiple chunks
            prefetch_depth: 2,
            ..default_test_config()
        };
        let result = train::<B>(&device, file.path(), 5, &config, None);
        assert!(
            result.final_train_loss < 10.0,
            "training with streaming should produce reasonable loss, got {}",
            result.final_train_loss
        );
    }
```

**Step 3: Update overfit_single_batch test**

The old test set `reservoir_turnover: 0.0` to disable refresh. With the streaming dataloader, the equivalent is just using a small dataset (which it already does). Update:
```rust
    #[test]
    fn overfit_single_batch() {
        let file = write_test_data(16);

        let device = Default::default();
        let config = TrainConfig {
            epochs: 200,
            ..default_test_config()
        };

        let result = train::<B>(&device, file.path(), 5, &config, None);
        assert!(
            result.final_train_loss < 0.05,
            "should overfit small data, got loss {}",
            result.final_train_loss
        );
    }
```

**Step 4: Update reservoir_training_reduces_loss test**

Rename and simplify:
```rust
    #[test]
    fn training_reduces_loss() {
        let file = write_test_data(16);
        let device = Default::default();
        let config = TrainConfig {
            epochs: 200,
            ..default_test_config()
        };
        let result = train::<B>(&device, file.path(), 5, &config, None);
        assert!(
            result.final_train_loss < 0.05,
            "training should reduce loss, got {}",
            result.final_train_loss
        );
    }
```

**Step 5: Run all tests**

Run: `cargo test -p cfvnet`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "test(cfvnet): update training tests for streaming dataloader"
```

---

### Task 6: Update sample config and documentation

**Files:**
- Modify: `sample_configurations/river_cfvnet.yaml` (replace reservoir params)
- Modify: `docs/training.md` (update training config table and description)

**Step 1: Update sample config**

In `sample_configurations/river_cfvnet.yaml`, replace:
```yaml
  reservoir_size: 400000
  reservoir_turnover: 0.05
```
with:
```yaml
  shuffle_buffer_size: 262144
  prefetch_depth: 4
```

**Step 2: Update docs/training.md**

Replace the reservoir config table rows (lines ~162-163) with:
```
| `shuffle_buffer_size` | 262144 | Number of records read and shuffled per chunk before splitting into batches |
| `prefetch_depth` | 4 | Number of pre-encoded batches buffered in the channel ahead of the training loop |
```

Replace the reservoir description paragraph (line ~165) with:
```
The training loop uses a **streaming dataloader**: a background thread reads `shuffle_buffer_size` records sequentially from disk, shuffles them in-place, splits into `batch_size`-sized chunks, encodes each into tensors, and sends them through a bounded channel (capacity = `prefetch_depth`). The training loop simply receives the next batch. This provides uniform dataset coverage (every record seen once per epoch) with I/O overlapping compute.
```

**Step 3: Run full test suite**

Run: `cargo test`
Expected: All tests pass (including integration test).

**Step 4: Commit**

```bash
git add sample_configurations/river_cfvnet.yaml docs/training.md
git commit -m "docs: update config and training docs for streaming dataloader"
```
