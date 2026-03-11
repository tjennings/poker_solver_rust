# CFVnet Training Pipeline Completion

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Complete the cfvnet train/evaluate/compare pipeline so models can be saved, loaded, resumed, and evaluated.

**Architecture:** Wire burn's built-in `Module::save_file`/`load_file` with `NamedMpkGzFileRecorder` into the training loop, add per-epoch indicatif progress bars with train/val loss, implement evaluate and compare CLI commands using existing metrics and comparison infrastructure.

**Tech Stack:** burn 0.16 (`Module` derive, `NamedMpkGzFileRecorder`), indicatif, existing `eval::metrics` and `eval::compare` modules.

---

### Task 1: Rename config field and update YAML

**Files:**
- Modify: `crates/cfvnet/src/config.rs:121` (field name)
- Modify: `crates/cfvnet/src/config.rs:136` (Default impl)
- Modify: `crates/cfvnet/src/config.rs:168-170` (default fn)
- Modify: `crates/cfvnet/src/config.rs:235` (test)
- Modify: `crates/cfvnet/src/model/training.rs:24` (TrainConfig field)
- Modify: `crates/cfvnet/src/model/training.rs:204` (test)
- Modify: `crates/cfvnet/src/main.rs:131` (mapping)
- Modify: `sample_configurations/river_cfvnet.yaml:28`

**Step 1: Rename `checkpoint_every_n_batches` to `checkpoint_every_n_epochs` everywhere**

In `config.rs`, change the field and serde attribute:

```rust
// config.rs line 120-121
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_every_n_epochs: usize,
```

In `Default for TrainingConfig`:
```rust
// config.rs line 136
            checkpoint_every_n_epochs: 1,
```

In `default_checkpoint_interval`:
```rust
// config.rs line 168-170
fn default_checkpoint_interval() -> usize {
    1
}
```

In `TrainConfig` in `training.rs`:
```rust
// training.rs line 24
    pub checkpoint_every_n_epochs: usize,
```

In `cmd_train` in `main.rs`:
```rust
// main.rs line 131
        checkpoint_every_n_epochs: cfg.training.checkpoint_every_n_epochs,
```

In `training.rs` test:
```rust
// training.rs line 204
            checkpoint_every_n_epochs: 0,
```

In `river_cfvnet.yaml`:
```yaml
  checkpoint_every_n_epochs: 1
```

**Step 2: Run tests to verify rename**

Run: `cargo test -p cfvnet`
Expected: all tests pass

**Step 3: Commit**

```bash
git add crates/cfvnet/src/config.rs crates/cfvnet/src/model/training.rs crates/cfvnet/src/main.rs sample_configurations/river_cfvnet.yaml
git commit -m "refactor(cfvnet): rename checkpoint_every_n_batches to checkpoint_every_n_epochs"
```

---

### Task 2: Add model save/load and progress bar to training loop

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs`

This is the core task. The training loop currently ignores `_output_dir` and has no progress output.

**Step 1: Write the failing test**

Add to the `tests` module in `training.rs`:

```rust
    #[test]
    fn train_saves_model_to_output_dir() {
        let file = write_test_data(16);
        let dataset = CfvDataset::from_file(file.path()).unwrap();
        let dir = tempfile::tempdir().unwrap();

        let device = Default::default();
        let config = TrainConfig {
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
        };

        let result = train::<B>(&device, &dataset, &config, Some(dir.path()));
        assert!(result.final_train_loss < 1.0);

        // Final model should be saved
        assert!(
            dir.path().join("model.mpk.gz").exists(),
            "model.mpk.gz should exist after training"
        );
    }

    #[test]
    fn train_saves_checkpoints() {
        let file = write_test_data(16);
        let dataset = CfvDataset::from_file(file.path()).unwrap();
        let dir = tempfile::tempdir().unwrap();

        let device = Default::default();
        let config = TrainConfig {
            hidden_layers: 2,
            hidden_size: 64,
            batch_size: 16,
            epochs: 4,
            learning_rate: 0.001,
            lr_min: 0.001,
            huber_delta: 1.0,
            aux_loss_weight: 0.0,
            validation_split: 0.0,
            checkpoint_every_n_epochs: 2,
        };

        train::<B>(&device, &dataset, &config, Some(dir.path()));

        // Checkpoint at epoch 2 should exist
        assert!(
            dir.path().join("checkpoint_epoch2.mpk.gz").exists(),
            "checkpoint_epoch2.mpk.gz should exist"
        );
        // Checkpoint at epoch 4 should exist
        assert!(
            dir.path().join("checkpoint_epoch4.mpk.gz").exists(),
            "checkpoint_epoch4.mpk.gz should exist"
        );
        // Final model too
        assert!(dir.path().join("model.mpk.gz").exists());
    }

    #[test]
    fn train_resumes_from_checkpoint() {
        let file = write_test_data(16);
        let dataset = CfvDataset::from_file(file.path()).unwrap();
        let dir = tempfile::tempdir().unwrap();

        let device = Default::default();
        let config = TrainConfig {
            hidden_layers: 2,
            hidden_size: 64,
            batch_size: 16,
            epochs: 5,
            learning_rate: 0.001,
            lr_min: 0.001,
            huber_delta: 1.0,
            aux_loss_weight: 0.0,
            validation_split: 0.0,
            checkpoint_every_n_epochs: 0,
        };

        // First training run
        let r1 = train::<B>(&device, &dataset, &config, Some(dir.path()));

        // Second training run — should resume from saved model
        let r2 = train::<B>(&device, &dataset, &config, Some(dir.path()));

        // Both should complete successfully (loss values may differ due to continued training)
        assert!(r1.final_train_loss < 10.0);
        assert!(r2.final_train_loss < 10.0);
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p cfvnet train_saves_model -- --nocapture`
Expected: FAIL — `model.mpk.gz` doesn't exist because `_output_dir` is unused.

**Step 3: Implement model save/load, progress bar, and validation**

Replace the entire `train` function in `training.rs` with:

```rust
use burn::record::{NamedMpkGzFileRecorder, FullPrecisionSettings};
use indicatif::{ProgressBar, ProgressStyle};

/// Train a `CfvNet` on the given dataset using a custom Adam training loop.
///
/// Saves the final model to `output_dir/model.mpk.gz`. If that file already
/// exists, resumes training from the saved weights. Periodic checkpoints are
/// saved every `config.checkpoint_every_n_epochs` epochs.
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    dataset: &CfvDataset,
    config: &TrainConfig,
    output_dir: Option<&std::path::Path>,
) -> TrainResult {
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

    // Build or resume model
    let mut model = CfvNet::<B>::new(device, config.hidden_layers, config.hidden_size);
    if let Some(dir) = output_dir {
        let model_path = dir.join("model");
        if model_path.with_extension("mpk.gz").exists() {
            eprintln!("Resuming from checkpoint: {}", model_path.display());
            model = model
                .load_file(&model_path, &recorder, device)
                .unwrap_or_else(|e| {
                    eprintln!("failed to load checkpoint: {e}");
                    std::process::exit(1);
                });
        }
    }

    let mut optim = AdamConfig::new().init::<B, CfvNet<B>>();

    // Split dataset into train and validation
    let n = dataset.len();
    let val_count = (n as f64 * config.validation_split) as usize;
    let train_count = n - val_count;

    let mut all_indices: Vec<usize> = (0..n).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    all_indices.shuffle(&mut rng);

    let train_indices: Vec<usize> = all_indices[..train_count].to_vec();
    let val_indices: Vec<usize> = all_indices[train_count..].to_vec();

    if val_count == 0 && config.validation_split > 0.0 {
        eprintln!("Warning: validation_split too small, no validation samples");
    }

    let mut final_loss = f32::MAX;

    for epoch in 1..=config.epochs {
        let mut epoch_indices = train_indices.clone();
        epoch_indices.shuffle(&mut rng);

        let num_batches = (train_count + config.batch_size - 1) / config.batch_size;

        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Epoch {msg} {wide_bar} {pos}/{len} [{elapsed}] ETA {eta}")
                .expect("valid template"),
        );
        pb.set_message(format!("{epoch}/{}", config.epochs));

        let mut epoch_loss_sum = 0.0_f64;
        let mut epoch_loss_count = 0_u64;

        for batch_start in (0..train_count).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(train_count);
            let batch_indices = &epoch_indices[batch_start..batch_end];

            let items: Vec<CfvItem> = batch_indices
                .iter()
                .filter_map(|&idx| dataset.get(idx))
                .collect();

            if items.is_empty() {
                pb.inc(1);
                continue;
            }

            let batch = collate_batch::<B>(&items, device);
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

            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            epoch_loss_sum += f64::from(loss_val);
            epoch_loss_count += 1;
            final_loss = loss_val;

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads_params);

            pb.inc(1);
        }

        // Compute validation loss
        let train_avg = if epoch_loss_count > 0 {
            epoch_loss_sum / epoch_loss_count as f64
        } else {
            0.0
        };

        let val_msg = if !val_indices.is_empty() {
            let val_loss = compute_val_loss::<B>(&model, dataset, &val_indices, config, device);
            format!("train={train_avg:.6} val={val_loss:.6}")
        } else {
            format!("train={train_avg:.6}")
        };

        pb.finish_with_message(format!("{epoch}/{} {val_msg}", config.epochs));

        // Checkpoint
        if let Some(dir) = output_dir {
            if config.checkpoint_every_n_epochs > 0
                && epoch % config.checkpoint_every_n_epochs == 0
            {
                let path = dir.join(format!("checkpoint_epoch{epoch}"));
                if let Err(e) = model.save_file(&path, &recorder) {
                    eprintln!("failed to save checkpoint: {e}");
                    std::process::exit(1);
                }
            }
        }
    }

    // Save final model
    if let Some(dir) = output_dir {
        let path = dir.join("model");
        if let Err(e) = model.save_file(&path, &recorder) {
            eprintln!("failed to save model: {e}");
            std::process::exit(1);
        }
    }

    TrainResult {
        final_train_loss: final_loss,
    }
}

/// Compute average loss over validation set (no gradient tracking).
fn compute_val_loss<B: AutodiffBackend>(
    model: &CfvNet<B>,
    dataset: &CfvDataset,
    val_indices: &[usize],
    config: &TrainConfig,
    device: &B::Device,
) -> f64 {
    let mut loss_sum = 0.0_f64;
    let mut count = 0_u64;

    for batch_start in (0..val_indices.len()).step_by(config.batch_size) {
        let batch_end = (batch_start + config.batch_size).min(val_indices.len());
        let items: Vec<CfvItem> = val_indices[batch_start..batch_end]
            .iter()
            .filter_map(|&idx| dataset.get(idx))
            .collect();
        if items.is_empty() {
            continue;
        }
        let batch = collate_batch::<B>(&items, device);
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
        let val: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];
        loss_sum += f64::from(val);
        count += 1;
    }

    if count > 0 {
        loss_sum / count as f64
    } else {
        0.0
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p cfvnet`
Expected: all tests pass (including the 3 new ones)

**Step 5: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "feat(cfvnet): model save/load, checkpoints, progress bar, validation loss"
```

---

### Task 3: Implement evaluate command

**Files:**
- Modify: `crates/cfvnet/src/main.rs:85-91` (evaluate match arm)

**Step 1: Implement `cmd_evaluate`**

Replace the `Evaluate` match arm and add the handler function:

```rust
        Commands::Evaluate { model, data } => cmd_evaluate(model, data),
```

Add the function:

```rust
fn cmd_evaluate(model_path: PathBuf, data_path: PathBuf) {
    use burn::backend::NdArray;
    use burn::record::{NamedMpkGzFileRecorder, FullPrecisionSettings};
    use cfvnet::eval::metrics::compute_prediction_metrics;
    use cfvnet::model::network::{CfvNet, OUTPUT_SIZE};

    type B = NdArray;
    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

    // Load model
    // CfvNet::new creates a fresh model; load_file overwrites the weights.
    // We need to know hidden_layers and hidden_size — read from a sidecar config
    // or use defaults. For now, use defaults (7 layers, 500 hidden).
    let model = CfvNet::<B>::new(&device, 7, 500);
    let model = model.load_file(&model_path, &recorder, &device).unwrap_or_else(|e| {
        eprintln!("failed to load model from {}: {e}", model_path.display());
        std::process::exit(1);
    });

    // Load dataset
    let dataset = cfvnet::model::dataset::CfvDataset::from_file(&data_path).unwrap_or_else(|e| {
        eprintln!("failed to load dataset: {e}");
        std::process::exit(1);
    });
    println!("Evaluating on {} records", dataset.len());

    // Run inference and compute metrics
    let mut total_mae = 0.0_f64;
    let mut total_max = 0.0_f64;
    let mut total_mbb = 0.0_f64;
    let mut count = 0_u64;

    use burn::tensor::{Tensor, TensorData};
    use cfvnet::model::network::INPUT_SIZE;

    for i in 0..dataset.len() {
        let item = match dataset.get(i) {
            Some(item) => item,
            None => continue,
        };

        let input = Tensor::<B, 2>::from_data(
            TensorData::new(item.input.clone(), [1, INPUT_SIZE]),
            &device,
        );
        let pred = model.forward(input);
        let pred_data: Vec<f32> = pred.into_data().to_vec().unwrap();

        let mask: Vec<bool> = item.mask.iter().map(|&v| v > 0.5).collect();
        // Reconstruct pot from the normalized input: pot = input[2657] * 400.0
        let pot = item.input[2657] * 400.0;

        let metrics = compute_prediction_metrics(&pred_data, &item.target, &mask, pot);
        total_mae += metrics.mae;
        total_max += metrics.max_error;
        total_mbb += metrics.mbb_error;
        count += 1;
    }

    if count == 0 {
        eprintln!("No valid records to evaluate");
        std::process::exit(1);
    }

    let n = count as f64;
    println!();
    println!("Results ({count} samples):");
    println!("  MAE (pot-relative): {:.6}", total_mae / n);
    println!("  Max error:          {:.6}", total_max / n);
    println!("  mBB error:          {:.2}", total_mbb / n);
}
```

**Step 2: Run to verify it compiles**

Run: `cargo check -p cfvnet`
Expected: compiles cleanly

**Step 3: Commit**

```bash
git add crates/cfvnet/src/main.rs
git commit -m "feat(cfvnet): implement evaluate command"
```

---

### Task 4: Implement compare command

**Files:**
- Modify: `crates/cfvnet/src/main.rs:92-100` (compare match arm)

**Step 1: Implement `cmd_compare`**

Replace the `Compare` match arm:

```rust
        Commands::Compare {
            model,
            num_spots,
            threads,
            config,
        } => cmd_compare(model, num_spots, threads, config),
```

Add the function:

```rust
fn cmd_compare(
    model_path: PathBuf,
    num_spots: usize,
    threads: Option<usize>,
    config_path: Option<PathBuf>,
) {
    use burn::backend::NdArray;
    use burn::record::{NamedMpkGzFileRecorder, FullPrecisionSettings};
    use burn::tensor::{Tensor, TensorData};
    use cfvnet::model::network::{CfvNet, INPUT_SIZE};
    use cfvnet::model::dataset::encode_record_for_inference;

    type B = NdArray;
    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

    let model = CfvNet::<B>::new(&device, 7, 500);
    let model = model.load_file(&model_path, &recorder, &device).unwrap_or_else(|e| {
        eprintln!("failed to load model from {}: {e}", model_path.display());
        std::process::exit(1);
    });

    // Load game config if provided, otherwise use defaults
    let game_config = if let Some(ref path) = config_path {
        let yaml = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("failed to read config: {e}");
            std::process::exit(1);
        });
        let cfg: cfvnet::config::CfvnetConfig = serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
            eprintln!("failed to parse config: {e}");
            std::process::exit(1);
        });
        cfg.game
    } else {
        cfvnet::config::GameConfig::default()
    };

    if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .ok(); // ignore if already set
    }

    println!("Comparing model vs exact solver on {num_spots} spots...");

    let summary = cfvnet::eval::compare::run_comparison(&game_config, num_spots, 12345, |sit, _result| {
        // Build input from situation for OOP (player=0)
        let input_data = encode_situation_for_inference(sit, 0);
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(input_data, [1, INPUT_SIZE]),
            &device,
        );
        let pred = model.forward(input);
        pred.into_data().to_vec::<f32>().unwrap()
    })
    .unwrap_or_else(|e| {
        eprintln!("comparison failed: {e}");
        std::process::exit(1);
    });

    println!();
    println!("Results ({} spots):", summary.num_spots);
    println!("  Mean MAE:       {:.6}", summary.mean_mae);
    println!("  Mean max error: {:.6}", summary.mean_max_error);
    println!("  Mean mBB:       {:.2}", summary.mean_mbb);
    println!("  Worst MAE:      {:.6}", summary.worst_mae);
    println!("  Worst mBB:      {:.2}", summary.worst_mbb);
}
```

**Step 2: Add `encode_situation_for_inference` helper**

This needs a public function to encode a `Situation` into the same 2660-float input vector that `encode_record` produces. Add to `crates/cfvnet/src/model/dataset.rs`:

```rust
use crate::datagen::sampler::Situation;
use crate::model::network::OUTPUT_SIZE;

/// Encode a `Situation` into a model input vector for inference.
///
/// `player` is 0 for OOP, 1 for IP.
pub fn encode_situation_for_inference(sit: &Situation, player: u8) -> Vec<f32> {
    let mut input = Vec::with_capacity(INPUT_SIZE);

    // OOP range (1326 floats)
    for i in 0..OUTPUT_SIZE {
        input.push(sit.ranges[0][i] as f32);
    }
    // IP range (1326 floats)
    for i in 0..OUTPUT_SIZE {
        input.push(sit.ranges[1][i] as f32);
    }
    // Board cards (5 floats, normalized)
    for &card in &sit.board {
        input.push(f32::from(card) / 51.0);
    }
    // Pot normalized
    input.push(sit.pot as f32 / 400.0);
    // Effective stack normalized
    input.push(sit.effective_stack as f32 / 400.0);
    // Player indicator
    input.push(f32::from(player));

    debug_assert_eq!(input.len(), INPUT_SIZE);
    input
}
```

Also need to make the `Situation` fields `pub` if they aren't already. Check `crates/cfvnet/src/datagen/sampler.rs` for the `Situation` struct visibility.

**Step 3: Run to verify it compiles**

Run: `cargo check -p cfvnet`
Expected: compiles cleanly

**Step 4: Commit**

```bash
git add crates/cfvnet/src/main.rs crates/cfvnet/src/model/dataset.rs
git commit -m "feat(cfvnet): implement compare command"
```

---

### Task 5: End-to-end smoke test

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs` (tests section)

**Step 1: Write integration test that runs generate + train + evaluate in sequence**

Add to training.rs tests:

```rust
    #[test]
    fn train_with_validation_reports_val_loss() {
        let file = write_test_data(20);
        let dataset = CfvDataset::from_file(file.path()).unwrap();
        let dir = tempfile::tempdir().unwrap();

        let device = Default::default();
        let config = TrainConfig {
            hidden_layers: 2,
            hidden_size: 64,
            batch_size: 8,
            epochs: 3,
            learning_rate: 0.001,
            lr_min: 0.001,
            huber_delta: 1.0,
            aux_loss_weight: 0.0,
            validation_split: 0.2, // 4 val samples out of 20
            checkpoint_every_n_epochs: 0,
        };

        let result = train::<B>(&device, &dataset, &config, Some(dir.path()));
        assert!(result.final_train_loss < 10.0);
        assert!(dir.path().join("model.mpk.gz").exists());
    }
```

**Step 2: Run full test suite**

Run: `cargo test -p cfvnet`
Expected: all tests pass

**Step 3: Run full workspace tests and verify < 1 minute**

Run: `cargo test --workspace`
Expected: all pass, total time < 60s

**Step 4: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "test(cfvnet): add validation split and e2e smoke tests"
```
