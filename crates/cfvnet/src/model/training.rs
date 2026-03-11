use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};

use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::model::dataset::{CfvDataset, CfvItem};
use crate::model::loss::cfvnet_loss;
use crate::model::network::{CfvNet, INPUT_SIZE, OUTPUT_SIZE};

/// Configuration for the CFVnet training loop.
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
}

/// Result returned after training completes.
pub struct TrainResult {
    pub final_train_loss: f32,
}

/// All tensors needed for one training batch.
struct Batch<B: AutodiffBackend> {
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    range: Tensor<B, 2>,
    game_value: Tensor<B, 1>,
}

/// Collate a batch of `CfvItem`s into stacked tensors for the forward pass.
fn collate_batch<B: AutodiffBackend>(
    items: &[CfvItem],
    device: &B::Device,
) -> Batch<B> {
    let bs = items.len();

    let mut input_data = Vec::with_capacity(bs * INPUT_SIZE);
    let mut target_data = Vec::with_capacity(bs * OUTPUT_SIZE);
    let mut mask_data = Vec::with_capacity(bs * OUTPUT_SIZE);
    let mut range_data = Vec::with_capacity(bs * OUTPUT_SIZE);
    let mut gv_data = Vec::with_capacity(bs);

    for item in items {
        input_data.extend_from_slice(&item.input);
        target_data.extend_from_slice(&item.target);
        mask_data.extend_from_slice(&item.mask);
        range_data.extend_from_slice(&item.range);
        gv_data.push(item.game_value);
    }

    let input = Tensor::from_data(
        TensorData::new(input_data, [bs, INPUT_SIZE]),
        device,
    );
    let target = Tensor::from_data(
        TensorData::new(target_data, [bs, OUTPUT_SIZE]),
        device,
    );
    let mask = Tensor::from_data(
        TensorData::new(mask_data, [bs, OUTPUT_SIZE]),
        device,
    );
    let range_t = Tensor::from_data(
        TensorData::new(range_data, [bs, OUTPUT_SIZE]),
        device,
    );
    let game_value = Tensor::from_data(
        TensorData::new(gv_data, [bs]),
        device,
    );

    Batch { input, target, mask, range: range_t, game_value }
}

/// Compute average validation loss over the given indices without gradient tracking.
fn compute_val_loss<B: AutodiffBackend>(
    model: &CfvNet<B>,
    dataset: &CfvDataset,
    val_indices: &[usize],
    config: &TrainConfig,
    device: &B::Device,
) -> f64 {
    let mut total_loss = 0.0_f64;
    let mut batch_count = 0_u64;

    for batch_start in (0..val_indices.len()).step_by(config.batch_size) {
        let batch_end = (batch_start + config.batch_size).min(val_indices.len());
        let batch_idx = &val_indices[batch_start..batch_end];

        let items: Vec<CfvItem> = batch_idx
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

        // INVARIANT: loss is shape [1], so to_vec always has exactly one element.
        let val: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];
        total_loss += val as f64;
        batch_count += 1;
    }

    if batch_count == 0 {
        0.0
    } else {
        total_loss / batch_count as f64
    }
}

/// Save the model to `dir/name` using NamedMpkGz format.
///
/// Logs a warning on failure instead of panicking so training can continue.
fn save_model<B: AutodiffBackend>(
    model: &CfvNet<B>,
    dir: &std::path::Path,
    name: &str,
) {
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let path = dir.join(name);
    if let Err(e) = model.clone().save_file(path, &recorder) {
        eprintln!("Warning: failed to save model '{}': {}", name, e);
    }
}

/// Train a `CfvNet` on the given dataset using a custom Adam training loop.
///
/// Returns the final training loss. Shuffles indices each epoch, processes
/// mini-batches, and updates the model via backpropagation. Optionally saves
/// checkpoints and the final model to `output_dir`.
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    dataset: &CfvDataset,
    config: &TrainConfig,
    output_dir: Option<&std::path::Path>,
) -> TrainResult {
    let mut model = CfvNet::<B>::new(device, config.hidden_layers, config.hidden_size);

    // Resume from checkpoint if a saved model exists.
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

    // Split dataset into train/val using deterministic shuffle.
    let n = dataset.len();
    let mut all_indices: Vec<usize> = (0..n).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    all_indices.shuffle(&mut rng);

    let val_count = (n as f64 * config.validation_split) as usize;
    let val_indices = all_indices[..val_count].to_vec();
    let mut train_indices = all_indices[val_count..].to_vec();

    let num_train_batches = (train_indices.len() + config.batch_size - 1) / config.batch_size;
    let mut final_loss = f32::MAX;

    for epoch in 1..=config.epochs {
        train_indices.shuffle(&mut rng);

        let pb = ProgressBar::new(num_train_batches as u64);
        // INVARIANT: template is valid — all placeholders are known indicatif fields.
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Epoch {msg} {wide_bar} {pos}/{len} [{elapsed}] ETA {eta}")
                .unwrap(),
        );

        let mut epoch_loss = 0.0_f64;
        let mut epoch_batches = 0_u64;

        for batch_start in (0..train_indices.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(train_indices.len());
            let batch_idx = &train_indices[batch_start..batch_end];

            let items: Vec<CfvItem> = batch_idx
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

            // INVARIANT: loss is shape [1], so to_vec always has exactly one element.
            final_loss = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            epoch_loss += final_loss as f64;
            epoch_batches += 1;

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads_params);
            pb.inc(1);
        }

        let avg_train = if epoch_batches > 0 {
            epoch_loss / epoch_batches as f64
        } else {
            0.0
        };

        // Compute validation loss if we have a validation set.
        let summary = if !val_indices.is_empty() {
            let val_loss = compute_val_loss(&model, dataset, &val_indices, config, device);
            format!("{epoch}/{} train={avg_train:.6} val={val_loss:.6}", config.epochs)
        } else {
            format!("{epoch}/{} train={avg_train:.6}", config.epochs)
        };

        pb.finish_with_message(summary);

        // Checkpoint every N epochs if configured.
        if config.checkpoint_every_n_epochs > 0
            && epoch % config.checkpoint_every_n_epochs == 0
        {
            if let Some(dir) = output_dir {
                save_model(&model, dir, &format!("checkpoint_epoch{epoch}"));
            }
        }
    }

    // Save final model.
    if let Some(dir) = output_dir {
        save_model(&model, dir, "model");
    }

    TrainResult {
        final_train_loss: final_loss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use std::io::Write;
    use tempfile::NamedTempFile;

    use crate::datagen::storage::{write_record, TrainingRecord};

    type B = Autodiff<NdArray>;

    fn write_test_data(n: usize) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..n {
            let mut rec = TrainingRecord {
                board: [0, 4, 8, 12, 16],
                pot: 100.0,
                effective_stack: 50.0,
                player: (i % 2) as u8,
                game_value: 0.1 * i as f32,
                oop_range: [0.0; 1326],
                ip_range: [0.0; 1326],
                cfvs: [0.0; 1326],
                valid_mask: [1; 1326],
            };
            // Set some non-zero values so the target is not all zeros.
            for j in 0..10 {
                rec.cfvs[j] = (i as f32 + j as f32) * 0.01;
                rec.oop_range[j] = 0.1;
                rec.ip_range[j] = 0.1;
            }
            write_record(&mut file, &rec).unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    fn overfit_single_batch() {
        let file = write_test_data(16);
        let dataset = CfvDataset::from_file(file.path()).unwrap();

        let device = Default::default();
        let config = TrainConfig {
            hidden_layers: 2,
            hidden_size: 64,
            batch_size: 16,
            epochs: 200,
            learning_rate: 0.001,
            lr_min: 0.001,
            huber_delta: 1.0,
            aux_loss_weight: 0.0,
            validation_split: 0.0,
            checkpoint_every_n_epochs: 0,
        };

        let result = train::<B>(&device, &dataset, &config, None);
        assert!(
            result.final_train_loss < 0.01,
            "should overfit small data, got loss {}",
            result.final_train_loss
        );
    }

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
        assert!(
            dir.path().join("model.mpk.gz").exists(),
            "model.mpk.gz should exist"
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
        assert!(dir.path().join("checkpoint_epoch2.mpk.gz").exists());
        assert!(dir.path().join("checkpoint_epoch4.mpk.gz").exists());
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
        let r1 = train::<B>(&device, &dataset, &config, Some(dir.path()));
        let r2 = train::<B>(&device, &dataset, &config, Some(dir.path()));
        assert!(r1.final_train_loss < 10.0);
        assert!(r2.final_train_loss < 10.0);
    }
}
