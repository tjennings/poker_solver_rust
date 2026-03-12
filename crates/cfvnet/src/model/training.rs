use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};

use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::model::dataset::{encode_record, CfvDataset};
use crate::model::loss::cfvnet_loss;
use crate::model::network::{CfvNet, OUTPUT_SIZE};

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
    pub gpu_chunk_size: usize,
    pub epochs_per_chunk: usize,
}

/// Result returned after training completes.
pub struct TrainResult {
    pub final_train_loss: f32,
}

/// Pre-encoded dataset stored as contiguous flat buffers for fast tensor creation.
struct PreEncoded {
    input: Vec<f32>,
    target: Vec<f32>,
    mask: Vec<f32>,
    range: Vec<f32>,
    game_value: Vec<f32>,
    in_size: usize,
    len: usize,
}

impl PreEncoded {
    /// Encode all records from the dataset into contiguous flat arrays.
    fn from_dataset(dataset: &CfvDataset) -> Self {
        let n = dataset.len();
        let in_size = dataset.input_size();
        let board_cards = dataset.board_cards();

        let mut input = Vec::with_capacity(n * in_size);
        let mut target = Vec::with_capacity(n * OUTPUT_SIZE);
        let mut mask = Vec::with_capacity(n * OUTPUT_SIZE);
        let mut range = Vec::with_capacity(n * OUTPUT_SIZE);
        let mut game_value = Vec::with_capacity(n);

        for rec in dataset.records() {
            let item = encode_record(rec, board_cards);
            input.extend_from_slice(&item.input);
            target.extend_from_slice(&item.target);
            mask.extend_from_slice(&item.mask);
            range.extend_from_slice(&item.range);
            game_value.push(item.game_value);
        }

        Self { input, target, mask, range, game_value, in_size, len: n }
    }

    /// Extract a contiguous slice of records into 5 tensors on `device`.
    fn chunk_tensors<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> ChunkTensors<B> {
        let n = indices.len();
        let in_size = self.in_size;

        let mut input = Vec::with_capacity(n * in_size);
        let mut target = Vec::with_capacity(n * OUTPUT_SIZE);
        let mut mask = Vec::with_capacity(n * OUTPUT_SIZE);
        let mut range = Vec::with_capacity(n * OUTPUT_SIZE);
        let mut gv = Vec::with_capacity(n);

        for &idx in indices {
            let i_start = idx * in_size;
            input.extend_from_slice(&self.input[i_start..i_start + in_size]);
            let o_start = idx * OUTPUT_SIZE;
            target.extend_from_slice(&self.target[o_start..o_start + OUTPUT_SIZE]);
            mask.extend_from_slice(&self.mask[o_start..o_start + OUTPUT_SIZE]);
            range.extend_from_slice(&self.range[o_start..o_start + OUTPUT_SIZE]);
            gv.push(self.game_value[idx]);
        }

        ChunkTensors {
            input: Tensor::from_data(TensorData::new(input, [n, in_size]), device),
            target: Tensor::from_data(TensorData::new(target, [n, OUTPUT_SIZE]), device),
            mask: Tensor::from_data(TensorData::new(mask, [n, OUTPUT_SIZE]), device),
            range: Tensor::from_data(TensorData::new(range, [n, OUTPUT_SIZE]), device),
            game_value: Tensor::from_data(TensorData::new(gv, [n]), device),
            len: n,
        }
    }
}

/// A chunk of tensors resident on the compute device.
struct ChunkTensors<B: Backend> {
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    range: Tensor<B, 2>,
    game_value: Tensor<B, 1>,
    len: usize,
}

impl<B: Backend> ChunkTensors<B> {
    /// Reorder all tensors in-place using the given permutation indices.
    fn index_select(&self, perm: Tensor<B, 1, Int>) -> Self {
        Self {
            input: self.input.clone().select(0, perm.clone()),
            target: self.target.clone().select(0, perm.clone()),
            mask: self.mask.clone().select(0, perm.clone()),
            range: self.range.clone().select(0, perm.clone()),
            game_value: self.game_value.clone().select(0, perm),
            len: self.len,
        }
    }

    /// Slice a mini-batch from [start..end) using narrow operations.
    fn slice_batch(&self, start: usize, end: usize) -> MiniBatch<B> {
        let len = end - start;
        MiniBatch {
            input: self.input.clone().narrow(0, start, len),
            target: self.target.clone().narrow(0, start, len),
            mask: self.mask.clone().narrow(0, start, len),
            range: self.range.clone().narrow(0, start, len),
            game_value: self.game_value.clone().narrow(0, start, len),
        }
    }
}

/// A single mini-batch of tensors.
struct MiniBatch<B: Backend> {
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    range: Tensor<B, 2>,
    game_value: Tensor<B, 1>,
}

/// Cosine annealing learning rate schedule.
///
/// Returns `lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / total_steps))`.
fn cosine_lr(lr_max: f64, lr_min: f64, step: usize, total_steps: usize) -> f64 {
    if total_steps == 0 {
        return lr_max;
    }
    let t = step.min(total_steps) as f64;
    let total = total_steps as f64;
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * t / total).cos())
}

/// Compute average validation loss using the inner (no-grad) backend.
fn compute_val_loss<B: AutodiffBackend>(
    model: &CfvNet<B>,
    encoded: &PreEncoded,
    val_indices: &[usize],
    config: &TrainConfig,
    device: &B::Device,
) -> f64 {
    let valid_model = model.valid();
    let chunk = encoded.chunk_tensors::<B::InnerBackend>(val_indices, device);

    let mut total_loss = 0.0_f64;
    let mut batch_count = 0_u64;
    let batch_size = config.batch_size;

    for batch_start in (0..chunk.len).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(chunk.len);
        let batch = chunk.slice_batch(batch_start, batch_end);

        let pred = valid_model.forward(batch.input);
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

    if batch_count == 0 { 0.0 } else { total_loss / batch_count as f64 }
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

/// How often (in batches) to read loss from GPU. Between reads, we skip
/// the GPU->CPU sync to avoid pipeline stalls.
const LOSS_READ_INTERVAL: usize = 50;

/// Train a `CfvNet` on the given dataset using a custom Adam training loop.
///
/// Returns the final training loss. Uses chunked GPU-resident tensors,
/// cosine LR annealing, reduced GPU->CPU loss sync, and no-grad validation.
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    dataset: &CfvDataset,
    config: &TrainConfig,
    output_dir: Option<&std::path::Path>,
) -> TrainResult {
    let in_size = dataset.input_size();
    let mut model = CfvNet::<B>::new(device, config.hidden_layers, config.hidden_size, in_size);

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

    // Pre-encode all records into flat CPU buffers (done once).
    let encoded = PreEncoded::from_dataset(dataset);

    // Split dataset into train/val using deterministic shuffle.
    let n = encoded.len;
    let mut all_indices: Vec<usize> = (0..n).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    all_indices.shuffle(&mut rng);

    let val_count = (n as f64 * config.validation_split) as usize;
    let val_indices = all_indices[..val_count].to_vec();
    let train_indices = all_indices[val_count..].to_vec();

    let num_train = train_indices.len();
    let batch_size = config.batch_size;
    let batches_per_epoch = num_train.div_ceil(batch_size);
    let total_steps = batches_per_epoch * config.epochs;

    // Determine chunk parameters.
    let chunk_size = config.gpu_chunk_size.max(batch_size).min(num_train);
    let epochs_per_chunk = config.epochs_per_chunk.max(1);

    // Split train indices into chunks.
    let chunks: Vec<Vec<usize>> = train_indices
        .chunks(chunk_size)
        .map(|c| c.to_vec())
        .collect();

    let mut final_loss = f32::MAX;
    let mut global_step: usize = 0;
    let mut total_epochs_done: usize = 0;

    'outer: loop {
        for chunk_indices in &chunks {
            // Upload this chunk's tensors to the device once.
            let chunk_tensors = encoded.chunk_tensors::<B>(chunk_indices, device);
            let chunk_len = chunk_tensors.len;
            let chunk_batches = chunk_len.div_ceil(batch_size);

            for _ in 0..epochs_per_chunk {
                if total_epochs_done >= config.epochs {
                    break 'outer;
                }

                let pb = ProgressBar::new(chunk_batches as u64);
                // INVARIANT: template is valid — all placeholders are known indicatif fields.
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template(
                            "  Epoch {msg} {wide_bar} {pos}/{len} [{elapsed}] ETA {eta}",
                        )
                        .unwrap(),
                );

                // Shuffle within the chunk: create a random permutation.
                let mut perm: Vec<usize> = (0..chunk_len).collect();
                perm.shuffle(&mut rng);
                let perm_data: Vec<i64> = perm.iter().map(|&i| i as i64).collect();
                let perm_tensor: Tensor<B, 1, Int> = Tensor::from_data(
                    TensorData::new(perm_data, [chunk_len]),
                    device,
                );
                let shuffled = chunk_tensors.index_select(perm_tensor);

                let mut epoch_loss = 0.0_f64;
                let mut epoch_batches = 0_u64;

                for batch_idx in 0..chunk_batches {
                    let start = batch_idx * batch_size;
                    let end = (start + batch_size).min(chunk_len);
                    let batch = shuffled.slice_batch(start, end);

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

                    // Only read loss from GPU every LOSS_READ_INTERVAL batches
                    // or on the last batch of the epoch to reduce sync overhead.
                    let is_last_batch = batch_idx + 1 == chunk_batches;
                    if batch_idx.is_multiple_of(LOSS_READ_INTERVAL) || is_last_batch {
                        // INVARIANT: loss is shape [1], so to_vec always has one element.
                        final_loss = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
                        epoch_loss += final_loss as f64;
                        epoch_batches += 1;
                    }

                    let lr = cosine_lr(config.learning_rate, config.lr_min, global_step, total_steps);
                    let grads = loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, &model);
                    model = optim.step(lr, model, grads_params);

                    global_step += 1;
                    pb.inc(1);
                }

                total_epochs_done += 1;

                let avg_train = if epoch_batches > 0 {
                    epoch_loss / epoch_batches as f64
                } else {
                    0.0
                };

                let lr_now = cosine_lr(
                    config.learning_rate,
                    config.lr_min,
                    global_step.saturating_sub(1),
                    total_steps,
                );

                // Compute validation loss if we have a validation set.
                let summary = if !val_indices.is_empty() {
                    let val_loss =
                        compute_val_loss(&model, &encoded, &val_indices, config, device);
                    format!(
                        "{}/{} lr={lr_now:.2e} train={avg_train:.6} val={val_loss:.6}",
                        total_epochs_done, config.epochs
                    )
                } else {
                    format!(
                        "{}/{} lr={lr_now:.2e} train={avg_train:.6}",
                        total_epochs_done, config.epochs
                    )
                };

                pb.finish_with_message(summary);

                // Checkpoint every N epochs if configured.
                if config.checkpoint_every_n_epochs > 0
                    && total_epochs_done.is_multiple_of(config.checkpoint_every_n_epochs)
                    && let Some(dir) = output_dir
                {
                    save_model(
                        &model,
                        dir,
                        &format!("checkpoint_epoch{total_epochs_done}"),
                    );
                }

                if total_epochs_done >= config.epochs {
                    break 'outer;
                }
            }
            // chunk_tensors dropped here — GPU memory freed.
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
                board: vec![0, 4, 8, 12, 16],
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
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();

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
            gpu_chunk_size: 100,
            epochs_per_chunk: 1,
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
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
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
            gpu_chunk_size: 100,
            epochs_per_chunk: 1,
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
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
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
            gpu_chunk_size: 100,
            epochs_per_chunk: 1,
        };
        train::<B>(&device, &dataset, &config, Some(dir.path()));
        assert!(dir.path().join("checkpoint_epoch2.mpk.gz").exists());
        assert!(dir.path().join("checkpoint_epoch4.mpk.gz").exists());
        assert!(dir.path().join("model.mpk.gz").exists());
    }

    #[test]
    fn train_with_validation_reports_val_loss() {
        let file = write_test_data(20);
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
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
            gpu_chunk_size: 100,
            epochs_per_chunk: 1,
        };

        let result = train::<B>(&device, &dataset, &config, Some(dir.path()));
        assert!(result.final_train_loss < 10.0);
        assert!(dir.path().join("model.mpk.gz").exists());
    }

    #[test]
    fn train_resumes_from_checkpoint() {
        let file = write_test_data(16);
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
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
            gpu_chunk_size: 100,
            epochs_per_chunk: 1,
        };
        let r1 = train::<B>(&device, &dataset, &config, Some(dir.path()));
        let r2 = train::<B>(&device, &dataset, &config, Some(dir.path()));
        assert!(r1.final_train_loss < 10.0);
        assert!(r2.final_train_loss < 10.0);
    }

    #[test]
    fn cosine_lr_boundaries() {
        let lr = cosine_lr(0.01, 0.001, 0, 100);
        assert!((lr - 0.01).abs() < 1e-9, "at step 0, lr should be lr_max, got {lr}");

        let lr = cosine_lr(0.01, 0.001, 100, 100);
        assert!((lr - 0.001).abs() < 1e-9, "at final step, lr should be lr_min, got {lr}");

        let lr_mid = cosine_lr(0.01, 0.001, 50, 100);
        assert!(lr_mid > 0.001 && lr_mid < 0.01, "mid lr should be between min and max, got {lr_mid}");

        // Midpoint of cosine = (lr_max + lr_min) / 2
        let expected_mid = (0.01 + 0.001) / 2.0;
        assert!((lr_mid - expected_mid).abs() < 1e-9, "mid lr should be {expected_mid}, got {lr_mid}");
    }

    #[test]
    fn multi_epoch_per_chunk() {
        let file = write_test_data(16);
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
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
            checkpoint_every_n_epochs: 0,
            gpu_chunk_size: 8,
            epochs_per_chunk: 2,
        };
        let result = train::<B>(&device, &dataset, &config, None);
        assert!(result.final_train_loss < 10.0);
    }
}
