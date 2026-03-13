use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};

use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use crate::datagen::storage::{read_record, TrainingRecord};
use crate::model::dataset::encode_record;
use crate::model::loss::cfvnet_loss;
use crate::model::network::{CfvNet, OUTPUT_SIZE, input_size};

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
    pub reservoir_size: usize,
    pub reservoir_turnover: f64,
}

/// Result returned after training completes.
pub struct TrainResult {
    pub final_train_loss: f32,
}

/// Pre-encoded chunk stored as contiguous flat buffers for fast tensor creation.
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
    /// Encode a slice of records into contiguous flat arrays.
    ///
    /// Encoding is parallelized across CPU cores with rayon.
    fn from_records(records: &[TrainingRecord], board_cards: usize) -> Self {
        let n = records.len();
        let in_size = input_size(board_cards);

        // Encode all records in parallel.
        let items: Vec<_> = records
            .par_iter()
            .map(|rec| encode_record(rec, board_cards))
            .collect();

        // Flatten into contiguous arrays.
        let mut input = Vec::with_capacity(n * in_size);
        let mut target = Vec::with_capacity(n * OUTPUT_SIZE);
        let mut mask = Vec::with_capacity(n * OUTPUT_SIZE);
        let mut range = Vec::with_capacity(n * OUTPUT_SIZE);
        let mut game_value = Vec::with_capacity(n);

        for item in &items {
            input.extend_from_slice(&item.input);
            target.extend_from_slice(&item.target);
            mask.extend_from_slice(&item.mask);
            range.extend_from_slice(&item.range);
            game_value.push(item.game_value);
        }

        Self { input, target, mask, range, game_value, in_size, len: n }
    }

}

/// GPU-resident reservoir of training data.
///
/// Holds pre-encoded training records as large tensors in GPU memory.
/// The training loop samples random batches via `select` (pure GPU).
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
        let n_actual = n.min(self.capacity);

        self.input = Tensor::from_data(
            TensorData::new(encoded.input, [n_actual, self.in_size]),
            &device,
        );
        self.target = Tensor::from_data(
            TensorData::new(encoded.target, [n_actual, OUTPUT_SIZE]),
            &device,
        );
        self.mask = Tensor::from_data(
            TensorData::new(encoded.mask, [n_actual, OUTPUT_SIZE]),
            &device,
        );
        self.range = Tensor::from_data(
            TensorData::new(encoded.range, [n_actual, OUTPUT_SIZE]),
            &device,
        );
        self.game_value = Tensor::from_data(
            TensorData::new(encoded.game_value, [n_actual]),
            &device,
        );
        self.capacity = n_actual;
        n_actual
    }

    /// Sample a random mini-batch from the reservoir.
    ///
    /// Uses `rand::seq::index::sample` for O(batch_size) index generation
    /// instead of shuffling the full reservoir. Indices are unique to avoid
    /// gradient accumulation issues in some backends.
    fn sample_batch(&self, batch_size: usize, rng: &mut ChaCha8Rng, device: &B::Device) -> SampledBatch<B> {
        use rand::seq::index;
        let actual_batch = batch_size.min(self.capacity);
        let sampled = index::sample(rng, self.capacity, actual_batch);
        let indices_vec: Vec<i64> = sampled.iter().map(|i| i as i64).collect();

        let indices: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(indices_vec, [actual_batch]),
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
        encoded: PreEncoded,
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
        let new_input: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(encoded.input, [n, self.in_size]),
            device,
        );
        let new_target: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(encoded.target, [n, OUTPUT_SIZE]),
            device,
        );
        let new_mask: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(encoded.mask, [n, OUTPUT_SIZE]),
            device,
        );
        let new_range: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(encoded.range, [n, OUTPUT_SIZE]),
            device,
        );
        let new_gv: Tensor<B, 1> = Tensor::from_data(
            TensorData::new(encoded.game_value, [n]),
            device,
        );

        // Use select_assign to scatter new values into the reservoir.
        let idx: Tensor<B, 1, Int> = Tensor::from_data(
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

/// Compute average validation loss from a small set of records.
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

/// Streaming record reader that reads from a sequence of files, filling
/// buffers up to a requested chunk size. Handles file boundaries transparently.
struct StreamingReader {
    files: Vec<PathBuf>,
    current_file_idx: usize,
    reader: Option<BufReader<std::fs::File>>,
    exhausted: bool,
}

impl StreamingReader {
    fn new(files: Vec<PathBuf>) -> Self {
        Self {
            files,
            current_file_idx: 0,
            reader: None,
            exhausted: false,
        }
    }

    /// Reset to the beginning of the file list for a new epoch pass.
    fn reset(&mut self) {
        self.current_file_idx = 0;
        self.reader = None;
        self.exhausted = false;
    }

    /// Read up to `max_records` records, spanning file boundaries as needed.
    /// Returns fewer than `max_records` only when all files are exhausted.
    fn read_chunk(&mut self, max_records: usize) -> Vec<TrainingRecord> {
        let mut records = Vec::with_capacity(max_records);

        while records.len() < max_records && !self.exhausted {
            // Open next file if no reader active.
            if self.reader.is_none() {
                if self.current_file_idx >= self.files.len() {
                    self.exhausted = true;
                    break;
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

            // Read records from current file.
            let reader = self.reader.as_mut().unwrap();
            match read_record(reader) {
                Ok(rec) => {
                    records.push(rec);
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Current file exhausted, move to next.
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

        records
    }
}

/// Collect sorted file paths from a path (file or directory).
fn collect_data_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_dir() {
        let mut paths: Vec<_> = std::fs::read_dir(path)
            .map_err(|e| format!("read directory {}: {e}", path.display()))?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if entry.file_type().ok()?.is_file() {
                    Some(entry.path())
                } else {
                    None
                }
            })
            .collect();
        if paths.is_empty() {
            return Err(format!("no files found in {}", path.display()));
        }
        paths.sort();
        Ok(paths)
    } else {
        Ok(vec![path.to_path_buf()])
    }
}

/// Count total records across all files without keeping them in memory.
fn count_total_records(files: &[PathBuf]) -> u64 {
    let mut total = 0u64;
    for path in files {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mut reader = BufReader::new(file);
        loop {
            match read_record(&mut reader) {
                Ok(_) => total += 1,
                Err(_) => break,
            }
        }
    }
    total
}

/// Train a `CfvNet` using a GPU-resident reservoir.
///
/// The reservoir holds `reservoir_size` records in GPU memory. Training samples
/// random batches directly from GPU tensors (zero PCIe transfer). A background
/// thread continuously refreshes records from disk. One epoch = `total_records /
/// batch_size` training steps.
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    data_path: &Path,
    board_cards: usize,
    config: &TrainConfig,
    output_dir: Option<&std::path::Path>,
) -> TrainResult {
    let in_size = input_size(board_cards);
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
        // Skip past validation records so they don't leak into the reservoir.
        if val_count > 0 {
            let _ = fill_reader.read_chunk(val_count);
        }
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
    // Only spawn if turnover > 0, otherwise no refresh needed.
    let (refresh_tx, refresh_rx) = mpsc::sync_channel::<PreEncoded>(4);
    let refresh_thread = if refresh_per_step > 0 {
        let refresh_files = files.clone();
        Some(std::thread::spawn(move || {
            let mut reader = StreamingReader::new(refresh_files);
            let mut consecutive_empty = 0u32;
            loop {
                let records = reader.read_chunk(refresh_per_step);
                if records.is_empty() {
                    consecutive_empty += 1;
                    if consecutive_empty > 3 {
                        eprintln!("Warning: refresh thread exhausted all files after 3 retries, stopping");
                        return;
                    }
                    reader.reset();
                    continue;
                }
                consecutive_empty = 0;
                let encoded = PreEncoded::from_records(&records, board_cards);
                if refresh_tx.send(encoded).is_err() {
                    return; // Training done, channel closed.
                }
            }
        }))
    } else {
        drop(refresh_tx);
        None
    };

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
                reservoir.scatter_refresh(encoded, &mut rng, device);
            }

            // Sample batch from reservoir and train.
            let batch = reservoir.sample_batch(batch_size, &mut rng, device);
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
    if let Some(thread) = refresh_thread {
        let _ = thread.join();
    }

    // Save final model.
    if let Some(dir) = output_dir {
        save_model(&model, dir, "model");
    }

    TrainResult { final_train_loss: final_loss }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use std::io::Write;
    use tempfile::NamedTempFile;

    use crate::datagen::storage::write_record;

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
            reservoir_size: 100,
            reservoir_turnover: 0.0,
        };

        let result = train::<B>(&device, file.path(), 5, &config, None);
        assert!(
            result.final_train_loss < 0.05,
            "should overfit small data, got loss {}",
            result.final_train_loss
        );
    }

    #[test]
    fn train_saves_model_to_output_dir() {
        let file = write_test_data(16);
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
            reservoir_size: 100,
            reservoir_turnover: 1.0,
        };
        let result = train::<B>(&device, file.path(), 5, &config, Some(dir.path()));
        assert!(result.final_train_loss < 1.0);
        assert!(
            dir.path().join("model.mpk.gz").exists(),
            "model.mpk.gz should exist"
        );
    }

    #[test]
    fn train_saves_checkpoints() {
        let file = write_test_data(16);
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
            reservoir_size: 100,
            reservoir_turnover: 1.0,
        };
        train::<B>(&device, file.path(), 5, &config, Some(dir.path()));
        assert!(dir.path().join("checkpoint_epoch2.mpk.gz").exists());
        assert!(dir.path().join("checkpoint_epoch4.mpk.gz").exists());
        assert!(dir.path().join("model.mpk.gz").exists());
    }

    #[test]
    fn train_with_validation_reports_val_loss() {
        let file = write_test_data(20);
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
            reservoir_size: 100,
            reservoir_turnover: 1.0,
        };

        let result = train::<B>(&device, file.path(), 5, &config, Some(dir.path()));
        assert!(result.final_train_loss < 10.0);
        assert!(dir.path().join("model.mpk.gz").exists());
    }

    #[test]
    fn train_resumes_from_checkpoint() {
        let file = write_test_data(16);
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
            reservoir_size: 100,
            reservoir_turnover: 1.0,
        };
        let r1 = train::<B>(&device, file.path(), 5, &config, Some(dir.path()));
        let r2 = train::<B>(&device, file.path(), 5, &config, Some(dir.path()));
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
    fn reservoir_training_reduces_loss() {
        let file = write_test_data(16);
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
            reservoir_size: 16,
            reservoir_turnover: 0.0,
        };
        let result = train::<B>(&device, file.path(), 5, &config, None);
        assert!(
            result.final_train_loss < 0.05,
            "reservoir training should reduce loss, got {}",
            result.final_train_loss
        );
    }

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
            reservoir_size: 16, // smaller than data -> refresh thread runs
            reservoir_turnover: 2.0,
        };
        let result = train::<B>(&device, file.path(), 5, &config, None);
        assert!(
            result.final_train_loss < 10.0,
            "training with refresh should produce reasonable loss, got {}",
            result.final_train_loss
        );
    }

    #[test]
    fn train_from_directory() {
        // Write data across two files in a temp directory.
        let dir = tempfile::tempdir().unwrap();
        let path1 = dir.path().join("data_01.bin");
        let path2 = dir.path().join("data_02.bin");

        for path in [&path1, &path2] {
            let mut file = std::fs::File::create(path).unwrap();
            for i in 0..8 {
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
                for j in 0..10 {
                    rec.cfvs[j] = (i as f32 + j as f32) * 0.01;
                    rec.oop_range[j] = 0.1;
                    rec.ip_range[j] = 0.1;
                }
                write_record(&mut file, &rec).unwrap();
            }
        }

        let device = Default::default();
        let config = TrainConfig {
            hidden_layers: 2,
            hidden_size: 64,
            batch_size: 8,
            epochs: 2,
            learning_rate: 0.001,
            lr_min: 0.001,
            huber_delta: 1.0,
            aux_loss_weight: 0.0,
            validation_split: 0.0,
            checkpoint_every_n_epochs: 0,
            reservoir_size: 100,
            reservoir_turnover: 1.0,
        };

        let result = train::<B>(&device, dir.path(), 5, &config, None);
        assert!(result.final_train_loss < 10.0);
    }

    #[test]
    fn streaming_reader_spans_files() {
        let dir = tempfile::tempdir().unwrap();
        let path1 = dir.path().join("a.bin");
        let path2 = dir.path().join("b.bin");

        // Write 3 records to first file, 2 to second.
        for (path, count) in [(&path1, 3), (&path2, 2)] {
            let mut file = std::fs::File::create(path).unwrap();
            for i in 0..count {
                let rec = TrainingRecord {
                    board: vec![0, 4, 8, 12, 16],
                    pot: 100.0,
                    effective_stack: 50.0,
                    player: (i % 2) as u8,
                    game_value: i as f32,
                    oop_range: [0.0; 1326],
                    ip_range: [0.0; 1326],
                    cfvs: [0.0; 1326],
                    valid_mask: [1; 1326],
                };
                write_record(&mut file, &rec).unwrap();
            }
        }

        let mut reader = StreamingReader::new(vec![path1, path2]);

        // Read 4 records (spans both files: 3 from first + 1 from second).
        let chunk1 = reader.read_chunk(4);
        assert_eq!(chunk1.len(), 4);

        // Read remaining 1.
        let chunk2 = reader.read_chunk(4);
        assert_eq!(chunk2.len(), 1);

        // Exhausted.
        let chunk3 = reader.read_chunk(4);
        assert_eq!(chunk3.len(), 0);

        // Reset and read again.
        reader.reset();
        let all = reader.read_chunk(100);
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn reservoir_fill_and_sample() {
        let file = write_test_data(20);
        let device = Default::default();
        let files = collect_data_files(file.path()).unwrap();

        let mut reservoir = GpuReservoir::<B>::new(&device, 20, 5);
        let mut reader = StreamingReader::new(files);
        let filled = reservoir.fill(&mut reader, 5);
        assert_eq!(filled, 20);

        // Sample a batch -- should not panic, should return correct shapes.
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let batch = reservoir.sample_batch(4, &mut rng, &device);
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
        reservoir.scatter_refresh(encoded, &mut rng, &device);
        // No panic = success. Reservoir still works.
        let batch = reservoir.sample_batch(4, &mut rng, &device);
        assert_eq!(batch.input.dims(), [4, input_size(5)]);
    }
}
