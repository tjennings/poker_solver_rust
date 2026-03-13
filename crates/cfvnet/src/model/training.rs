use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};

use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use crate::datagen::storage::{read_record, TrainingRecord};
use crate::model::dataset::encode_record;
use crate::model::loss::cfvnet_loss;
use crate::model::network::{CfvNet, NUM_COMBOS, OUTPUT_SIZE, input_size};

/// Target size: OOP CFVs (1326) + IP CFVs (1326).
const TARGET_SIZE: usize = 2 * OUTPUT_SIZE;
/// Mask size: 1326 (shared for both players).
const MASK_SIZE: usize = OUTPUT_SIZE;

/// Configuration for the CFVnet training loop.
pub struct TrainConfig {
    pub hidden_layers: usize,
    pub hidden_size: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub lr_min: f64,
    pub huber_delta: f64,
    pub validation_split: f64,
    pub checkpoint_every_n_epochs: usize,
    pub gpu_chunk_size: usize,
    pub epochs_per_chunk: usize,
    pub prefetch_chunks: usize,
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
    oop_range: Vec<f32>,
    ip_range: Vec<f32>,
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
        let mut target = Vec::with_capacity(n * TARGET_SIZE);
        let mut mask = Vec::with_capacity(n * MASK_SIZE);
        let mut oop_range = Vec::with_capacity(n * NUM_COMBOS);
        let mut ip_range = Vec::with_capacity(n * NUM_COMBOS);

        for item in &items {
            input.extend_from_slice(&item.input);
            // Concatenate OOP + IP targets into a single [2652] row.
            target.extend_from_slice(&item.oop_target);
            target.extend_from_slice(&item.ip_target);
            mask.extend_from_slice(&item.mask);
            oop_range.extend_from_slice(&item.oop_range);
            ip_range.extend_from_slice(&item.ip_range);
        }

        Self { input, target, mask, oop_range, ip_range, in_size, len: n }
    }

    /// Consume self and create tensors on `device`, avoiding copies.
    fn into_tensors<B: Backend>(self, device: &B::Device) -> ChunkTensors<B> {
        let n = self.len;
        let in_size = self.in_size;
        ChunkTensors {
            input: Tensor::from_data(TensorData::new(self.input, [n, in_size]), device),
            target: Tensor::from_data(TensorData::new(self.target, [n, TARGET_SIZE]), device),
            mask: Tensor::from_data(TensorData::new(self.mask, [n, MASK_SIZE]), device),
            oop_range: Tensor::from_data(TensorData::new(self.oop_range, [n, NUM_COMBOS]), device),
            ip_range: Tensor::from_data(TensorData::new(self.ip_range, [n, NUM_COMBOS]), device),
            len: n,
        }
    }

    /// Create tensors on `device` by cloning (for reusable data like validation set).
    fn to_tensors<B: Backend>(&self, device: &B::Device) -> ChunkTensors<B> {
        let n = self.len;
        let in_size = self.in_size;
        ChunkTensors {
            input: Tensor::from_data(TensorData::new(self.input.clone(), [n, in_size]), device),
            target: Tensor::from_data(TensorData::new(self.target.clone(), [n, TARGET_SIZE]), device),
            mask: Tensor::from_data(TensorData::new(self.mask.clone(), [n, MASK_SIZE]), device),
            oop_range: Tensor::from_data(TensorData::new(self.oop_range.clone(), [n, NUM_COMBOS]), device),
            ip_range: Tensor::from_data(TensorData::new(self.ip_range.clone(), [n, NUM_COMBOS]), device),
            len: n,
        }
    }

    /// Extract a subset by indices into tensors on `device`.
    fn chunk_tensors<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> ChunkTensors<B> {
        let n = indices.len();
        let in_size = self.in_size;

        let mut input = Vec::with_capacity(n * in_size);
        let mut target = Vec::with_capacity(n * TARGET_SIZE);
        let mut mask = Vec::with_capacity(n * MASK_SIZE);
        let mut oop_range = Vec::with_capacity(n * NUM_COMBOS);
        let mut ip_range = Vec::with_capacity(n * NUM_COMBOS);

        for &idx in indices {
            let i_start = idx * in_size;
            input.extend_from_slice(&self.input[i_start..i_start + in_size]);
            let t_start = idx * TARGET_SIZE;
            target.extend_from_slice(&self.target[t_start..t_start + TARGET_SIZE]);
            let m_start = idx * MASK_SIZE;
            mask.extend_from_slice(&self.mask[m_start..m_start + MASK_SIZE]);
            let r_start = idx * NUM_COMBOS;
            oop_range.extend_from_slice(&self.oop_range[r_start..r_start + NUM_COMBOS]);
            ip_range.extend_from_slice(&self.ip_range[r_start..r_start + NUM_COMBOS]);
        }

        ChunkTensors {
            input: Tensor::from_data(TensorData::new(input, [n, in_size]), device),
            target: Tensor::from_data(TensorData::new(target, [n, TARGET_SIZE]), device),
            mask: Tensor::from_data(TensorData::new(mask, [n, MASK_SIZE]), device),
            oop_range: Tensor::from_data(TensorData::new(oop_range, [n, NUM_COMBOS]), device),
            ip_range: Tensor::from_data(TensorData::new(ip_range, [n, NUM_COMBOS]), device),
            len: n,
        }
    }
}

/// A chunk of tensors resident on the compute device.
struct ChunkTensors<B: Backend> {
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    oop_range: Tensor<B, 2>,
    ip_range: Tensor<B, 2>,
    len: usize,
}

impl<B: Backend> ChunkTensors<B> {
    /// Reorder all tensors in-place using the given permutation indices.
    fn index_select(&self, perm: Tensor<B, 1, Int>) -> Self {
        Self {
            input: self.input.clone().select(0, perm.clone()),
            target: self.target.clone().select(0, perm.clone()),
            mask: self.mask.clone().select(0, perm.clone()),
            oop_range: self.oop_range.clone().select(0, perm.clone()),
            ip_range: self.ip_range.clone().select(0, perm),
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
            oop_range: self.oop_range.clone().narrow(0, start, len),
            ip_range: self.ip_range.clone().narrow(0, start, len),
        }
    }
}

/// A single mini-batch of tensors.
struct MiniBatch<B: Backend> {
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    oop_range: Tensor<B, 2>,
    ip_range: Tensor<B, 2>,
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
    let chunk = encoded.to_tensors::<B::InnerBackend>(device);

    let mut total_loss = 0.0_f64;
    let mut batch_count = 0_u64;
    let batch_size = config.batch_size;

    for batch_start in (0..chunk.len).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(chunk.len);
        let batch = chunk.slice_batch(batch_start, batch_end);

        let pred = valid_model.forward(batch.input, batch.oop_range, batch.ip_range);
        let loss = cfvnet_loss(
            pred,
            batch.target,
            batch.mask,
            config.huber_delta,
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

/// Messages sent from the prefetch producer thread to the training consumer.
enum ChunkMsg {
    /// A pre-encoded chunk ready for GPU upload.
    Data(PreEncoded),
    /// All files for this epoch have been read.
    EndOfEpoch,
}


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

/// Train a `CfvNet` on data files using streaming GPU-chunked training.
///
/// Loads records on demand in chunks of `gpu_chunk_size`, never holding the
/// full dataset in memory. Files are read sequentially; chunk boundaries
/// can span files. Each pass through all files counts as one epoch.
///
/// A small validation set is loaded once at startup (first `validation_split`
/// fraction of the first pass's records).
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

    // Load validation set: read first pass, take validation_split fraction.
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

    let num_train = total_records;
    let batch_size = config.batch_size;
    let chunk_size = config.gpu_chunk_size.max(batch_size);
    let epochs_per_chunk = config.epochs_per_chunk.max(1);

    // Estimate total steps for LR schedule.
    let batches_per_epoch = num_train.div_ceil(batch_size);
    let total_steps = batches_per_epoch * config.epochs;

    let mut final_loss = f32::MAX;
    let mut global_step: usize = 0;
    let mut total_epochs_done: usize = 0;

    // Prefetch pipeline: background thread reads + encodes chunks, sends
    // through a bounded channel so N chunks are always ready.
    let prefetch_slots = config.prefetch_chunks.max(1);

    let num_epochs = config.epochs;
    let producer_files = files.clone();
    let producer_seed = rng.clone();

    let (tx, rx) = mpsc::sync_channel::<ChunkMsg>(prefetch_slots);

    let producer = std::thread::spawn(move || {
        let mut rng = producer_seed;
        let mut shuffled = producer_files;

        for _epoch in 0..num_epochs {
            shuffled.shuffle(&mut rng);
            let mut reader = StreamingReader::new(shuffled.clone());

            loop {
                let records = reader.read_chunk(chunk_size);
                if records.is_empty() {
                    break;
                }
                let encoded = PreEncoded::from_records(&records, board_cards);
                if tx.send(ChunkMsg::Data(encoded)).is_err() {
                    return; // Consumer dropped — training ended early.
                }
            }

            if tx.send(ChunkMsg::EndOfEpoch).is_err() {
                return;
            }
        }
        // Channel drops when producer exits, signaling completion.
    });

    // Advance the main rng past the producer's usage so they don't share state.
    // The producer got a clone; we skip ahead to keep them independent.
    for _ in 0..config.epochs {
        rng.set_word_pos(rng.get_word_pos() + 1);
    }

    let mut chunks_in_epoch = 0u64;

    for msg in rx {
        match msg {
            ChunkMsg::EndOfEpoch => {
                total_epochs_done += 1;

                // Compute validation loss at epoch boundary (not per-chunk).
                if let Some(ref val_enc) = val_encoded {
                    let val_loss = compute_val_loss(&model, val_enc, config, device);
                    eprintln!(
                        "  Epoch {}/{} complete — val={val_loss:.6}",
                        total_epochs_done, config.epochs
                    );
                }

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

                chunks_in_epoch = 0;

                if total_epochs_done >= config.epochs {
                    break;
                }
            }
            ChunkMsg::Data(encoded) => {
                let chunk_len = encoded.len;
                let chunk_tensors = encoded.into_tensors::<B>(device);

                let chunk_batches = chunk_len.div_ceil(batch_size);

                for _ in 0..epochs_per_chunk {
                    if total_epochs_done >= config.epochs {
                        break;
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

                        let pred = model.forward(batch.input, batch.oop_range, batch.ip_range);
                        let loss = cfvnet_loss(
                            pred,
                            batch.target,
                            batch.mask,
                            config.huber_delta,
                        );

                        // Only read loss from GPU every LOSS_READ_INTERVAL batches
                        // or on the last batch to reduce sync overhead.
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

                    chunks_in_epoch += 1;

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

                    let summary = format!(
                        "{}/{} chunk={} lr={lr_now:.2e} train={avg_train:.6}",
                        total_epochs_done + 1, config.epochs, chunks_in_epoch
                    );

                    pb.finish_with_message(summary);
                }
                // chunk_tensors dropped here — GPU memory freed.
            }
        }
    }

    // Wait for producer thread to finish.
    let _ = producer.join();

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

    use crate::datagen::storage::write_record;

    type B = Autodiff<NdArray>;

    fn write_test_data(n: usize) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..n {
            let mut rec = TrainingRecord {
                board: vec![0, 4, 8, 12, 16],
                pot: 100.0,
                effective_stack: 50.0,
                oop_range: [0.0; 1326],
                ip_range: [0.0; 1326],
                oop_cfvs: [0.0; 1326],
                ip_cfvs: [0.0; 1326],
                valid_mask: [1; 1326],
            };
            // Set some non-zero values so the target is not all zeros.
            for j in 0..10 {
                rec.oop_cfvs[j] = (i as f32 + j as f32) * 0.01;
                rec.ip_cfvs[j] = (i as f32 + j as f32) * -0.01;
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

            validation_split: 0.0,
            checkpoint_every_n_epochs: 0,
            gpu_chunk_size: 100,
            epochs_per_chunk: 1,
            prefetch_chunks: 1,

        };

        let result = train::<B>(&device, file.path(), 5, &config, None);
        assert!(
            result.final_train_loss < 0.01,
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

            validation_split: 0.0,
            checkpoint_every_n_epochs: 0,
            gpu_chunk_size: 100,
            epochs_per_chunk: 1,
            prefetch_chunks: 1,

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

            validation_split: 0.0,
            checkpoint_every_n_epochs: 2,
            gpu_chunk_size: 100,
            epochs_per_chunk: 1,
            prefetch_chunks: 1,

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

            validation_split: 0.2, // 4 val samples out of 20
            checkpoint_every_n_epochs: 0,
            gpu_chunk_size: 100,
            epochs_per_chunk: 1,
            prefetch_chunks: 1,

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

            validation_split: 0.0,
            checkpoint_every_n_epochs: 0,
            gpu_chunk_size: 100,
            epochs_per_chunk: 1,
            prefetch_chunks: 1,

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
    fn multi_epoch_per_chunk() {
        let file = write_test_data(16);
        let device = Default::default();
        let config = TrainConfig {
            hidden_layers: 2,
            hidden_size: 64,
            batch_size: 16,
            epochs: 4,
            learning_rate: 0.001,
            lr_min: 0.001,
            huber_delta: 1.0,

            validation_split: 0.0,
            checkpoint_every_n_epochs: 0,
            gpu_chunk_size: 8,
            epochs_per_chunk: 2,
            prefetch_chunks: 1,

        };
        let result = train::<B>(&device, file.path(), 5, &config, None);
        assert!(result.final_train_loss < 10.0);
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
                    oop_range: [0.0; 1326],
                    ip_range: [0.0; 1326],
                    oop_cfvs: [0.0; 1326],
                    ip_cfvs: [0.0; 1326],
                    valid_mask: [1; 1326],
                };
                for j in 0..10 {
                    rec.oop_cfvs[j] = (i as f32 + j as f32) * 0.01;
                    rec.ip_cfvs[j] = (i as f32 + j as f32) * -0.01;
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

            validation_split: 0.0,
            checkpoint_every_n_epochs: 0,
            gpu_chunk_size: 10, // smaller than a file, forces cross-file chunking
            epochs_per_chunk: 1,
            prefetch_chunks: 1,

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
            for _i in 0..count {
                let rec = TrainingRecord {
                    board: vec![0, 4, 8, 12, 16],
                    pot: 100.0,
                    effective_stack: 50.0,
                    oop_range: [0.0; 1326],
                    ip_range: [0.0; 1326],
                    oop_cfvs: [0.0; 1326],
                    ip_cfvs: [0.0; 1326],
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
}
