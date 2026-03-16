use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use crate::datagen::bucketed_storage::{
    count_bucketed_records_in_files, read_bucketed_header, read_bucketed_record, BucketedRecord,
};
use crate::model::bucketed_network::BucketedCfvNet;
use crate::model::loss::masked_huber_loss;
use crate::model::training::{collect_data_files, cosine_lr};

/// Configuration for bucketed training.
pub struct BucketedTrainConfig {
    pub num_buckets: usize,
    pub hidden_layers: usize,
    pub hidden_size: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub lr_min: f64,
    pub huber_delta: f64,
    pub validation_split: f64,
    pub checkpoint_every_n_epochs: usize,
    pub shuffle_buffer_size: usize,
    pub prefetch_depth: usize,
    pub encoder_threads: usize,
}

/// Result returned after bucketed training completes.
pub struct BucketedTrainResult {
    pub final_train_loss: f32,
}

/// Pre-encoded chunk stored as contiguous flat buffers for fast tensor creation.
struct BucketedPreEncoded {
    input: Vec<f32>,   // flattened [batch * input_size]
    target: Vec<f32>,  // flattened [batch * output_size]
    input_size: usize, // 2*num_buckets + 1
    output_size: usize, // 2*num_buckets
    len: usize,        // batch size
}

/// Tensors created on a device from pre-encoded bucketed data.
struct BucketedDeviceTensors<B: Backend> {
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
}

impl BucketedPreEncoded {
    /// Encode a slice of bucketed records into contiguous flat arrays.
    fn from_records(records: &[BucketedRecord], num_buckets: usize) -> Self {
        let n = records.len();
        let input_size = 2 * num_buckets + 1;
        let output_size = 2 * num_buckets;
        let mut input = Vec::with_capacity(n * input_size);
        let mut target = Vec::with_capacity(n * output_size);
        for rec in records {
            input.extend_from_slice(&rec.input);
            target.extend_from_slice(&rec.target);
        }
        Self {
            input,
            target,
            input_size,
            output_size,
            len: n,
        }
    }

    /// Create tensors on `device`, consuming the pre-encoded data.
    ///
    /// The mask is derived from the input: for each record, mask[i] = 1.0 where
    /// the reach (first 2*num_buckets elements of input) is > 0.
    fn into_device_tensors<B: Backend>(self, device: &B::Device) -> BucketedDeviceTensors<B> {
        let n = self.len;
        let input_size = self.input_size;
        let output_size = self.output_size;

        // Compute mask from the reach portion of input (first 2*num_buckets elements).
        let mut mask_data = Vec::with_capacity(n * output_size);
        for i in 0..n {
            let base = i * input_size;
            for j in 0..output_size {
                mask_data.push(if self.input[base + j] > 0.0 { 1.0f32 } else { 0.0f32 });
            }
        }

        BucketedDeviceTensors {
            input: Tensor::from_data(TensorData::new(self.input, [n, input_size]), device),
            target: Tensor::from_data(TensorData::new(self.target, [n, output_size]), device),
            mask: Tensor::from_data(TensorData::new(mask_data, [n, output_size]), device),
        }
    }

    /// Create tensors on `device` by cloning (for reusable data like validation).
    fn to_device_tensors<B: Backend>(&self, device: &B::Device) -> BucketedDeviceTensors<B> {
        let cloned = BucketedPreEncoded {
            input: self.input.clone(),
            target: self.target.clone(),
            input_size: self.input_size,
            output_size: self.output_size,
            len: self.len,
        };
        cloned.into_device_tensors(device)
    }
}

/// Compute average validation loss from a small set of bucketed records.
fn compute_bucketed_val_loss<B: AutodiffBackend>(
    model: &BucketedCfvNet<B>,
    encoded: &BucketedPreEncoded,
    config: &BucketedTrainConfig,
    device: &B::Device,
) -> f64 {
    let valid_model = model.valid();
    let n = encoded.len;
    let batch_size = config.batch_size;

    let tensors = encoded.to_device_tensors::<B::InnerBackend>(device);

    let mut total_loss = 0.0_f64;
    let mut batch_count = 0_u64;

    for batch_start in (0..n).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n);
        let len = batch_end - batch_start;

        let b_input = tensors.input.clone().narrow(0, batch_start, len);
        let b_target = tensors.target.clone().narrow(0, batch_start, len);
        let b_mask = tensors.mask.clone().narrow(0, batch_start, len);

        let pred = valid_model.forward(b_input);
        let loss = masked_huber_loss(pred, b_target, b_mask, config.huber_delta);

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

/// Save a bucketed model to `dir/name` using NamedMpkGz format.
///
/// Logs a warning on failure instead of panicking so training can continue.
fn save_bucketed_model<B: AutodiffBackend>(
    model: &BucketedCfvNet<B>,
    dir: &Path,
    name: &str,
) {
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let path = dir.join(name);
    if let Err(e) = model.clone().save_file(path, &recorder) {
        eprintln!("Warning: failed to save model '{}': {}", name, e);
    }
}

/// Streaming record reader that reads bucketed records from a sequence of files.
///
/// Each file starts with a u32 header (num_buckets), followed by records.
struct StreamingBucketedReader {
    files: Vec<PathBuf>,
    current_file_idx: usize,
    reader: Option<BufReader<std::fs::File>>,
    num_buckets: usize,
    exhausted: bool,
}

impl StreamingBucketedReader {
    fn new(files: Vec<PathBuf>, num_buckets: usize) -> Self {
        Self {
            files,
            current_file_idx: 0,
            reader: None,
            num_buckets,
            exhausted: false,
        }
    }

    /// Reset to the beginning of the file list for a new epoch pass.
    #[allow(dead_code)]
    fn reset(&mut self) {
        self.current_file_idx = 0;
        self.reader = None;
        self.exhausted = false;
    }

    /// Open the next file, reading and validating the header.
    fn open_next_file(&mut self) -> bool {
        while self.current_file_idx < self.files.len() {
            let path = &self.files[self.current_file_idx];
            match std::fs::File::open(path) {
                Ok(f) => {
                    let mut reader = BufReader::new(f);
                    match read_bucketed_header(&mut reader) {
                        Ok(nb) => {
                            if nb as usize != self.num_buckets {
                                eprintln!(
                                    "Warning: skipping {} (num_buckets={nb}, expected {})",
                                    path.display(),
                                    self.num_buckets
                                );
                                self.current_file_idx += 1;
                                continue;
                            }
                            self.reader = Some(reader);
                            return true;
                        }
                        Err(e) => {
                            eprintln!("Warning: skipping {} (header read error: {e})", path.display());
                            self.current_file_idx += 1;
                            continue;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Warning: skipping {}: {e}", path.display());
                    self.current_file_idx += 1;
                    continue;
                }
            }
        }
        false
    }

    /// Read a single record, advancing through files as needed.
    fn read_one(&mut self) -> Option<BucketedRecord> {
        while !self.exhausted {
            if self.reader.is_none() && !self.open_next_file() {
                self.exhausted = true;
                return None;
            }

            let reader = self.reader.as_mut().unwrap();
            match read_bucketed_record(reader, self.num_buckets) {
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

    /// Read up to `max_records` records, spanning file boundaries as needed.
    fn read_chunk(&mut self, max_records: usize) -> Vec<BucketedRecord> {
        let mut records = Vec::with_capacity(max_records);

        while records.len() < max_records && !self.exhausted {
            if self.reader.is_none() && !self.open_next_file() {
                self.exhausted = true;
                break;
            }

            let reader = self.reader.as_mut().unwrap();
            match read_bucketed_record(reader, self.num_buckets) {
                Ok(rec) => {
                    records.push(rec);
                }
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

        records
    }
}

/// Load the latest checkpoint from `output_dir` if one exists.
fn load_or_create_bucketed_model<B: AutodiffBackend>(
    model: BucketedCfvNet<B>,
    output_dir: Option<&Path>,
    device: &B::Device,
) -> (BucketedCfvNet<B>, usize) {
    let Some(dir) = output_dir else {
        return (model, 0);
    };
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

    // Scan for checkpoint_epochN.mpk.gz and find the highest N.
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut best_epoch = 0usize;
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(n) = name
                .strip_prefix("checkpoint_epoch")
                .and_then(|rest| rest.strip_suffix(".mpk.gz"))
                .and_then(|n_str| n_str.parse::<usize>().ok())
            {
                best_epoch = best_epoch.max(n);
            }
        }
        if best_epoch > 0 {
            let ckpt_path = dir.join(format!("checkpoint_epoch{best_epoch}"));
            match model.clone().load_file(ckpt_path, &recorder, device) {
                Ok(loaded) => {
                    eprintln!("Resuming from checkpoint epoch {best_epoch}");
                    return (loaded, best_epoch);
                }
                Err(e) => {
                    eprintln!("Warning: failed to load checkpoint epoch {best_epoch}: {e}");
                }
            }
        }
    }

    // Fall back to model.mpk.gz.
    let model_path = dir.join("model");
    if model_path.with_extension("mpk.gz").exists() {
        match model.clone().load_file(model_path, &recorder, device) {
            Ok(loaded) => {
                eprintln!("Resuming from saved model");
                return (loaded, 0);
            }
            Err(e) => {
                eprintln!("Warning: failed to load model, starting fresh: {e}");
            }
        }
    }

    (model, 0)
}

/// Load bucketed validation records and return them pre-encoded.
fn load_bucketed_validation_set(
    files: &[PathBuf],
    val_count: usize,
    num_buckets: usize,
) -> Option<BucketedPreEncoded> {
    if val_count == 0 {
        return None;
    }
    eprintln!("Loading {val_count} bucketed validation records...");
    let mut val_reader = StreamingBucketedReader::new(files.to_vec(), num_buckets);
    let val_records = val_reader.read_chunk(val_count);
    let actual_val = val_records.len();
    eprintln!("Loaded {actual_val} bucketed validation records");
    Some(BucketedPreEncoded::from_records(&val_records, num_buckets))
}

/// Spawn a two-stage dataloader pipeline for bucketed records.
///
/// Stage 1: Reader thread with streaming shuffle buffer.
/// Stage 2: Encoder threads that flatten records into pre-encoded batches.
fn spawn_bucketed_dataloader_thread(
    files: &[PathBuf],
    config: &BucketedTrainConfig,
    val_count: usize,
) -> (
    mpsc::Receiver<BucketedPreEncoded>,
    Vec<std::thread::JoinHandle<()>>,
) {
    let batch_size = config.batch_size;
    let shuffle_buffer_size = config.shuffle_buffer_size;
    let prefetch_depth = config.prefetch_depth;
    let num_buckets = config.num_buckets;

    // Stage 1 -> Stage 2: un-encoded record batches.
    let (record_tx, record_rx) =
        mpsc::sync_channel::<Vec<BucketedRecord>>(prefetch_depth * 2);
    // Stage 2 -> Training loop: encoded batches.
    let (batch_tx, batch_rx) = mpsc::sync_channel::<BucketedPreEncoded>(prefetch_depth);

    // Stage 1: Reader thread with streaming shuffle buffer.
    let reader_files = files.to_vec();
    let reader_thread = std::thread::spawn(move || {
        use rand::seq::SliceRandom;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut files = reader_files;
        let mut epoch = 0u64;

        loop {
            let mut reader = StreamingBucketedReader::new(files.clone(), num_buckets);
            if val_count > 0 {
                let _ = reader.read_chunk(val_count);
            }

            // Phase 1: Fill buffer.
            let mut buffer = reader.read_chunk(shuffle_buffer_size);
            if buffer.is_empty() {
                eprintln!("Warning: no bucketed training records found, stopping dataloader");
                return;
            }
            buffer.shuffle(&mut rng);

            // Phase 2: Stream with eviction.
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

            // Phase 3: Drain remaining buffer.
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
            if !batch_buf.is_empty() && record_tx.send(batch_buf).is_err() {
                return;
            }

            // Phase 4: Prepare next epoch.
            epoch += 1;
            rng = ChaCha8Rng::seed_from_u64(42 + epoch);
            files.shuffle(&mut rng);
        }
    });

    // Stage 2: Encoder threads.
    let record_rx = std::sync::Arc::new(std::sync::Mutex::new(record_rx));
    let num_encoders = config.encoder_threads;
    let mut handles = vec![reader_thread];
    for _ in 0..num_encoders {
        let rx = record_rx.clone();
        let tx = batch_tx.clone();
        handles.push(std::thread::spawn(move || {
            loop {
                let records = {
                    let lock = rx.lock().unwrap();
                    match lock.recv() {
                        Ok(r) => r,
                        Err(_) => return,
                    }
                };
                let encoded = BucketedPreEncoded::from_records(&records, num_buckets);
                if tx.send(encoded).is_err() {
                    return;
                }
            }
        }));
    }
    drop(batch_tx);

    (batch_rx, handles)
}

/// Train a `BucketedCfvNet` using a streaming dataloader.
///
/// A background thread reads bucketed records from disk, shuffles them in a
/// buffer, splits into batches, and sends pre-encoded data over a channel.
/// One epoch = `train_records / batch_size` training steps.
pub fn train_bucketed<B: AutodiffBackend>(
    device: &B::Device,
    data_path: &Path,
    config: &BucketedTrainConfig,
    output_dir: Option<&Path>,
) -> BucketedTrainResult {
    let model = BucketedCfvNet::<B>::new(
        device,
        config.hidden_layers,
        config.hidden_size,
        config.num_buckets,
    );
    let (mut model, start_epoch) = load_or_create_bucketed_model(model, output_dir, device);

    let mut optim = AdamConfig::new().init::<B, BucketedCfvNet<B>>();

    let files = collect_data_files(data_path).unwrap_or_else(|e| {
        eprintln!("failed to collect data files: {e}");
        std::process::exit(1);
    });

    eprintln!("Counting bucketed records across {} file(s)...", files.len());
    let (file_buckets, total_records_u64) =
        count_bucketed_records_in_files(&files).unwrap_or_else(|e| {
            eprintln!("failed to count records: {e}");
            std::process::exit(1);
        });
    let total_records = total_records_u64 as usize;
    eprintln!(
        "Total records: {total_records} (num_buckets={file_buckets})"
    );

    if file_buckets as usize != config.num_buckets {
        eprintln!(
            "Warning: config num_buckets={} but files have num_buckets={file_buckets}",
            config.num_buckets
        );
    }

    if total_records == 0 {
        eprintln!("No bucketed training records found.");
        return BucketedTrainResult {
            final_train_loss: f32::MAX,
        };
    }

    let val_count = ((total_records as f64 * config.validation_split) as usize).min(total_records);
    let val_encoded = load_bucketed_validation_set(&files, val_count, config.num_buckets);

    // Compute training schedule.
    let train_records = total_records - val_count;
    let batch_size = config.batch_size;
    let steps_per_epoch = train_records.div_ceil(batch_size);
    let total_steps = steps_per_epoch * config.epochs;

    if start_epoch >= config.epochs {
        eprintln!(
            "Already completed {start_epoch}/{} epochs, nothing to do.",
            config.epochs
        );
        return BucketedTrainResult {
            final_train_loss: f32::MAX,
        };
    }

    let remaining_epochs = config.epochs - start_epoch;
    eprintln!(
        "Training: epochs {}-{} ({remaining_epochs} remaining) x {} steps/epoch = {} total steps",
        start_epoch + 1,
        config.epochs,
        steps_per_epoch,
        total_steps
    );

    let (data_rx, loader_threads) =
        spawn_bucketed_dataloader_thread(&files, config, val_count);

    let mut final_loss = f32::MAX;
    let mut global_step: usize = start_epoch * steps_per_epoch;

    for epoch in start_epoch..config.epochs {
        let pb = ProgressBar::new(steps_per_epoch as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Epoch {msg} {wide_bar} {pos}/{len} [{elapsed}] ETA {eta}")
                .unwrap(),
        );

        let mut epoch_loss = 0.0_f64;

        for step_in_epoch in 0..steps_per_epoch {
            let encoded = match data_rx.recv() {
                Ok(e) => e,
                Err(_) => break,
            };

            let batch = encoded.into_device_tensors::<B::InnerBackend>(device);
            let input = Tensor::<B, 2>::from_inner(batch.input);
            let target = Tensor::<B, 2>::from_inner(batch.target);
            let mask = Tensor::<B, 2>::from_inner(batch.mask);

            let pred = model.forward(input);
            let loss = masked_huber_loss(pred, target, mask, config.huber_delta);

            // Read loss only at epoch end to avoid GPU sync stalls.
            if step_in_epoch + 1 == steps_per_epoch {
                final_loss = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
                epoch_loss = final_loss as f64;
            }

            let lr = cosine_lr(config.learning_rate, config.lr_min, global_step, total_steps);
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads_params);

            global_step += 1;
            pb.inc(1);
        }

        let lr_now = cosine_lr(
            config.learning_rate,
            config.lr_min,
            global_step.saturating_sub(1),
            total_steps,
        );

        let mut summary = format!(
            "{}/{} lr={lr_now:.2e} train={epoch_loss:.6}",
            epoch + 1,
            config.epochs,
        );

        // Validation loss at epoch boundary.
        if let Some(ref val_enc) = val_encoded {
            let val_loss = compute_bucketed_val_loss(&model, val_enc, config, device);
            summary.push_str(&format!(" val={val_loss:.6}"));
        }

        pb.finish_with_message(summary);

        // Checkpoint.
        if config.checkpoint_every_n_epochs > 0
            && (epoch + 1) % config.checkpoint_every_n_epochs == 0
            && let Some(dir) = output_dir
        {
            save_bucketed_model(&model, dir, &format!("checkpoint_epoch{}", epoch + 1));
        }
    }

    // Drop data channel to signal threads to exit, then join.
    drop(data_rx);
    for handle in loader_threads {
        if let Err(e) = handle.join() {
            eprintln!("ERROR: dataloader thread panicked: {e:?}");
        }
    }

    // Save final model.
    if let Some(dir) = output_dir {
        save_bucketed_model(&model, dir, "model");
    }

    BucketedTrainResult {
        final_train_loss: final_loss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use std::io::Write;
    use tempfile::NamedTempFile;

    use crate::datagen::bucketed_storage::{write_bucketed_header, write_bucketed_record};

    type B = Autodiff<NdArray>;

    fn default_bucketed_test_config(num_buckets: usize) -> BucketedTrainConfig {
        BucketedTrainConfig {
            num_buckets,
            hidden_layers: 2,
            hidden_size: 64,
            batch_size: 16,
            epochs: 2,
            learning_rate: 0.001,
            lr_min: 0.001,
            huber_delta: 1.0,
            validation_split: 0.0,
            checkpoint_every_n_epochs: 0,
            shuffle_buffer_size: 100,
            prefetch_depth: 2,
            encoder_threads: 2,
        }
    }

    fn write_bucketed_test_data(num_records: usize, num_buckets: usize) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        write_bucketed_header(&mut file, num_buckets as u32).unwrap();
        for i in 0..num_records {
            let input_len = 2 * num_buckets + 1;
            let target_len = 2 * num_buckets;
            let mut input = vec![1.0 / num_buckets as f32; input_len]; // uniform reach
            input[2 * num_buckets] = 0.5; // pot/stack
            let target: Vec<f32> = (0..target_len)
                .map(|j| (i as f32 + j as f32) * 0.001)
                .collect();
            let rec = BucketedRecord { input, target };
            write_bucketed_record(&mut file, &rec).unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    fn overfit_single_batch() {
        let num_buckets = 10;
        let file = write_bucketed_test_data(16, num_buckets);

        let device = Default::default();
        let config = BucketedTrainConfig {
            epochs: 200,
            ..default_bucketed_test_config(num_buckets)
        };

        let result = train_bucketed::<B>(&device, file.path(), &config, None);
        assert!(
            result.final_train_loss < 0.05,
            "should overfit small bucketed data, got loss {}",
            result.final_train_loss
        );
    }

    #[test]
    fn train_saves_model_to_output_dir() {
        let num_buckets = 10;
        let file = write_bucketed_test_data(16, num_buckets);
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let config = default_bucketed_test_config(num_buckets);
        let result = train_bucketed::<B>(&device, file.path(), &config, Some(dir.path()));
        assert!(result.final_train_loss < 1.0);
        assert!(
            dir.path().join("model.mpk.gz").exists(),
            "model.mpk.gz should exist"
        );
    }

    #[test]
    fn train_saves_checkpoints() {
        let num_buckets = 10;
        let file = write_bucketed_test_data(16, num_buckets);
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let config = BucketedTrainConfig {
            epochs: 4,
            checkpoint_every_n_epochs: 2,
            ..default_bucketed_test_config(num_buckets)
        };
        train_bucketed::<B>(&device, file.path(), &config, Some(dir.path()));
        assert!(dir.path().join("checkpoint_epoch2.mpk.gz").exists());
        assert!(dir.path().join("checkpoint_epoch4.mpk.gz").exists());
        assert!(dir.path().join("model.mpk.gz").exists());
    }

    #[test]
    fn train_resumes_from_checkpoint() {
        let num_buckets = 10;
        let file = write_bucketed_test_data(16, num_buckets);
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();

        // Train 4 epochs with checkpoints every 2.
        let config = BucketedTrainConfig {
            epochs: 4,
            checkpoint_every_n_epochs: 2,
            ..default_bucketed_test_config(num_buckets)
        };
        train_bucketed::<B>(&device, file.path(), &config, Some(dir.path()));
        assert!(dir.path().join("checkpoint_epoch4.mpk.gz").exists());

        // Resume with 6 total epochs — should pick up from epoch 4.
        let config2 = BucketedTrainConfig {
            epochs: 6,
            checkpoint_every_n_epochs: 2,
            ..default_bucketed_test_config(num_buckets)
        };
        let r2 = train_bucketed::<B>(&device, file.path(), &config2, Some(dir.path()));
        assert!(r2.final_train_loss < 10.0);
        assert!(dir.path().join("checkpoint_epoch6.mpk.gz").exists());
    }

    #[test]
    fn cosine_lr_boundaries() {
        // Reuse from training module since it's now pub(crate).
        let lr = cosine_lr(0.01, 0.001, 0, 100);
        assert!(
            (lr - 0.01).abs() < 1e-9,
            "at step 0, lr should be lr_max, got {lr}"
        );

        let lr = cosine_lr(0.01, 0.001, 100, 100);
        assert!(
            (lr - 0.001).abs() < 1e-9,
            "at final step, lr should be lr_min, got {lr}"
        );

        let lr_mid = cosine_lr(0.01, 0.001, 50, 100);
        assert!(
            lr_mid > 0.001 && lr_mid < 0.01,
            "mid lr should be between min and max, got {lr_mid}"
        );

        let expected_mid = (0.01 + 0.001) / 2.0;
        assert!(
            (lr_mid - expected_mid).abs() < 1e-9,
            "mid lr should be {expected_mid}, got {lr_mid}"
        );
    }
}
