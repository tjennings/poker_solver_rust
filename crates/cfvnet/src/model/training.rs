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

use crate::datagen::storage::{read_record, TrainingRecord};
use crate::model::dataset::encode_record;
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
    pub shuffle_buffer_size: usize,
    pub prefetch_depth: usize,
    pub encoder_threads: usize,
    pub gpu_prefetch: usize,
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

/// Tensors created on a device from pre-encoded data.
struct DeviceTensors<B: Backend> {
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    range: Tensor<B, 2>,
    game_value: Tensor<B, 1>,
}

impl PreEncoded {
    /// Encode a slice of records into contiguous flat arrays.
    fn from_records(records: &[TrainingRecord]) -> Self {
        let n = records.len();
        let in_size = INPUT_SIZE;

        let items: Vec<_> = records
            .iter()
            .map(encode_record)
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

    /// Create tensors on `device`, consuming the pre-encoded data.
    fn into_device_tensors<B: Backend>(self, device: &B::Device) -> DeviceTensors<B> {
        let n = self.len;
        let in_size = self.in_size;
        DeviceTensors {
            input: Tensor::from_data(TensorData::new(self.input, [n, in_size]), device),
            target: Tensor::from_data(TensorData::new(self.target, [n, OUTPUT_SIZE]), device),
            mask: Tensor::from_data(TensorData::new(self.mask, [n, OUTPUT_SIZE]), device),
            range: Tensor::from_data(TensorData::new(self.range, [n, OUTPUT_SIZE]), device),
            game_value: Tensor::from_data(TensorData::new(self.game_value, [n]), device),
        }
    }

    /// Create tensors on `device` by cloning (for reusable data like validation).
    fn to_device_tensors<B: Backend>(&self, device: &B::Device) -> DeviceTensors<B> {
        let cloned = PreEncoded {
            input: self.input.clone(),
            target: self.target.clone(),
            mask: self.mask.clone(),
            range: self.range.clone(),
            game_value: self.game_value.clone(),
            in_size: self.in_size,
            len: self.len,
        };
        cloned.into_device_tensors(device)
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
    let batch_size = config.batch_size;

    // Create full validation tensors on the inner (non-autodiff) backend.
    let tensors = encoded.to_device_tensors::<B::InnerBackend>(device);

    let mut total_loss = 0.0_f64;
    let mut batch_count = 0_u64;

    for batch_start in (0..n).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n);
        let len = batch_end - batch_start;

        let b_input = tensors.input.clone().narrow(0, batch_start, len);
        let b_target = tensors.target.clone().narrow(0, batch_start, len);
        let b_mask = tensors.mask.clone().narrow(0, batch_start, len);
        let b_range = tensors.range.clone().narrow(0, batch_start, len);
        let b_gv = tensors.game_value.clone().narrow(0, batch_start, len);

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
    #[allow(dead_code)]
    fn reset(&mut self) {
        self.current_file_idx = 0;
        self.reader = None;
        self.exhausted = false;
    }

    /// Read a single record, advancing through files as needed.
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

/// Load the latest checkpoint from `output_dir` if one exists.
///
/// Scans for `checkpoint_epochN.mpk.gz` files and picks the highest N.
/// Falls back to `model.mpk.gz` if no numbered checkpoints exist.
/// Returns `(model, start_epoch)` where `start_epoch` is the epoch to
/// resume from (0 if starting fresh).
fn load_or_create_model<B: AutodiffBackend>(
    model: CfvNet<B>,
    output_dir: Option<&Path>,
    device: &B::Device,
) -> (CfvNet<B>, usize) {
    let Some(dir) = output_dir else { return (model, 0) };
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

/// Load validation records and return them pre-encoded.
fn load_validation_set(
    files: &[PathBuf],
    val_count: usize,
) -> Option<PreEncoded> {
    if val_count == 0 {
        return None;
    }
    eprintln!("Loading {val_count} validation records...");
    let mut val_reader = StreamingReader::new(files.to_vec());
    let val_records = val_reader.read_chunk(val_count);
    let actual_val = val_records.len();
    eprintln!("Loaded {actual_val} validation records");
    Some(PreEncoded::from_records(&val_records))
}

/// Spawn a two-stage dataloader pipeline: a reader thread with a streaming
/// shuffle buffer, and an encoder thread that encodes batches in parallel
/// via rayon. The reader continuously reads records one at a time, evicting
/// shuffled records into batches, eliminating pipeline stalls between chunks.
///
/// Returns `(batch_receiver, thread_handles)`. The threads exit cleanly when
/// the receiver is dropped (cascading channel close).
fn spawn_dataloader_thread(
    files: &[PathBuf],
    config: &TrainConfig,
    val_count: usize,
) -> (mpsc::Receiver<PreEncoded>, Vec<std::thread::JoinHandle<()>>) {
    let batch_size = config.batch_size;
    let shuffle_buffer_size = config.shuffle_buffer_size;
    let prefetch_depth = config.prefetch_depth;

    // Stage 1 → Stage 2: un-encoded record batches.
    let (record_tx, record_rx) = mpsc::sync_channel::<Vec<TrainingRecord>>(prefetch_depth * 2);
    // Stage 2 → Training loop: encoded batches.
    let (batch_tx, batch_rx) = mpsc::sync_channel::<PreEncoded>(prefetch_depth);

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
            if !batch_buf.is_empty() && record_tx.send(batch_buf).is_err() {
                return;
            }

            // Phase 4: Prepare next epoch.
            epoch += 1;
            rng = ChaCha8Rng::seed_from_u64(42 + epoch);
            files.shuffle(&mut rng);
        }
    });

    // Stage 2: Encoder threads — multiple threads pull from the shared record
    // channel, encode via rayon, and push to the batch channel. This keeps CPU
    // cores busy continuously instead of idling between sequential encode calls.
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
                let encoded = PreEncoded::from_records(&records);
                if tx.send(encoded).is_err() {
                    return;
                }
            }
        }));
    }
    drop(batch_tx); // Drop the original sender so channel closes when all encoders exit.

    (batch_rx, handles)
}

/// Train a `CfvNet` using a streaming dataloader.
///
/// A background thread reads records from disk, shuffles them in a buffer,
/// splits into batches, and sends pre-encoded data over a channel. One epoch =
/// `train_records / batch_size` training steps.
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    data_path: &Path,
    board_cards: usize,
    config: &TrainConfig,
    output_dir: Option<&std::path::Path>,
) -> TrainResult {
    let model = CfvNet::<B>::new(device, config.hidden_layers, config.hidden_size, INPUT_SIZE);
    let (mut model, start_epoch) = load_or_create_model(model, output_dir, device);

    let mut optim = AdamConfig::new().init::<B, CfvNet<B>>();

    let files = collect_data_files(data_path).unwrap_or_else(|e| {
        eprintln!("failed to collect data files: {e}");
        std::process::exit(1);
    });

    eprintln!("Counting records across {} file(s)...", files.len());
    let total_records = count_total_records(&files, board_cards) as usize;
    eprintln!("Total records: {total_records}");

    if total_records == 0 {
        eprintln!("No training records found.");
        return TrainResult { final_train_loss: f32::MAX };
    }

    let val_count = ((total_records as f64 * config.validation_split) as usize)
        .min(total_records);
    let val_encoded = load_validation_set(&files, val_count);

    // Compute training schedule.
    let train_records = total_records - val_count;
    let batch_size = config.batch_size;
    let steps_per_epoch = train_records.div_ceil(batch_size);
    let total_steps = steps_per_epoch * config.epochs;

    if start_epoch >= config.epochs {
        eprintln!("Already completed {start_epoch}/{} epochs, nothing to do.", config.epochs);
        return TrainResult { final_train_loss: f32::MAX };
    }

    let remaining_epochs = config.epochs - start_epoch;
    eprintln!(
        "Training: epochs {}-{} ({remaining_epochs} remaining) x {} steps/epoch = {} total steps",
        start_epoch + 1, config.epochs, steps_per_epoch, total_steps
    );

    let (data_rx, loader_threads) = spawn_dataloader_thread(
        &files,
        config,
        val_count,
    );

    // Spawn GPU upload thread to overlap CPU->GPU transfer with compute.
    let gpu_prefetch = config.gpu_prefetch.max(1);
    let (gpu_tx, gpu_rx) = mpsc::sync_channel::<DeviceTensors<B::InnerBackend>>(gpu_prefetch);
    let upload_device = device.clone();
    let upload_handle = std::thread::spawn(move || {
        while let Ok(encoded) = data_rx.recv() {
            let tensors = encoded.into_device_tensors::<B::InnerBackend>(&upload_device);
            if gpu_tx.send(tensors).is_err() {
                break;
            }
        }
    });

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
            // Receive next GPU-resident batch from the upload thread (blocking).
            let batch = match gpu_rx.recv() {
                Ok(b) => b,
                Err(_) => break, // Channel closed unexpectedly.
            };
            let input = Tensor::<B, 2>::from_inner(batch.input);
            let target = Tensor::<B, 2>::from_inner(batch.target);
            let mask = Tensor::<B, 2>::from_inner(batch.mask);
            let range = Tensor::<B, 2>::from_inner(batch.range);
            let game_value = Tensor::<B, 1>::from_inner(batch.game_value);
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

    // Drop gpu channel to signal upload thread to exit, then join all threads.
    drop(gpu_rx);
    if let Err(e) = upload_handle.join() {
        eprintln!("ERROR: GPU upload thread panicked: {e:?}");
    }
    for handle in loader_threads {
        if let Err(e) = handle.join() {
            eprintln!("ERROR: dataloader thread panicked: {e:?}");
        }
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
            encoder_threads: 2,
            gpu_prefetch: 1,
        }
    }

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

    #[test]
    fn train_saves_model_to_output_dir() {
        let file = write_test_data(16);
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let config = default_test_config();
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
            epochs: 4,
            checkpoint_every_n_epochs: 2,
            ..default_test_config()
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
            batch_size: 8,
            epochs: 3,
            validation_split: 0.2, // 4 val samples out of 20
            ..default_test_config()
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
            epochs: 5,
            ..default_test_config()
        };
        let r1 = train::<B>(&device, file.path(), 5, &config, Some(dir.path()));
        let r2 = train::<B>(&device, file.path(), 5, &config, Some(dir.path()));
        assert!(r1.final_train_loss < 10.0);
        assert!(r2.final_train_loss < 10.0);
    }

    #[test]
    fn train_resumes_from_numbered_checkpoint() {
        let file = write_test_data(16);
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();

        // Train 4 epochs with checkpoints every 2.
        let config = TrainConfig {
            epochs: 4,
            checkpoint_every_n_epochs: 2,
            ..default_test_config()
        };
        train::<B>(&device, file.path(), 5, &config, Some(dir.path()));
        assert!(dir.path().join("checkpoint_epoch4.mpk.gz").exists());

        // Resume with 6 total epochs — should pick up from epoch 4.
        let config2 = TrainConfig {
            epochs: 6,
            checkpoint_every_n_epochs: 2,
            ..default_test_config()
        };
        let r2 = train::<B>(&device, file.path(), 5, &config2, Some(dir.path()));
        assert!(r2.final_train_loss < 10.0);
        assert!(dir.path().join("checkpoint_epoch6.mpk.gz").exists());
    }

    #[test]
    fn train_skips_when_already_complete() {
        let file = write_test_data(16);
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();

        // Train 4 epochs with checkpoints every 2.
        let config = TrainConfig {
            epochs: 4,
            checkpoint_every_n_epochs: 2,
            ..default_test_config()
        };
        train::<B>(&device, file.path(), 5, &config, Some(dir.path()));

        // Run again with same epoch count — should detect completion.
        let r2 = train::<B>(&device, file.path(), 5, &config, Some(dir.path()));
        assert_eq!(r2.final_train_loss, f32::MAX);
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

    #[test]
    fn training_with_streaming() {
        // Use more data than shuffle_buffer_size to exercise the streaming path.
        let file = write_test_data(64);
        let device = Default::default();
        let config = TrainConfig {
            batch_size: 8,
            epochs: 5,
            shuffle_buffer_size: 16,
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
            batch_size: 8,
            ..default_test_config()
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
    fn dataloader_stops_on_empty_files() {
        // Create a directory with an empty file (no valid records).
        let dir = tempfile::tempdir().unwrap();
        let empty_path = dir.path().join("empty.bin");
        std::fs::File::create(&empty_path).unwrap();

        let config = TrainConfig {
            batch_size: 8,
            shuffle_buffer_size: 100,
            prefetch_depth: 2,
            ..default_test_config()
        };

        let (rx, handles) = spawn_dataloader_thread(
            &[empty_path],
            &config,
            0,
        );

        // The reader thread should stop after seeing empty buffer,
        // cascading close through the encoder, causing recv() to return Err.
        let result = rx.recv();
        assert!(result.is_err(), "expected channel to close after empty reads");

        // Both threads should have exited cleanly (no panic).
        for handle in handles {
            assert!(handle.join().is_ok(), "dataloader thread should not panic");
        }
    }

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

        let r0 = reader.read_one().unwrap();
        assert!((r0.pot - 0.0).abs() < 1e-6);
        let r1 = reader.read_one().unwrap();
        assert!((r1.pot - 10.0).abs() < 1e-6);
        let r2 = reader.read_one().unwrap();
        assert!((r2.pot - 20.0).abs() < 1e-6);

        assert!(reader.read_one().is_none());

        reader.reset();
        assert!(reader.read_one().is_some());
    }

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

    #[test]
    fn streaming_shuffle_buffer_coverage() {
        // 64 records, shuffle buffer of 16, batch size 8.
        // With streaming eviction, all 64 records should be delivered.
        let file = write_test_data(64);
        let config = TrainConfig {
            batch_size: 8,
            shuffle_buffer_size: 16,
            prefetch_depth: 2,
            epochs: 2,
            ..default_test_config()
        };

        let (rx, handles) = spawn_dataloader_thread(
            &[file.path().to_path_buf()],
            &config,
            0,
        );

        // One epoch = ceil(64 / 8) = 8 batches.
        // Receive two epochs worth (16 batches).
        let mut total_records = 0;
        let mut total_batches = 0;
        for _ in 0..16 {
            match rx.recv() {
                Ok(batch) => {
                    total_records += batch.len;
                    total_batches += 1;
                }
                Err(_) => break,
            }
        }
        assert_eq!(total_batches, 16, "should receive 16 batches for 2 epochs");
        assert_eq!(total_records, 128, "should receive 64 records per epoch x 2");

        drop(rx);
        for handle in handles {
            handle.join().expect("thread should not panic");
        }
    }

}
