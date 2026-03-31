//! GPU inference server with async batching.
//!
//! A dedicated thread owns the [`CfvNet`] model, receives leaf-evaluation
//! requests from CPU solver workers via channels, batches them, runs forward
//! passes, and periodically triggers training steps.
//!
//! Workers call [`InferenceHandle::evaluate`] which blocks until the GPU batch
//! containing their request completes. [`InferenceHandle::notify_solve_complete`]
//! increments the solve counter so the server knows when to train.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use crossbeam_channel::{bounded, Receiver, Sender};

use cfvnet::model::network::{CfvNet, INPUT_SIZE, OUTPUT_SIZE};

use crate::replay_buffer::ReplayBuffer;

/// Request from a solver worker to the inference server.
pub struct InferenceRequest {
    /// Encoded [`INPUT_SIZE`]-element input vector.
    pub input: Vec<f32>,
    /// Channel to send the [`OUTPUT_SIZE`]-element CFV result back on.
    pub response_tx: Sender<Vec<f32>>,
}

/// Configuration for the inference server.
#[derive(Clone, Debug)]
pub struct InferenceServerConfig {
    /// Max requests to batch before running a forward pass.
    pub batch_size: usize,
    /// Max microseconds to wait for a full batch before running a partial one.
    pub batch_timeout_us: u64,
    /// Run one training step after this many subgame solves complete.
    pub train_every_n_solves: usize,
    /// Batch size for training steps.
    pub train_batch_size: usize,
}

/// Handle for workers to submit inference requests.
///
/// Cloneable — each worker thread gets its own copy.
#[derive(Clone)]
pub struct InferenceHandle {
    request_tx: Sender<InferenceRequest>,
    solve_counter: Arc<AtomicUsize>,
}

impl InferenceHandle {
    /// Submit a leaf evaluation request and block until the result arrives.
    ///
    /// The server batches this request with others and runs a single forward
    /// pass on the GPU, then sends the result back via the response channel.
    pub fn evaluate(&self, input: Vec<f32>) -> Vec<f32> {
        let (resp_tx, resp_rx) = bounded(1);
        self.request_tx
            .send(InferenceRequest {
                input,
                response_tx: resp_tx,
            })
            .expect("inference server shut down");
        resp_rx.recv().expect("inference server dropped response")
    }

    /// Notify the server that one subgame solve completed.
    ///
    /// Call this after extracting root CFVs and pushing to the replay buffer.
    /// The server uses this counter to decide when to run training steps.
    pub fn notify_solve_complete(&self) {
        self.solve_counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Return current solve count (for diagnostics / testing).
    pub fn solve_count(&self) -> usize {
        self.solve_counter.load(Ordering::Relaxed)
    }
}

/// Spawn the inference server on a dedicated thread.
///
/// Returns an [`InferenceHandle`] that workers use to submit requests,
/// and a [`JoinHandle`](std::thread::JoinHandle) for the server thread.
///
/// The server runs until `shutdown` is set to `true` and all pending
/// requests have been drained, or until the request channel disconnects.
pub fn spawn_inference_server<B: Backend + 'static>(
    model: CfvNet<B>,
    device: B::Device,
    config: InferenceServerConfig,
    replay_buffer: Arc<ReplayBuffer>,
    shutdown: Arc<AtomicBool>,
) -> (InferenceHandle, std::thread::JoinHandle<()>)
where
    B::Device: Send,
    CfvNet<B>: Send,
{
    let (request_tx, request_rx) = crossbeam_channel::unbounded();
    let solve_counter = Arc::new(AtomicUsize::new(0));
    let handle = InferenceHandle {
        request_tx,
        solve_counter: Arc::clone(&solve_counter),
    };

    let thread = std::thread::spawn(move || {
        run_server_loop(
            model,
            device,
            config,
            request_rx,
            solve_counter,
            replay_buffer,
            shutdown,
        );
    });

    (handle, thread)
}

/// Main server loop: collect requests -> batch forward pass -> send responses.
///
/// Periodically checks whether a training step is due.
fn run_server_loop<B: Backend>(
    model: CfvNet<B>,
    device: B::Device,
    config: InferenceServerConfig,
    request_rx: Receiver<InferenceRequest>,
    solve_counter: Arc<AtomicUsize>,
    replay_buffer: Arc<ReplayBuffer>,
    shutdown: Arc<AtomicBool>,
) {
    let timeout = Duration::from_micros(config.batch_timeout_us);
    let mut last_train_at = 0usize;

    while !shutdown.load(Ordering::Relaxed) {
        // Collect a batch of requests.
        let mut batch: Vec<InferenceRequest> = Vec::with_capacity(config.batch_size);

        // Block on first request (or check shutdown periodically).
        match request_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(req) => batch.push(req),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // No requests — check if training is due.
                maybe_train(
                    &config,
                    &replay_buffer,
                    &solve_counter,
                    &mut last_train_at,
                );
                continue;
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }

        // Try to fill the batch up to batch_size or timeout.
        let deadline = std::time::Instant::now() + timeout;
        while batch.len() < config.batch_size {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match request_rx.recv_timeout(remaining) {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }
        }

        if batch.is_empty() {
            continue;
        }

        // Run batched forward pass.
        let batch_len = batch.len();
        let mut flat_input: Vec<f32> = Vec::with_capacity(batch_len * INPUT_SIZE);
        for req in &batch {
            flat_input.extend_from_slice(&req.input);
        }

        let input_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(flat_input, [batch_len, INPUT_SIZE]),
            &device,
        );

        let output_tensor = model.forward(input_tensor);
        let output_data: Vec<f32> = output_tensor
            .into_data()
            .to_vec::<f32>()
            .expect("output tensor conversion");

        // Send results back to each worker.
        for (i, req) in batch.into_iter().enumerate() {
            let start = i * OUTPUT_SIZE;
            let cfvs = output_data[start..start + OUTPUT_SIZE].to_vec();
            let _ = req.response_tx.send(cfvs);
        }

        // Check if training is due.
        maybe_train(
            &config,
            &replay_buffer,
            &solve_counter,
            &mut last_train_at,
        );
    }
}

/// Run one training step if enough solves have completed since the last one.
///
/// TODO (Task 5): Replace this stub with actual training logic that requires
/// an `AutodiffBackend`, optimizer state, and loss function. For now, this
/// just logs that training would happen and updates the counter.
fn maybe_train(
    config: &InferenceServerConfig,
    replay_buffer: &ReplayBuffer,
    solve_counter: &AtomicUsize,
    last_train_at: &mut usize,
) {
    let current = solve_counter.load(Ordering::Relaxed);
    if current.wrapping_sub(*last_train_at) >= config.train_every_n_solves
        && replay_buffer.len() >= config.train_batch_size
    {
        // Sample batch from replay buffer.
        let samples = replay_buffer.sample(config.train_batch_size);

        // TODO (Task 5): Build input/target tensors and run one training step.
        // This requires an AutodiffBackend, optimizer state, and loss function.
        // Full training integration is Task 5.
        eprintln!(
            "  [InferenceServer] Training step stub: {} samples, solves={}",
            samples.len(),
            current
        );

        *last_train_at = current;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_request_response_roundtrip() {
        let (resp_tx, resp_rx) = crossbeam_channel::bounded(1);
        let req = InferenceRequest {
            input: vec![1.0; INPUT_SIZE],
            response_tx: resp_tx,
        };
        // Simulate server sending response.
        req.response_tx.send(vec![0.5; OUTPUT_SIZE]).unwrap();
        let result = resp_rx.recv().unwrap();
        assert_eq!(result.len(), OUTPUT_SIZE);
        assert_eq!(result[0], 0.5);
    }

    #[test]
    fn test_server_config_defaults() {
        let config = InferenceServerConfig {
            batch_size: 256,
            batch_timeout_us: 100,
            train_every_n_solves: 50,
            train_batch_size: 512,
        };
        assert_eq!(config.batch_size, 256);
        assert_eq!(config.batch_timeout_us, 100);
        assert_eq!(config.train_every_n_solves, 50);
        assert_eq!(config.train_batch_size, 512);
    }

    #[test]
    fn test_inference_handle_evaluate_end_to_end() {
        use burn::backend::NdArray;
        type TestBackend = NdArray;

        let device = Default::default();
        // Use a small model for fast tests.
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let replay_buffer = Arc::new(ReplayBuffer::new(100));
        let shutdown = Arc::new(AtomicBool::new(false));

        let (handle, server_thread) = spawn_inference_server(
            model,
            device,
            InferenceServerConfig {
                batch_size: 4,
                batch_timeout_us: 1000,
                train_every_n_solves: 100,
                train_batch_size: 32,
            },
            replay_buffer,
            Arc::clone(&shutdown),
        );

        // Submit a single request.
        let input = vec![0.1_f32; INPUT_SIZE];
        let result = handle.evaluate(input);
        assert_eq!(result.len(), OUTPUT_SIZE, "should return 1326 CFVs");

        // All values should be finite.
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "CFV[{i}] is not finite: {v}");
        }

        // Shutdown.
        shutdown.store(true, Ordering::Relaxed);
        server_thread.join().expect("server thread should not panic");
    }

    #[test]
    fn test_inference_handle_concurrent_workers() {
        use burn::backend::NdArray;
        type TestBackend = NdArray;

        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let replay_buffer = Arc::new(ReplayBuffer::new(100));
        let shutdown = Arc::new(AtomicBool::new(false));

        let (handle, server_thread) = spawn_inference_server(
            model,
            device,
            InferenceServerConfig {
                batch_size: 8,
                batch_timeout_us: 5000,
                train_every_n_solves: 100,
                train_batch_size: 32,
            },
            replay_buffer,
            Arc::clone(&shutdown),
        );

        // Spawn 4 worker threads, each submitting 3 requests.
        let mut worker_handles = Vec::new();
        for worker_id in 0..4 {
            let h = handle.clone();
            worker_handles.push(std::thread::spawn(move || {
                for req_id in 0..3 {
                    let input = vec![(worker_id * 10 + req_id) as f32; INPUT_SIZE];
                    let result = h.evaluate(input);
                    assert_eq!(result.len(), OUTPUT_SIZE);
                    for &v in &result {
                        assert!(v.is_finite());
                    }
                }
            }));
        }

        // Wait for all workers to finish.
        for h in worker_handles {
            h.join().expect("worker thread should not panic");
        }

        // Shutdown.
        shutdown.store(true, Ordering::Relaxed);
        server_thread.join().expect("server thread should not panic");
    }

    #[test]
    fn test_notify_solve_complete_increments_counter() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let handle = InferenceHandle {
            request_tx: tx,
            solve_counter: Arc::new(AtomicUsize::new(0)),
        };

        assert_eq!(handle.solve_count(), 0);
        handle.notify_solve_complete();
        assert_eq!(handle.solve_count(), 1);
        handle.notify_solve_complete();
        handle.notify_solve_complete();
        assert_eq!(handle.solve_count(), 3);
    }

    #[test]
    fn test_maybe_train_fires_when_threshold_met() {
        let config = InferenceServerConfig {
            batch_size: 256,
            batch_timeout_us: 100,
            train_every_n_solves: 2,
            train_batch_size: 4,
        };
        let replay_buffer = ReplayBuffer::new(100);
        // Fill replay buffer with enough entries.
        for i in 0..10 {
            replay_buffer.push(crate::replay_buffer::ReplayEntry {
                input: vec![i as f32; INPUT_SIZE],
                target: vec![0.0; OUTPUT_SIZE],
            });
        }

        let solve_counter = AtomicUsize::new(5);
        let mut last_train_at = 0;

        // Should trigger training (5 - 0 = 5 >= 2, and buffer has 10 >= 4).
        maybe_train(&config, &replay_buffer, &solve_counter, &mut last_train_at);
        assert_eq!(last_train_at, 5, "last_train_at should update to current solve count");
    }

    #[test]
    fn test_maybe_train_skips_when_below_threshold() {
        let config = InferenceServerConfig {
            batch_size: 256,
            batch_timeout_us: 100,
            train_every_n_solves: 10,
            train_batch_size: 4,
        };
        let replay_buffer = ReplayBuffer::new(100);
        for i in 0..10 {
            replay_buffer.push(crate::replay_buffer::ReplayEntry {
                input: vec![i as f32; INPUT_SIZE],
                target: vec![0.0; OUTPUT_SIZE],
            });
        }

        let solve_counter = AtomicUsize::new(3);
        let mut last_train_at = 0;

        // Should NOT trigger (3 - 0 = 3 < 10).
        maybe_train(&config, &replay_buffer, &solve_counter, &mut last_train_at);
        assert_eq!(last_train_at, 0, "last_train_at should not change");
    }

    #[test]
    fn test_maybe_train_skips_when_buffer_too_small() {
        let config = InferenceServerConfig {
            batch_size: 256,
            batch_timeout_us: 100,
            train_every_n_solves: 1,
            train_batch_size: 100,
        };
        let replay_buffer = ReplayBuffer::new(1000);
        // Only 5 entries, but train_batch_size requires 100.
        for i in 0..5 {
            replay_buffer.push(crate::replay_buffer::ReplayEntry {
                input: vec![i as f32; INPUT_SIZE],
                target: vec![0.0; OUTPUT_SIZE],
            });
        }

        let solve_counter = AtomicUsize::new(10);
        let mut last_train_at = 0;

        maybe_train(&config, &replay_buffer, &solve_counter, &mut last_train_at);
        assert_eq!(last_train_at, 0, "should not train with insufficient buffer");
    }

    #[test]
    fn test_server_shutdown_clean() {
        use burn::backend::NdArray;
        type TestBackend = NdArray;

        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let replay_buffer = Arc::new(ReplayBuffer::new(100));
        let shutdown = Arc::new(AtomicBool::new(false));

        let (_handle, server_thread) = spawn_inference_server(
            model,
            device,
            InferenceServerConfig {
                batch_size: 4,
                batch_timeout_us: 100,
                train_every_n_solves: 100,
                train_batch_size: 32,
            },
            replay_buffer,
            Arc::clone(&shutdown),
        );

        // Immediately signal shutdown without sending any requests.
        shutdown.store(true, Ordering::Relaxed);
        server_thread.join().expect("server should shut down cleanly");
    }

    #[test]
    fn test_server_handles_different_inputs() {
        use burn::backend::NdArray;
        type TestBackend = NdArray;

        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let replay_buffer = Arc::new(ReplayBuffer::new(100));
        let shutdown = Arc::new(AtomicBool::new(false));

        let (handle, server_thread) = spawn_inference_server(
            model,
            device,
            InferenceServerConfig {
                batch_size: 4,
                batch_timeout_us: 5000,
                train_every_n_solves: 100,
                train_batch_size: 32,
            },
            replay_buffer,
            Arc::clone(&shutdown),
        );

        // Two different inputs should produce different outputs.
        let result_zeros = handle.evaluate(vec![0.0; INPUT_SIZE]);
        let result_ones = handle.evaluate(vec![1.0; INPUT_SIZE]);

        let diff: f32 = result_zeros
            .iter()
            .zip(result_ones.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "different inputs should produce different outputs, diff={diff}"
        );

        shutdown.store(true, Ordering::Relaxed);
        server_thread.join().expect("server thread should not panic");
    }
}
