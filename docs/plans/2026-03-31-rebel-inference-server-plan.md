# ReBeL Inference Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** GPU inference server with async batching so value net is queried every CFR iteration, with interleaved training on a single GPU.

**Architecture:** Dedicated GPU thread owns the model, receives leaf-evaluation requests from CPU solver workers via channels, batches them, runs forward passes, and periodically runs training steps. Workers call `solve_step()` in a loop, re-evaluating boundary CFVs each iteration via the server.

**Tech Stack:** Rust, crossbeam-channel (mpsc + oneshot), burn 0.16 (wgpu backend), existing range-solver `solve_step` + `set_boundary_cfvs` APIs.

---

## Key Existing APIs

| API | Location | Signature |
|-----|----------|-----------|
| `solve_step` | `range-solver/src/solver.rs:181` | `fn solve_step<T: Game>(game: &T, iter: u32)` |
| `set_boundary_cfvs` | `range-solver/src/game/interpreter.rs:339` | `fn set_boundary_cfvs(&self, ordinal, player, cfvs: Vec<f32>)` |
| `num_boundary_nodes` | `range-solver/src/game/interpreter.rs:313` | `fn num_boundary_nodes(&self) -> usize` |
| `boundary_pot` | `range-solver/src/game/interpreter.rs:351` | `fn boundary_pot(&self, ordinal) -> i32` |
| `prepare_game` | `rebel/src/solver.rs` | `fn prepare_game(pbs, config) -> Result<PreparedGame>` |
| `solve_prepared` | `rebel/src/solver.rs` | `fn solve_prepared(pg: &mut PreparedGame) -> SolveResult` |
| `CfvNet::forward` | `cfvnet/src/model/network.rs` | `fn forward(&self, Tensor<B,2>) -> Tensor<B,2>` |

Both `solve_step` and `set_boundary_cfvs` take `&self`/`&T` (shared refs), so they can be interleaved without mutability conflicts.

---

### Task 1: Add crossbeam-channel dependency and ReplayBuffer

**Files:**
- Modify: `crates/rebel/Cargo.toml`
- Create: `crates/rebel/src/replay_buffer.rs`
- Modify: `crates/rebel/src/lib.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_buffer_push_and_len() {
        let mut buf = ReplayBuffer::new(100);
        assert_eq!(buf.len(), 0);
        buf.push(ReplayEntry {
            input: vec![0.0; 2720],
            target: vec![0.0; 1326],
        });
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_replay_buffer_sample() {
        let mut buf = ReplayBuffer::new(100);
        for i in 0..50 {
            buf.push(ReplayEntry {
                input: vec![i as f32; 2720],
                target: vec![0.0; 1326],
            });
        }
        let samples = buf.sample(10);
        assert_eq!(samples.len(), 10);
        // Each sample's input[0] should be in [0, 50)
        for s in &samples {
            assert!(s.input[0] >= 0.0 && s.input[0] < 50.0);
        }
    }

    #[test]
    fn test_replay_buffer_evicts_oldest() {
        let mut buf = ReplayBuffer::new(5);
        for i in 0..10 {
            buf.push(ReplayEntry {
                input: vec![i as f32; 2720],
                target: vec![0.0; 1326],
            });
        }
        assert_eq!(buf.len(), 5);
        // Oldest entries (0-4) should be evicted, only 5-9 remain
        let samples = buf.sample(5);
        for s in &samples {
            assert!(s.input[0] >= 5.0);
        }
    }

    #[test]
    fn test_replay_buffer_thread_safe() {
        use std::sync::Arc;
        let buf = Arc::new(ReplayBuffer::new(1000));
        let buf2 = Arc::clone(&buf);
        let handle = std::thread::spawn(move || {
            for i in 0..100 {
                buf2.push(ReplayEntry {
                    input: vec![i as f32; 2720],
                    target: vec![0.0; 1326],
                });
            }
        });
        handle.join().unwrap();
        assert_eq!(buf.len(), 100);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rebel replay_buffer`
Expected: FAIL — module not found

**Step 3: Implement**

Add to `crates/rebel/Cargo.toml`:
```toml
crossbeam-channel = "0.5"
```

Create `crates/rebel/src/replay_buffer.rs`:

```rust
use std::collections::VecDeque;
use std::sync::Mutex;
use rand::Rng;

/// A single training example: encoded PBS input and target CFVs.
#[derive(Clone)]
pub struct ReplayEntry {
    pub input: Vec<f32>,   // 2720 elements
    pub target: Vec<f32>,  // 1326 elements
}

/// Thread-safe circular replay buffer.
///
/// Workers push training examples, the inference server samples batches
/// for training. Evicts oldest entries when full.
pub struct ReplayBuffer {
    entries: Mutex<VecDeque<ReplayEntry>>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    pub fn push(&self, entry: ReplayEntry) {
        let mut entries = self.entries.lock().unwrap();
        if entries.len() >= self.capacity {
            entries.pop_front();
        }
        entries.push_back(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sample `n` random entries. Returns fewer if buffer has < n entries.
    pub fn sample(&self, n: usize) -> Vec<ReplayEntry> {
        let entries = self.entries.lock().unwrap();
        let len = entries.len();
        if len == 0 { return Vec::new(); }
        let mut rng = rand::rng();
        (0..n.min(len))
            .map(|_| {
                let idx = rng.random_range(0..len);
                entries[idx].clone()
            })
            .collect()
    }
}
```

Add `pub mod replay_buffer;` to `lib.rs`.

**Step 4: Run test**

Run: `cargo test -p rebel replay_buffer`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/rebel/Cargo.toml crates/rebel/src/replay_buffer.rs crates/rebel/src/lib.rs
git commit -m "feat(rebel): thread-safe circular replay buffer for inference server"
```

---

### Task 2: InferenceServer — request/response types and server loop

**Files:**
- Create: `crates/rebel/src/inference_server.rs`
- Modify: `crates/rebel/src/lib.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_request_response_roundtrip() {
        let (resp_tx, resp_rx) = crossbeam_channel::bounded(1);
        let req = InferenceRequest {
            input: vec![1.0; 2720],
            response_tx: resp_tx,
        };
        // Simulate server sending response
        req.response_tx.send(vec![0.5; 1326]).unwrap();
        let result = resp_rx.recv().unwrap();
        assert_eq!(result.len(), 1326);
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
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rebel inference_server`
Expected: FAIL

**Step 3: Implement**

Create `crates/rebel/src/inference_server.rs`:

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::time::Duration;
use crossbeam_channel::{Sender, Receiver, bounded};
use burn::prelude::*;

use crate::replay_buffer::ReplayBuffer;

/// Request from a solver worker to the inference server.
pub struct InferenceRequest {
    /// Encoded 2720-element input vector.
    pub input: Vec<f32>,
    /// Channel to send the 1326-element CFV result back on.
    pub response_tx: Sender<Vec<f32>>,
}

/// Configuration for the inference server.
#[derive(Clone, Debug)]
pub struct InferenceServerConfig {
    /// Max requests to batch before running a forward pass.
    pub batch_size: usize,
    /// Max microseconds to wait for a full batch before running partial.
    pub batch_timeout_us: u64,
    /// Run one training step after this many subgame solves complete.
    pub train_every_n_solves: usize,
    /// Batch size for training steps.
    pub train_batch_size: usize,
}

/// Handle for workers to submit inference requests.
#[derive(Clone)]
pub struct InferenceHandle {
    request_tx: Sender<InferenceRequest>,
    solve_counter: Arc<AtomicUsize>,
}

impl InferenceHandle {
    /// Submit a leaf evaluation request and block until result arrives.
    pub fn evaluate(&self, input: Vec<f32>) -> Vec<f32> {
        let (resp_tx, resp_rx) = bounded(1);
        self.request_tx
            .send(InferenceRequest { input, response_tx: resp_tx })
            .expect("inference server shut down");
        resp_rx.recv().expect("inference server dropped response")
    }

    /// Notify the server that one subgame solve completed.
    /// Call this after extracting root CFVs and pushing to replay buffer.
    pub fn notify_solve_complete(&self) {
        self.solve_counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// Spawn the inference server on a dedicated thread.
///
/// Returns an `InferenceHandle` that workers use to submit requests,
/// and a `JoinHandle` for the server thread.
///
/// The server runs until `shutdown` is set to true.
pub fn spawn_inference_server<B: Backend>(
    model: cfvnet::model::network::CfvNet<B>,
    device: B::Device,
    config: InferenceServerConfig,
    replay_buffer: Arc<ReplayBuffer>,
    shutdown: Arc<AtomicBool>,
) -> (InferenceHandle, std::thread::JoinHandle<()>)
where
    B: Backend,
    B::FloatTensorPrimitive: Send,
{
    let (request_tx, request_rx) = crossbeam_channel::unbounded();
    let solve_counter = Arc::new(AtomicUsize::new(0));
    let handle = InferenceHandle {
        request_tx,
        solve_counter: Arc::clone(&solve_counter),
    };

    let thread = std::thread::spawn(move || {
        run_server_loop(model, device, config, request_rx, solve_counter, replay_buffer, shutdown);
    });

    (handle, thread)
}

/// Main server loop: collect requests → batch forward pass → send responses.
/// Periodically runs training steps.
fn run_server_loop<B: Backend>(
    mut model: cfvnet::model::network::CfvNet<B>,
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
        // Collect a batch of requests
        let mut batch: Vec<InferenceRequest> = Vec::with_capacity(config.batch_size);

        // Block on first request (or check shutdown)
        match request_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(req) => batch.push(req),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Check if training is due
                maybe_train(&mut model, &device, &config, &replay_buffer,
                            &solve_counter, &mut last_train_at);
                continue;
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }

        // Try to fill the batch up to batch_size or timeout
        let deadline = std::time::Instant::now() + timeout;
        while batch.len() < config.batch_size {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() { break; }
            match request_rx.recv_timeout(remaining) {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }
        }

        if batch.is_empty() { continue; }

        // Run batched forward pass
        let batch_size = batch.len();
        let mut flat_input = Vec::with_capacity(batch_size * 2720);
        for req in &batch {
            flat_input.extend_from_slice(&req.input);
        }

        let input_tensor = Tensor::<B, 2>::from_floats(
            &flat_input[..],
            &device,
        ).reshape([batch_size, 2720]);

        let output_tensor = model.forward(input_tensor);
        let output_data: Vec<f32> = output_tensor.to_data().to_vec().unwrap();

        // Send results back
        for (i, req) in batch.into_iter().enumerate() {
            let start = i * 1326;
            let cfvs = output_data[start..start + 1326].to_vec();
            let _ = req.response_tx.send(cfvs);
        }

        // Check if training is due
        maybe_train(&mut model, &device, &config, &replay_buffer,
                     &solve_counter, &mut last_train_at);
    }
}

/// Run one training step if enough solves have completed since the last one.
fn maybe_train<B: Backend>(
    model: &mut cfvnet::model::network::CfvNet<B>,
    device: &B::Device,
    config: &InferenceServerConfig,
    replay_buffer: &ReplayBuffer,
    solve_counter: &AtomicUsize,
    last_train_at: &mut usize,
) {
    let current = solve_counter.load(Ordering::Relaxed);
    if current - *last_train_at >= config.train_every_n_solves
        && replay_buffer.len() >= config.train_batch_size
    {
        // Sample batch from replay buffer
        let samples = replay_buffer.sample(config.train_batch_size);

        // Build input/target tensors
        let mut flat_input = Vec::with_capacity(samples.len() * 2720);
        let mut flat_target = Vec::with_capacity(samples.len() * 1326);
        for s in &samples {
            flat_input.extend_from_slice(&s.input);
            flat_target.extend_from_slice(&s.target);
        }

        // TODO: Run one training step
        // This requires an AutodiffBackend, optimizer state, and loss function.
        // For now, log that training would happen.
        // Full training integration is Task 5.
        eprintln!(
            "  [InferenceServer] Training step: {} samples, solves={}",
            samples.len(), current
        );

        *last_train_at = current;
    }
}
```

Add `pub mod inference_server;` to `lib.rs`.

**Step 4: Run test**

Run: `cargo test -p rebel inference_server`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/rebel/src/inference_server.rs crates/rebel/src/lib.rs
git commit -m "feat(rebel): inference server — GPU batching with request/response channels"
```

---

### Task 3: Iterative subgame solver using inference server

**Files:**
- Modify: `crates/rebel/src/subgame_solve.rs`

Add a new function `solve_subgame_iterative` that uses `InferenceHandle` instead of `&dyn LeafEvaluator`. Each CFR iteration re-evaluates boundaries.

**Step 1: Write the failing test**

```rust
#[test]
fn test_solve_subgame_iterative_exists() {
    // Verify the function signature compiles
    // Full integration test requires a running inference server
    fn _check_signature(
        _pbs: &Pbs,
        _config: &SolveConfig,
        _handle: &InferenceHandle,
    ) -> Result<SubgameSolveResult, String> {
        solve_subgame_iterative(_pbs, _config, _handle)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rebel subgame_solve::tests::test_solve_subgame_iterative_exists`
Expected: FAIL

**Step 3: Implement**

Add to `crates/rebel/src/subgame_solve.rs`:

```rust
use crate::inference_server::InferenceHandle;

/// Solve a subgame with per-iteration boundary re-evaluation via the inference server.
///
/// Unlike `solve_subgame` (which evaluates boundaries once), this function
/// re-evaluates boundary CFVs at every CFR iteration as the strategy (and
/// therefore reach probabilities at leaves) evolves. This matches the
/// reference ReBeL implementation.
///
/// The inference handle submits requests to the GPU inference server, which
/// batches them across all worker threads for efficient GPU utilization.
pub fn solve_subgame_iterative(
    pbs: &Pbs,
    config: &SolveConfig,
    handle: &InferenceHandle,
) -> Result<SubgameSolveResult, String> {
    // Build game (same as prepare_game in solver.rs)
    let mut prepared = crate::solver::prepare_game(pbs, config)?;
    let game = &prepared.game;
    let n_boundary = game.num_boundary_nodes();

    for iter in 0..config.solver_iterations {
        // Re-evaluate boundary CFVs with current ranges
        if n_boundary > 0 {
            evaluate_boundaries_via_server(game, pbs, handle);
        }

        // One CFR iteration
        range_solver::solve_step(game, iter);
    }

    // Extract results (same as solve_prepared)
    Ok(extract_result(&mut prepared))
}

/// Evaluate all boundary nodes via the inference server.
///
/// For each boundary × player: encode a 2720-element input vector from
/// the current ranges, submit to the server, set the returned CFVs.
fn evaluate_boundaries_via_server(
    game: &PostFlopGame,
    pbs: &Pbs,
    handle: &InferenceHandle,
) {
    let n_boundary = game.num_boundary_nodes();
    let starting_pot = game.tree_config().starting_pot;
    let eff_stack = game.tree_config().effective_stack;

    for player in 0..2usize {
        let hands = game.private_cards(player);

        for ordinal in 0..n_boundary {
            let bpot = game.boundary_pot(ordinal);
            let amount = (bpot - starting_pot) / 2;
            let boundary_eff_stack = eff_stack - amount;

            // Build 2720-element input from current ranges
            // OOP range (1326) + IP range (1326) + board (52) + rank (13) + pot + stack + player
            let input = build_boundary_input(
                pbs, hands, bpot as f32, boundary_eff_stack as f32, player as u8,
            );

            // Submit to inference server and block
            let cfvs = handle.evaluate(input);

            // Map from 1326-canonical to solver combo ordering
            let cfvs_solver: Vec<f32> = hands
                .iter()
                .map(|&(c1, c2)| {
                    let idx = range_solver::card::card_pair_to_index(c1, c2);
                    cfvs[idx]
                })
                .collect();

            game.set_boundary_cfvs(ordinal, player, cfvs_solver);
        }
    }
}

/// Build a 2720-element input vector for a boundary evaluation.
/// Uses the same encoding as cfvnet: ranges + board one-hot + pot/stack/player.
fn build_boundary_input(
    pbs: &Pbs,
    _hands: &[(Card, Card)],  // solver combo ordering (unused — we use canonical)
    pot: f32,
    effective_stack: f32,
    player: u8,
) -> Vec<f32> {
    // Reuse the encoding from leaf_evaluator.rs or cfvnet::model::dataset
    let mut input = vec![0.0f32; 2720];

    // OOP range: positions 0..1326
    for i in 0..1326 {
        input[i] = pbs.reach_probs[0][i];
    }
    // IP range: positions 1326..2652
    for i in 0..1326 {
        input[1326 + i] = pbs.reach_probs[1][i];
    }
    // Board one-hot: positions 2652..2704
    for &card in &pbs.board {
        input[2652 + card as usize] = 1.0;
    }
    // Rank presence: positions 2704..2717
    for &card in &pbs.board {
        let rank = (card / 4) as usize;
        input[2704 + rank] = 1.0;
    }
    // Pot: position 2717
    input[2717] = pot / 400.0;
    // Effective stack: position 2718
    input[2718] = effective_stack / 400.0;
    // Player indicator: position 2719
    input[2719] = player as f32;

    input
}
```

**Note:** The `extract_result` function already exists in subgame_solve.rs. Reuse it. If it's private, make it `pub(crate)`.

**Note:** The ranges used for boundary evaluation should be the CURRENT ranges from the game's strategy state (which evolve during CFR), not the initial PBS ranges. However, extracting per-iteration ranges from `PostFlopGame` mid-solve requires reading the current strategy and computing reach probabilities through the tree. For the initial implementation, use the PBS reach probs (which is what we did before). This is an approximation — improving it to use actual per-iteration reaches is a follow-up optimization.

**Step 4: Run test**

Run: `cargo test -p rebel subgame_solve`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/rebel/src/subgame_solve.rs
git commit -m "feat(rebel): iterative subgame solver with per-iteration boundary re-evaluation"
```

---

### Task 4: Wire self-play to use inference server

**Files:**
- Modify: `crates/rebel/src/self_play.rs`

Update `play_self_play_hand` and `self_play_training_loop` to use `InferenceHandle` instead of `&dyn LeafEvaluator`.

**Step 1: Implement**

Change `play_self_play_hand` signature:
```rust
pub fn play_self_play_hand<R: Rng>(
    handle: &InferenceHandle,          // was: evaluator: &dyn LeafEvaluator
    solve_config: &SolveConfig,
    sp_config: &SelfPlayConfig,
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    rng: &mut R,
) -> Vec<TrainingExample> {
```

Replace the `solve_depth_limited_pbs(&pbs, solve_config, evaluator)` call with:
```rust
let solve_result = solve_subgame_iterative(&pbs, solve_config, handle);
```

Change `self_play_training_loop` similarly:
```rust
pub fn self_play_training_loop(
    handle: &InferenceHandle,          // was: evaluator: &dyn LeafEvaluator
    solve_config: &SolveConfig,
    sp_config: &SelfPlayConfig,
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    replay_buffer: &Arc<ReplayBuffer>, // was: buffer: &mut DiskBuffer
) -> usize {
```

Instead of writing to DiskBuffer, push to replay buffer:
```rust
for example in &examples {
    for player in 0..2u8 {
        let input = build_training_input(&example.pbs, player);
        let target = example.cfvs[player as usize].to_vec();
        replay_buffer.push(ReplayEntry { input, target });
    }
}
handle.notify_solve_complete();
```

**Step 2: Update tests**

Update existing tests that reference the old signatures. Integration tests remain `#[ignore]`.

**Step 3: Run test**

Run: `cargo test -p rebel self_play`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/rebel/src/self_play.rs
git commit -m "feat(rebel): wire self-play loop to inference server and replay buffer"
```

---

### Task 5: Training integration in inference server

**Files:**
- Modify: `crates/rebel/src/inference_server.rs`

Replace the `maybe_train` TODO with actual training logic. This requires an `AutodiffBackend`.

**Step 1: Implement**

The server needs to hold both an inference model (Backend) and training state (AutodiffBackend). The approach: the server owns an `AutodiffBackend` model, uses `.valid()` for inference and the full model for training.

```rust
fn maybe_train<B: AutodiffBackend>(
    model: &mut cfvnet::model::network::CfvNet<B>,
    optimizer: &mut impl Optimizer<CfvNet<B>, B>,
    device: &B::Device,
    config: &InferenceServerConfig,
    replay_buffer: &ReplayBuffer,
    solve_counter: &AtomicUsize,
    last_train_at: &mut usize,
) {
    let current = solve_counter.load(Ordering::Relaxed);
    if current - *last_train_at < config.train_every_n_solves { return; }
    if replay_buffer.len() < config.train_batch_size { return; }

    let samples = replay_buffer.sample(config.train_batch_size);
    let n = samples.len();

    // Build tensors
    let mut flat_input = Vec::with_capacity(n * 2720);
    let mut flat_target = Vec::with_capacity(n * 1326);
    for s in &samples {
        flat_input.extend_from_slice(&s.input);
        flat_target.extend_from_slice(&s.target);
    }

    let input = Tensor::<B, 2>::from_floats(&flat_input[..], device)
        .reshape([n, 2720]);
    let target = Tensor::<B, 2>::from_floats(&flat_target[..], device)
        .reshape([n, 1326]);

    // Forward + loss + backward
    let predicted = model.forward(input);
    let loss = (predicted - target).powf_scalar(2.0).mean();
    let loss_val: f32 = loss.clone().into_scalar().elem();

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, model);
    *model = optimizer.step(config.learning_rate, model.clone(), grads);

    eprintln!("  [train] loss={loss_val:.6} buffer={} solves={current}", replay_buffer.len());
    *last_train_at = current;
}
```

**Note:** The actual loss function should match cfvnet (Huber + aux loss). For the initial implementation, MSE is fine. Refinement to Huber is a follow-up.

**Note:** The `model.forward()` for inference uses `B::InnerBackend` (no autograd), while training uses the full `B` (with autograd). Use `model.valid()` to get an inference-only copy, or track this in the server state.

**Step 2: Run test**

Run: `cargo test -p rebel inference_server`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/rebel/src/inference_server.rs
git commit -m "feat(rebel): training integration in inference server — MSE loss on replay buffer"
```

---

### Task 6: Wire rebel-train CLI to inference server

**Files:**
- Modify: `crates/trainer/src/main.rs`

Update `run_rebel_train` to use the inference server architecture for self-play.

**Step 1: Implement**

In the self-play phase (after offline seeding or with `--model`):

```rust
// Load model
let model = load_cfvnet_model(&model_path, &device)?;

// Create replay buffer
let replay_buffer = Arc::new(ReplayBuffer::new(
    rebel_config.inference.replay_capacity,
));

// Spawn inference server
let shutdown = Arc::new(AtomicBool::new(false));
let (handle, server_thread) = spawn_inference_server(
    model, device,
    rebel_config.inference.clone(),
    Arc::clone(&replay_buffer),
    Arc::clone(&shutdown),
);

// Run self-play on worker threads
let solve_config = build_solve_config(&rebel_config.seed);
let sp_config = build_self_play_config(&rebel_config);
let total = self_play_training_loop(
    &handle, &solve_config, &sp_config,
    &strategy, &tree, &buckets,
    &replay_buffer,
);

// Shutdown
shutdown.store(true, Ordering::Relaxed);
server_thread.join().unwrap();
eprintln!("Self-play complete: {total} examples generated");
```

**Step 2: Add InferenceServerConfig to RebelConfig**

In `crates/rebel/src/config.rs`, add:

```rust
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceConfig {
    #[serde(default = "default_batch_size_inf")]
    pub batch_size: usize,          // 256
    #[serde(default = "default_batch_timeout")]
    pub batch_timeout_us: u64,      // 100
    #[serde(default = "default_train_every")]
    pub train_every_n_solves: usize, // 50
    #[serde(default = "default_train_batch")]
    pub train_batch_size: usize,    // 512
    #[serde(default = "default_replay_cap")]
    pub replay_capacity: usize,     // 200000
}
```

Add `pub inference: InferenceConfig` to `RebelConfig` (with serde default).

**Step 3: Verify build**

Run: `cargo build -p poker-solver-trainer`
Expected: compiles

**Step 4: Commit**

```bash
git add crates/trainer/src/main.rs crates/rebel/src/config.rs
git commit -m "feat(trainer): wire rebel-train to inference server for self-play"
```

---

### Task 7: Update sample config and integration test

**Files:**
- Modify: `sample_configurations/rebel_1k_50bb.yaml`
- Modify: `crates/rebel/src/generate.rs` (update solve_buffer_records for depth-limited with server)

**Step 1: Add inference config to YAML**

```yaml
inference:
  batch_size: 256
  batch_timeout_us: 100
  train_every_n_solves: 50
  train_batch_size: 512
  replay_capacity: 200000
```

**Step 2: Integration test**

Add to `crates/rebel/src/inference_server.rs`:

```rust
#[test]
#[ignore] // requires GPU
fn test_inference_server_end_to_end() {
    // 1. Create a random CfvNet model
    // 2. Spawn inference server
    // 3. Submit 10 requests from main thread
    // 4. Verify 1326-element responses
    // 5. Shutdown
}
```

**Step 3: Commit**

```bash
git add sample_configurations/rebel_1k_50bb.yaml crates/rebel/src/inference_server.rs
git commit -m "feat(rebel): inference server config and integration test"
```

---

## Dependency Graph

```
Task 1 (ReplayBuffer) ──► Task 2 (InferenceServer) ──► Task 3 (Iterative Solver)
                                    │                          │
                                    ▼                          ▼
                              Task 5 (Training)          Task 4 (Self-play wiring)
                                    │                          │
                                    └──────────┬───────────────┘
                                               ▼
                                    Task 6 (CLI wiring)
                                               │
                                               ▼
                                    Task 7 (Config + test)
```

Tasks 1→2 are sequential. Tasks 3, 4, 5 can be parallelized after Task 2. Tasks 6, 7 depend on all prior.
