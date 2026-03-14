# GPU-Accelerated Turn Datagen Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU inference for the river model during turn datagen, with batched forward passes and mutex contention tracking.

**Architecture:** Shared `Arc<Mutex<CfvNet<CudaJit>>>` model on GPU. Rayon threads do DCFR tree traversal on CPU. At depth boundaries, `SharedRiverNetEvaluator` acquires the mutex, batches all ~48 river card inputs into one `[N, INPUT_SIZE]` tensor, runs a single GPU forward pass, releases the lock. Contention tracked via `AtomicU64`.

**Tech Stack:** Rust, burn 0.16 (CudaJit backend), rayon, indicatif

---

### Task 1: Add `--backend` CLI flag to Generate command

**Files:**
- Modify: `crates/cfvnet/src/main.rs:20-38` (Generate variant)
- Modify: `crates/cfvnet/src/main.rs:164-178` (match arm)
- Modify: `crates/cfvnet/src/main.rs:290-296` (cmd_generate signature)

**Step 1: Add backend field to Generate command**

In `Commands::Generate`, add after `per_file`:

```rust
        /// Backend for inference: ndarray (CPU, default) or cuda (GPU, requires --features cuda).
        #[arg(long, default_value = "ndarray")]
        backend: String,
```

**Step 2: Plumb backend through to cmd_generate**

Update the `Commands::Generate` match arm to pass `backend`:

```rust
Commands::Generate {
    config,
    output,
    num_samples,
    threads,
    per_file,
    backend,
} => {
    ensure_parent_dir(&output);
    cmd_generate(config, output, num_samples, threads, per_file, &backend);
}
```

Update `cmd_generate` signature:

```rust
fn cmd_generate(
    config_path: PathBuf,
    output: PathBuf,
    num_samples: Option<u64>,
    threads: Option<usize>,
    per_file: Option<u64>,
    backend: &str,
) {
```

Pass `backend` to the turn generate call (for now, ignore it — Task 3 will wire it up):

```rust
let result = match street {
    "turn" => cfvnet::datagen::turn_generate::generate_turn_training_data(&cfg, &file_output, backend),
    _ => cfvnet::datagen::generate::generate_training_data(&cfg, &file_output),
};
```

**Step 3: Update `generate_turn_training_data` signature**

Add `backend: &str` parameter. For now, ignore it and use NdArray as before. This makes the function compile while we build the GPU path.

```rust
pub fn generate_turn_training_data(
    config: &CfvnetConfig,
    output_path: &Path,
    backend: &str,
) -> Result<(), String> {
```

**Step 4: Run `cargo check -p cfvnet` — should compile**

**Step 5: Commit**

```
feat(cfvnet): add --backend flag to generate command
```

---

### Task 2: Create `SharedRiverNetEvaluator` with batched inference and contention tracking

**Files:**
- Modify: `crates/cfvnet/src/eval/river_net_evaluator.rs`

This is the core component. It wraps a shared GPU model behind a mutex, batches all river card inputs into a single forward pass, and tracks mutex wait time.

**Step 1: Add imports**

At the top of `river_net_evaluator.rs`, add:

```rust
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
```

**Step 2: Create `SharedRiverNetEvaluator` struct**

Add after the existing `RiverNetEvaluator` impl block (before the `impl LeafEvaluator` block):

```rust
/// GPU-accelerated evaluator that shares a single model behind a mutex.
///
/// Batches all valid river card inputs into a single `[N, INPUT_SIZE]` tensor
/// for one forward pass per `evaluate()` call. Tracks cumulative mutex wait time.
pub struct SharedRiverNetEvaluator<B: Backend> {
    model: Arc<Mutex<CfvNet<B>>>,
    device: B::Device,
    /// Cumulative nanoseconds spent waiting for the mutex across all threads.
    pub wait_nanos: Arc<AtomicU64>,
}

impl<B: Backend> SharedRiverNetEvaluator<B>
where
    B::Device: Clone,
{
    pub fn new(
        model: Arc<Mutex<CfvNet<B>>>,
        device: B::Device,
        wait_nanos: Arc<AtomicU64>,
    ) -> Self {
        Self { model, device, wait_nanos }
    }
}
```

**Step 3: Implement `LeafEvaluator` with batched forward pass**

```rust
impl<B: Backend> LeafEvaluator for SharedRiverNetEvaluator<B>
where
    B::Device: Clone,
{
    fn evaluate(
        &self,
        combos: &[[Card; 2]],
        board: &[Card],
        pot: f64,
        effective_stack: f64,
        oop_range: &[f64],
        ip_range: &[f64],
        traverser: u8,
    ) -> Vec<f64> {
        assert_eq!(board.len(), 4, "SharedRiverNetEvaluator requires a 4-card turn board");
        assert_eq!(combos.len(), oop_range.len());
        assert_eq!(combos.len(), ip_range.len());

        let num_combos = combos.len();

        // Pre-convert combos to u8 pairs and their 1326 indices.
        let combos_u8: Vec<[u8; 2]> = combos
            .iter()
            .map(|c| [rs_card_to_u8(c[0]), rs_card_to_u8(c[1])])
            .collect();
        let combo_indices: Vec<usize> = combos_u8
            .iter()
            .map(|c| card_pair_to_index(c[0], c[1]))
            .collect();

        let board_u8: Vec<u8> = board.iter().map(|c| rs_card_to_u8(*c)).collect();

        // Build all river card inputs CPU-side.
        let mut inputs: Vec<f32> = Vec::new();
        let mut river_cards: Vec<u8> = Vec::new();
        let mut valid_combo_masks: Vec<Vec<bool>> = Vec::new();

        for river_u8 in 0u8..52 {
            if board_u8.contains(&river_u8) {
                continue;
            }

            let river_board_u8: [u8; 5] = [
                board_u8[0], board_u8[1], board_u8[2], board_u8[3], river_u8,
            ];

            let mut oop_1326 = [0.0_f32; OUTPUT_SIZE];
            let mut ip_1326 = [0.0_f32; OUTPUT_SIZE];
            let mut valid_combo_mask = vec![false; num_combos];

            for (i, &idx) in combo_indices.iter().enumerate() {
                if combos_u8[i][0] == river_u8 || combos_u8[i][1] == river_u8 {
                    continue;
                }
                valid_combo_mask[i] = true;
                oop_1326[idx] = oop_range[i] as f32;
                ip_1326[idx] = ip_range[i] as f32;
            }

            let input_vec = build_input(
                &oop_1326, &ip_1326, &river_board_u8,
                pot, effective_stack, traverser,
            );
            inputs.extend_from_slice(&input_vec);
            river_cards.push(river_u8);
            valid_combo_masks.push(valid_combo_mask);
        }

        let batch_size = river_cards.len();

        // Acquire mutex, run batched forward pass on GPU.
        let t0 = Instant::now();
        let model = self.model.lock().unwrap();
        let wait = t0.elapsed();
        self.wait_nanos.fetch_add(wait.as_nanos() as u64, Ordering::Relaxed);

        let data = TensorData::new(inputs, [batch_size, INPUT_SIZE]);
        let input_tensor = Tensor::<B, 2>::from_data(data, &self.device);
        let output = model.forward(input_tensor);

        drop(model); // release mutex before post-processing

        let out_data = output.into_data();
        let out_vec: Vec<f32> = out_data.to_vec().expect("output tensor conversion");

        // Post-process: average over river cards per combo.
        let mut cfv_sum = vec![0.0_f64; num_combos];
        let mut cfv_count = vec![0_u32; num_combos];

        for (river_idx, mask) in valid_combo_masks.iter().enumerate() {
            let row_start = river_idx * OUTPUT_SIZE;
            for (i, &idx) in combo_indices.iter().enumerate() {
                if mask[i] {
                    cfv_sum[i] += f64::from(out_vec[row_start + idx]);
                    cfv_count[i] += 1;
                }
            }
        }

        cfv_sum
            .iter()
            .zip(cfv_count.iter())
            .map(|(&sum, &count)| {
                if count > 0 { sum / f64::from(count) } else { 0.0 }
            })
            .collect()
    }
}
```

**Step 4: Add test for `SharedRiverNetEvaluator`**

In the `#[cfg(test)] mod tests` block, add:

```rust
    #[test]
    fn shared_evaluator_matches_sequential() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let wait_nanos = Arc::new(AtomicU64::new(0));

        let shared = SharedRiverNetEvaluator::new(
            Arc::new(Mutex::new(model.clone())),
            device,
            wait_nanos.clone(),
        );
        let sequential = RiverNetEvaluator::new(model, Default::default());

        let board = test_board();
        let hands = SubgameHands::enumerate(&board);
        let n = hands.combos.len().min(20);
        let combos = &hands.combos[..n];
        let oop_range = vec![1.0 / n as f64; n];
        let ip_range = vec![1.0 / n as f64; n];

        let shared_result = shared.evaluate(combos, &board, 100.0, 200.0, &oop_range, &ip_range, 0);
        let seq_result = sequential.evaluate(combos, &board, 100.0, 200.0, &oop_range, &ip_range, 0);

        assert_eq!(shared_result.len(), seq_result.len());
        for (i, (s, q)) in shared_result.iter().zip(seq_result.iter()).enumerate() {
            assert!(
                (s - q).abs() < 1e-4,
                "combo {i}: shared={s} vs sequential={q}, diff={}",
                (s - q).abs()
            );
        }

        // Verify wait tracking recorded something (mutex was acquired).
        // With single-threaded test, wait should be minimal but non-zero.
        // Just verify it didn't panic.
        let _wait = wait_nanos.load(Ordering::Relaxed);
    }
```

**Step 5: Run tests**

```bash
cargo test -p cfvnet -- river_net_evaluator
```

Expected: all existing tests pass + new `shared_evaluator_matches_sequential` passes.

**Step 6: Commit**

```
feat(cfvnet): add SharedRiverNetEvaluator with batched inference and contention tracking
```

---

### Task 3: Wire GPU backend into turn datagen

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs`

**Step 1: Add CUDA-path function**

Add a new public function `generate_turn_training_data_cuda` that loads the model on CudaJit, wraps in `Arc<Mutex>`, creates the wait counter, and passes `SharedRiverNetEvaluator` to each solve. It follows the same chunked rayon pattern but doesn't need `SyncModel` — each thread creates its own `SharedRiverNetEvaluator` from the shared `Arc`.

The function should be `#[cfg(feature = "cuda")]` gated.

Add these imports at the top (inside `#[cfg(feature = "cuda")]`):

```rust
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, Ordering};
```

The function structure mirrors `generate_turn_training_data` but:
- Uses `burn::backend::CudaJit` as `B` instead of `NdArray`
- Wraps model in `Arc<Mutex<CfvNet<CudaJit<f32>>>>`
- Creates `Arc<AtomicU64>` for wait tracking
- Each solve closure creates a `SharedRiverNetEvaluator` from cloned Arc refs
- After completion, prints contention stats

```rust
#[cfg(feature = "cuda")]
pub fn generate_turn_training_data_cuda(
    config: &CfvnetConfig,
    output_path: &Path,
) -> Result<(), String> {
    use burn::backend::{CudaJit, cuda_jit::CudaDevice};
    use crate::eval::river_net_evaluator::SharedRiverNetEvaluator;

    type CB = CudaJit<f32>;

    let river_model_path = config.game.river_model_path.as_deref()
        .ok_or("river_model_path is required for turn datagen")?;

    let num_samples = config.datagen.num_samples;
    let seed = crate::config::resolve_seed(config.datagen.seed);
    let threads = config.datagen.threads;
    let solver_iterations = config.datagen.solver_iterations;
    let bet_sizes_f64 = parse_bet_sizes(&config.game.bet_sizes);
    if bet_sizes_f64.is_empty() {
        return Err("no valid percentage bet sizes found in config".into());
    }
    let bet_sizes_vec = vec![bet_sizes_f64];

    // Load river model on CUDA.
    let device = CudaDevice::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model = CfvNet::<CB>::new(
        &device,
        config.training.hidden_layers,
        config.training.hidden_size,
        INPUT_SIZE,
    )
    .load_file(river_model_path, &recorder, &device)
    .map_err(|e| format!("failed to load river model on CUDA: {e}"))?;

    println!("River model loaded on CUDA");

    let model = Arc::new(Mutex::new(model));
    let wait_nanos = Arc::new(AtomicU64::new(0));

    let pb = ProgressBar::new(num_samples);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}")
            .expect("valid progress bar template"),
    );
    pb.enable_steady_tick(std::time::Duration::from_secs(1));

    let file = std::fs::File::create(output_path)
        .map_err(|e| format!("create output: {e}"))?;
    let mut writer = BufWriter::new(file);

    let pool = if threads > 1 {
        Some(rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| format!("thread pool: {e}"))?)
    } else {
        None
    };

    let wall_start = std::time::Instant::now();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut remaining = num_samples;

    while remaining > 0 {
        let chunk_len = remaining.min(CHUNK_SIZE);
        remaining -= chunk_len;

        let situations: Vec<_> = (0..chunk_len)
            .map(|_| sample_situation(&config.datagen, config.game.initial_stack, 4, &mut rng))
            .collect();

        let solve_one = |sit: &super::sampler::Situation|
            -> Option<([f32; NUM_COMBOS], [f32; NUM_COMBOS], [u8; NUM_COMBOS], f32, f32)>
        {
            if sit.effective_stack <= 0 {
                pb.inc(1);
                return None;
            }

            let evaluator: Box<dyn LeafEvaluator> = Box::new(
                SharedRiverNetEvaluator::new(
                    Arc::clone(&model),
                    device,
                    Arc::clone(&wait_nanos),
                )
            );

            let result = solve_turn_situation(
                sit.board_cards(),
                f64::from(sit.pot),
                f64::from(sit.effective_stack),
                &sit.ranges,
                &bet_sizes_vec,
                solver_iterations,
                evaluator,
            );

            pb.inc(1);
            Some(result)
        };

        let results: Vec<_> = match &pool {
            Some(pool) => pool.install(|| {
                use rayon::prelude::*;
                situations.par_iter().map(solve_one).collect()
            }),
            None => situations.iter().map(solve_one).collect(),
        };

        for (sit, result) in situations.iter().zip(results) {
            if let Some((oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv)) = result {
                let board_vec = sit.board_cards().to_vec();

                let oop_rec = TrainingRecord {
                    board: board_vec.clone(),
                    pot: sit.pot as f32,
                    effective_stack: sit.effective_stack as f32,
                    player: 0,
                    game_value: oop_gv,
                    oop_range: sit.ranges[0],
                    ip_range: sit.ranges[1],
                    cfvs: oop_cfvs,
                    valid_mask,
                };
                write_record(&mut writer, &oop_rec).map_err(|e| format!("write OOP: {e}"))?;

                let ip_rec = TrainingRecord {
                    board: board_vec,
                    pot: sit.pot as f32,
                    effective_stack: sit.effective_stack as f32,
                    player: 1,
                    game_value: ip_gv,
                    oop_range: sit.ranges[0],
                    ip_range: sit.ranges[1],
                    cfvs: ip_cfvs,
                    valid_mask,
                };
                write_record(&mut writer, &ip_rec).map_err(|e| format!("write IP: {e}"))?;
            }
        }
    }

    pb.finish_with_message("done");

    // Print contention stats.
    let wall_secs = wall_start.elapsed().as_secs_f64();
    let wait_secs = wait_nanos.load(Ordering::Relaxed) as f64 / 1e9;
    let pct = if wall_secs > 0.0 { wait_secs / wall_secs * 100.0 } else { 0.0 };
    println!("GPU mutex wait: {wait_secs:.1}s total ({pct:.1}% of {wall_secs:.1}s wall time)");

    Ok(())
}
```

**Step 2: Update `generate_turn_training_data` to dispatch on backend**

Modify the existing function to check the `backend` parameter:

```rust
pub fn generate_turn_training_data(
    config: &CfvnetConfig,
    output_path: &Path,
    backend: &str,
) -> Result<(), String> {
    match backend {
        #[cfg(feature = "cuda")]
        "cuda" => return generate_turn_training_data_cuda(config, output_path),
        #[cfg(not(feature = "cuda"))]
        "cuda" => return Err("CUDA backend not enabled. Rebuild with: cargo build -p cfvnet --features cuda --release".into()),
        _ => {} // fall through to NdArray path below
    }

    // ... existing NdArray implementation unchanged ...
```

**Step 3: Run `cargo check -p cfvnet` and `cargo check -p cfvnet --features cuda`**

**Step 4: Commit**

```
feat(cfvnet): wire GPU backend into turn datagen with CUDA dispatch
```

---

### Task 4: End-to-end test

**Step 1: Build with CUDA and run a small datagen**

```bash
cargo run -p cfvnet --release --features cuda -- generate \
  -c sample_configurations/turn_cfvnet.yaml \
  -o /tmp/turn_gpu_test.bin \
  --num-samples 10 \
  --backend cuda
```

Expected: River model loads on CUDA, generates 10 samples, prints mutex contention stats.

**Step 2: Compare output against CPU path**

```bash
cargo run -p cfvnet --release -- generate \
  -c sample_configurations/turn_cfvnet.yaml \
  -o /tmp/turn_cpu_test.bin \
  --num-samples 10 \
  --backend ndarray
```

Both should produce valid training data. Use `datagen-eval` to verify:

```bash
cargo run -p cfvnet --release -- datagen-eval -d /tmp/turn_gpu_test.bin
cargo run -p cfvnet --release -- datagen-eval -d /tmp/turn_cpu_test.bin
```

**Step 3: Commit any fixes**

```
fix(cfvnet): address issues from GPU datagen end-to-end test
```
