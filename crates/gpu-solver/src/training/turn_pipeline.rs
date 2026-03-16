//! Supremus-style GPU turn CFVNet training pipeline.
//!
//! Same structure as the river pipeline but:
//! - Samples 4-card boards instead of 5
//! - Uses `TurnBatchSolver` with neural leaf evaluation
//! - Loads a trained river model for leaf evaluation
//! - Trains a turn CFVNet (same architecture as river)
//!
//! Pipeline: sample turn boards -> build turn solver -> solve with leaf eval
//!           -> insert root CFVs into reservoir -> train -> validate

#[cfg(feature = "training")]
use std::io::IsTerminal;
#[cfg(feature = "training")]
use std::path::PathBuf;
#[cfg(feature = "training")]
use std::time::Instant;

#[cfg(feature = "training")]
use burn::tensor::backend::AutodiffBackend;
#[cfg(feature = "training")]
use indicatif::{ProgressBar, ProgressStyle};

#[cfg(feature = "training")]
use crate::batch::BatchConfig;
#[cfg(feature = "training")]
use crate::gpu::GpuContext;
#[cfg(feature = "training")]
use crate::training::builder::GpuBatchBuilder;
#[cfg(feature = "training")]
use crate::training::leaf_eval::GpuLeafEvaluator;
#[cfg(feature = "training")]
use crate::training::reservoir::GpuReservoir;
#[cfg(feature = "training")]
use crate::training::sampler::GpuTurnSampler;
#[cfg(feature = "training")]
use crate::training::trainer::GpuTrainer;
#[cfg(feature = "training")]
use crate::training::turn_solver::TurnBatchSolver;
#[cfg(feature = "training")]
use crate::training::turn_solver::TurnBatchSolverCuda;
#[cfg(feature = "training")]
use crate::training::cuda_net::GpuLeafEvaluatorCuda;
#[cfg(feature = "training")]
use crate::training::validation::GroundTruthValidator;
#[cfg(feature = "training")]
use crate::tree::FlatTree;

/// Configuration for the GPU turn CFVNet training pipeline.
#[cfg(feature = "training")]
#[derive(Debug, Clone)]
pub struct TurnTrainingConfig {
    /// Path to trained river model for leaf evaluation.
    pub river_model_path: PathBuf,
    /// Number of hidden layers in the river model (must match trained model).
    pub river_hidden_layers: usize,
    /// Hidden size of the river model (must match trained model).
    pub river_hidden_size: usize,
    /// Total number of training samples to generate.
    pub num_samples: u64,
    /// DCFR+ iterations per solve batch.
    pub solve_iterations: u32,
    /// Number of turn spots per solve batch.
    pub batch_size: usize,
    /// Maximum capacity of the GPU reservoir buffer.
    pub reservoir_capacity: usize,
    /// Number of hidden layers in the turn CFVNet.
    pub hidden_layers: usize,
    /// Hidden layer width.
    pub hidden_size: usize,
    /// Mini-batch size for training.
    pub train_batch_size: usize,
    /// Number of training steps per solve batch.
    pub train_steps_per_batch: usize,
    /// Initial learning rate.
    pub learning_rate: f64,
    /// Huber loss delta.
    pub huber_delta: f64,
    /// Auxiliary game-value loss weight.
    pub aux_loss_weight: f64,
    /// Print validation metrics every N samples.
    pub validation_interval: u64,
    /// Save checkpoint every N samples.
    pub checkpoint_interval: u64,
    /// Number of ground-truth validation positions.
    pub gt_validation_positions: usize,
    /// Iterations for ground-truth solves.
    pub gt_solve_iterations: u32,
    /// Output directory for checkpoints and final model.
    pub output_dir: PathBuf,
    /// Random seed.
    pub seed: u64,
    /// Reference pot size for shared tree topology.
    pub ref_pot: i32,
    /// Reference effective stack for shared tree topology.
    pub ref_stack: i32,
}

#[cfg(feature = "training")]
impl Default for TurnTrainingConfig {
    fn default() -> Self {
        Self {
            river_model_path: PathBuf::from("models/river_cfvnet/model"),
            river_hidden_layers: 7,
            river_hidden_size: 500,
            num_samples: 20_000_000,
            solve_iterations: 4000,
            batch_size: 1000,
            reservoir_capacity: 100_000,
            hidden_layers: 7,
            hidden_size: 500,
            train_batch_size: 1024,
            train_steps_per_batch: 10,
            learning_rate: 0.001,
            huber_delta: 1.0,
            aux_loss_weight: 1.0,
            validation_interval: 10_000,
            checkpoint_interval: 1_000_000,
            gt_validation_positions: 10,
            gt_solve_iterations: 500,
            output_dir: PathBuf::from("models/turn_cfvnet"),
            seed: 42,
            ref_pot: 100,
            ref_stack: 100,
        }
    }
}

/// Per-batch timing accumulator for performance reporting.
#[cfg(feature = "training")]
struct PhaseTiming {
    sample_ms: f64,
    build_ms: f64,
    solve_ms: f64,
    insert_ms: f64,
    train_ms: f64,
}

/// Build a turn-specific `GpuBatchBuilder` using a depth-limited turn tree.
///
/// The reference tree has fold terminals and depth-boundary leaves (no showdowns,
/// no chance nodes). Returns the builder (whose shared topology contains the
/// reference FlatTree with boundary info).
#[cfg(feature = "training")]
fn build_turn_batch_builder(
    gpu: &GpuContext,
    config: &BatchConfig,
    ref_pot: i32,
    ref_stack: i32,
) -> Result<GpuBatchBuilder, String> {
    GpuBatchBuilder::new_turn(gpu, config, ref_pot, ref_stack)
}

/// Pad 4-card turn boards to 5-card format for the reservoir encoder.
///
/// The 5th card is set to 255 (>= 52), which the kernel ignores in
/// one-hot encoding and masking. Input: `[num_spots * 4]`.
/// Output: `[num_spots * 5]`.
#[cfg(feature = "training")]
fn pad_boards_to_5(boards_4: &[u32]) -> Vec<u32> {
    let num_spots = boards_4.len() / 4;
    let mut out = Vec::with_capacity(num_spots * 5);
    for i in 0..num_spots {
        out.push(boards_4[i * 4]);
        out.push(boards_4[i * 4 + 1]);
        out.push(boards_4[i * 4 + 2]);
        out.push(boards_4[i * 4 + 3]);
        out.push(255); // dummy 5th card (>= 52, ignored by kernel)
    }
    out
}

/// Run the full GPU turn CFVNet training pipeline.
///
/// Loads the trained river model, then runs the Supremus-style loop:
/// sample turn boards -> solve with leaf eval -> reservoir insert -> train -> validate.
#[cfg(feature = "training")]
pub fn train_turn_cfvnet<B: AutodiffBackend>(
    config: &TurnTrainingConfig,
    device: &B::Device,
) -> Result<(), String> {
    let gpu = GpuContext::new(0).map_err(|e| format!("CUDA init: {e}"))?;

    // 1. Load river model for leaf evaluation
    eprintln!(
        "Loading river model from {:?} ({}x{})...",
        config.river_model_path, config.river_hidden_layers, config.river_hidden_size
    );
    let leaf_eval = GpuLeafEvaluator::<B::InnerBackend>::load(
        &config.river_model_path,
        device,
        config.river_hidden_layers,
        config.river_hidden_size,
    )?;

    // 2. Build shared turn tree topology (with depth boundaries)
    let batch_config = BatchConfig::default();
    let builder = build_turn_batch_builder(
        &gpu,
        &batch_config,
        config.ref_pot,
        config.ref_stack,
    )?;

    let ref_tree = &builder.topology().ref_tree;
    let num_boundaries = ref_tree.boundary_indices.len();
    eprintln!(
        "Turn tree: {} nodes, {} infosets, {} boundaries, {} fold terminals",
        ref_tree.num_nodes(),
        ref_tree.num_infosets,
        num_boundaries,
        ref_tree.node_types.iter().filter(|t| **t == crate::tree::NodeType::TerminalFold).count(),
    );

    // 3. Initialize components
    let mut reservoir = GpuReservoir::new(&gpu, config.reservoir_capacity)
        .map_err(|e| format!("reservoir: {e}"))?;
    let mut trainer = GpuTrainer::<B>::new(
        device,
        config.hidden_layers,
        config.hidden_size,
        config.learning_rate,
        config.huber_delta,
        config.aux_loss_weight,
    );

    // Initialize GPU turn sampler (4-card boards)
    let mut sampler = GpuTurnSampler::new(&gpu, config.batch_size, config.seed + 3_000_000)?;

    // Pre-allocate pot/stack GPU buffers (constant across all batches)
    let pots = vec![config.ref_pot as f32; config.batch_size];
    let stacks = vec![config.ref_stack as f32; config.batch_size];
    let gpu_pots = gpu.upload(&pots).map_err(|e| format!("upload pots: {e}"))?;
    let gpu_stacks = gpu.upload(&stacks).map_err(|e| format!("upload stacks: {e}"))?;

    // Pre-compute ground-truth validation set
    // For turn, we solve turn positions with the same river model as leaf evaluator.
    // For now, skip ground-truth validation since it requires solving turn positions
    // on CPU with leaf evaluation which is complex. Use hold-out validation only.
    eprintln!("Note: Ground-truth validation for turn uses river-model leaf eval");
    eprintln!(
        "  Skipping pre-compute (hold-out validation only for now)"
    );

    let mut total_samples = 0u64;
    let mut last_loss = f32::MAX;
    let mut sample_seed = config.seed + 2_000_000;
    let start = Instant::now();

    let mut timing_accum = PhaseTiming {
        sample_ms: 0.0,
        build_ms: 0.0,
        solve_ms: 0.0,
        insert_ms: 0.0,
        train_ms: 0.0,
    };
    let mut batches_since_report = 0u64;

    // Create output directory
    std::fs::create_dir_all(&config.output_dir)
        .map_err(|e| format!("create output dir: {e}"))?;

    eprintln!("Starting GPU turn training pipeline");
    eprintln!("  Batch size: {} spots", config.batch_size);
    eprintln!("  Solve iterations: {}", config.solve_iterations);
    eprintln!("  Reservoir capacity: {}", config.reservoir_capacity);
    eprintln!(
        "  Turn model: {} layers x {} hidden",
        config.hidden_layers, config.hidden_size
    );
    eprintln!(
        "  River model: {} layers x {} hidden",
        config.river_hidden_layers, config.river_hidden_size
    );
    eprintln!("  Depth boundaries: {}", num_boundaries);
    eprintln!(
        "  Train batch: {} x {} steps/batch",
        config.train_batch_size, config.train_steps_per_batch
    );
    eprintln!("  Learning rate: {}", config.learning_rate);
    eprintln!("  Sampling: GPU-native (GpuTurnSampler)");
    eprintln!("  Output: {}", config.output_dir.display());
    eprintln!();

    let pb = if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new(config.num_samples);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{prefix} [{bar:30}] {pos}/{len} ({per_sec}, ETA {eta}) {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        pb.set_prefix("TURN");
        pb
    } else {
        ProgressBar::hidden()
    };

    while total_samples < config.num_samples {
        // === SAMPLE PHASE (GPU) ===
        let t0 = Instant::now();
        sampler.sample(&gpu)?;
        let t1 = Instant::now();

        // === BUILD SOLVER PHASE ===
        // Download boards to CPU for turn solver setup (needed for river card computation)
        let boards_host: Vec<u32> = gpu
            .download(sampler.boards())
            .map_err(|e| format!("download boards: {e}"))?;

        let batch_solver = builder.build_depth_limited(
            &gpu,
            sampler.ranges_oop(),
            sampler.ranges_ip(),
            config.batch_size,
        )?;

        let topo = &builder.topology().ref_tree;
        let mut turn_solver = TurnBatchSolver::new(
            &gpu,
            batch_solver,
            &leaf_eval,
            &topo.boundary_indices,
            &topo.boundary_pots,
            &topo.boundary_stacks,
            &boards_host,
            config.batch_size,
        )?;
        let t2 = Instant::now();

        // === SOLVE PHASE ===
        let solve_result = turn_solver.solve_with_cfvs(config.solve_iterations)?;
        let t3 = Instant::now();

        // === INSERT PHASE (GPU) ===
        // Pad 4-card boards to 5-card for the reservoir encoder kernel.
        // The 5th card is set to 255 (>= 52), which the kernel ignores
        // in one-hot encoding (`if (b[i] < 52)`) and masking.
        let padded_boards = pad_boards_to_5(&boards_host);
        let gpu_padded_boards = gpu
            .upload(&padded_boards)
            .map_err(|e| format!("upload padded boards: {e}"))?;
        reservoir
            .insert_batch_gpu(
                &gpu,
                sampler.ranges_oop(),
                sampler.ranges_ip(),
                &gpu_padded_boards,
                &gpu_pots,
                &gpu_stacks,
                &solve_result.cfvs_oop,
                &solve_result.cfvs_ip,
                config.batch_size,
            )
            .map_err(|e| format!("reservoir insert: {e}"))?;
        let t4 = Instant::now();

        total_samples += (config.batch_size as u64) * 2; // 2 records per spot

        // === TRAIN PHASE ===
        if reservoir.size() >= config.train_batch_size {
            for _ in 0..config.train_steps_per_batch {
                sample_seed += 1;
                let batch = reservoir
                    .sample_minibatch_gpu(
                        &gpu,
                        config.train_batch_size,
                        sample_seed as u32,
                    )
                    .map_err(|e| format!("sample minibatch: {e}"))?;

                last_loss = trainer.train_step(&gpu, &batch)?;
            }
        }
        let t5 = Instant::now();

        // Update progress bar
        pb.set_position(total_samples);
        if last_loss < f32::MAX {
            pb.set_message(format!("loss={:.4}", last_loss));
        }

        // Accumulate timing
        timing_accum.sample_ms += (t1 - t0).as_secs_f64() * 1000.0;
        timing_accum.build_ms += (t2 - t1).as_secs_f64() * 1000.0;
        timing_accum.solve_ms += (t3 - t2).as_secs_f64() * 1000.0;
        timing_accum.insert_ms += (t4 - t3).as_secs_f64() * 1000.0;
        timing_accum.train_ms += (t5 - t4).as_secs_f64() * 1000.0;
        batches_since_report += 1;

        // === VALIDATION (periodic) ===
        if total_samples % config.validation_interval < (config.batch_size as u64 * 2) {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = total_samples as f64 / elapsed;

            // Hold-out validation loss
            let val_loss = if reservoir.size() >= config.train_batch_size {
                sample_seed += 1;
                let val_batch = reservoir
                    .sample_minibatch_gpu(
                        &gpu,
                        config.train_batch_size,
                        sample_seed as u32,
                    )
                    .map_err(|e| format!("val batch: {e}"))?;
                trainer.validation_loss(&gpu, &val_batch)?
            } else {
                f32::NAN
            };

            let msg = format!(
                "[{:>12} samples] loss={:.6} val={:.6} rate={:.0} samples/s  reservoir={}",
                total_samples, last_loss, val_loss, rate, reservoir.size()
            );
            pb.println(&msg);
            eprintln!("{}", msg);

            // Print per-phase timing breakdown
            if batches_since_report > 0 {
                let n = batches_since_report as f64;
                let msg = format!(
                    "  timing: sample={:.1}ms build={:.1}ms solve={:.1}ms insert={:.1}ms train={:.1}ms  ({} batches avg)",
                    timing_accum.sample_ms / n,
                    timing_accum.build_ms / n,
                    timing_accum.solve_ms / n,
                    timing_accum.insert_ms / n,
                    timing_accum.train_ms / n,
                    batches_since_report,
                );
                pb.println(&msg);
                eprintln!("{}", msg);
            }

            // Reset accumulator
            timing_accum = PhaseTiming {
                sample_ms: 0.0,
                build_ms: 0.0,
                solve_ms: 0.0,
                insert_ms: 0.0,
                train_ms: 0.0,
            };
            batches_since_report = 0;
        }

        // === CHECKPOINT (periodic) ===
        if config.checkpoint_interval > 0
            && total_samples % config.checkpoint_interval < (config.batch_size as u64 * 2)
            && total_samples > 0
        {
            let label = format!("checkpoint_{}", total_samples);
            trainer.save_checkpoint(&config.output_dir, &label);
            let msg = format!("  Saved checkpoint: {}", label);
            pb.println(&msg);
            eprintln!("{}", msg);
        }
    }

    pb.finish_with_message("done");

    // Save final model
    trainer.save_final(&config.output_dir)?;
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!();
    eprintln!(
        "Turn training complete. {} samples in {:.1}s ({:.0} samples/s)",
        total_samples,
        elapsed,
        total_samples as f64 / elapsed
    );
    eprintln!("Model saved to {:?}", config.output_dir);
    Ok(())
}

/// Run the GPU turn CFVNet training pipeline with CUDA-native leaf evaluation.
///
/// Uses `CudaNetInference` for the river model forward pass, keeping all data
/// on the GPU throughout inference. This eliminates the GPU->CPU->GPU bounce
/// that made burn-based inference slow for turn training.
///
/// This is the preferred entry point for turn training. The burn-based
/// `train_turn_cfvnet` is kept for reference/testing.
#[cfg(feature = "training")]
pub fn train_turn_cfvnet_cuda<B: AutodiffBackend>(
    config: &TurnTrainingConfig,
    device: &B::Device,
) -> Result<(), String> {
    let gpu = GpuContext::new(0).map_err(|e| format!("CUDA init: {e}"))?;

    // 1. Load river model using CUDA-native inference (no burn runtime for forward pass)
    eprintln!(
        "Loading river model from {:?} ({}x{}) with CUDA-native inference...",
        config.river_model_path, config.river_hidden_layers, config.river_hidden_size
    );

    // Compute max batch size for leaf evaluation:
    // num_boundaries * 48 rivers * batch_size spots
    // We don't know num_boundaries yet, but we can use a generous upper bound.
    // A typical turn tree has ~4-8 boundaries, so 8 * 48 * batch_size is safe.
    let max_leaf_batch = 8 * 48 * config.batch_size;
    let mut leaf_eval = GpuLeafEvaluatorCuda::load(
        &config.river_model_path,
        &gpu,
        config.river_hidden_layers,
        config.river_hidden_size,
        max_leaf_batch,
    )?;

    // 2. Build shared turn tree topology (with depth boundaries)
    let batch_config = BatchConfig::default();
    let builder = build_turn_batch_builder(
        &gpu,
        &batch_config,
        config.ref_pot,
        config.ref_stack,
    )?;

    let ref_tree = &builder.topology().ref_tree;
    let num_boundaries = ref_tree.boundary_indices.len();
    eprintln!(
        "Turn tree: {} nodes, {} infosets, {} boundaries, {} fold terminals",
        ref_tree.num_nodes(),
        ref_tree.num_infosets,
        num_boundaries,
        ref_tree.node_types.iter().filter(|t| **t == crate::tree::NodeType::TerminalFold).count(),
    );

    // Verify max_leaf_batch is sufficient
    let actual_leaf_batch = num_boundaries * 48 * config.batch_size;
    assert!(
        actual_leaf_batch <= max_leaf_batch,
        "Leaf batch size {} exceeds pre-allocated max {}. Increase estimate.",
        actual_leaf_batch,
        max_leaf_batch,
    );

    // 3. Initialize components
    let mut reservoir = GpuReservoir::new(&gpu, config.reservoir_capacity)
        .map_err(|e| format!("reservoir: {e}"))?;
    let mut trainer = GpuTrainer::<B>::new(
        device,
        config.hidden_layers,
        config.hidden_size,
        config.learning_rate,
        config.huber_delta,
        config.aux_loss_weight,
    );

    let mut sampler = GpuTurnSampler::new(&gpu, config.batch_size, config.seed + 3_000_000)?;

    let pots = vec![config.ref_pot as f32; config.batch_size];
    let stacks = vec![config.ref_stack as f32; config.batch_size];
    let gpu_pots = gpu.upload(&pots).map_err(|e| format!("upload pots: {e}"))?;
    let gpu_stacks = gpu.upload(&stacks).map_err(|e| format!("upload stacks: {e}"))?;

    eprintln!("Note: Ground-truth validation for turn uses river-model leaf eval");
    eprintln!("  Skipping pre-compute (hold-out validation only for now)");

    let mut total_samples = 0u64;
    let mut last_loss = f32::MAX;
    let mut sample_seed = config.seed + 2_000_000;
    let start = Instant::now();

    let mut timing_accum = PhaseTiming {
        sample_ms: 0.0,
        build_ms: 0.0,
        solve_ms: 0.0,
        insert_ms: 0.0,
        train_ms: 0.0,
    };
    let mut batches_since_report = 0u64;

    std::fs::create_dir_all(&config.output_dir)
        .map_err(|e| format!("create output dir: {e}"))?;

    eprintln!("Starting GPU turn training pipeline (CUDA-native leaf eval)");
    eprintln!("  Batch size: {} spots", config.batch_size);
    eprintln!("  Solve iterations: {}", config.solve_iterations);
    eprintln!("  Reservoir capacity: {}", config.reservoir_capacity);
    eprintln!(
        "  Turn model: {} layers x {} hidden",
        config.hidden_layers, config.hidden_size
    );
    eprintln!(
        "  River model: {} layers x {} hidden",
        config.river_hidden_layers, config.river_hidden_size
    );
    eprintln!("  Depth boundaries: {}", num_boundaries);
    eprintln!("  Leaf eval: CUDA-native (cuBLAS + custom kernels)");
    eprintln!(
        "  Train batch: {} x {} steps/batch",
        config.train_batch_size, config.train_steps_per_batch
    );
    eprintln!("  Learning rate: {}", config.learning_rate);
    eprintln!("  Output: {}", config.output_dir.display());
    eprintln!();

    let pb = if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new(config.num_samples);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{prefix} [{bar:30}] {pos}/{len} ({per_sec}, ETA {eta}) {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        pb.set_prefix("TURN");
        pb
    } else {
        ProgressBar::hidden()
    };

    while total_samples < config.num_samples {
        // === SAMPLE PHASE ===
        let t0 = Instant::now();
        sampler.sample(&gpu)?;
        let t1 = Instant::now();

        // === BUILD SOLVER PHASE ===
        let boards_host: Vec<u32> = gpu
            .download(sampler.boards())
            .map_err(|e| format!("download boards: {e}"))?;

        let batch_solver = builder.build_depth_limited(
            &gpu,
            sampler.ranges_oop(),
            sampler.ranges_ip(),
            config.batch_size,
        )?;

        let topo = &builder.topology().ref_tree;
        let mut turn_solver = TurnBatchSolverCuda::new(
            &gpu,
            batch_solver,
            &mut leaf_eval,
            &topo.boundary_indices,
            &topo.boundary_pots,
            &topo.boundary_stacks,
            &boards_host,
            config.batch_size,
        )?;
        let t2 = Instant::now();

        // === SOLVE PHASE ===
        let mut solve_result = turn_solver.solve_with_cfvs(config.solve_iterations)?;
        let t3 = Instant::now();

        // === INSERT PHASE ===
        let padded_boards = pad_boards_to_5(&boards_host);
        let gpu_padded_boards = gpu
            .upload(&padded_boards)
            .map_err(|e| format!("upload padded boards: {e}"))?;
        reservoir
            .insert_batch_gpu(
                &gpu,
                sampler.ranges_oop(),
                sampler.ranges_ip(),
                &gpu_padded_boards,
                &gpu_pots,
                &gpu_stacks,
                &solve_result.cfvs_oop,
                &solve_result.cfvs_ip,
                config.batch_size,
            )
            .map_err(|e| format!("reservoir insert: {e}"))?;
        let t4 = Instant::now();

        total_samples += (config.batch_size as u64) * 2;

        // === TRAIN PHASE ===
        if reservoir.size() >= config.train_batch_size {
            for _ in 0..config.train_steps_per_batch {
                sample_seed += 1;
                let batch = reservoir
                    .sample_minibatch_gpu(
                        &gpu,
                        config.train_batch_size,
                        sample_seed as u32,
                    )
                    .map_err(|e| format!("sample minibatch: {e}"))?;

                last_loss = trainer.train_step(&gpu, &batch)?;
            }
        }
        let t5 = Instant::now();

        // Update progress bar
        pb.set_position(total_samples);
        if last_loss < f32::MAX {
            pb.set_message(format!("loss={:.4}", last_loss));
        }

        timing_accum.sample_ms += (t1 - t0).as_secs_f64() * 1000.0;
        timing_accum.build_ms += (t2 - t1).as_secs_f64() * 1000.0;
        timing_accum.solve_ms += (t3 - t2).as_secs_f64() * 1000.0;
        timing_accum.insert_ms += (t4 - t3).as_secs_f64() * 1000.0;
        timing_accum.train_ms += (t5 - t4).as_secs_f64() * 1000.0;
        batches_since_report += 1;

        // === VALIDATION ===
        if total_samples % config.validation_interval < (config.batch_size as u64 * 2) {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = total_samples as f64 / elapsed;

            let val_loss = if reservoir.size() >= config.train_batch_size {
                sample_seed += 1;
                let val_batch = reservoir
                    .sample_minibatch_gpu(
                        &gpu,
                        config.train_batch_size,
                        sample_seed as u32,
                    )
                    .map_err(|e| format!("val batch: {e}"))?;
                trainer.validation_loss(&gpu, &val_batch)?
            } else {
                f32::NAN
            };

            let msg = format!(
                "[{:>12} samples] loss={:.6} val={:.6} rate={:.0} samples/s  reservoir={}",
                total_samples, last_loss, val_loss, rate, reservoir.size()
            );
            pb.println(&msg);
            eprintln!("{}", msg);

            if batches_since_report > 0 {
                let n = batches_since_report as f64;
                let msg = format!(
                    "  timing: sample={:.1}ms build={:.1}ms solve={:.1}ms insert={:.1}ms train={:.1}ms  ({} batches avg)",
                    timing_accum.sample_ms / n,
                    timing_accum.build_ms / n,
                    timing_accum.solve_ms / n,
                    timing_accum.insert_ms / n,
                    timing_accum.train_ms / n,
                    batches_since_report,
                );
                pb.println(&msg);
                eprintln!("{}", msg);
            }

            timing_accum = PhaseTiming {
                sample_ms: 0.0,
                build_ms: 0.0,
                solve_ms: 0.0,
                insert_ms: 0.0,
                train_ms: 0.0,
            };
            batches_since_report = 0;
        }

        // === CHECKPOINT ===
        if config.checkpoint_interval > 0
            && total_samples % config.checkpoint_interval < (config.batch_size as u64 * 2)
            && total_samples > 0
        {
            let label = format!("checkpoint_{}", total_samples);
            trainer.save_checkpoint(&config.output_dir, &label);
            let msg = format!("  Saved checkpoint: {}", label);
            pb.println(&msg);
            eprintln!("{}", msg);
        }
    }

    pb.finish_with_message("done");

    trainer.save_final(&config.output_dir)?;
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!();
    eprintln!(
        "Turn training complete (CUDA-native). {} samples in {:.1}s ({:.0} samples/s)",
        total_samples,
        elapsed,
        total_samples as f64 / elapsed
    );
    eprintln!("Model saved to {:?}", config.output_dir);
    Ok(())
}

#[cfg(all(test, feature = "training"))]
mod tests {
    use super::*;

    #[test]
    fn test_turn_pipeline_config_defaults() {
        let config = TurnTrainingConfig::default();
        assert_eq!(config.batch_size, 1000);
        assert_eq!(config.solve_iterations, 4000);
        assert_eq!(config.reservoir_capacity, 100_000);
        assert_eq!(config.river_hidden_layers, 7);
        assert_eq!(config.river_hidden_size, 500);
    }

    #[test]
    fn test_build_turn_tree_has_boundaries() {
        use range_solver::bet_size::BetSizeOptions;
        use crate::tree::NodeType;

        let bet_sizes = BetSizeOptions::try_from(("50%,a", "")).unwrap();
        let flop = [0u8, 1, 2];
        let turn = 3u8;

        let mut game = crate::tree::build_turn_game(flop, turn, 100, 100, &bet_sizes);
        let flat = FlatTree::from_postflop_game(&mut game);

        // Should have depth boundaries
        assert!(
            !flat.boundary_indices.is_empty(),
            "Turn tree should have depth boundaries"
        );

        // Should have fold terminals
        let num_fold = flat.node_types.iter().filter(|t| **t == NodeType::TerminalFold).count();
        assert!(num_fold > 0, "Should have fold terminals");

        // Should NOT have showdown terminals
        let num_sd = flat.node_types.iter().filter(|t| **t == NodeType::TerminalShowdown).count();
        assert_eq!(num_sd, 0, "Should have no showdown terminals");
    }
}
