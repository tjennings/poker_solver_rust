//! Supremus-style GPU flop CFVNet training pipeline.
//!
//! Same structure as the turn pipeline but:
//! - Samples 3-card boards instead of 4
//! - Uses `FlopBatchSolverCuda` with neural leaf evaluation at turn boundaries
//! - Loads a trained turn model for leaf evaluation
//! - Trains a flop CFVNet (same architecture as turn/river)
//!
//! Pipeline: sample flop boards -> build flop solver -> solve with leaf eval
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
use crate::training::reservoir::GpuReservoir;
#[cfg(feature = "training")]
use crate::training::sampler::GpuFlopSampler;
#[cfg(feature = "training")]
use crate::training::trainer::GpuTrainer;
#[cfg(feature = "training")]
use crate::training::cuda_net::GpuLeafEvaluatorCuda;
#[cfg(feature = "training")]
use crate::training::flop_solver::FlopBatchSolverCuda;
#[cfg(feature = "training")]
use crate::tree::FlatTree;

/// Configuration for the GPU flop CFVNet training pipeline.
#[cfg(feature = "training")]
#[derive(Debug, Clone)]
pub struct FlopTrainingConfig {
    /// Path to trained turn model for leaf evaluation.
    pub turn_model_path: PathBuf,
    /// Number of hidden layers in the turn model (must match trained model).
    pub turn_hidden_layers: usize,
    /// Hidden size of the turn model (must match trained model).
    pub turn_hidden_size: usize,
    /// Total number of training samples to generate.
    pub num_samples: u64,
    /// DCFR+ iterations per solve batch.
    pub solve_iterations: u32,
    /// Number of flop spots per solve batch.
    pub batch_size: usize,
    /// Maximum capacity of the GPU reservoir buffer.
    pub reservoir_capacity: usize,
    /// Number of hidden layers in the flop CFVNet.
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
impl Default for FlopTrainingConfig {
    fn default() -> Self {
        Self {
            turn_model_path: PathBuf::from("models/turn_cfvnet/model"),
            turn_hidden_layers: 7,
            turn_hidden_size: 500,
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
            output_dir: PathBuf::from("models/flop_cfvnet"),
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

/// Build a flop-specific `GpuBatchBuilder` using a depth-limited flop tree.
#[cfg(feature = "training")]
fn build_flop_batch_builder(
    gpu: &GpuContext,
    config: &BatchConfig,
    ref_pot: i32,
    ref_stack: i32,
) -> Result<GpuBatchBuilder, String> {
    GpuBatchBuilder::new_flop(gpu, config, ref_pot, ref_stack)
}

/// Pad 3-card flop boards to 5-card format for the reservoir encoder.
///
/// The 4th and 5th cards are set to 255 (>= 52), which the kernel ignores in
/// one-hot encoding and masking. Input: `[num_spots * 3]`.
/// Output: `[num_spots * 5]`.
#[cfg(feature = "training")]
fn pad_boards_to_5_from_flop(boards_3: &[u32]) -> Vec<u32> {
    let num_spots = boards_3.len() / 3;
    let mut out = Vec::with_capacity(num_spots * 5);
    for i in 0..num_spots {
        out.push(boards_3[i * 3]);
        out.push(boards_3[i * 3 + 1]);
        out.push(boards_3[i * 3 + 2]);
        out.push(255); // dummy 4th card (>= 52, ignored by kernel)
        out.push(255); // dummy 5th card (>= 52, ignored by kernel)
    }
    out
}

/// Run the GPU flop CFVNet training pipeline with CUDA-native leaf evaluation.
///
/// Loads the trained turn model, then runs the Supremus-style loop:
/// sample flop boards -> solve with leaf eval -> reservoir insert -> train -> validate.
#[cfg(feature = "training")]
pub fn train_flop_cfvnet_cuda<B: AutodiffBackend>(
    config: &FlopTrainingConfig,
    device: &B::Device,
) -> Result<(), String> {
    let gpu = GpuContext::new(0).map_err(|e| format!("CUDA init: {e}"))?;

    // 1. Load turn model using CUDA-native inference
    eprintln!(
        "Loading turn model from {:?} ({}x{}) with CUDA-native inference...",
        config.turn_model_path, config.turn_hidden_layers, config.turn_hidden_size
    );

    // Max batch size for leaf evaluation:
    // num_boundaries * 49 turn cards * batch_size spots
    // Typical flop tree has ~4-8 boundaries, so 8 * 49 * batch_size is safe.
    let max_leaf_batch = 8 * 49 * config.batch_size;
    let mut leaf_eval = GpuLeafEvaluatorCuda::load(
        &config.turn_model_path,
        &gpu,
        config.turn_hidden_layers,
        config.turn_hidden_size,
        max_leaf_batch,
    )?;

    // 2. Build shared flop tree topology (with depth boundaries)
    let batch_config = BatchConfig::default();
    let builder = build_flop_batch_builder(
        &gpu,
        &batch_config,
        config.ref_pot,
        config.ref_stack,
    )?;

    let ref_tree = &builder.topology().ref_tree;
    let num_boundaries = ref_tree.boundary_indices.len();
    eprintln!(
        "Flop tree: {} nodes, {} infosets, {} boundaries, {} fold terminals",
        ref_tree.num_nodes(),
        ref_tree.num_infosets,
        num_boundaries,
        ref_tree.node_types.iter().filter(|t| **t == crate::tree::NodeType::TerminalFold).count(),
    );

    // Verify max_leaf_batch is sufficient
    let actual_leaf_batch = num_boundaries * 49 * config.batch_size;
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

    let mut sampler = GpuFlopSampler::new(&gpu, config.batch_size, config.seed + 3_000_000)?;

    let pots = vec![config.ref_pot as f32; config.batch_size];
    let stacks = vec![config.ref_stack as f32; config.batch_size];
    let gpu_pots = gpu.upload(&pots).map_err(|e| format!("upload pots: {e}"))?;
    let gpu_stacks = gpu.upload(&stacks).map_err(|e| format!("upload stacks: {e}"))?;

    eprintln!("Note: Ground-truth validation for flop uses turn-model leaf eval");
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

    eprintln!("Starting GPU flop training pipeline (CUDA-native leaf eval)");
    eprintln!("  Batch size: {} spots", config.batch_size);
    eprintln!("  Solve iterations: {}", config.solve_iterations);
    eprintln!("  Reservoir capacity: {}", config.reservoir_capacity);
    eprintln!(
        "  Flop model: {} layers x {} hidden",
        config.hidden_layers, config.hidden_size
    );
    eprintln!(
        "  Turn model: {} layers x {} hidden",
        config.turn_hidden_layers, config.turn_hidden_size
    );
    eprintln!("  Depth boundaries: {}", num_boundaries);
    eprintln!("  Turn cards per boundary: 49");
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
        pb.set_prefix("FLOP");
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
        let mut flop_solver = FlopBatchSolverCuda::new(
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
        let mut solve_result = flop_solver.solve_with_cfvs(config.solve_iterations)?;
        let t3 = Instant::now();

        // === INSERT PHASE ===
        // Pad 3-card boards to 5-card for the reservoir encoder kernel.
        // The 4th and 5th cards are set to 255 (>= 52), which the kernel ignores.
        let padded_boards = pad_boards_to_5_from_flop(&boards_host);
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
    fn test_flop_pipeline_config_defaults() {
        let config = FlopTrainingConfig::default();
        assert_eq!(config.batch_size, 1000);
        assert_eq!(config.solve_iterations, 4000);
        assert_eq!(config.reservoir_capacity, 100_000);
        assert_eq!(config.turn_hidden_layers, 7);
        assert_eq!(config.turn_hidden_size, 500);
    }

    #[test]
    fn test_build_flop_tree_has_boundaries() {
        use range_solver::bet_size::BetSizeOptions;
        use crate::tree::NodeType;

        let bet_sizes = BetSizeOptions::try_from(("50%,a", "")).unwrap();
        let flop = [0u8, 1, 2];

        let mut game = crate::tree::build_flop_game(flop, 100, 100, &bet_sizes);
        let flat = FlatTree::from_postflop_game(&mut game);

        // Should have depth boundaries
        assert!(
            !flat.boundary_indices.is_empty(),
            "Flop tree should have depth boundaries"
        );

        // Should have fold terminals
        let num_fold = flat.node_types.iter().filter(|t| **t == NodeType::TerminalFold).count();
        assert!(num_fold > 0, "Should have fold terminals");

        // Should NOT have showdown terminals
        let num_sd = flat.node_types.iter().filter(|t| **t == NodeType::TerminalShowdown).count();
        assert_eq!(num_sd, 0, "Should have no showdown terminals");
    }

    #[test]
    fn test_pad_boards_to_5_from_flop() {
        let boards_3 = vec![0u32, 1, 2, 10, 11, 12];
        let padded = pad_boards_to_5_from_flop(&boards_3);
        assert_eq!(padded.len(), 10);
        assert_eq!(padded[0..3], [0, 1, 2]);
        assert_eq!(padded[3], 255);
        assert_eq!(padded[4], 255);
        assert_eq!(padded[5..8], [10, 11, 12]);
        assert_eq!(padded[8], 255);
        assert_eq!(padded[9], 255);
    }
}
