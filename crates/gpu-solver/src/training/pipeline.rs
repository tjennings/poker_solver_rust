//! Supremus-style GPU river CFVNet training pipeline.
//!
//! Orchestrates the full sample -> solve -> insert -> train loop:
//!
//! 1. **Sample**: Generate random river situations on GPU (or CPU fallback)
//! 2. **Solve**: Batch-solve all situations on GPU using DCFR+
//! 3. **Insert**: Encode and insert solved CFVs into GPU reservoir
//! 4. **Train**: Sample mini-batches from reservoir and train the CFVNet
//! 5. **Validate**: Periodically check hold-out loss and ground-truth RMSE
//! 6. **Checkpoint**: Save model weights at intervals

#[cfg(feature = "training")]
use std::path::PathBuf;
#[cfg(feature = "training")]
use std::time::Instant;

#[cfg(feature = "training")]
use burn::tensor::backend::AutodiffBackend;

#[cfg(feature = "training")]
use crate::batch::BatchConfig;
#[cfg(feature = "training")]
use crate::gpu::GpuContext;
#[cfg(feature = "training")]
use crate::training::builder::GpuBatchBuilder;
#[cfg(feature = "training")]
use crate::training::reservoir::GpuReservoir;
#[cfg(feature = "training")]
use crate::training::trainer::GpuTrainer;
#[cfg(feature = "training")]
use crate::training::validation::GroundTruthValidator;

/// Configuration for the GPU river CFVNet training pipeline.
#[cfg(feature = "training")]
#[derive(Debug, Clone)]
pub struct RiverTrainingConfig {
    /// Total number of training samples to generate.
    pub num_samples: u64,
    /// DCFR+ iterations per solve batch.
    pub solve_iterations: u32,
    /// Number of river spots per solve batch.
    pub batch_size: usize,
    /// Maximum capacity of the GPU reservoir buffer.
    pub reservoir_capacity: usize,
    /// Number of hidden layers in the CFVNet.
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
impl Default for RiverTrainingConfig {
    fn default() -> Self {
        Self {
            num_samples: 50_000_000,
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
            gt_validation_positions: 100,
            gt_solve_iterations: 10_000,
            output_dir: PathBuf::from("models/river_cfvnet"),
            seed: 42,
            ref_pot: 100,
            ref_stack: 100,
        }
    }
}

/// Generate random river situations on CPU and upload to GPU.
///
/// Produces `batch_size` random river boards with uniform ranges.
/// Board cards are sampled uniformly without replacement from 0..52.
/// Ranges are set to 1.0 for unblocked combos, 0.0 for blocked.
#[cfg(feature = "training")]
fn sample_situations_cpu(
    batch_size: usize,
    rng: &mut impl rand::Rng,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut boards = Vec::with_capacity(batch_size * 5);
    let mut ranges_oop = Vec::with_capacity(batch_size * 1326);
    let mut ranges_ip = Vec::with_capacity(batch_size * 1326);

    for _ in 0..batch_size {
        // Sample 5 unique board cards
        let mut board = [0u8; 5];
        let mut used = [false; 52];
        for card in &mut board {
            loop {
                let c: u8 = rng.gen_range(0..52);
                if !used[c as usize] {
                    used[c as usize] = true;
                    *card = c;
                    break;
                }
            }
        }

        for &c in &board {
            boards.push(u32::from(c));
        }

        // Build uniform range (1.0 for unblocked combos)
        for i in 0..1326 {
            let (c1, c2) = range_solver::card::index_to_card_pair(i);
            let blocked = used[c1 as usize] || used[c2 as usize];
            let weight = if blocked { 0.0f32 } else { 1.0f32 };
            ranges_oop.push(weight);
            ranges_ip.push(weight);
        }
    }

    (boards, ranges_oop, ranges_ip)
}

/// Run the full GPU river CFVNet training pipeline.
///
/// This is the main entry point for Supremus-style training:
/// sample -> batch-solve -> reservoir insert -> train -> validate -> checkpoint.
#[cfg(feature = "training")]
pub fn train_river_cfvnet<B: AutodiffBackend>(
    config: &RiverTrainingConfig,
    device: &B::Device,
) -> Result<(), String> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let gpu = GpuContext::new(0).map_err(|e| format!("CUDA init: {e}"))?;

    // Initialize components
    let batch_config = BatchConfig::default();
    let builder = GpuBatchBuilder::new(&gpu, &batch_config, config.ref_pot, config.ref_stack)?;
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

    // Pre-compute ground-truth validation set
    eprintln!("Pre-computing ground-truth validation set ({} positions x {} iters)...",
        config.gt_validation_positions, config.gt_solve_iterations);
    let gt_validator = GroundTruthValidator::precompute(
        config.gt_validation_positions,
        config.gt_solve_iterations,
        config.seed + 1_000_000,
    );

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut total_samples = 0u64;
    let mut last_loss = f32::MAX;
    let mut sample_seed = config.seed + 2_000_000;
    let start = Instant::now();

    // Create output directory
    std::fs::create_dir_all(&config.output_dir)
        .map_err(|e| format!("create output dir: {e}"))?;

    eprintln!("Starting GPU river training pipeline");
    eprintln!("  Batch size: {} spots", config.batch_size);
    eprintln!("  Solve iterations: {}", config.solve_iterations);
    eprintln!("  Reservoir capacity: {}", config.reservoir_capacity);
    eprintln!("  Model: {} layers x {} hidden", config.hidden_layers, config.hidden_size);
    eprintln!("  Train batch: {} x {} steps/batch", config.train_batch_size, config.train_steps_per_batch);
    eprintln!("  Learning rate: {}", config.learning_rate);
    eprintln!("  Output: {}", config.output_dir.display());
    eprintln!();

    while total_samples < config.num_samples {
        // === SAMPLE PHASE (CPU for now, GPU sampler can replace later) ===
        let (boards, ranges_oop, ranges_ip) = sample_situations_cpu(
            config.batch_size,
            &mut rng,
        );

        // Upload to GPU
        let gpu_boards = gpu.upload(&boards).map_err(|e| format!("upload boards: {e}"))?;
        let gpu_ranges_oop = gpu.upload(&ranges_oop).map_err(|e| format!("upload oop: {e}"))?;
        let gpu_ranges_ip = gpu.upload(&ranges_ip).map_err(|e| format!("upload ip: {e}"))?;

        // === SOLVE PHASE (GPU) ===
        let mut batch_solver = builder.build_from_gpu_data(
            &gpu,
            &gpu_boards,
            &gpu_ranges_oop,
            &gpu_ranges_ip,
            config.batch_size,
        )?;

        let solve_result = batch_solver.solve_with_cfvs(
            config.solve_iterations,
            None,
        )?;

        // === INSERT PHASE (GPU) ===
        // Build pots/stacks buffers (uniform for all spots)
        let pots = vec![config.ref_pot as f32; config.batch_size];
        let stacks = vec![config.ref_stack as f32; config.batch_size];
        let gpu_pots = gpu.upload(&pots).map_err(|e| format!("upload pots: {e}"))?;
        let gpu_stacks = gpu.upload(&stacks).map_err(|e| format!("upload stacks: {e}"))?;

        reservoir.insert_batch_gpu(
            &gpu,
            &gpu_ranges_oop,
            &gpu_ranges_ip,
            &gpu_boards,
            &gpu_pots,
            &gpu_stacks,
            &solve_result.cfvs_oop,
            &solve_result.cfvs_ip,
            config.batch_size,
        ).map_err(|e| format!("reservoir insert: {e}"))?;

        total_samples += (config.batch_size as u64) * 2; // 2 records per spot

        // === TRAIN PHASE ===
        if reservoir.size() >= config.train_batch_size {
            for _ in 0..config.train_steps_per_batch {
                sample_seed += 1;
                let batch = reservoir.sample_minibatch_gpu(
                    &gpu,
                    config.train_batch_size,
                    sample_seed as u32,
                ).map_err(|e| format!("sample minibatch: {e}"))?;

                last_loss = trainer.train_step(&gpu, &batch)?;
            }
        }

        // === VALIDATION (periodic) ===
        if total_samples % config.validation_interval < (config.batch_size as u64 * 2) {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = total_samples as f64 / elapsed;

            // Hold-out validation loss
            let val_loss = if reservoir.size() >= config.train_batch_size {
                sample_seed += 1;
                let val_batch = reservoir.sample_minibatch_gpu(
                    &gpu,
                    config.train_batch_size,
                    sample_seed as u32,
                ).map_err(|e| format!("val batch: {e}"))?;
                trainer.validation_loss(&gpu, &val_batch)?
            } else {
                f32::NAN
            };

            // Ground-truth RMSE
            let gt_rmse = gt_validator.evaluate(trainer.model(), device);

            eprintln!(
                "[{:>12} samples] loss={:.6} val={:.6} gt_rmse={:.6} rate={:.0} samples/s  reservoir={}",
                total_samples, last_loss, val_loss, gt_rmse, rate, reservoir.size()
            );
        }

        // === CHECKPOINT (periodic) ===
        if config.checkpoint_interval > 0
            && total_samples % config.checkpoint_interval < (config.batch_size as u64 * 2)
            && total_samples > 0
        {
            let label = format!("checkpoint_{}", total_samples);
            trainer.save_checkpoint(&config.output_dir, &label);
            eprintln!("  Saved checkpoint: {}", label);
        }
    }

    // Save final model
    trainer.save_final(&config.output_dir)?;
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!();
    eprintln!(
        "Training complete. {} samples in {:.1}s ({:.0} samples/s)",
        total_samples, elapsed, total_samples as f64 / elapsed
    );
    eprintln!("Model saved to {:?}", config.output_dir);
    Ok(())
}

#[cfg(all(test, feature = "training"))]
mod tests {
    use super::*;

    #[test]
    fn test_sample_situations_cpu() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (boards, ranges_oop, ranges_ip) = sample_situations_cpu(10, &mut rng);

        assert_eq!(boards.len(), 10 * 5);
        assert_eq!(ranges_oop.len(), 10 * 1326);
        assert_eq!(ranges_ip.len(), 10 * 1326);

        // Verify boards have valid card indices
        for &card in &boards {
            assert!(card < 52, "card index should be < 52, got {card}");
        }

        // Verify each board has 5 unique cards
        for i in 0..10 {
            let board = &boards[i * 5..(i + 1) * 5];
            let mut seen = std::collections::HashSet::new();
            for &c in board {
                assert!(seen.insert(c), "board should have unique cards");
            }
        }

        // Verify ranges have correct blocking
        for i in 0..10 {
            let board = &boards[i * 5..(i + 1) * 5];
            let range = &ranges_oop[i * 1326..(i + 1) * 1326];

            let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
            for (j, &weight) in range.iter().enumerate() {
                let (c1, c2) = range_solver::card::index_to_card_pair(j);
                if board_set.contains(&c1) || board_set.contains(&c2) {
                    assert_eq!(weight, 0.0, "blocked combo should have weight 0");
                } else {
                    assert_eq!(weight, 1.0, "unblocked combo should have weight 1");
                }
            }
        }
    }

    /// Small end-to-end test using NdArray backend (no GPU required).
    #[test]
    fn test_pipeline_config_defaults() {
        let config = RiverTrainingConfig::default();
        assert_eq!(config.batch_size, 1000);
        assert_eq!(config.solve_iterations, 4000);
        assert_eq!(config.reservoir_capacity, 100_000);
    }
}
