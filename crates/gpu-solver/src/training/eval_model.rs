//! GPU model evaluation: compare neural predictions against exact GPU solves.
//!
//! For each of `num_spots` random situations:
//! 1. Sample a random board and uniform ranges
//! 2. Solve exactly on GPU at high iteration count
//! 3. Encode the input and run the model forward pass
//! 4. Compare predicted CFVs against ground-truth CFVs (MAE, max error)
//!
//! Supports both river (5-card board, direct BatchGpuSolver) and
//! turn (4-card board, TurnBatchSolverCuda with leaf model) evaluation.

#[cfg(feature = "training")]
use std::time::Instant;

#[cfg(feature = "training")]
use crate::batch::BatchConfig;
#[cfg(feature = "training")]
use crate::gpu::GpuContext;
#[cfg(feature = "training")]
use crate::training::builder::GpuBatchBuilder;
#[cfg(feature = "training")]
use crate::training::cuda_net::{CudaNetInference, GpuLeafEvaluatorCuda};
#[cfg(feature = "training")]
use crate::training::reservoir::{encode_input, OUTPUT_SIZE};
#[cfg(feature = "training")]
use crate::training::sampler::{GpuSampler, GpuTurnSampler};
#[cfg(feature = "training")]
use crate::training::turn_solver::TurnBatchSolverCuda;

#[cfg(feature = "training")]
use range_solver::bet_size::BetSizeOptions;

/// Configuration for the model evaluation command.
#[cfg(feature = "training")]
#[derive(Debug, Clone)]
pub struct EvalModelConfig {
    /// Path to the model checkpoint (without .mpk.gz extension).
    pub model_path: std::path::PathBuf,
    /// Number of hidden layers in the model.
    pub hidden_layers: usize,
    /// Hidden layer width.
    pub hidden_size: usize,
    /// Street to evaluate: "river" or "turn".
    pub street: String,
    /// Number of random spots to evaluate.
    pub num_spots: usize,
    /// DCFR+ iterations for ground-truth solves.
    pub solve_iterations: u32,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Reference pot size.
    pub ref_pot: i32,
    /// Reference effective stack.
    pub ref_stack: i32,
    /// Path to leaf model (required for turn eval).
    pub leaf_model: Option<std::path::PathBuf>,
    /// Number of hidden layers in the leaf model.
    pub leaf_hidden_layers: usize,
    /// Hidden layer width of the leaf model.
    pub leaf_hidden_size: usize,
    /// Bet size string (e.g. "50%,a").
    pub bet_sizes: String,
}

/// Per-spot evaluation result.
#[cfg(feature = "training")]
struct SpotResult {
    mae: f32,
    max_error: f32,
    valid_combos: usize,
}

/// Run the model evaluation pipeline.
///
/// Solves random spots on GPU, runs model predictions, and prints MAE.
#[cfg(feature = "training")]
pub fn run_eval_model(config: &EvalModelConfig) -> Result<(), String> {
    match config.street.as_str() {
        "river" => run_eval_river(config),
        "turn" => run_eval_turn(config),
        other => Err(format!("Unknown street: {other}. Use 'river' or 'turn'.")),
    }
}

// ---------------------------------------------------------------------------
// River evaluation
// ---------------------------------------------------------------------------

#[cfg(feature = "training")]
fn run_eval_river(config: &EvalModelConfig) -> Result<(), String> {
    let gpu = GpuContext::new(0).map_err(|e| format!("CUDA init: {e}"))?;

    eprintln!(
        "Evaluating river model: {} ({}x{})",
        config.model_path.display(),
        config.hidden_layers,
        config.hidden_size,
    );
    eprintln!(
        "Solving {} spots at {} iterations...\n",
        config.num_spots, config.solve_iterations,
    );

    // Load the model for prediction
    let mut model = CudaNetInference::load_from_burn(
        &config.model_path,
        &gpu,
        config.hidden_layers,
        config.hidden_size,
        1, // max_batch_size = 1 (one encoded input at a time)
    )?;

    // Build shared river tree topology
    let bet_sizes = BetSizeOptions::try_from((config.bet_sizes.as_str(), ""))
        .map_err(|e| format!("Invalid bet sizes: {e}"))?;
    let batch_config = BatchConfig {
        oop_bet_sizes: bet_sizes.clone(),
        ip_bet_sizes: bet_sizes,
    };
    let builder = GpuBatchBuilder::new(&gpu, &batch_config, config.ref_pot, config.ref_stack)?;
    let num_combinations = builder.topology().ref_tree.num_combinations;

    // Sampler for random river boards
    let mut sampler = GpuSampler::new(&gpu, 1, config.seed)?;

    let mut results: Vec<SpotResult> = Vec::with_capacity(config.num_spots);
    let start = Instant::now();

    for spot_idx in 0..config.num_spots {
        // 1. Sample a random river situation (batch_size=1)
        sampler.sample(&gpu)?;

        // 2. Build solver and solve
        let mut solver = builder.build_from_gpu_data(
            &gpu,
            sampler.boards(),
            sampler.ranges_oop(),
            sampler.ranges_ip(),
            1, // num_spots=1
        )?;

        let solve_result = solver.solve_with_cfvs(config.solve_iterations, None)?;

        // 3. Download ground-truth CFVs
        let gt_cfvs_oop: Vec<f32> = gpu
            .download(&solve_result.cfvs_oop)
            .map_err(|e| format!("download cfvs_oop: {e}"))?;
        let gt_cfvs_ip: Vec<f32> = gpu
            .download(&solve_result.cfvs_ip)
            .map_err(|e| format!("download cfvs_ip: {e}"))?;

        // 4. Download board and ranges for encoding
        let boards_host: Vec<u32> = gpu
            .download(sampler.boards())
            .map_err(|e| format!("download boards: {e}"))?;
        let ranges_oop_host: Vec<f32> = gpu
            .download(sampler.ranges_oop())
            .map_err(|e| format!("download ranges_oop: {e}"))?;
        let ranges_ip_host: Vec<f32> = gpu
            .download(sampler.ranges_ip())
            .map_err(|e| format!("download ranges_ip: {e}"))?;

        // 5. Evaluate both perspectives and combine
        let result = evaluate_both_perspectives(
            &gpu,
            &mut model,
            &boards_host,
            &ranges_oop_host,
            &ranges_ip_host,
            &gt_cfvs_oop,
            &gt_cfvs_ip,
            config.ref_pot as f32,
            config.ref_stack as f32,
            num_combinations,
        )?;

        eprintln!(
            "  Spot {:3}/{}: MAE={:.6}  max={:.6}  valid={}",
            spot_idx + 1,
            config.num_spots,
            result.mae,
            result.max_error,
            result.valid_combos,
        );

        results.push(result);
    }

    print_summary(&results, start.elapsed());
    Ok(())
}

// ---------------------------------------------------------------------------
// Turn evaluation
// ---------------------------------------------------------------------------

#[cfg(feature = "training")]
fn run_eval_turn(config: &EvalModelConfig) -> Result<(), String> {
    let leaf_model_path = config
        .leaf_model
        .as_ref()
        .ok_or("--leaf-model is required for turn evaluation")?;

    let gpu = GpuContext::new(0).map_err(|e| format!("CUDA init: {e}"))?;

    eprintln!(
        "Evaluating turn model: {} ({}x{})",
        config.model_path.display(),
        config.hidden_layers,
        config.hidden_size,
    );
    eprintln!(
        "Leaf model: {} ({}x{})",
        leaf_model_path.display(),
        config.leaf_hidden_layers,
        config.leaf_hidden_size,
    );
    eprintln!(
        "Solving {} spots at {} iterations...\n",
        config.num_spots, config.solve_iterations,
    );

    // Load the turn model for prediction
    let mut model = CudaNetInference::load_from_burn(
        &config.model_path,
        &gpu,
        config.hidden_layers,
        config.hidden_size,
        1, // max_batch_size = 1
    )?;

    // Load leaf (river) model for solving
    // Max leaf batch = num_boundaries * 48 rivers * 1 spot
    // Typical turn tree has ~4-8 boundaries, use generous upper bound.
    let max_leaf_batch = 8 * 48;
    let mut leaf_eval = GpuLeafEvaluatorCuda::load(
        leaf_model_path,
        &gpu,
        config.leaf_hidden_layers,
        config.leaf_hidden_size,
        max_leaf_batch,
    )?;

    // Build shared turn tree topology
    let bet_sizes = BetSizeOptions::try_from((config.bet_sizes.as_str(), ""))
        .map_err(|e| format!("Invalid bet sizes: {e}"))?;
    let batch_config = BatchConfig {
        oop_bet_sizes: bet_sizes.clone(),
        ip_bet_sizes: bet_sizes,
    };
    let builder = GpuBatchBuilder::new_turn(&gpu, &batch_config, config.ref_pot, config.ref_stack)?;
    let ref_tree = &builder.topology().ref_tree;
    let num_boundaries = ref_tree.boundary_indices.len();
    let num_combinations = ref_tree.num_combinations;
    eprintln!(
        "Turn tree: {} nodes, {} boundaries",
        ref_tree.num_nodes(),
        num_boundaries,
    );

    // Verify leaf batch size
    let actual_leaf_batch = num_boundaries * 48;
    assert!(
        actual_leaf_batch <= max_leaf_batch,
        "Leaf batch size {} exceeds pre-allocated max {}",
        actual_leaf_batch,
        max_leaf_batch,
    );

    // Sampler for random 4-card boards
    let mut sampler = GpuTurnSampler::new(&gpu, 1, config.seed)?;

    let mut results: Vec<SpotResult> = Vec::with_capacity(config.num_spots);
    let start = Instant::now();

    for spot_idx in 0..config.num_spots {
        // 1. Sample a random turn situation (batch_size=1)
        sampler.sample(&gpu)?;

        // 2. Download boards for the turn solver constructor
        let boards_host: Vec<u32> = gpu
            .download(sampler.boards())
            .map_err(|e| format!("download boards: {e}"))?;

        // 3. Build depth-limited solver + turn solver with leaf eval
        let batch_solver = builder.build_depth_limited(
            &gpu,
            sampler.ranges_oop(),
            sampler.ranges_ip(),
            1, // num_spots=1
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
            1, // num_spots=1
            num_combinations,
        )?;

        // 4. Solve
        let solve_result = turn_solver.solve_with_cfvs(config.solve_iterations)?;

        // 5. Download ground-truth CFVs
        let gt_cfvs_oop: Vec<f32> = gpu
            .download(&solve_result.cfvs_oop)
            .map_err(|e| format!("download cfvs_oop: {e}"))?;
        let gt_cfvs_ip: Vec<f32> = gpu
            .download(&solve_result.cfvs_ip)
            .map_err(|e| format!("download cfvs_ip: {e}"))?;

        // 6. Download ranges for encoding
        let ranges_oop_host: Vec<f32> = gpu
            .download(sampler.ranges_oop())
            .map_err(|e| format!("download ranges_oop: {e}"))?;
        let ranges_ip_host: Vec<f32> = gpu
            .download(sampler.ranges_ip())
            .map_err(|e| format!("download ranges_ip: {e}"))?;

        // 7. For encoding, pad 4-card board to 5-card (5th card = 255)
        let boards_padded: Vec<u32> = pad_board_to_5(&boards_host);

        // 8. Evaluate both perspectives
        let result = evaluate_both_perspectives(
            &gpu,
            &mut model,
            &boards_padded,
            &ranges_oop_host,
            &ranges_ip_host,
            &gt_cfvs_oop,
            &gt_cfvs_ip,
            config.ref_pot as f32,
            config.ref_stack as f32,
            num_combinations,
        )?;

        eprintln!(
            "  Spot {:3}/{}: MAE={:.6}  max={:.6}  valid={}",
            spot_idx + 1,
            config.num_spots,
            result.mae,
            result.max_error,
            result.valid_combos,
        );

        results.push(result);
    }

    print_summary(&results, start.elapsed());
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Evaluate model predictions against ground-truth CFVs for both players.
///
/// Encodes the situation from both OOP and IP perspectives, runs the model
/// forward pass, and computes MAE and max error across all valid combos.
///
/// Ground-truth CFVs are in raw DCFR+ units. Model predictions are in
/// pot-relative units. We convert ground truth to pot-relative for comparison:
///   pot_relative = raw * num_combinations / pot + 0.5
#[cfg(feature = "training")]
fn evaluate_both_perspectives(
    gpu: &GpuContext,
    model: &mut CudaNetInference,
    board: &[u32],
    ranges_oop: &[f32],
    ranges_ip: &[f32],
    gt_cfvs_oop: &[f32],
    gt_cfvs_ip: &[f32],
    pot: f32,
    stack: f32,
    num_combinations: f64,
) -> Result<SpotResult, String> {
    let mut total_abs_error = 0.0f64;
    let mut max_error = 0.0f32;
    let mut total_valid = 0usize;

    // Convert ground-truth CFVs from raw DCFR+ units to pot-relative
    let num_combs_f32 = num_combinations as f32;
    let gt_oop_pr: Vec<f32> = gt_cfvs_oop.iter()
        .map(|&v| v * num_combs_f32 / pot + 0.5)
        .collect();
    let gt_ip_pr: Vec<f32> = gt_cfvs_ip.iter()
        .map(|&v| v * num_combs_f32 / pot + 0.5)
        .collect();

    for player in 0..2u8 {
        let (input, mask, _game_value) = encode_input(
            ranges_oop,
            ranges_ip,
            board,
            pot,
            stack,
            player,
        );
        let gt_cfvs = if player == 0 { &gt_oop_pr } else { &gt_ip_pr };

        // Upload encoded input to GPU and run forward pass
        let gpu_input = gpu
            .upload(&input)
            .map_err(|e| format!("upload input: {e}"))?;
        let gpu_output = model
            .forward(gpu, &gpu_input, 1)
            .map_err(|e| format!("forward: {e}"))?;
        let predicted: Vec<f32> = gpu
            .download(gpu_output)
            .map_err(|e| format!("download output: {e}"))?;

        // Compare only valid (unblocked) combos
        let mut player_gt_min = f32::MAX;
        let mut player_gt_max = f32::MIN;
        let mut player_pred_min = f32::MAX;
        let mut player_pred_max = f32::MIN;
        for i in 0..OUTPUT_SIZE {
            if mask[i] > 0.5 {
                let err = (predicted[i] - gt_cfvs[i]).abs();
                total_abs_error += f64::from(err);
                if err > max_error {
                    max_error = err;
                }
                total_valid += 1;
                player_gt_min = player_gt_min.min(gt_cfvs[i]);
                player_gt_max = player_gt_max.max(gt_cfvs[i]);
                player_pred_min = player_pred_min.min(predicted[i]);
                player_pred_max = player_pred_max.max(predicted[i]);
            }
        }
        if player == 0 {
            eprintln!("    OOP gt=[{:.4}..{:.4}] pred=[{:.4}..{:.4}]",
                player_gt_min, player_gt_max, player_pred_min, player_pred_max);
        }
    }

    let mae = if total_valid > 0 {
        (total_abs_error / total_valid as f64) as f32
    } else {
        0.0
    };

    Ok(SpotResult {
        mae,
        max_error,
        valid_combos: total_valid,
    })
}

/// Print the summary of all spot results.
#[cfg(feature = "training")]
fn print_summary(results: &[SpotResult], elapsed: std::time::Duration) {
    if results.is_empty() {
        eprintln!("\nNo spots evaluated.");
        return;
    }

    let total_combos: usize = results.iter().map(|r| r.valid_combos).sum();
    let global_mae = if total_combos > 0 {
        // Weighted average: sum(mae * valid) / total_valid
        let weighted_sum: f64 = results
            .iter()
            .map(|r| f64::from(r.mae) * r.valid_combos as f64)
            .sum();
        (weighted_sum / total_combos as f64) as f32
    } else {
        0.0
    };
    let global_max: f32 = results
        .iter()
        .map(|r| r.max_error)
        .fold(0.0f32, f32::max);

    // MAE is now in pot-relative units (model and ground truth both pot-relative).
    // mBB = pot-relative MAE * 1000
    let mbb = global_mae * 1000.0;

    eprintln!();
    eprintln!("Summary:");
    eprintln!("  Pot-relative MAE:           {global_mae:.6} ({mbb:.1} mBB/hand)");
    eprintln!("  Max Error (pot-relative):   {global_max:.6}");
    eprintln!("  Spots evaluated:            {}", results.len());
    eprintln!("  Total combos evaluated:     {total_combos}");
    eprintln!("  Elapsed:                    {:.1}s", elapsed.as_secs_f64());
}

/// Pad a 4-card turn board to a 5-card format with a sentinel (255).
///
/// The reservoir encoder treats card values >= 52 as absent.
#[cfg(feature = "training")]
fn pad_board_to_5(boards_4: &[u32]) -> Vec<u32> {
    let num_spots = boards_4.len() / 4;
    let mut out = Vec::with_capacity(num_spots * 5);
    for s in 0..num_spots {
        out.extend_from_slice(&boards_4[s * 4..(s + 1) * 4]);
        out.push(255); // sentinel for missing river card
    }
    out
}
