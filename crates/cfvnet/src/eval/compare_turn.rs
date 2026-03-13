//! Turn model comparison against CfvSubgameSolver ground truth.
//!
//! Two modes:
//! - **compare-net**: solver uses `RiverNetEvaluator` at leaves (fast, approximate)
//! - **compare-exact**: solver uses `ExactRiverEvaluator` at leaves (slow, exact)

use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use burn::tensor::{Tensor, TensorData};
use poker_solver_core::blueprint_v2::cfv_subgame_solver::{
    CfvSubgameSolver, ExactRiverEvaluator, LeafEvaluator,
};
use poker_solver_core::blueprint_v2::game_tree::GameTree;
use poker_solver_core::blueprint_v2::subgame_cfr::SubgameHands;
use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::poker::{Card, Suit, Value};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use range_solver::card::card_pair_to_index;

use crate::config::CfvnetConfig;
use crate::datagen::range_gen::NUM_COMBOS;
use crate::datagen::sampler::{sample_situation, Situation};
use crate::eval::compare::{ComparisonSummary, SpotResult};
use crate::eval::metrics::compute_prediction_metrics;
use crate::eval::river_net_evaluator::RiverNetEvaluator;
use crate::model::network::{CfvNet, NUM_COMBOS as NET_NUM_COMBOS, input_size};

use std::path::Path;

type B = NdArray;

/// Convert a range-solver `u8` card to an `rs_poker::core::Card`.
fn u8_to_rs_card(id: u8) -> Card {
    let rank = id / 4;
    let suit_id = id % 4;
    let value = Value::from(rank);
    let suit = match suit_id {
        0 => Suit::Club,
        1 => Suit::Diamond,
        2 => Suit::Heart,
        3 => Suit::Spade,
        _ => unreachable!(),
    };
    Card::new(value, suit)
}

/// Convert an `rs_poker::core::Card` to a range-solver `u8` card.
fn rs_card_to_u8(card: Card) -> u8 {
    let rank = card.value as u8;
    let suit = match card.suit {
        Suit::Club => 0,
        Suit::Diamond => 1,
        Suit::Heart => 2,
        Suit::Spade => 3,
    };
    4 * rank + suit
}

/// Parse config bet size strings into pot fractions, skipping "a" (all-in).
fn parse_bet_sizes(sizes: &[String]) -> Vec<f64> {
    sizes
        .iter()
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.eq_ignore_ascii_case("a") {
                return None;
            }
            let num_str = trimmed.trim_end_matches('%');
            num_str.parse::<f64>().ok().map(|v| v / 100.0)
        })
        .collect()
}

/// Solve a turn situation with the given evaluator and return 1326-indexed CFVs.
///
/// Returns `(cfvs_1326, valid_mask)` for the given traverser.
fn solve_and_extract(
    sit: &Situation,
    bet_sizes: &[Vec<f64>],
    solver_iterations: u32,
    evaluator: Box<dyn LeafEvaluator>,
    traverser: u8,
) -> ([f32; NUM_COMBOS], [bool; NUM_COMBOS]) {
    let board_cards: Vec<Card> = sit.board_cards().iter().map(|&c| u8_to_rs_card(c)).collect();
    let pot = f64::from(sit.pot);
    let effective_stack = f64::from(sit.effective_stack);
    let invested = [pot / 2.0; 2];
    let starting_stack = effective_stack + pot / 2.0;

    let tree = GameTree::build_subgame(
        Street::Turn,
        pot,
        invested,
        starting_stack,
        bet_sizes,
        Some(1),
    );

    let hands = SubgameHands::enumerate(&board_cards);
    let mut solver =
        CfvSubgameSolver::new(tree, hands.clone(), &board_cards, evaluator, starting_stack);
    solver.train(solver_iterations);

    let cfvs_combo = solver.root_cfvs(traverser);

    let mut cfvs_1326 = [0.0_f32; NUM_COMBOS];
    let mut valid_mask = [false; NUM_COMBOS];
    let half_pot = pot / 2.0;
    let norm = if half_pot > 0.0 { half_pot } else { 1.0 };

    for (combo_idx, combo) in hands.combos.iter().enumerate() {
        let c0 = rs_card_to_u8(combo[0]);
        let c1 = rs_card_to_u8(combo[1]);
        let idx = card_pair_to_index(c0, c1);
        cfvs_1326[idx] = (cfvs_combo[combo_idx] / norm) as f32;
        valid_mask[idx] = true;
    }

    (cfvs_1326, valid_mask)
}

/// Run the turn model forward pass and return 1326-indexed predicted CFVs
/// for the given traverser (0 = OOP, 1 = IP).
fn predict_with_model(
    model: &CfvNet<B>,
    device: &<B as burn::tensor::backend::Backend>::Device,
    sit: &Situation,
    traverser: u8,
) -> Vec<f32> {
    let in_size = input_size(4);
    let mut input = Vec::with_capacity(in_size);
    input.extend_from_slice(&sit.ranges[0]);
    input.extend_from_slice(&sit.ranges[1]);
    // 52-dim one-hot board encoding
    let mut board_one_hot = [0.0f32; 52];
    for &card in sit.board_cards() {
        board_one_hot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_one_hot);
    // SPR = effective_stack / pot (guard against div-by-zero)
    let pot_f32 = sit.pot as f32;
    let stack_f32 = sit.effective_stack as f32;
    let spr = if pot_f32 > 0.0 { stack_f32 / pot_f32 } else { 0.0 };
    input.push(spr);
    debug_assert_eq!(input.len(), in_size);

    let data = TensorData::new(input, [1, in_size]);
    let input_tensor = Tensor::<B, 2>::from_data(data, device);
    let range_oop = Tensor::<B, 2>::from_data(
        TensorData::new(sit.ranges[0].to_vec(), [1, NET_NUM_COMBOS]),
        device,
    );
    let range_ip = Tensor::<B, 2>::from_data(
        TensorData::new(sit.ranges[1].to_vec(), [1, NET_NUM_COMBOS]),
        device,
    );
    let output = model.forward(input_tensor, range_oop, range_ip);
    let full_vec: Vec<f32> = output.into_data().to_vec::<f32>().expect("output tensor conversion");
    // Dual output: first 1326 = OOP CFVs, last 1326 = IP CFVs.
    let offset = if traverser == 0 { 0 } else { NET_NUM_COMBOS };
    full_vec[offset..offset + NET_NUM_COMBOS].to_vec()
}

/// Compare a single spot: model prediction vs ground truth.
fn compare_single(
    model: &CfvNet<B>,
    device: &<B as burn::tensor::backend::Backend>::Device,
    sit: &Situation,
    bet_sizes: &[Vec<f64>],
    solver_iterations: u32,
    evaluator: Box<dyn LeafEvaluator>,
) -> (f64, f64, f64, SpotResult) {
    let (actual, valid_mask) = solve_and_extract(sit, bet_sizes, solver_iterations, evaluator, 0);
    let predicted = predict_with_model(model, device, sit, 0);

    let mask_bool: Vec<bool> = valid_mask.to_vec();
    let metrics = compute_prediction_metrics(&predicted, &actual, &mask_bool, sit.pot as f32);
    (metrics.mae, metrics.max_error, metrics.mbb_error, SpotResult {
        board: sit.board,
        board_size: sit.board_size,
        pot: sit.pot,
        effective_stack: sit.effective_stack,
        mae: metrics.mae,
        mbb: metrics.mbb_error,
    })
}

/// Aggregate per-spot metrics into a `ComparisonSummary`.
fn aggregate(results: Vec<(f64, f64, f64, SpotResult)>) -> ComparisonSummary {
    let n = results.len() as f64;
    let mut sum_mae = 0.0;
    let mut sum_max = 0.0;
    let mut sum_mbb = 0.0;
    let mut worst_mae = 0.0_f64;
    let mut worst_mbb = 0.0_f64;
    let mut spots = Vec::with_capacity(results.len());

    for (mae, max_err, mbb, spot) in results {
        sum_mae += mae;
        sum_max += max_err;
        sum_mbb += mbb;
        worst_mae = worst_mae.max(mae);
        worst_mbb = worst_mbb.max(mbb);
        spots.push(spot);
    }

    ComparisonSummary {
        num_spots: spots.len(),
        mean_mae: sum_mae / n,
        mean_max_error: sum_max / n,
        mean_mbb: sum_mbb / n,
        worst_mae,
        worst_mbb,
        spots,
    }
}

/// Compare a turn model against `CfvSubgameSolver` + `RiverNetEvaluator`.
///
/// Loads the turn model from `turn_model_path` and the river model from
/// `river_model_path`. For each random turn spot, solves it with the
/// river net as the leaf evaluator, then compares solver CFVs against
/// the turn model's predictions.
///
/// # Errors
///
/// Returns an error if model loading fails or config is invalid.
pub fn run_turn_comparison_net(
    config: &CfvnetConfig,
    turn_model_path: &Path,
    river_model_path: &Path,
    num_spots: usize,
    seed: u64,
) -> Result<ComparisonSummary, String> {
    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

    let turn_in_size = input_size(4);
    let turn_model = CfvNet::<B>::new(
        &device,
        config.training.hidden_layers,
        config.training.hidden_size,
        turn_in_size,
    )
    .load_file(turn_model_path, &recorder, &device)
    .map_err(|e| format!("failed to load turn model: {e}"))?;

    let river_in_size = input_size(5);
    let river_model = CfvNet::<B>::new(
        &device,
        config.training.hidden_layers,
        config.training.hidden_size,
        river_in_size,
    )
    .load_file(river_model_path, &recorder, &device)
    .map_err(|e| format!("failed to load river model: {e}"))?;

    let bet_sizes_f64 = parse_bet_sizes(&config.game.bet_sizes);
    if bet_sizes_f64.is_empty() {
        return Err("no valid percentage bet sizes found in config".into());
    }
    let bet_sizes_vec = vec![bet_sizes_f64];
    let solver_iterations = config.datagen.solver_iterations;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut results = Vec::with_capacity(num_spots);

    for i in 0..num_spots {
        let sit = sample_situation(&config.datagen, config.game.initial_stack, 4, &mut rng);
        if sit.effective_stack <= 0 {
            continue;
        }

        let eval_model = river_model.clone();
        let evaluator: Box<dyn LeafEvaluator> =
            Box::new(RiverNetEvaluator::new(eval_model, device));

        let result = compare_single(
            &turn_model,
            &device,
            &sit,
            &bet_sizes_vec,
            solver_iterations,
            evaluator,
        );
        eprintln!("  spot {}/{num_spots}: MAE={:.6}, mBB={:.2}", i + 1, result.0, result.2);
        results.push(result);
    }

    if results.is_empty() {
        return Err("all spots had zero effective stack".into());
    }

    Ok(aggregate(results))
}

/// Compare a turn model against `CfvSubgameSolver` + `ExactRiverEvaluator`.
///
/// Loads the turn model from `turn_model_path`. For each random turn spot,
/// solves it exactly by enumerating all river runouts at depth boundaries,
/// then compares solver CFVs against the turn model's predictions.
///
/// This is much slower than `run_turn_comparison_net` but produces exact
/// ground truth without relying on a river model.
///
/// # Errors
///
/// Returns an error if model loading fails or config is invalid.
pub fn run_turn_comparison_exact(
    config: &CfvnetConfig,
    turn_model_path: &Path,
    num_spots: usize,
    seed: u64,
) -> Result<ComparisonSummary, String> {
    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

    let turn_in_size = input_size(4);
    let turn_model = CfvNet::<B>::new(
        &device,
        config.training.hidden_layers,
        config.training.hidden_size,
        turn_in_size,
    )
    .load_file(turn_model_path, &recorder, &device)
    .map_err(|e| format!("failed to load turn model: {e}"))?;

    let bet_sizes_f64 = parse_bet_sizes(&config.game.bet_sizes);
    if bet_sizes_f64.is_empty() {
        return Err("no valid percentage bet sizes found in config".into());
    }
    let bet_sizes_vec = vec![bet_sizes_f64.clone()];
    let solver_iterations = config.datagen.solver_iterations;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut results = Vec::with_capacity(num_spots);

    for i in 0..num_spots {
        let sit = sample_situation(&config.datagen, config.game.initial_stack, 4, &mut rng);
        if sit.effective_stack <= 0 {
            continue;
        }

        let evaluator: Box<dyn LeafEvaluator> = Box::new(ExactRiverEvaluator {
            bet_sizes: vec![1.0],
            iterations: 50,
        });

        let result = compare_single(
            &turn_model,
            &device,
            &sit,
            &bet_sizes_vec,
            solver_iterations,
            evaluator,
        );
        eprintln!("  spot {}/{num_spots}: MAE={:.6}, mBB={:.2}", i + 1, result.0, result.2);
        results.push(result);
    }

    if results.is_empty() {
        return Err("all spots had zero effective stack".into());
    }

    Ok(aggregate(results))
}

/// Run comparison using pre-loaded models (for testing).
///
/// Avoids file I/O by accepting already-constructed models.
pub fn run_turn_comparison_net_with_models(
    config: &CfvnetConfig,
    turn_model: &CfvNet<B>,
    river_model: &CfvNet<B>,
    num_spots: usize,
    seed: u64,
) -> Result<ComparisonSummary, String> {
    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let bet_sizes_f64 = parse_bet_sizes(&config.game.bet_sizes);
    if bet_sizes_f64.is_empty() {
        return Err("no valid percentage bet sizes found in config".into());
    }
    let bet_sizes_vec = vec![bet_sizes_f64];
    let solver_iterations = config.datagen.solver_iterations;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut results = Vec::with_capacity(num_spots);

    for _ in 0..num_spots {
        let sit = sample_situation(&config.datagen, config.game.initial_stack, 4, &mut rng);
        if sit.effective_stack <= 0 {
            continue;
        }

        let eval_model = river_model.clone();
        let evaluator: Box<dyn LeafEvaluator> =
            Box::new(RiverNetEvaluator::new(eval_model, device));

        let result = compare_single(
            turn_model,
            &device,
            &sit,
            &bet_sizes_vec,
            solver_iterations,
            evaluator,
        );
        results.push(result);
    }

    if results.is_empty() {
        return Err("all spots had zero effective stack".into());
    }

    Ok(aggregate(results))
}

/// Run exact comparison using a pre-loaded turn model (for testing).
///
/// `exact_iterations` controls how many CFR iterations the
/// `ExactRiverEvaluator` runs per river runout. Use a small value
/// (e.g. 2) in tests for speed.
pub fn run_turn_comparison_exact_with_model(
    config: &CfvnetConfig,
    turn_model: &CfvNet<B>,
    num_spots: usize,
    seed: u64,
    exact_iterations: u32,
) -> Result<ComparisonSummary, String> {
    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let bet_sizes_f64 = parse_bet_sizes(&config.game.bet_sizes);
    if bet_sizes_f64.is_empty() {
        return Err("no valid percentage bet sizes found in config".into());
    }
    let bet_sizes_vec = vec![bet_sizes_f64.clone()];
    let solver_iterations = config.datagen.solver_iterations;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut results = Vec::with_capacity(num_spots);

    for _ in 0..num_spots {
        let sit = sample_situation(&config.datagen, config.game.initial_stack, 4, &mut rng);
        if sit.effective_stack <= 0 {
            continue;
        }

        let evaluator: Box<dyn LeafEvaluator> = Box::new(ExactRiverEvaluator {
            bet_sizes: vec![1.0],
            iterations: exact_iterations,
        });

        let result = compare_single(
            turn_model,
            &device,
            &sit,
            &bet_sizes_vec,
            solver_iterations,
            evaluator,
        );
        results.push(result);
    }

    if results.is_empty() {
        return Err("all spots had zero effective stack".into());
    }

    Ok(aggregate(results))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DatagenConfig, EvaluationConfig, GameConfig, TrainingConfig};

    fn test_config() -> CfvnetConfig {
        CfvnetConfig {
            game: GameConfig {
                initial_stack: 200,
                bet_sizes: vec!["50%".into(), "a".into()],
                board_size: 4,
                river_model_path: None,
                ..Default::default()
            },
            datagen: DatagenConfig {
                num_samples: 1,
                street: "turn".into(),
                solver_iterations: 20,
                target_exploitability: 0.05,
                threads: 1,
                seed: 42,
                ..Default::default()
            },
            training: TrainingConfig {
                hidden_layers: 1,
                hidden_size: 8,
                ..Default::default()
            },
            evaluation: EvaluationConfig::default(),
        }
    }

    #[test]
    fn compare_net_pipeline_runs() {
        let config = test_config();
        let device = <B as burn::tensor::backend::Backend>::Device::default();

        let turn_in_size = input_size(4);
        let turn_model = CfvNet::<B>::new(&device, 1, 8, turn_in_size);

        let river_in_size = input_size(5);
        let river_model = CfvNet::<B>::new(&device, 1, 8, river_in_size);

        let summary =
            run_turn_comparison_net_with_models(&config, &turn_model, &river_model, 1, 42)
                .unwrap();

        assert_eq!(summary.num_spots, 1);
        assert!(summary.mean_mae.is_finite(), "mean_mae not finite");
        assert!(summary.mean_max_error.is_finite(), "mean_max_error not finite");
        assert!(summary.mean_mbb.is_finite(), "mean_mbb not finite");
        assert!(summary.worst_mae.is_finite(), "worst_mae not finite");
        assert!(summary.worst_mbb.is_finite(), "worst_mbb not finite");
    }

    /// Smoke test for the exact comparison pipeline. Ignored by default
    /// because `ExactRiverEvaluator` solves ~48 river subgames per boundary
    /// and is too slow in debug mode. Run with `cargo test --release -- --ignored`.
    #[test]
    #[ignore]
    fn compare_exact_pipeline_runs() {
        let mut config = test_config();
        config.datagen.solver_iterations = 5;
        let device = <B as burn::tensor::backend::Backend>::Device::default();

        let turn_in_size = input_size(4);
        let turn_model = CfvNet::<B>::new(&device, 1, 8, turn_in_size);

        let summary =
            run_turn_comparison_exact_with_model(&config, &turn_model, 1, 42, 2).unwrap();

        assert_eq!(summary.num_spots, 1);
        assert!(summary.mean_mae.is_finite(), "mean_mae not finite");
        assert!(summary.mean_max_error.is_finite(), "mean_max_error not finite");
        assert!(summary.mean_mbb.is_finite(), "mean_mbb not finite");
    }

    #[test]
    fn compare_net_with_saved_models() {
        let config = test_config();
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

        let turn_in_size = input_size(4);
        let turn_model = CfvNet::<B>::new(&device, 1, 8, turn_in_size);

        let river_in_size = input_size(5);
        let river_model = CfvNet::<B>::new(&device, 1, 8, river_in_size);

        // Save to temp dirs and reload via the file-based API.
        let turn_dir = tempfile::tempdir().unwrap();
        let river_dir = tempfile::tempdir().unwrap();
        let turn_path = turn_dir.path().join("model");
        let river_path = river_dir.path().join("model");

        turn_model
            .clone()
            .save_file(&turn_path, &recorder)
            .unwrap();
        river_model
            .clone()
            .save_file(&river_path, &recorder)
            .unwrap();

        let summary =
            run_turn_comparison_net(&config, &turn_path, &river_path, 1, 42).unwrap();

        assert_eq!(summary.num_spots, 1);
        assert!(summary.mean_mae.is_finite());
    }

    /// File-based model loading + exact comparison. Ignored for the same
    /// reason as `compare_exact_pipeline_runs`.
    #[test]
    #[ignore]
    fn compare_exact_with_saved_model_loads_from_disk() {
        let mut config = test_config();
        config.datagen.solver_iterations = 5;
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

        let turn_in_size = input_size(4);
        let turn_model = CfvNet::<B>::new(&device, 1, 8, turn_in_size);

        let turn_dir = tempfile::tempdir().unwrap();
        let turn_path = turn_dir.path().join("model");
        turn_model.save_file(&turn_path, &recorder).unwrap();

        let loaded = CfvNet::<B>::new(&device, 1, 8, turn_in_size)
            .load_file(&turn_path, &recorder, &device)
            .unwrap();

        let summary =
            run_turn_comparison_exact_with_model(&config, &loaded, 1, 42, 2).unwrap();

        assert_eq!(summary.num_spots, 1);
        assert!(summary.mean_mae.is_finite());
    }

    #[test]
    fn parse_bet_sizes_filters_allin() {
        let sizes = vec!["50%".into(), "100%".into(), "a".into()];
        let parsed = parse_bet_sizes(&sizes);
        assert_eq!(parsed.len(), 2);
        assert!((parsed[0] - 0.5).abs() < 1e-10);
        assert!((parsed[1] - 1.0).abs() < 1e-10);
    }
}
