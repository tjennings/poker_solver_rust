//! Turn model comparison against range-solver ground truth.
//!
//! Two modes:
//! - **compare-net**: solver uses `NetBoundaryEvaluator` (river net) at depth boundaries
//! - **compare-exact**: solver uses `PostFlopGame` with `depth_limit: None` (full solve through river)

use std::path::Path;
use std::sync::{Arc, Mutex};

use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use burn::tensor::{Tensor, TensorData};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::{BetSize, BetSizeOptions};
use range_solver::card::{card_pair_to_index, CardConfig, NOT_DEALT};
use range_solver::game::{BoundaryEvaluator, PostFlopGame};
use range_solver::range::Range as RsRange;
use range_solver::solve;

use crate::config::CfvnetConfig;
use crate::datagen::range_gen::NUM_COMBOS;
use crate::datagen::sampler::{sample_situation, Situation};
use crate::eval::compare::{ComparisonSummary, SpotResult};
use crate::eval::metrics::compute_prediction_metrics;
use crate::eval::river_net_evaluator::build_input;
use crate::model::network::{CfvNet, DECK_SIZE, INPUT_SIZE, NUM_RANKS, OUTPUT_SIZE};

type B = NdArray;

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

/// Build a `PostFlopGame` for a turn situation.
///
/// - `depth_limit: Some(0)` stops at the river boundary (for net evaluation)
/// - `depth_limit: None` solves through river to showdown (for exact evaluation)
fn build_turn_postflop_game(
    board_u8: &[u8],
    pot: f64,
    effective_stack: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    bet_sizes: &[Vec<f64>],
    depth_limit: Option<u8>,
) -> Option<PostFlopGame> {
    if effective_stack <= 0.0 {
        return None;
    }

    let oop_range = RsRange::from_raw_data(&ranges[0]).expect("valid OOP range");
    let ip_range = RsRange::from_raw_data(&ranges[1]).expect("valid IP range");

    let sizes: Vec<BetSize> = bet_sizes
        .iter()
        .flat_map(|v| v.iter().map(|&f| BetSize::PotRelative(f)))
        .collect();
    let bet_size_opts = BetSizeOptions {
        bet: sizes.clone(),
        raise: Vec::new(),
    };

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [board_u8[0], board_u8[1], board_u8[2]],
        turn: board_u8[3],
        river: NOT_DEALT,
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: pot as i32,
        effective_stack: effective_stack as i32,
        turn_bet_sizes: [bet_size_opts.clone(), bet_size_opts],
        river_bet_sizes: [BetSizeOptions::default(), BetSizeOptions::default()],
        depth_limit,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).expect("valid action tree");
    let mut game = PostFlopGame::with_config(card_config, action_tree).expect("valid game");
    game.allocate_memory(false);
    Some(game)
}

/// Boundary evaluator wrapping a river neural network for turn depth boundaries.
///
/// When the solver reaches a turn-river boundary, this evaluator averages
/// the river CFV network predictions over all possible river cards.
///
/// The model is held behind a `Mutex` because `CfvNet<NdArray>` is not `Sync`
/// (burn's `Param` contains `OnceCell`). The mutex is uncontended in practice
/// since boundary evaluation is single-threaded.
struct NetBoundaryEvaluator {
    model: Mutex<CfvNet<B>>,
    device: <B as burn::tensor::backend::Backend>::Device,
    board_u8: [u8; 4],
    /// Private cards per player, in game ordering.
    private_cards: [Vec<(u8, u8)>; 2],
}

impl NetBoundaryEvaluator {
    fn new(
        model: CfvNet<B>,
        device: <B as burn::tensor::backend::Backend>::Device,
        board_u8: [u8; 4],
        private_cards: [Vec<(u8, u8)>; 2],
    ) -> Self {
        Self {
            model: Mutex::new(model),
            device,
            board_u8,
            private_cards,
        }
    }
}

impl BoundaryEvaluator for NetBoundaryEvaluator {
    fn compute_cfvs(
        &self,
        player: usize,
        pot: i32,
        remaining_stack: f64,
        opponent_reach: &[f32],
        num_hands: usize,
    ) -> Vec<f32> {
        let opp = player ^ 1;
        let hero_cards = &self.private_cards[player];
        let opp_cards = &self.private_cards[opp];

        // Map opponent reach from game ordering to 1326-indexed.
        let mut opp_1326 = [0.0_f32; OUTPUT_SIZE];
        for (i, &(c0, c1)) in opp_cards.iter().enumerate() {
            if i < opponent_reach.len() {
                let idx = card_pair_to_index(c0, c1);
                opp_1326[idx] = opponent_reach[i];
            }
        }

        // For the hero side, use uniform reach (solver handles weighting externally).
        let mut hero_1326 = [0.0_f32; OUTPUT_SIZE];
        for &(c0, c1) in hero_cards.iter() {
            let idx = card_pair_to_index(c0, c1);
            hero_1326[idx] = 1.0;
        }

        // Map to (oop, ip) ordering for the network input.
        let (oop_1326, ip_1326) = if player == 0 {
            (&hero_1326, &opp_1326)
        } else {
            (&opp_1326, &hero_1326)
        };

        let effective_stack = remaining_stack;
        let pot_f64 = f64::from(pot);
        let board_u8 = &self.board_u8;

        // Accumulate CFVs per hero hand, averaged over river cards.
        let mut cfv_sum = vec![0.0_f64; num_hands];
        let mut cfv_count = vec![0_u32; num_hands];

        // Pre-compute hero card 1326 indices.
        let hero_indices: Vec<usize> = hero_cards
            .iter()
            .map(|&(c0, c1)| card_pair_to_index(c0, c1))
            .collect();

        for river_u8 in 0u8..52 {
            if board_u8.contains(&river_u8) {
                continue;
            }

            let river_board: [u8; 5] = [
                board_u8[0], board_u8[1], board_u8[2], board_u8[3], river_u8,
            ];

            // Filter out combos that conflict with the river card.
            let mut oop_filtered = *oop_1326;
            let mut ip_filtered = *ip_1326;
            // Zero out all combos containing the river card.
            for other in 0u8..52 {
                if other == river_u8 {
                    continue;
                }
                let (lo, hi) = if river_u8 < other {
                    (river_u8, other)
                } else {
                    (other, river_u8)
                };
                let idx = card_pair_to_index(lo, hi);
                oop_filtered[idx] = 0.0;
                ip_filtered[idx] = 0.0;
            }

            let input_vec = build_input(
                &oop_filtered,
                &ip_filtered,
                &river_board,
                pot_f64,
                effective_stack,
                player as u8,
            );

            let data = TensorData::new(input_vec, [1, INPUT_SIZE]);
            let input_tensor = Tensor::<B, 2>::from_data(data, &self.device);
            let model = self.model.lock().unwrap();
            let output = model.forward(input_tensor);
            drop(model);
            let out_vec: Vec<f32> = output
                .into_data()
                .to_vec()
                .expect("output tensor conversion");

            // Map 1326-indexed output back to hero's game ordering.
            for (i, &idx) in hero_indices.iter().enumerate() {
                let (c0, c1) = hero_cards[i];
                if c0 != river_u8 && c1 != river_u8 {
                    cfv_sum[i] += f64::from(out_vec[idx]);
                    cfv_count[i] += 1;
                }
            }
        }

        // Average over river cards.
        cfv_sum
            .iter()
            .zip(cfv_count.iter())
            .map(|(&sum, &count)| {
                if count > 0 {
                    (sum / f64::from(count)) as f32
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Extract 1326-indexed CFVs from a solved `PostFlopGame`.
fn extract_cfvs(
    game: &mut PostFlopGame,
    pot: f64,
    traverser: u8,
) -> ([f32; NUM_COMBOS], [bool; NUM_COMBOS]) {
    game.back_to_root();
    game.cache_normalized_weights();
    let raw_evs = game.expected_values(traverser as usize);
    let hands = game.private_cards(traverser as usize);

    let half_pot = pot / 2.0;
    let norm = if half_pot > 0.0 { half_pot } else { 1.0 };

    let mut cfvs_1326 = [0.0_f32; NUM_COMBOS];
    let mut valid_mask = [false; NUM_COMBOS];

    for (i, &(c0, c1)) in hands.iter().enumerate() {
        let idx = card_pair_to_index(c0, c1);
        cfvs_1326[idx] = ((f64::from(raw_evs[i]) - half_pot) / norm) as f32;
        valid_mask[idx] = true;
    }

    (cfvs_1326, valid_mask)
}

/// Solve a turn situation with a river net boundary evaluator and return 1326-indexed CFVs.
fn solve_and_extract_net(
    sit: &Situation,
    bet_sizes: &[Vec<f64>],
    solver_iterations: u32,
    river_model: CfvNet<B>,
    device: <B as burn::tensor::backend::Backend>::Device,
    traverser: u8,
) -> ([f32; NUM_COMBOS], [bool; NUM_COMBOS]) {
    let pot = f64::from(sit.pot);
    let effective_stack = f64::from(sit.effective_stack);
    let board_u8 = sit.board_cards();

    let mut game = build_turn_postflop_game(
        board_u8,
        pot,
        effective_stack,
        &sit.ranges,
        bet_sizes,
        Some(0), // depth-limited: stop at river boundary
    )
    .expect("game should build for non-degenerate situation");

    // Build boundary evaluator with the game's private cards.
    let private_cards = [
        game.private_cards(0).to_vec(),
        game.private_cards(1).to_vec(),
    ];
    let board_arr: [u8; 4] = [board_u8[0], board_u8[1], board_u8[2], board_u8[3]];
    let evaluator = NetBoundaryEvaluator::new(river_model, device, board_arr, private_cards);
    game.boundary_evaluator = Some(Arc::new(evaluator));

    let abs_target = 0.0; // no early stop
    solve(&mut game, solver_iterations, abs_target, false);

    extract_cfvs(&mut game, pot, traverser)
}

/// Solve a turn situation exactly (through river to showdown) and return 1326-indexed CFVs.
fn solve_and_extract_exact(
    sit: &Situation,
    bet_sizes: &[Vec<f64>],
    solver_iterations: u32,
    traverser: u8,
) -> ([f32; NUM_COMBOS], [bool; NUM_COMBOS]) {
    let pot = f64::from(sit.pot);
    let effective_stack = f64::from(sit.effective_stack);
    let board_u8 = sit.board_cards();

    let mut game = build_turn_postflop_game(
        board_u8,
        pot,
        effective_stack,
        &sit.ranges,
        bet_sizes,
        None, // no depth limit: solve through river to showdown
    )
    .expect("game should build for non-degenerate situation");

    let abs_target = 0.0;
    solve(&mut game, solver_iterations, abs_target, false);

    extract_cfvs(&mut game, pot, traverser)
}

/// Run the turn model forward pass and return 1326-indexed predicted CFVs.
fn predict_with_model(
    model: &CfvNet<B>,
    device: &<B as burn::tensor::backend::Backend>::Device,
    sit: &Situation,
    traverser: u8,
) -> Vec<f32> {
    let mut input = Vec::with_capacity(INPUT_SIZE);
    input.extend_from_slice(&sit.ranges[0]);
    input.extend_from_slice(&sit.ranges[1]);
    let mut board_onehot = [0.0_f32; DECK_SIZE];
    for &card in sit.board_cards() {
        debug_assert!((card as usize) < DECK_SIZE, "card id {card} out of range");
        board_onehot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_onehot);
    let mut rank_presence = [0.0_f32; NUM_RANKS];
    for &card in sit.board_cards() {
        debug_assert!((card as usize) < DECK_SIZE, "card id {card} out of range");
        rank_presence[(card / 4) as usize] = 1.0;
    }
    input.extend_from_slice(&rank_presence);
    input.push(sit.pot as f32 / 400.0);
    input.push(sit.effective_stack as f32 / 400.0);
    input.push(f32::from(traverser));
    debug_assert_eq!(input.len(), INPUT_SIZE);

    let data = TensorData::new(input, [1, INPUT_SIZE]);
    let input_tensor = Tensor::<B, 2>::from_data(data, device);
    let output = model.forward(input_tensor);
    output
        .into_data()
        .to_vec::<f32>()
        .expect("output tensor conversion")
}

/// Compare a single spot (net mode): model prediction vs ground truth.
fn compare_single_net(
    model: &CfvNet<B>,
    device: &<B as burn::tensor::backend::Backend>::Device,
    sit: &Situation,
    bet_sizes: &[Vec<f64>],
    solver_iterations: u32,
    river_model: CfvNet<B>,
    river_device: <B as burn::tensor::backend::Backend>::Device,
) -> (f64, f64, f64, SpotResult) {
    let (actual, valid_mask) =
        solve_and_extract_net(sit, bet_sizes, solver_iterations, river_model, river_device, 0);
    let predicted = predict_with_model(model, device, sit, 0);

    let mask_bool: Vec<bool> = valid_mask.to_vec();
    let metrics = compute_prediction_metrics(&predicted, &actual, &mask_bool, sit.pot as f32);
    (
        metrics.mae,
        metrics.max_error,
        metrics.mbb_error,
        SpotResult {
            board: sit.board,
            board_size: sit.board_size,
            pot: sit.pot,
            effective_stack: sit.effective_stack,
            mae: metrics.mae,
            mbb: metrics.mbb_error,
        },
    )
}

/// Compare a single spot (exact mode): model prediction vs ground truth.
fn compare_single_exact(
    model: &CfvNet<B>,
    device: &<B as burn::tensor::backend::Backend>::Device,
    sit: &Situation,
    bet_sizes: &[Vec<f64>],
    solver_iterations: u32,
) -> (f64, f64, f64, SpotResult) {
    let (actual, valid_mask) = solve_and_extract_exact(sit, bet_sizes, solver_iterations, 0);
    let predicted = predict_with_model(model, device, sit, 0);

    let mask_bool: Vec<bool> = valid_mask.to_vec();
    let metrics = compute_prediction_metrics(&predicted, &actual, &mask_bool, sit.pot as f32);
    (
        metrics.mae,
        metrics.max_error,
        metrics.mbb_error,
        SpotResult {
            board: sit.board,
            board_size: sit.board_size,
            pot: sit.pot,
            effective_stack: sit.effective_stack,
            mae: metrics.mae,
            mbb: metrics.mbb_error,
        },
    )
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

/// Compare a turn model against `PostFlopGame` + `NetBoundaryEvaluator` (river net).
///
/// Loads the turn model from `turn_model_path` and the river model from
/// `river_model_path`. For each random turn spot, solves it with the
/// river net at depth boundaries, then compares solver CFVs against
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

    let turn_model = CfvNet::<B>::new(
        &device,
        config.training.hidden_layers,
        config.training.hidden_size,
        INPUT_SIZE,
    )
    .load_file(turn_model_path, &recorder, &device)
    .map_err(|e| format!("failed to load turn model: {e}"))?;

    let river_model = CfvNet::<B>::new(
        &device,
        config.training.hidden_layers,
        config.training.hidden_size,
        INPUT_SIZE,
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

        let result = compare_single_net(
            &turn_model,
            &device,
            &sit,
            &bet_sizes_vec,
            solver_iterations,
            river_model.clone(),
            device,
        );
        eprintln!(
            "  spot {}/{num_spots}: MAE={:.6}, mBB={:.2}",
            i + 1,
            result.0,
            result.2
        );
        results.push(result);
    }

    if results.is_empty() {
        return Err("all spots had zero effective stack".into());
    }

    Ok(aggregate(results))
}

/// Compare a turn model against `PostFlopGame` with exact river solving.
///
/// Loads the turn model from `turn_model_path`. For each random turn spot,
/// solves it exactly through the river to showdown, then compares solver
/// CFVs against the turn model's predictions.
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

    let turn_model = CfvNet::<B>::new(
        &device,
        config.training.hidden_layers,
        config.training.hidden_size,
        INPUT_SIZE,
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

        let result = compare_single_exact(
            &turn_model,
            &device,
            &sit,
            &bet_sizes_vec,
            solver_iterations,
        );
        eprintln!(
            "  spot {}/{num_spots}: MAE={:.6}, mBB={:.2}",
            i + 1,
            result.0,
            result.2
        );
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

        let result = compare_single_net(
            turn_model,
            &device,
            &sit,
            &bet_sizes_vec,
            solver_iterations,
            river_model.clone(),
            device,
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
/// Unlike the old implementation, `exact_iterations` is no longer needed
/// since `PostFlopGame` with `depth_limit: None` solves through river
/// to showdown natively. The parameter is kept for API compatibility.
pub fn run_turn_comparison_exact_with_model(
    config: &CfvnetConfig,
    turn_model: &CfvNet<B>,
    num_spots: usize,
    seed: u64,
    _exact_iterations: u32,
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

        let result = compare_single_exact(
            turn_model,
            &device,
            &sit,
            &bet_sizes_vec,
            solver_iterations,
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
                seed: Some(42),
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

        let turn_model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);

        let river_model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);

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

    /// Smoke test for the exact comparison pipeline.
    /// Uses `PostFlopGame` with `depth_limit: None` (solve through river).
    #[test]
    #[ignore]
    fn compare_exact_pipeline_runs() {
        let mut config = test_config();
        config.datagen.solver_iterations = 5;
        let device = <B as burn::tensor::backend::Backend>::Device::default();

        let turn_model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);

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

        let turn_model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);

        let river_model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);

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

    /// File-based model loading + exact comparison. Ignored because
    /// full turn+river solving is slow in debug mode.
    #[test]
    #[ignore]
    fn compare_exact_with_saved_model_loads_from_disk() {
        let mut config = test_config();
        config.datagen.solver_iterations = 5;
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

        let turn_model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);

        let turn_dir = tempfile::tempdir().unwrap();
        let turn_path = turn_dir.path().join("model");
        turn_model.save_file(&turn_path, &recorder).unwrap();

        let loaded = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE)
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

    #[test]
    fn net_boundary_evaluator_returns_values_per_hand() {
        use range_solver::game::BoundaryEvaluator;

        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let river_model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);
        // Board: As Kh 7d 4c (turn)
        let board_u8: [u8; 4] = [
            4 * 12 + 3, // As
            4 * 11 + 2, // Kh
            4 * 5 + 1,  // 7d
            4 * 2 + 0,  // 4c
        ];
        // Build a PostFlopGame to get realistic private cards
        let bet_sizes_f64 = parse_bet_sizes(&vec!["50%".into()]);
        let game = build_turn_postflop_game(
            &board_u8,
            100.0,
            200.0,
            &[[1.0; NUM_COMBOS]; 2],
            &[bet_sizes_f64],
            Some(0),
        )
        .expect("game should build");
        let private_cards = [
            game.private_cards(0).to_vec(),
            game.private_cards(1).to_vec(),
        ];
        let evaluator = NetBoundaryEvaluator::new(river_model, device, board_u8, private_cards);

        let num_hands = game.private_cards(0).len();
        let opponent_reach = vec![1.0_f32; game.private_cards(1).len()];
        let result = evaluator.compute_cfvs(0, 100, 200.0, &opponent_reach, num_hands);
        assert_eq!(result.len(), num_hands, "should return one CFV per hand");
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "hand {i} has non-finite CFV: {v}");
        }
    }

    #[test]
    fn solve_and_extract_net_returns_valid_cfvs() {
        let config = test_config();
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let river_model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let sit = sample_situation(&config.datagen, config.game.initial_stack, 4, &mut rng);

        let bet_sizes_f64 = parse_bet_sizes(&config.game.bet_sizes);
        let bet_sizes_vec = vec![bet_sizes_f64];

        let (cfvs, valid) =
            solve_and_extract_net(&sit, &bet_sizes_vec, 10, river_model, device, 0);

        // At least some combos should be valid
        let valid_count = valid.iter().filter(|&&v| v).count();
        assert!(valid_count > 0, "should have valid combos");

        // All valid CFVs should be finite
        for (i, (&c, &v)) in cfvs.iter().zip(valid.iter()).enumerate() {
            if v {
                assert!(c.is_finite(), "combo {i} has non-finite CFV: {c}");
            }
        }
    }

    #[test]
    fn solve_and_extract_exact_returns_valid_cfvs() {
        let config = test_config();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let sit = sample_situation(&config.datagen, config.game.initial_stack, 4, &mut rng);

        let bet_sizes_f64 = parse_bet_sizes(&config.game.bet_sizes);
        let bet_sizes_vec = vec![bet_sizes_f64];

        let (cfvs, valid) = solve_and_extract_exact(&sit, &bet_sizes_vec, 5, 0);

        let valid_count = valid.iter().filter(|&&v| v).count();
        assert!(valid_count > 0, "should have valid combos");

        for (i, (&c, &v)) in cfvs.iter().zip(valid.iter()).enumerate() {
            if v {
                assert!(c.is_finite(), "combo {i} has non-finite CFV: {c}");
            }
        }
    }
}
