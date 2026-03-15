//! Two-tier validation for GPU river CFVNet training.
//!
//! 1. **Hold-out loss**: Run the model on a reservoir mini-batch without
//!    gradient updates. Fast, runs every validation interval.
//!
//! 2. **Ground-truth validation**: Pre-solve a small set of fixed river
//!    positions at high iteration count on CPU. Compare model predictions
//!    against exact CFVs. Runs periodically (more expensive).

#[cfg(feature = "training")]
use burn::tensor::backend::AutodiffBackend;
#[cfg(feature = "training")]
use burn::tensor::{Tensor, TensorData};
#[cfg(feature = "training")]
use burn::module::AutodiffModule;

#[cfg(feature = "training")]
use cfvnet::model::loss::cfvnet_loss;
#[cfg(feature = "training")]
use cfvnet::model::network::{CfvNet, INPUT_SIZE, OUTPUT_SIZE};

#[cfg(feature = "training")]
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
#[cfg(feature = "training")]
use range_solver::bet_size::BetSizeOptions;
#[cfg(feature = "training")]
use range_solver::range::Range;
#[cfg(feature = "training")]
use range_solver::CardConfig;

#[cfg(feature = "training")]
use super::reservoir::encode_input;

/// A pre-computed ground-truth validation set.
///
/// Contains a fixed set of river positions solved at high iteration count
/// on CPU. Used to measure the model's prediction quality against exact
/// CFVs.
#[cfg(feature = "training")]
pub struct GroundTruthValidator {
    /// Flat input features: `[num_positions * 2 * INPUT_SIZE]`
    /// (two records per position: OOP + IP perspective).
    inputs: Vec<f32>,
    /// Flat target CFVs: `[num_records * OUTPUT_SIZE]`.
    targets: Vec<f32>,
    /// Flat masks: `[num_records * OUTPUT_SIZE]`.
    masks: Vec<f32>,
    /// Flat ranges: `[num_records * OUTPUT_SIZE]`.
    ranges: Vec<f32>,
    /// Game values: `[num_records]`.
    game_values: Vec<f32>,
    /// Number of records (2 per position: OOP + IP).
    num_records: usize,
}

#[cfg(feature = "training")]
impl GroundTruthValidator {
    /// Pre-compute ground truth by solving `num_positions` fixed river spots
    /// on CPU with `solve_iterations` iterations each.
    ///
    /// Uses a deterministic seed for reproducibility. Each position is solved
    /// with the range-solver at full precision.
    pub fn precompute(
        num_positions: usize,
        solve_iterations: u32,
        seed: u64,
    ) -> Self {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        let mut masks = Vec::new();
        let mut ranges_out = Vec::new();
        let mut game_values = Vec::new();
        let mut solved = 0;

        while solved < num_positions {
            // Generate a random river board (5 unique cards)
            let mut board = [0u8; 5];
            let mut used = [false; 52];
            for card in &mut board {
                loop {
                    let c = rng.gen_range(0..52u8);
                    if !used[c as usize] {
                        used[c as usize] = true;
                        *card = c;
                        break;
                    }
                }
            }

            // Build uniform ranges (excluding board-blocked combos)
            let board_u32: Vec<u32> = board.iter().map(|&c| u32::from(c)).collect();
            let mut range_data = [0.0f32; 1326];
            for (i, r) in range_data.iter_mut().enumerate() {
                let (c1, c2) = range_solver::card::index_to_card_pair(i);
                if !used[c1 as usize] && !used[c2 as usize] {
                    *r = 1.0;
                }
            }

            let oop_range = match Range::from_raw_data(&range_data) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let ip_range = match Range::from_raw_data(&range_data) {
                Ok(r) => r,
                Err(_) => continue,
            };

            // Build game
            let card_config = CardConfig {
                range: [oop_range, ip_range],
                flop: [board[0], board[1], board[2]],
                turn: board[3],
                river: board[4],
            };

            let oop_sizes = BetSizeOptions::try_from(("50%,a", "")).unwrap();
            let ip_sizes = BetSizeOptions::try_from(("50%,a", "")).unwrap();

            let tree_config = TreeConfig {
                initial_state: BoardState::River,
                starting_pot: 100,
                effective_stack: 100,
                flop_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
                turn_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
                river_bet_sizes: [oop_sizes, ip_sizes],
                add_allin_threshold: 1.5,
                force_allin_threshold: 0.15,
                merging_threshold: 0.1,
                ..Default::default()
            };

            let action_tree = match ActionTree::new(tree_config) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let mut game = match range_solver::PostFlopGame::with_config(card_config, action_tree) {
                Ok(g) => g,
                Err(_) => continue,
            };

            game.allocate_memory(false);
            let _exploitability = range_solver::solve(&mut game, solve_iterations, 0.0, false);

            // Extract root CFVs
            game.back_to_root();
            game.cache_normalized_weights();
            let cfvs_oop = game.expected_values(0);
            let cfvs_ip = game.expected_values(1);

            // Encode both perspectives
            for player in 0..2u8 {
                let (input, mask, gv) = encode_input(
                    &range_data,
                    &range_data,
                    &board_u32,
                    100.0,
                    100.0,
                    player,
                );
                let cfvs = if player == 0 { &cfvs_oop } else { &cfvs_ip };
                let player_range = &range_data;

                inputs.extend_from_slice(&input);

                // Pad or truncate to OUTPUT_SIZE
                let mut target = vec![0.0f32; OUTPUT_SIZE];
                let copy_len = cfvs.len().min(OUTPUT_SIZE);
                target[..copy_len].copy_from_slice(&cfvs[..copy_len]);
                targets.extend_from_slice(&target);

                masks.extend_from_slice(&mask);
                ranges_out.extend_from_slice(player_range);
                game_values.push(gv);
            }

            solved += 1;
        }

        let num_records = num_positions * 2;
        eprintln!(
            "Ground-truth validator: solved {} positions ({} records)",
            num_positions, num_records
        );

        Self {
            inputs,
            targets,
            masks,
            ranges: ranges_out,
            game_values,
            num_records,
        }
    }

    /// Number of validation records.
    pub fn num_records(&self) -> usize {
        self.num_records
    }

    /// Evaluate the model against the ground-truth validation set.
    ///
    /// Returns the RMSE of predicted CFVs vs ground-truth CFVs (masked).
    pub fn evaluate<B: AutodiffBackend>(
        &self,
        model: &CfvNet<B>,
        device: &B::Device,
    ) -> f32 {
        if self.num_records == 0 {
            return 0.0;
        }

        let valid_model = model.valid();
        let n = self.num_records;

        let input = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(self.inputs.clone(), [n, INPUT_SIZE]),
            device,
        );
        let target = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(self.targets.clone(), [n, OUTPUT_SIZE]),
            device,
        );
        let mask = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(self.masks.clone(), [n, OUTPUT_SIZE]),
            device,
        );

        let pred = valid_model.forward(input);

        // Compute masked RMSE
        let diff = (pred - target) * mask.clone();
        let sq_diff = diff.powf_scalar(2.0);
        let masked_sq = sq_diff * mask.clone();
        let num_valid: Tensor<B::InnerBackend, 1> = mask.sum().clamp_min(1.0);
        let mse: Tensor<B::InnerBackend, 1> = masked_sq.sum().div(num_valid);
        let rmse = mse.sqrt();
        let rmse_val: f32 = rmse.into_data().to_vec::<f32>().unwrap()[0];
        rmse_val
    }

    /// Evaluate the model and return the combined cfvnet loss (Huber + aux).
    pub fn evaluate_loss<B: AutodiffBackend>(
        &self,
        model: &CfvNet<B>,
        device: &B::Device,
        huber_delta: f64,
        aux_loss_weight: f64,
    ) -> f32 {
        if self.num_records == 0 {
            return 0.0;
        }

        let valid_model = model.valid();
        let n = self.num_records;

        let input = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(self.inputs.clone(), [n, INPUT_SIZE]),
            device,
        );
        let target = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(self.targets.clone(), [n, OUTPUT_SIZE]),
            device,
        );
        let mask = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(self.masks.clone(), [n, OUTPUT_SIZE]),
            device,
        );
        let range = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(self.ranges.clone(), [n, OUTPUT_SIZE]),
            device,
        );
        let game_value = Tensor::<B::InnerBackend, 1>::from_data(
            TensorData::new(self.game_values.clone(), [n]),
            device,
        );

        let pred = valid_model.forward(input);
        let loss = cfvnet_loss(pred, target, mask, range, game_value, huber_delta, aux_loss_weight);
        loss.into_data().to_vec::<f32>().unwrap()[0]
    }
}

#[cfg(all(test, feature = "training"))]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    type TestB = Autodiff<NdArray>;

    #[test]
    fn test_ground_truth_validator_precompute() {
        // Solve 5 positions with low iterations for speed
        let validator = GroundTruthValidator::precompute(5, 100, 42);

        // 5 positions x 2 records each = 10
        assert_eq!(validator.num_records(), 10);
        assert_eq!(validator.inputs.len(), 10 * INPUT_SIZE);
        assert_eq!(validator.targets.len(), 10 * OUTPUT_SIZE);
        assert_eq!(validator.masks.len(), 10 * OUTPUT_SIZE);

        // Verify non-trivial content
        let non_zero_targets = validator.targets.iter().filter(|&&v| v != 0.0).count();
        assert!(non_zero_targets > 0, "targets should have non-zero CFVs");
    }

    #[test]
    fn test_ground_truth_evaluate() {
        let validator = GroundTruthValidator::precompute(3, 100, 42);

        let device = Default::default();
        let model = CfvNet::<TestB>::new(&device, 2, 64, INPUT_SIZE);

        let rmse = validator.evaluate(&model, &device);
        assert!(rmse.is_finite(), "RMSE should be finite");
        assert!(rmse >= 0.0, "RMSE should be non-negative");
    }

    #[test]
    fn test_ground_truth_evaluate_loss() {
        let validator = GroundTruthValidator::precompute(3, 100, 42);

        let device = Default::default();
        let model = CfvNet::<TestB>::new(&device, 2, 64, INPUT_SIZE);

        let loss = validator.evaluate_loss(&model, &device, 1.0, 0.0);
        assert!(loss.is_finite(), "loss should be finite");
        assert!(loss >= 0.0, "loss should be non-negative");
    }
}
