//! Boundary evaluator that adapts BoundaryNet for range-solver leaf evaluation.

use std::path::Path;
use std::sync::Mutex;

use crate::model::boundary_net::BoundaryNet;
use crate::model::network::{DECK_SIZE, INPUT_SIZE, NUM_COMBOS, NUM_RANKS};
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use burn::tensor::{Tensor, TensorData};
use range_solver::card::card_pair_to_index;

/// Encode inputs for boundary inference from raw game state.
pub fn encode_boundary_inference_input(
    oop_range: &[f32],
    ip_range: &[f32],
    board: &[u8],
    pot: f32,
    effective_stack: f32,
    player: u8,
) -> Vec<f32> {
    let total = pot + effective_stack;
    let norm = if total > 0.0 { total } else { 1.0 };

    let mut input = Vec::with_capacity(INPUT_SIZE);
    input.extend_from_slice(oop_range);
    input.extend_from_slice(ip_range);

    let mut board_onehot = [0.0_f32; DECK_SIZE];
    for &card in board {
        board_onehot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_onehot);

    let mut rank_presence = [0.0_f32; NUM_RANKS];
    for &card in board {
        rank_presence[(card / 4) as usize] = 1.0;
    }
    input.extend_from_slice(&rank_presence);

    input.push(pot / norm);
    input.push(effective_stack / norm);
    input.push(f32::from(player));

    debug_assert_eq!(input.len(), INPUT_SIZE);
    input
}

/// Convert normalized EVs back to chip EVs.
pub fn denormalize_ev(normalized: &[f32], pot: f32, effective_stack: f32) -> Vec<f32> {
    let total = pot + effective_stack;
    normalized.iter().map(|&v| v * total).collect()
}

/// Neural boundary evaluator wrapping a trained `BoundaryNet` model.
///
/// Maps between the range-solver's per-player private card ordering and
/// the 1326-combo index ordering used by the network. The model is wrapped
/// in a `Mutex` to satisfy the `Send + Sync` requirement of `BoundaryEvaluator`.
pub struct NeuralBoundaryEvaluator {
    model: Mutex<BoundaryNet<NdArray>>,
    board: Vec<u8>,
    private_cards: [Vec<(u8, u8)>; 2],
}

impl NeuralBoundaryEvaluator {
    pub fn new(
        model: BoundaryNet<NdArray>,
        board: Vec<u8>,
        private_cards: [Vec<(u8, u8)>; 2],
    ) -> Self {
        Self { model: Mutex::new(model), board, private_cards }
    }
}

impl range_solver::game::BoundaryEvaluator for NeuralBoundaryEvaluator {
    fn num_continuations(&self) -> usize { 1 }

    fn compute_cfvs(
        &self,
        player: usize,
        pot: i32,
        remaining_stack: f64,
        opponent_reach: &[f32],
        num_hands: usize,
        _continuation_index: usize,
    ) -> Vec<f32> {
        let opp = player ^ 1;

        // Check for zero total opponent reach (unreachable boundary).
        let opp_total: f32 = opponent_reach.iter().sum();
        if opp_total <= 0.0 {
            return vec![0.0; num_hands];
        }

        // Build 1326-element range vectors from game ordering.
        let mut opp_range = vec![0.0_f32; NUM_COMBOS];
        for (i, &(c1, c2)) in self.private_cards[opp].iter().enumerate() {
            if i < opponent_reach.len() {
                opp_range[card_pair_to_index(c1, c2)] = opponent_reach[i];
            }
        }

        // Hero range: uniform 1.0 (solver weights externally).
        let mut hero_range = vec![0.0_f32; NUM_COMBOS];
        for &(c1, c2) in &self.private_cards[player] {
            hero_range[card_pair_to_index(c1, c2)] = 1.0;
        }

        let (oop_range, ip_range) = if player == 0 {
            (&hero_range, &opp_range)
        } else {
            (&opp_range, &hero_range)
        };

        let pot_f32 = pot as f32;
        let eff_stack_f32 = remaining_stack as f32;

        let input = encode_boundary_inference_input(
            oop_range, ip_range, &self.board, pot_f32, eff_stack_f32, player as u8,
        );

        // Forward pass (mutex-locked for thread safety).
        let device = Default::default();
        let tensor = Tensor::<NdArray, 2>::from_data(
            TensorData::new(input, [1, INPUT_SIZE]),
            &device,
        );
        let model = self.model.lock().unwrap();
        let output = model.forward(tensor);
        drop(model);
        let normalized: Vec<f32> = output.into_data().to_vec::<f32>().unwrap();

        // Denormalize: multiply by (pot + effective_stack) to get chip EVs.
        let total = pot_f32 + eff_stack_f32;
        let chip_evs: Vec<f32> = normalized.iter().map(|&v| v * total).collect();

        // Map 1326-combo ordering back to game private_cards ordering.
        let mut cfvs = Vec::with_capacity(num_hands);
        for &(c1, c2) in &self.private_cards[player] {
            cfvs.push(chip_evs[card_pair_to_index(c1, c2)]);
        }
        cfvs
    }
}

/// Load a trained `BoundaryNet` from a model directory and wrap it as a
/// `NeuralBoundaryEvaluator` ready for range-solver integration.
///
/// The model directory must contain `config.yaml` (with `training.hidden_layers`
/// and `training.hidden_size`) and the model checkpoint file.
pub fn load_neural_boundary_evaluator(
    model_dir: &Path,
    board: Vec<u8>,
    private_cards: [Vec<(u8, u8)>; 2],
) -> Result<NeuralBoundaryEvaluator, String> {
    let config_path = model_dir.join("config.yaml");
    let yaml = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("read {}: {e}", config_path.display()))?;
    let cfg: crate::config::CfvnetConfig = serde_yaml::from_str(&yaml)
        .map_err(|e| format!("parse {}: {e}", config_path.display()))?;

    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model_path = if model_dir.is_dir() {
        model_dir.join("model")
    } else {
        model_dir.to_path_buf()
    };

    let model = BoundaryNet::<NdArray>::new(
        &device,
        cfg.training.hidden_layers,
        cfg.training.hidden_size,
    )
    .load_file(&model_path, &recorder, &device)
    .map_err(|e| format!("load model from {}: {e}", model_path.display()))?;

    Ok(NeuralBoundaryEvaluator::new(model, board, private_cards))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::network::POT_INDEX;

    #[test]
    fn denormalize_produces_chip_ev() {
        let normalized = vec![0.12_f32, -0.04, 0.0];
        let pot = 100.0;
        let effective_stack = 150.0;
        let chip_ev = denormalize_ev(&normalized, pot, effective_stack);
        assert!((chip_ev[0] - 30.0).abs() < 1e-4);
        assert!((chip_ev[1] - (-10.0)).abs() < 1e-4);
        assert!((chip_ev[2] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn encode_boundary_input_from_ranges() {
        let oop_range = vec![0.5; 1326];
        let ip_range = vec![0.5; 1326];
        let board = vec![0u8, 4, 8, 12, 16];
        let pot = 100.0_f32;
        let effective_stack = 150.0_f32;
        let player = 0u8;
        let input = encode_boundary_inference_input(
            &oop_range, &ip_range, &board, pot, effective_stack, player,
        );
        assert_eq!(input.len(), INPUT_SIZE);
        assert!((input[POT_INDEX] - 0.4).abs() < 1e-6);
        assert!((input[POT_INDEX + 1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn neural_evaluator_returns_correct_cfv_count() {
        use burn::backend::NdArray;
        use crate::model::boundary_net::BoundaryNet;
        use range_solver::game::BoundaryEvaluator;

        let device = Default::default();
        let model = BoundaryNet::<NdArray>::new(&device, 2, 64);

        // Board: 5 cards (flop+turn+river scenario, though boundary usually at turn)
        let board = vec![0u8, 4, 8, 12, 16];

        // Private cards for two players: a small set of non-board card pairs
        // Cards 1,2,3,5,6,7 are not on the board (board uses 0,4,8,12,16)
        let private_cards_p0 = vec![(1u8, 2u8), (1, 3), (2, 3), (5, 6)];
        let private_cards_p1 = vec![(5u8, 7u8), (6, 7), (1, 5)];

        let evaluator = NeuralBoundaryEvaluator::new(
            model,
            board,
            [private_cards_p0.clone(), private_cards_p1.clone()],
        );

        assert_eq!(evaluator.num_continuations(), 1);

        // Test player 0: opponent is player 1, so opponent_reach has p1's hand count
        let opponent_reach = vec![0.5_f32; private_cards_p1.len()];
        let cfvs = evaluator.compute_cfvs(0, 100, 150.0, &opponent_reach, private_cards_p0.len(), 0);
        assert_eq!(cfvs.len(), private_cards_p0.len(), "CFV count should match player 0 hand count");

        // Test player 1: opponent is player 0
        let opponent_reach = vec![0.5_f32; private_cards_p0.len()];
        let cfvs = evaluator.compute_cfvs(1, 100, 150.0, &opponent_reach, private_cards_p1.len(), 0);
        assert_eq!(cfvs.len(), private_cards_p1.len(), "CFV count should match player 1 hand count");
    }

    #[test]
    fn neural_evaluator_handles_zero_opponent_reach() {
        use burn::backend::NdArray;
        use crate::model::boundary_net::BoundaryNet;
        use range_solver::game::BoundaryEvaluator;

        let device = Default::default();
        let model = BoundaryNet::<NdArray>::new(&device, 2, 64);

        let board = vec![0u8, 4, 8, 12, 16];
        let private_cards_p0 = vec![(1u8, 2u8), (1, 3), (2, 3)];
        let private_cards_p1 = vec![(5u8, 6u8), (5, 7)];

        let evaluator = NeuralBoundaryEvaluator::new(
            model,
            board,
            [private_cards_p0.clone(), private_cards_p1.clone()],
        );

        // All-zero opponent reach: boundary is unreachable
        let opponent_reach = vec![0.0_f32; private_cards_p1.len()];
        let cfvs = evaluator.compute_cfvs(0, 100, 150.0, &opponent_reach, private_cards_p0.len(), 0);
        assert_eq!(cfvs.len(), private_cards_p0.len());
        // All CFVs should be zero when opponent has no reach
        for &v in &cfvs {
            assert!((v).abs() < 1e-6, "expected 0.0 for zero opponent reach, got {v}");
        }
    }
}
