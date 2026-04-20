//! Boundary evaluator that adapts BoundaryNet for range-solver leaf evaluation.
//!
//! Two implementations are available, selected at compile time:
//! - Default (burn-based): uses a `BoundaryNet<NdArray>` model loaded from burn checkpoint files.
//! - `onnx` feature: uses `ort::Session` loaded from a `.onnx` file exported by PyTorch.

use std::path::Path;
use std::sync::Arc;

use crate::model::network::{DECK_SIZE, INPUT_SIZE, NUM_COMBOS, NUM_RANKS};
use range_solver::card::card_pair_to_index;

#[cfg(not(feature = "onnx"))]
use std::sync::Mutex;
#[cfg(not(feature = "onnx"))]
use crate::model::boundary_net::BoundaryNet;
#[cfg(not(feature = "onnx"))]
use burn::backend::NdArray;
#[cfg(not(feature = "onnx"))]
use burn::module::Module;
#[cfg(not(feature = "onnx"))]
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
#[cfg(not(feature = "onnx"))]
use burn::tensor::{Tensor, TensorData};

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

// ---------------------------------------------------------------------------
// Burn-based implementation (default, when `onnx` feature is NOT enabled)
// ---------------------------------------------------------------------------

/// Neural boundary evaluator wrapping a trained `BoundaryNet` model.
///
/// Maps between the range-solver's per-player private card ordering and
/// the 1326-combo index ordering used by the network. The model is wrapped
/// in a `Mutex` to satisfy the `Send + Sync` requirement of `BoundaryEvaluator`.
#[cfg(not(feature = "onnx"))]
pub struct NeuralBoundaryEvaluator {
    model: Mutex<BoundaryNet<NdArray>>,
    board: Vec<u8>,
    private_cards: [Vec<(u8, u8)>; 2],
}

#[cfg(not(feature = "onnx"))]
impl NeuralBoundaryEvaluator {
    pub fn new(
        model: BoundaryNet<NdArray>,
        board: Vec<u8>,
        private_cards: [Vec<(u8, u8)>; 2],
    ) -> Self {
        Self { model: Mutex::new(model), board, private_cards }
    }
}

#[cfg(not(feature = "onnx"))]
impl range_solver::game::BoundaryEvaluator for NeuralBoundaryEvaluator {
    fn num_continuations(&self) -> usize { 1 }

    fn compute_cfvs(
        &self,
        _player: usize,
        _pot: i32,
        _remaining_stack: f64,
        _opponent_reach: &[f32],
        _num_hands: usize,
        _continuation_index: usize,
    ) -> Vec<f32> {
        // Learned boundary net was trained on (oop_range, ip_range) pairs of
        // realistic reach distributions. The single-side trait method doesn't
        // receive hero's reach — previous impl uniformed it, producing an
        // out-of-distribution input for the net. Force callers to use
        // compute_cfvs_both which threads both sides' reaches through.
        panic!(
            "NeuralBoundaryEvaluator: use compute_cfvs_both (net needs both \
             ranges; uniform-hero single-side call produces OOD input)"
        );
    }

    fn compute_cfvs_both(
        &self,
        pot: i32,
        remaining_stack: f64,
        oop_reach: &[f32],
        ip_reach: &[f32],
        _num_oop: usize,
        _num_ip: usize,
        _continuation_index: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let oop_cfvs = self.forward_for_player(0, pot, remaining_stack, oop_reach, ip_reach);
        let ip_cfvs  = self.forward_for_player(1, pot, remaining_stack, oop_reach, ip_reach);
        (oop_cfvs, ip_cfvs)
    }
}

#[cfg(not(feature = "onnx"))]
impl NeuralBoundaryEvaluator {
    /// Run one forward pass for a given traverser player, using the real
    /// game-space reach distributions for both sides (no uniform-hero hack).
    fn forward_for_player(
        &self,
        player: usize,
        pot: i32,
        remaining_stack: f64,
        oop_reach_game: &[f32],
        ip_reach_game: &[f32],
    ) -> Vec<f32> {
        // Build 1326-element range vectors from game ordering for BOTH sides.
        let mut oop_range_1326 = vec![0.0_f32; NUM_COMBOS];
        for (i, &(c1, c2)) in self.private_cards[0].iter().enumerate() {
            if i < oop_reach_game.len() {
                oop_range_1326[card_pair_to_index(c1, c2)] = oop_reach_game[i];
            }
        }
        let mut ip_range_1326 = vec![0.0_f32; NUM_COMBOS];
        for (i, &(c1, c2)) in self.private_cards[1].iter().enumerate() {
            if i < ip_reach_game.len() {
                ip_range_1326[card_pair_to_index(c1, c2)] = ip_reach_game[i];
            }
        }

        let pot_f32 = pot as f32;
        let eff_stack_f32 = remaining_stack as f32;

        let input = encode_boundary_inference_input(
            &oop_range_1326, &ip_range_1326, &self.board, pot_f32, eff_stack_f32, player as u8,
        );

        let device = Default::default();
        let tensor = Tensor::<NdArray, 2>::from_data(
            TensorData::new(input, [1, INPUT_SIZE]),
            &device,
        );
        let model = self.model.lock().unwrap();
        let output = model.forward(tensor);
        drop(model);
        let normalized: Vec<f32> = output.into_data().to_vec::<f32>().unwrap();

        let total = pot_f32 + eff_stack_f32;
        let chip_evs: Vec<f32> = normalized.iter().map(|&v| v * total).collect();

        // Map 1326-combo ordering back to game private_cards ordering.
        let hands = &self.private_cards[player];
        let mut cfvs = Vec::with_capacity(hands.len());
        for &(c1, c2) in hands {
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
#[cfg(not(feature = "onnx"))]
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

// ---------------------------------------------------------------------------
// ONNX-based implementation (when `onnx` feature IS enabled)
// ---------------------------------------------------------------------------

/// Neural boundary evaluator using ONNX Runtime for inference.
///
/// Loads a `.onnx` model exported from PyTorch. `ort::Session` is natively
/// `Send + Sync` and `run` takes `&self`, so no `Mutex` is needed.
/// The session is wrapped in `Arc` so multiple evaluators (one per boundary)
/// can share a single loaded model.
#[cfg(feature = "onnx")]
pub struct NeuralBoundaryEvaluator {
    session: Arc<ort::session::Session>,
    board: Vec<u8>,
    private_cards: [Vec<(u8, u8)>; 2],
}

#[cfg(feature = "onnx")]
impl range_solver::game::BoundaryEvaluator for NeuralBoundaryEvaluator {
    fn num_continuations(&self) -> usize { 1 }

    fn compute_cfvs(
        &self,
        _player: usize,
        _pot: i32,
        _remaining_stack: f64,
        _opponent_reach: &[f32],
        _num_hands: usize,
        _continuation_index: usize,
    ) -> Vec<f32> {
        // See rationale on the Burn variant above. Learned boundary net
        // requires both realistic reach distributions; single-side path
        // uniformed the hero side and produced OOD input.
        panic!(
            "NeuralBoundaryEvaluator (onnx): use compute_cfvs_both (net needs \
             both ranges; uniform-hero single-side call produces OOD input)"
        );
    }

    fn compute_cfvs_both(
        &self,
        pot: i32,
        remaining_stack: f64,
        oop_reach: &[f32],
        ip_reach: &[f32],
        _num_oop: usize,
        _num_ip: usize,
        _continuation_index: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let oop_cfvs = self.forward_for_player(0, pot, remaining_stack, oop_reach, ip_reach);
        let ip_cfvs  = self.forward_for_player(1, pot, remaining_stack, oop_reach, ip_reach);
        (oop_cfvs, ip_cfvs)
    }
}

#[cfg(feature = "onnx")]
impl NeuralBoundaryEvaluator {
    fn forward_for_player(
        &self,
        player: usize,
        pot: i32,
        remaining_stack: f64,
        oop_reach_game: &[f32],
        ip_reach_game: &[f32],
    ) -> Vec<f32> {
        // Build 1326-element range vectors from game ordering for BOTH sides.
        let mut oop_range_1326 = vec![0.0_f32; NUM_COMBOS];
        for (i, &(c1, c2)) in self.private_cards[0].iter().enumerate() {
            if i < oop_reach_game.len() {
                oop_range_1326[card_pair_to_index(c1, c2)] = oop_reach_game[i];
            }
        }
        let mut ip_range_1326 = vec![0.0_f32; NUM_COMBOS];
        for (i, &(c1, c2)) in self.private_cards[1].iter().enumerate() {
            if i < ip_reach_game.len() {
                ip_range_1326[card_pair_to_index(c1, c2)] = ip_reach_game[i];
            }
        }

        let pot_f32 = pot as f32;
        let eff_stack_f32 = remaining_stack as f32;

        let input_vec = encode_boundary_inference_input(
            &oop_range_1326, &ip_range_1326, &self.board, pot_f32, eff_stack_f32, player as u8,
        );

        let input_tensor = ort::value::Tensor::from_array(
            ([1_i64, INPUT_SIZE as i64], input_vec),
        ).expect("ort tensor creation");
        let outputs = self.session
            .run(ort::inputs![input_tensor].expect("ort inputs"))
            .expect("ort session run");
        let output_view = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("ort output extract f32");
        let normalized: Vec<f32> = output_view.iter().copied().collect();

        // Unit conversion: target = cfv_halfpot * pot / (pot + eff_stack)
        // → cfv_halfpot = output * (pot + eff_stack) / pot
        let scale = if pot_f32 > 0.0 {
            (pot_f32 + eff_stack_f32) / pot_f32
        } else {
            0.0
        };

        // Map 1326-combo ordering back to game private_cards ordering.
        let hands = &self.private_cards[player];
        let mut cfvs = Vec::with_capacity(hands.len());
        for &(c1, c2) in hands {
            cfvs.push(normalized[card_pair_to_index(c1, c2)] * scale);
        }
        cfvs
    }
}

/// Load a trained model from an `.onnx` file and wrap it as a
/// `NeuralBoundaryEvaluator` ready for range-solver integration.
#[cfg(feature = "onnx")]
pub fn load_neural_boundary_evaluator(
    model_path: &Path,
    board: Vec<u8>,
    private_cards: [Vec<(u8, u8)>; 2],
) -> Result<NeuralBoundaryEvaluator, String> {
    let session = load_shared_onnx_session(model_path)?;
    Ok(NeuralBoundaryEvaluator { session, board, private_cards })
}

/// Load an ONNX session from a model file, returning it wrapped in `Arc`.
///
/// Use this when multiple `NeuralBoundaryEvaluator` instances (one per boundary)
/// need to share a single model load. Loading is expensive (~100ms for a 26MB
/// model), so loading once and sharing via Arc is critical for depth-limited
/// solves with 100+ boundaries.
#[cfg(feature = "onnx")]
pub fn load_shared_onnx_session(model_path: &Path) -> Result<Arc<ort::session::Session>, String> {
    use ort::session::{Session, builder::GraphOptimizationLevel};

    let session = Session::builder()
        .map_err(|e| format!("ort session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("ort optimization: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("ort load {}: {e}", model_path.display()))?;
    Ok(Arc::new(session))
}

/// Create a `NeuralBoundaryEvaluator` from a pre-loaded shared session.
///
/// Each boundary in a depth-limited solve has a different board (e.g., flop +
/// different turn card). This constructor avoids reloading the ONNX model for
/// each boundary.
#[cfg(feature = "onnx")]
pub fn neural_boundary_evaluator_from_shared(
    session: Arc<ort::session::Session>,
    board: Vec<u8>,
    private_cards: [Vec<(u8, u8)>; 2],
) -> NeuralBoundaryEvaluator {
    NeuralBoundaryEvaluator { session, board, private_cards }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::network::POT_INDEX;

    #[cfg(feature = "onnx")]
    mod onnx_tests {
        use super::*;
        use std::path::PathBuf;
        use std::sync::Arc;

        #[test]
        fn load_onnx_evaluator_rejects_missing_file() {
            let board = vec![0u8, 4, 8, 12, 16];
            let private_cards_p0 = vec![(1u8, 2u8), (1, 3)];
            let private_cards_p1 = vec![(5u8, 6u8), (5, 7)];
            let result = load_neural_boundary_evaluator(
                &PathBuf::from("/nonexistent/model.onnx"),
                board,
                [private_cards_p0, private_cards_p1],
            );
            match result {
                Err(err) => {
                    assert!(err.contains("ort load"), "error should mention ort load, got: {err}");
                }
                Ok(_) => panic!("loading from non-existent path should fail"),
            }
        }

        #[test]
        fn onnx_evaluator_is_send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<NeuralBoundaryEvaluator>();
        }

        #[test]
        fn onnx_evaluator_implements_boundary_trait() {
            // Verify NeuralBoundaryEvaluator implements BoundaryEvaluator at compile time.
            fn _assert_trait<T: range_solver::game::BoundaryEvaluator>() {}
            _assert_trait::<NeuralBoundaryEvaluator>();
        }

        #[test]
        fn load_shared_session_rejects_missing_file() {
            let result = load_shared_onnx_session(&PathBuf::from("/nonexistent/model.onnx"));
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.contains("ort"), "error should mention ort, got: {err}");
        }

        #[test]
        fn from_shared_constructor_produces_valid_evaluator() {
            // This test validates the from_shared constructor compiles and
            // produces a NeuralBoundaryEvaluator that implements BoundaryEvaluator.
            // Uses an ignored gate since it requires a real ONNX model file.
            fn _assert_from_shared_compiles(session: Arc<ort::session::Session>) {
                let board = vec![0u8, 4, 8, 12];
                let private_cards = [
                    vec![(1u8, 2u8), (1, 3)],
                    vec![(5u8, 6u8), (5, 7)],
                ];
                let eval = neural_boundary_evaluator_from_shared(
                    session, board, private_cards,
                );
                // Verify it implements BoundaryEvaluator
                fn _check<T: range_solver::game::BoundaryEvaluator>(_: &T) {}
                _check(&eval);
            }
        }

        #[test]
        #[ignore] // Requires ONNX model file on disk
        fn shared_session_smoke_test_compute_cfvs() {
            use range_solver::game::BoundaryEvaluator;

            let model_path = PathBuf::from(
                "../../local_data/models/cfvnet_river_py_v2/checkpoint_epoch675.onnx"
            );
            if !model_path.exists() {
                eprintln!("skipping: ONNX model not found at {}", model_path.display());
                return;
            }

            let session = load_shared_onnx_session(&model_path)
                .expect("load shared session");
            assert_eq!(Arc::strong_count(&session), 1);

            // Create two evaluators sharing the same session (different boards)
            let board_a = vec![0u8, 4, 8, 12]; // 4-card turn board
            let board_b = vec![0u8, 4, 8, 16]; // different turn card

            // Generate non-conflicting private cards for board_a
            let private_cards_a: [Vec<(u8, u8)>; 2] = [
                (1..=20).step_by(2).map(|i| (i as u8, (i + 1) as u8))
                    .filter(|&(c1, c2)| ![0, 4, 8, 12].contains(&c1) && ![0, 4, 8, 12].contains(&c2))
                    .take(5)
                    .collect(),
                (21..=40).step_by(2).map(|i| (i as u8, (i + 1) as u8))
                    .filter(|&(c1, c2)| ![0, 4, 8, 12].contains(&c1) && ![0, 4, 8, 12].contains(&c2))
                    .take(5)
                    .collect(),
            ];
            let private_cards_b: [Vec<(u8, u8)>; 2] = [
                (1..=20).step_by(2).map(|i| (i as u8, (i + 1) as u8))
                    .filter(|&(c1, c2)| ![0, 4, 8, 16].contains(&c1) && ![0, 4, 8, 16].contains(&c2))
                    .take(5)
                    .collect(),
                (21..=40).step_by(2).map(|i| (i as u8, (i + 1) as u8))
                    .filter(|&(c1, c2)| ![0, 4, 8, 16].contains(&c1) && ![0, 4, 8, 16].contains(&c2))
                    .take(5)
                    .collect(),
            ];

            let eval_a = neural_boundary_evaluator_from_shared(
                Arc::clone(&session), board_a, private_cards_a.clone(),
            );
            let eval_b = neural_boundary_evaluator_from_shared(
                Arc::clone(&session), board_b, private_cards_b.clone(),
            );
            assert_eq!(Arc::strong_count(&session), 3);

            // compute_cfvs should return correct-length output
            let num_hands_p0 = private_cards_a[0].len();
            let opponent_reach = vec![1.0_f32; private_cards_a[1].len()];
            let cfvs = eval_a.compute_cfvs(0, 200, 100.0, &opponent_reach, num_hands_p0, 0);
            assert_eq!(cfvs.len(), num_hands_p0, "CFV count must match player 0 hands");

            // All CFVs should be finite
            for &v in &cfvs {
                assert!(v.is_finite(), "CFV must be finite, got {v}");
            }

            // eval_b should also work
            let num_hands_p0_b = private_cards_b[0].len();
            let opponent_reach_b = vec![1.0_f32; private_cards_b[1].len()];
            let cfvs_b = eval_b.compute_cfvs(0, 200, 100.0, &opponent_reach_b, num_hands_p0_b, 0);
            assert_eq!(cfvs_b.len(), num_hands_p0_b);
        }
    }

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

    #[cfg(not(feature = "onnx"))]
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

        // compute_cfvs_both takes both sides' reach distributions. Single-side
        // compute_cfvs panics by design (net was trained on both realistic
        // ranges; uniform-hero fallback produced OOD inputs).
        let oop_reach = vec![0.5_f32; private_cards_p0.len()];
        let ip_reach  = vec![0.5_f32; private_cards_p1.len()];
        let (oop_cfvs, ip_cfvs) = evaluator.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach,
            private_cards_p0.len(), private_cards_p1.len(), 0,
        );
        assert_eq!(oop_cfvs.len(), private_cards_p0.len(), "OOP CFV count should match p0 hand count");
        assert_eq!(ip_cfvs.len(),  private_cards_p1.len(), "IP CFV count should match p1 hand count");
    }

    #[cfg(not(feature = "onnx"))]
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

        // With compute_cfvs_both both reach vectors are passed through;
        // assert shape only (net may extrapolate on zero-range inputs).
        let zero_oop = vec![0.0_f32; private_cards_p0.len()];
        let zero_ip  = vec![0.0_f32; private_cards_p1.len()];
        let (oop_cfvs, ip_cfvs) = evaluator.compute_cfvs_both(
            100, 150.0, &zero_oop, &zero_ip,
            private_cards_p0.len(), private_cards_p1.len(), 0,
        );
        assert_eq!(oop_cfvs.len(), private_cards_p0.len());
        assert_eq!(ip_cfvs.len(),  private_cards_p1.len());
    }
}
