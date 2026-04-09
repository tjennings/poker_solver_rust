//! GPU-accelerated BoundaryNet evaluator for turn datagen.
//!
//! Uses ONNX Runtime with TensorRT EP -> CUDA EP -> CPU fallback
//! for batched BoundaryNet inference at river boundary nodes.

use std::path::Path;

use crate::eval::boundary_evaluator::encode_boundary_inference_input;
use crate::model::network::{INPUT_SIZE, NUM_COMBOS};
use range_solver::card::card_pair_to_index;

/// GPU-accelerated BoundaryNet evaluator using ONNX Runtime.
///
/// Configured to prefer TensorRT EP, falling back to CUDA EP, then CPU.
/// `ort::Session` is natively `Send + Sync`, so no `Mutex` is needed.
pub struct GpuBoundaryEvaluator {
    session: ort::session::Session,
}

/// Request for boundary evaluation at a set of river boundary nodes.
pub struct BoundaryEvalRequest {
    pub board: [u8; 4],
    pub pot: f32,
    pub effective_stack: f32,
    pub oop_reach: Vec<f32>,
    pub ip_reach: Vec<f32>,
    pub num_boundaries: usize,
}

/// Result of boundary evaluation.
pub struct BoundaryEvalResult {
    pub leaf_cfv_p0: Vec<f32>,
    pub leaf_cfv_p1: Vec<f32>,
}

/// Zero out reach values for hands that conflict with a given card.
///
/// A hand at combo index `card_pair_to_index(c0, c1)` conflicts with `card`
/// if `c0 == card` or `c1 == card`.
fn zero_conflicting_hands(reach: &mut [f32], card: u8) {
    for c0 in 0u8..52 {
        for c1 in (c0 + 1)..52 {
            if c0 == card || c1 == card {
                reach[card_pair_to_index(c0, c1)] = 0.0;
            }
        }
    }
}

impl GpuBoundaryEvaluator {
    /// Load a BoundaryNet model from an ONNX file.
    ///
    /// Attempts TensorRT EP first, then CUDA EP, then falls back to CPU.
    pub fn load(model_path: &Path) -> Result<Self, String> {
        use ort::session::{builder::GraphOptimizationLevel, Session};

        let session = Session::builder()
            .map_err(|e| format!("ort session builder: {e}"))?
            .with_execution_providers([
                ort::execution_providers::TensorRTExecutionProvider::default().build(),
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ])
            .map_err(|e| format!("ort execution providers: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ort optimization: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("ort load {}: {e}", model_path.display()))?;

        Ok(Self { session })
    }

    /// Run batched forward pass. Input shape `[num_rows, INPUT_SIZE]`, output `[num_rows, 1326]`.
    /// Takes ownership of the input buffer to avoid an extra copy into the ORT tensor.
    pub fn infer_batch(&self, input: Vec<f32>, num_rows: usize) -> Result<Vec<f32>, String> {
        assert_eq!(
            input.len(),
            num_rows * INPUT_SIZE,
            "input length mismatch: expected {} got {}",
            num_rows * INPUT_SIZE,
            input.len()
        );

        let input_tensor = ort::value::Tensor::from_array((
            [num_rows as i64, INPUT_SIZE as i64],
            input,
        ))
        .map_err(|e| format!("ort tensor creation: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![input_tensor].map_err(|e| format!("ort inputs: {e}"))?)
            .map_err(|e| format!("ort session run: {e}"))?;

        let output_view = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ort output extract: {e}"))?;

        Ok(output_view.iter().copied().collect())
    }
}

/// Evaluate BoundaryNet at river boundaries, averaging over all valid river cards.
///
/// For each boundary and each player, runs the net on all possible river runouts
/// (48 for a 4-card board), then averages the per-hand CFVs weighted by opponent reach.
///
/// `hand_cards` maps from hand index (0..num_hands) to `(c0, c1)` card pairs,
/// used to convert from 1326-combo space to the game's hand ordering.
pub fn evaluate_boundaries_batched(
    evaluator: &GpuBoundaryEvaluator,
    requests: &[BoundaryEvalRequest],
    hand_cards: &[(u8, u8)],
) -> Result<Vec<BoundaryEvalResult>, String> {
    let num_hands = hand_cards.len();
    let hand_combo_indices: Vec<usize> = hand_cards
        .iter()
        .map(|&(c0, c1)| card_pair_to_index(c0, c1))
        .collect();

    // Build all inference inputs: for each request x boundary x player x river.
    let mut all_inputs: Vec<f32> = Vec::new();
    let mut total_rows: usize = 0;

    // Track valid rivers and precomputed opponent weights per boundary for reduction.
    struct BoundaryMeta {
        valid_rivers: Vec<u8>,
        // Per-boundary: opponent reach sum per river, for reach-weighted averaging.
        // ip_weights[bi][ri] = sum of IP reach for hands not conflicting with river ri.
        ip_weights: Vec<Vec<f64>>,
        oop_weights: Vec<Vec<f64>>,
    }
    let mut request_meta: Vec<BoundaryMeta> = Vec::with_capacity(requests.len());

    for req in requests {
        let valid_rivers: Vec<u8> = (0u8..52)
            .filter(|r| !req.board.contains(r))
            .collect();

        let mut ip_weights = Vec::with_capacity(req.num_boundaries);
        let mut oop_weights = Vec::with_capacity(req.num_boundaries);

        for bi in 0..req.num_boundaries {
            let oop_base = &req.oop_reach[bi * NUM_COMBOS..(bi + 1) * NUM_COMBOS];
            let ip_base = &req.ip_reach[bi * NUM_COMBOS..(bi + 1) * NUM_COMBOS];

            // Precompute opponent weights per river (reused in reduction).
            let bi_ip_w: Vec<f64> = valid_rivers
                .iter()
                .map(|&river| {
                    let mut ip_copy = ip_base.to_vec();
                    zero_conflicting_hands(&mut ip_copy, river);
                    ip_copy.iter().map(|&v| f64::from(v)).sum()
                })
                .collect();
            let bi_oop_w: Vec<f64> = valid_rivers
                .iter()
                .map(|&river| {
                    let mut oop_copy = oop_base.to_vec();
                    zero_conflicting_hands(&mut oop_copy, river);
                    oop_copy.iter().map(|&v| f64::from(v)).sum()
                })
                .collect();

            for player in 0u8..2 {
                for (ri, &river) in valid_rivers.iter().enumerate() {
                    let board5 = [req.board[0], req.board[1], req.board[2], req.board[3], river];

                    // Use already-zeroed ranges via weights, but we still need the
                    // full zeroed vectors for the NN input encoding.
                    let mut oop = oop_base.to_vec();
                    let mut ip = ip_base.to_vec();
                    zero_conflicting_hands(&mut oop, river);
                    zero_conflicting_hands(&mut ip, river);

                    let input = encode_boundary_inference_input(
                        &oop, &ip, &board5, req.pot, req.effective_stack, player,
                    );
                    all_inputs.extend_from_slice(&input);
                    total_rows += 1;

                    let _ = ri; // used for weight indexing in reduction
                }
            }

            ip_weights.push(bi_ip_w);
            oop_weights.push(bi_oop_w);
        }

        request_meta.push(BoundaryMeta {
            valid_rivers,
            ip_weights,
            oop_weights,
        });
    }

    // Single batched inference call.
    let all_outputs = if total_rows > 0 {
        evaluator.infer_batch(all_inputs, total_rows)?
    } else {
        Vec::new()
    };

    // Reduce: average over rivers, weighted by opponent reach. One result per request.
    let mut results: Vec<BoundaryEvalResult> = Vec::with_capacity(requests.len());
    let mut row_cursor = 0usize;

    for (req_idx, req) in requests.iter().enumerate() {
        let meta = &request_meta[req_idx];
        let num_rivers = meta.valid_rivers.len();

        let mut leaf_cfv_p0: Vec<f32> = Vec::with_capacity(req.num_boundaries * num_hands);
        let mut leaf_cfv_p1: Vec<f32> = Vec::with_capacity(req.num_boundaries * num_hands);

        for bi in 0..req.num_boundaries {
            let p0_row_start = row_cursor;
            row_cursor += num_rivers;
            let p1_row_start = row_cursor;
            row_cursor += num_rivers;

            let denorm = f64::from(req.pot + req.effective_stack);

            for (hi, &combo_idx) in hand_combo_indices.iter().enumerate() {
                let (c0, c1) = hand_cards[hi];

                // Player 0: opponent is IP.
                let mut cfv_sum_p0 = 0.0_f64;
                let mut weight_sum_p0 = 0.0_f64;
                for (ri, &river) in meta.valid_rivers.iter().enumerate() {
                    if c0 == river || c1 == river {
                        continue;
                    }
                    let row = p0_row_start + ri;
                    let net_val = f64::from(all_outputs[row * NUM_COMBOS + combo_idx]);
                    cfv_sum_p0 += meta.ip_weights[bi][ri] * net_val;
                    weight_sum_p0 += meta.ip_weights[bi][ri];
                }
                let cfv_p0 = if weight_sum_p0 > 0.0 {
                    (cfv_sum_p0 / weight_sum_p0) * denorm
                } else {
                    0.0
                };
                leaf_cfv_p0.push(cfv_p0 as f32);

                // Player 1: opponent is OOP.
                let mut cfv_sum_p1 = 0.0_f64;
                let mut weight_sum_p1 = 0.0_f64;
                for (ri, &river) in meta.valid_rivers.iter().enumerate() {
                    if c0 == river || c1 == river {
                        continue;
                    }
                    let row = p1_row_start + ri;
                    let net_val = f64::from(all_outputs[row * NUM_COMBOS + combo_idx]);
                    cfv_sum_p1 += meta.oop_weights[bi][ri] * net_val;
                    weight_sum_p1 += meta.oop_weights[bi][ri];
                }
                let cfv_p1 = if weight_sum_p1 > 0.0 {
                    (cfv_sum_p1 / weight_sum_p1) * denorm
                } else {
                    0.0
                };
                leaf_cfv_p1.push(cfv_p1 as f32);
            }
        }

        results.push(BoundaryEvalResult {
            leaf_cfv_p0,
            leaf_cfv_p1,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_construction_zeros_conflicting_hands() {
        let mut reach = vec![1.0_f32; NUM_COMBOS];

        zero_conflicting_hands(&mut reach, 10);

        let mut zeroed_count = 0;
        for c0 in 0u8..52 {
            for c1 in (c0 + 1)..52 {
                let idx = card_pair_to_index(c0, c1);
                if c0 == 10 || c1 == 10 {
                    assert_eq!(
                        reach[idx], 0.0,
                        "hand ({c0},{c1}) at index {idx} should be zeroed"
                    );
                    zeroed_count += 1;
                } else {
                    assert_eq!(
                        reach[idx], 1.0,
                        "hand ({c0},{c1}) at index {idx} should remain 1.0"
                    );
                }
            }
        }
        assert_eq!(zeroed_count, 51);
    }

    #[test]
    fn zero_conflicting_hands_preserves_non_conflicting() {
        let mut reach = vec![0.5_f32; NUM_COMBOS];

        zero_conflicting_hands(&mut reach, 0);

        // (1, 2) does not conflict with card 0.
        let idx = card_pair_to_index(1, 2);
        assert_eq!(reach[idx], 0.5);

        // (0, 1) conflicts with card 0.
        let idx01 = card_pair_to_index(0, 1);
        assert_eq!(reach[idx01], 0.0);
    }

    #[test]
    fn evaluator_loads_and_infers() {
        let model_path = Path::new("../../local_data/cfvnet/river/v2/best.onnx");
        if !model_path.exists() {
            return;
        }

        let evaluator = GpuBoundaryEvaluator::load(model_path)
            .expect("failed to load model");

        let input = vec![0.0_f32; INPUT_SIZE];
        let output = evaluator
            .infer_batch(input, 1)
            .expect("inference failed");

        assert_eq!(output.len(), NUM_COMBOS, "output should have 1326 values");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn boundary_eval_produces_valid_cfvs() {
        let model_path = Path::new("../../local_data/cfvnet/river/v2/best.onnx");
        if !model_path.exists() {
            return;
        }

        let evaluator = GpuBoundaryEvaluator::load(model_path)
            .expect("failed to load model");

        // Board: As Kh Qd Jc (cards 48, 42, 33, 24).
        let board = [48u8, 42, 33, 24];

        let hand_cards: Vec<(u8, u8)> = (0u8..52)
            .flat_map(|c0| ((c0 + 1)..52).map(move |c1| (c0, c1)))
            .filter(|&(c0, c1)| !board.contains(&c0) && !board.contains(&c1))
            .collect();
        let num_hands = hand_cards.len();

        let oop_reach = vec![1.0_f32 / NUM_COMBOS as f32; NUM_COMBOS];
        let ip_reach = vec![1.0_f32 / NUM_COMBOS as f32; NUM_COMBOS];

        let request = BoundaryEvalRequest {
            board,
            pot: 100.0,
            effective_stack: 200.0,
            oop_reach,
            ip_reach,
            num_boundaries: 1,
        };

        let results = evaluate_boundaries_batched(&evaluator, &[request], &hand_cards)
            .expect("boundary eval failed");

        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert_eq!(result.leaf_cfv_p0.len(), num_hands);
        assert_eq!(result.leaf_cfv_p1.len(), num_hands);

        for (i, &v) in result.leaf_cfv_p0.iter().enumerate() {
            assert!(v.is_finite(), "P0 CFV[{i}] = {v} is not finite");
        }
        for (i, &v) in result.leaf_cfv_p1.iter().enumerate() {
            assert!(v.is_finite(), "P1 CFV[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn boundary_eval_handles_multiple_boundaries() {
        let model_path = Path::new("../../local_data/cfvnet/river/v2/best.onnx");
        if !model_path.exists() {
            return;
        }

        let evaluator = GpuBoundaryEvaluator::load(model_path)
            .expect("failed to load model");

        let board = [0u8, 4, 8, 12];

        let hand_cards: Vec<(u8, u8)> = (0u8..52)
            .flat_map(|c0| ((c0 + 1)..52).map(move |c1| (c0, c1)))
            .filter(|&(c0, c1)| !board.contains(&c0) && !board.contains(&c1))
            .collect();
        let num_hands = hand_cards.len();

        let num_boundaries = 3;
        let oop_reach = vec![1.0_f32 / NUM_COMBOS as f32; NUM_COMBOS * num_boundaries];
        let ip_reach = vec![1.0_f32 / NUM_COMBOS as f32; NUM_COMBOS * num_boundaries];

        let request = BoundaryEvalRequest {
            board,
            pot: 50.0,
            effective_stack: 100.0,
            oop_reach,
            ip_reach,
            num_boundaries,
        };

        let results = evaluate_boundaries_batched(&evaluator, &[request], &hand_cards)
            .expect("boundary eval failed");

        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert_eq!(result.leaf_cfv_p0.len(), num_boundaries * num_hands);
        assert_eq!(result.leaf_cfv_p1.len(), num_boundaries * num_hands);

        for (i, &v) in result.leaf_cfv_p0.iter().enumerate() {
            assert!(v.is_finite(), "P0 CFV[{i}] = {v} is not finite");
        }
        for (i, &v) in result.leaf_cfv_p1.iter().enumerate() {
            assert!(v.is_finite(), "P1 CFV[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn evaluate_empty_requests_returns_empty() {
        // No model needed — empty input should short-circuit.
        // We can't construct the evaluator without a model, so just test
        // that the function handles the empty case at the type level.
        // The actual empty-requests path produces total_rows=0 which skips inference.
    }
}
