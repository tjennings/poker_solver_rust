//! CfvNet-backed [`LeafEvaluator`] for depth-boundary evaluation.
//!
//! Unlike [`SharedRiverNetEvaluator`](cfvnet::eval::river_net_evaluator::SharedRiverNetEvaluator),
//! which averages over 48 possible river cards for turn boundaries,
//! [`RebelLeafEvaluator`] evaluates the public belief state (PBS) directly.
//! The value network has learned to predict CFVs at any street boundary,
//! so no river enumeration is needed.

use std::sync::{Arc, Mutex};

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use cfvnet::eval::river_net_evaluator::{build_input, rs_card_to_u8};
use cfvnet::model::network::{CfvNet, INPUT_SIZE, OUTPUT_SIZE};
use poker_solver_core::blueprint_v2::LeafEvaluator;
use poker_solver_core::poker::Card;
use range_solver::card::card_pair_to_index;

/// A [`LeafEvaluator`] backed by a CfvNet value network.
///
/// Evaluates depth-boundary CFVs in a single forward pass, without river
/// enumeration. The model predicts pot-relative CFVs for all 1326 combos
/// given the current ranges, board, pot, stack, and traverser.
pub struct RebelLeafEvaluator<B: Backend> {
    model: Arc<Mutex<CfvNet<B>>>,
    device: B::Device,
}

impl<B: Backend> RebelLeafEvaluator<B> {
    /// Create a new evaluator wrapping the given model and device.
    pub fn new(model: CfvNet<B>, device: B::Device) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            device,
        }
    }

    /// Create from a shared model (useful when multiple evaluators share one GPU model).
    pub fn from_shared(model: Arc<Mutex<CfvNet<B>>>, device: B::Device) -> Self {
        Self { model, device }
    }
}

/// Build the 2720-element input vector for a single boundary evaluation.
///
/// This mirrors [`build_input`](cfvnet::eval::river_net_evaluator::build_input)
/// but accepts variable-length boards (3-5 cards) instead of requiring exactly 5.
///
/// Layout:
///   - `[0..1326)`: OOP range probabilities
///   - `[1326..2652)`: IP range probabilities
///   - `[2652..2704)`: board one-hot (52 elements)
///   - `[2704..2717)`: rank presence (13 elements)
///   - `[2717]`: pot / 400.0
///   - `[2718]`: effective_stack / 400.0
///   - `[2719]`: player indicator (0.0=OOP, 1.0=IP)
pub fn build_boundary_input(
    oop_1326: &[f32; OUTPUT_SIZE],
    ip_1326: &[f32; OUTPUT_SIZE],
    board_u8: &[u8],
    pot: f64,
    effective_stack: f64,
    traverser: u8,
) -> Vec<f32> {
    // If the board has exactly 5 cards, delegate to the existing build_input.
    if board_u8.len() == 5 {
        let board_arr: [u8; 5] = [board_u8[0], board_u8[1], board_u8[2], board_u8[3], board_u8[4]];
        return build_input(oop_1326, ip_1326, &board_arr, pot, effective_stack, traverser);
    }

    let mut input = Vec::with_capacity(INPUT_SIZE);
    input.extend_from_slice(oop_1326);
    input.extend_from_slice(ip_1326);

    // Board one-hot (52 elements).
    let mut board_onehot = [0.0_f32; 52];
    for &card in board_u8 {
        debug_assert!((card as usize) < 52, "card id {card} out of range");
        board_onehot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_onehot);

    // Rank presence (13 elements).
    let mut rank_presence = [0.0_f32; 13];
    for &card in board_u8 {
        rank_presence[(card / 4) as usize] = 1.0;
    }
    input.extend_from_slice(&rank_presence);

    // Pot, stack, player indicator.
    input.push(pot as f32 / 400.0);
    input.push(effective_stack as f32 / 400.0);
    input.push(if traverser == 0 { 0.0 } else { 1.0 });

    debug_assert_eq!(input.len(), INPUT_SIZE);
    input
}

impl<B: Backend> LeafEvaluator for RebelLeafEvaluator<B>
where
    B::Device: Clone,
{
    fn evaluate(
        &self,
        combos: &[[Card; 2]],
        board: &[Card],
        pot: f64,
        effective_stack: f64,
        oop_range: &[f64],
        ip_range: &[f64],
        traverser: u8,
    ) -> Vec<f64> {
        assert!(
            (3..=5).contains(&board.len()),
            "RebelLeafEvaluator: board must have 3-5 cards, got {}",
            board.len()
        );
        assert_eq!(combos.len(), oop_range.len());
        assert_eq!(combos.len(), ip_range.len());

        let num_combos = combos.len();

        // Convert combos to u8 pairs and canonical 1326-indices.
        let combos_u8: Vec<[u8; 2]> = combos
            .iter()
            .map(|c| [rs_card_to_u8(c[0]), rs_card_to_u8(c[1])])
            .collect();
        let combo_indices: Vec<usize> = combos_u8
            .iter()
            .map(|c| card_pair_to_index(c[0], c[1]))
            .collect();

        // Convert board to u8.
        let board_u8: Vec<u8> = board.iter().map(|c| rs_card_to_u8(*c)).collect();

        // Map solver ranges to 1326-indexed arrays.
        let mut oop_1326 = [0.0_f32; OUTPUT_SIZE];
        let mut ip_1326 = [0.0_f32; OUTPUT_SIZE];
        for (i, &idx) in combo_indices.iter().enumerate() {
            oop_1326[idx] = oop_range[i] as f32;
            ip_1326[idx] = ip_range[i] as f32;
        }

        // Build input vector.
        let input_vec =
            build_boundary_input(&oop_1326, &ip_1326, &board_u8, pot, effective_stack, traverser);

        // Forward pass.
        let model = self.model.lock().unwrap();
        let data = TensorData::new(input_vec, [1, INPUT_SIZE]);
        let input_tensor = Tensor::<B, 2>::from_data(data, &self.device);
        let output = model.forward(input_tensor);
        drop(model);

        // Extract output and map back to solver combo ordering.
        let out_data = output.into_data();
        let out_vec: Vec<f32> = out_data.to_vec().expect("output tensor conversion");

        // Map 1326-indexed outputs back to solver combo order.
        // The network returns pot-relative CFVs, which is what the solver expects.
        let mut result = Vec::with_capacity(num_combos);
        for &idx in &combo_indices {
            result.push(f64::from(out_vec[idx]));
        }
        result
    }

    fn evaluate_boundaries(
        &self,
        combos: &[[Card; 2]],
        board: &[Card],
        oop_range: &[f64],
        ip_range: &[f64],
        requests: &[(f64, f64, u8)],
    ) -> Vec<Vec<f64>> {
        assert!(
            (3..=5).contains(&board.len()),
            "RebelLeafEvaluator: board must have 3-5 cards, got {}",
            board.len()
        );
        assert_eq!(combos.len(), oop_range.len());
        assert_eq!(combos.len(), ip_range.len());
        if requests.is_empty() {
            return Vec::new();
        }

        let num_combos = combos.len();

        // Convert combos to u8 pairs and canonical 1326-indices.
        let combos_u8: Vec<[u8; 2]> = combos
            .iter()
            .map(|c| [rs_card_to_u8(c[0]), rs_card_to_u8(c[1])])
            .collect();
        let combo_indices: Vec<usize> = combos_u8
            .iter()
            .map(|c| card_pair_to_index(c[0], c[1]))
            .collect();

        // Convert board to u8.
        let board_u8: Vec<u8> = board.iter().map(|c| rs_card_to_u8(*c)).collect();

        // Map solver ranges to 1326-indexed arrays (shared across all requests).
        let mut oop_1326 = [0.0_f32; OUTPUT_SIZE];
        let mut ip_1326 = [0.0_f32; OUTPUT_SIZE];
        for (i, &idx) in combo_indices.iter().enumerate() {
            oop_1326[idx] = oop_range[i] as f32;
            ip_1326[idx] = ip_range[i] as f32;
        }

        // Build N input vectors (one per request) and stack into a batch.
        let batch_size = requests.len();
        let mut inputs: Vec<f32> = Vec::with_capacity(batch_size * INPUT_SIZE);
        for &(pot, effective_stack, traverser) in requests {
            let input_vec = build_boundary_input(
                &oop_1326,
                &ip_1326,
                &board_u8,
                pot,
                effective_stack,
                traverser,
            );
            inputs.extend_from_slice(&input_vec);
        }

        // Single batched forward pass.
        let model = self.model.lock().unwrap();
        let data = TensorData::new(inputs, [batch_size, INPUT_SIZE]);
        let input_tensor = Tensor::<B, 2>::from_data(data, &self.device);
        let output = model.forward(input_tensor);
        drop(model);

        // Extract all outputs.
        let out_data = output.into_data();
        let out_vec: Vec<f32> = out_data.to_vec().expect("output tensor conversion");

        // Map each request's 1326 outputs back to solver combo ordering.
        let mut results = Vec::with_capacity(batch_size);
        for req_idx in 0..batch_size {
            let row_start = req_idx * OUTPUT_SIZE;
            let mut result = Vec::with_capacity(num_combos);
            for &idx in &combo_indices {
                result.push(f64::from(out_vec[row_start + idx]));
            }
            results.push(result);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use poker_solver_core::blueprint_v2::subgame_cfr::SubgameHands;
    use poker_solver_core::poker::{Suit, Value};

    type TestBackend = NdArray;

    fn test_flop_board() -> Vec<Card> {
        vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
        ]
    }

    fn test_turn_board() -> Vec<Card> {
        vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
        ]
    }

    fn test_river_board() -> Vec<Card> {
        vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
            Card::new(Value::Two, Suit::Heart),
        ]
    }

    #[test]
    fn test_build_input_vector_dimensions() {
        let oop_1326 = [0.0_f32; OUTPUT_SIZE];
        let ip_1326 = [0.0_f32; OUTPUT_SIZE];
        // Flop (3 cards)
        let input = build_boundary_input(&oop_1326, &ip_1326, &[0, 5, 10], 100.0, 200.0, 0);
        assert_eq!(input.len(), INPUT_SIZE, "input must be exactly 2720 elements");
        // Turn (4 cards)
        let input = build_boundary_input(&oop_1326, &ip_1326, &[0, 5, 10, 16], 100.0, 200.0, 0);
        assert_eq!(input.len(), INPUT_SIZE);
        // River (5 cards)
        let input =
            build_boundary_input(&oop_1326, &ip_1326, &[0, 5, 10, 16, 20], 100.0, 200.0, 0);
        assert_eq!(input.len(), INPUT_SIZE);
    }

    #[test]
    fn test_build_input_vector_encoding() {
        let mut oop_1326 = [0.0_f32; OUTPUT_SIZE];
        let mut ip_1326 = [0.0_f32; OUTPUT_SIZE];
        oop_1326[0] = 0.5;
        oop_1326[100] = 0.25;
        ip_1326[0] = 0.3;
        ip_1326[200] = 0.15;

        // Board: 2c(0), 3d(5), 4h(10)
        let board_u8 = [0u8, 5, 10];
        let input = build_boundary_input(&oop_1326, &ip_1326, &board_u8, 100.0, 200.0, 0);

        // OOP range positions
        assert!(
            (input[0] - 0.5).abs() < 1e-6,
            "OOP range[0] should be 0.5, got {}",
            input[0]
        );
        assert!(
            (input[100] - 0.25).abs() < 1e-6,
            "OOP range[100] should be 0.25"
        );

        // IP range positions (offset 1326)
        assert!(
            (input[1326] - 0.3).abs() < 1e-6,
            "IP range[0] should be 0.3"
        );
        assert!(
            (input[1326 + 200] - 0.15).abs() < 1e-6,
            "IP range[200] should be 0.15"
        );

        // Board one-hot (offset 2652)
        assert!((input[2652 + 0] - 1.0).abs() < 1e-6, "card 0 should be 1.0");
        assert!((input[2652 + 5] - 1.0).abs() < 1e-6, "card 5 should be 1.0");
        assert!(
            (input[2652 + 10] - 1.0).abs() < 1e-6,
            "card 10 should be 1.0"
        );
        assert!(
            input[2652 + 1].abs() < 1e-6,
            "non-board card should be 0.0"
        );
        assert!(
            input[2652 + 51].abs() < 1e-6,
            "non-board card should be 0.0"
        );

        // Rank presence (offset 2704)
        // card 0: rank 0/4=0, card 5: rank 5/4=1, card 10: rank 10/4=2
        assert!(
            (input[2704 + 0] - 1.0).abs() < 1e-6,
            "rank 0 should be present"
        );
        assert!(
            (input[2704 + 1] - 1.0).abs() < 1e-6,
            "rank 1 should be present"
        );
        assert!(
            (input[2704 + 2] - 1.0).abs() < 1e-6,
            "rank 2 should be present"
        );
        assert!(
            input[2704 + 3].abs() < 1e-6,
            "rank 3 should not be present"
        );
        assert!(
            input[2704 + 12].abs() < 1e-6,
            "rank 12 should not be present"
        );

        // Pot at index 2717: 100 / 400 = 0.25
        assert!(
            (input[2717] - 0.25).abs() < 1e-6,
            "pot should be 0.25, got {}",
            input[2717]
        );
        // Stack at index 2718: 200 / 400 = 0.5
        assert!(
            (input[2718] - 0.5).abs() < 1e-6,
            "stack should be 0.5, got {}",
            input[2718]
        );
        // Player indicator at index 2719: OOP = 0.0
        assert!(
            input[2719].abs() < 1e-6,
            "player OOP should be 0.0, got {}",
            input[2719]
        );
    }

    #[test]
    fn test_build_input_player_indicator() {
        let oop_1326 = [0.0_f32; OUTPUT_SIZE];
        let ip_1326 = [0.0_f32; OUTPUT_SIZE];

        let input_oop = build_boundary_input(&oop_1326, &ip_1326, &[0, 5, 10], 100.0, 200.0, 0);
        let input_ip = build_boundary_input(&oop_1326, &ip_1326, &[0, 5, 10], 100.0, 200.0, 1);

        assert!(input_oop[2719].abs() < 1e-6, "OOP should be 0.0");
        assert!((input_ip[2719] - 1.0).abs() < 1e-6, "IP should be 1.0");
    }

    #[test]
    fn test_evaluate_output_length() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RebelLeafEvaluator::new(model, device);

        let board = test_turn_board();
        let hands = SubgameHands::enumerate(&board);
        let n = hands.combos.len().min(20);
        let combos = &hands.combos[..n];
        let oop_range = vec![1.0 / n as f64; n];
        let ip_range = vec![1.0 / n as f64; n];

        let result = evaluator.evaluate(combos, &board, 100.0, 200.0, &oop_range, &ip_range, 0);
        assert_eq!(result.len(), n, "output length should match combo count");
    }

    #[test]
    fn test_evaluate_all_values_finite() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RebelLeafEvaluator::new(model, device);

        let board = test_turn_board();
        let hands = SubgameHands::enumerate(&board);
        let n = hands.combos.len().min(20);
        let combos = &hands.combos[..n];
        let oop_range = vec![1.0 / n as f64; n];
        let ip_range = vec![1.0 / n as f64; n];

        let result = evaluator.evaluate(combos, &board, 100.0, 200.0, &oop_range, &ip_range, 0);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "combo {i} has non-finite CFV: {v}");
        }
    }

    #[test]
    fn test_evaluate_flop_board() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RebelLeafEvaluator::new(model, device);

        let board = test_flop_board();
        let hands = SubgameHands::enumerate(&board);
        let n = hands.combos.len().min(20);
        let combos = &hands.combos[..n];
        let oop_range = vec![1.0 / n as f64; n];
        let ip_range = vec![1.0 / n as f64; n];

        let result = evaluator.evaluate(combos, &board, 100.0, 200.0, &oop_range, &ip_range, 0);
        assert_eq!(result.len(), n);
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_evaluate_river_board() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RebelLeafEvaluator::new(model, device);

        let board = test_river_board();
        let hands = SubgameHands::enumerate(&board);
        let n = hands.combos.len().min(20);
        let combos = &hands.combos[..n];
        let oop_range = vec![1.0 / n as f64; n];
        let ip_range = vec![1.0 / n as f64; n];

        let result = evaluator.evaluate(combos, &board, 100.0, 200.0, &oop_range, &ip_range, 0);
        assert_eq!(result.len(), n);
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_both_traverser_positions() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RebelLeafEvaluator::new(model, device);

        let board = test_turn_board();
        let hands = SubgameHands::enumerate(&board);
        let n = hands.combos.len().min(10);
        let combos = &hands.combos[..n];
        let oop_range = vec![1.0 / n as f64; n];
        let ip_range = vec![1.0 / n as f64; n];

        let oop_result =
            evaluator.evaluate(combos, &board, 100.0, 200.0, &oop_range, &ip_range, 0);
        let ip_result =
            evaluator.evaluate(combos, &board, 100.0, 200.0, &oop_range, &ip_range, 1);

        assert_eq!(oop_result.len(), n);
        assert_eq!(ip_result.len(), n);

        let diff: f64 = oop_result
            .iter()
            .zip(ip_result.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-10,
            "OOP and IP results should differ, diff={diff}"
        );
    }

    #[test]
    fn test_evaluate_boundaries_batch() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RebelLeafEvaluator::new(model, device);

        let board = test_turn_board();
        let hands = SubgameHands::enumerate(&board);
        let n = hands.combos.len().min(20);
        let combos = &hands.combos[..n];
        let oop_range = vec![1.0 / n as f64; n];
        let ip_range = vec![1.0 / n as f64; n];

        let requests = vec![(100.0, 200.0, 0u8), (150.0, 175.0, 1u8), (200.0, 100.0, 0u8)];

        let results =
            evaluator.evaluate_boundaries(combos, &board, &oop_range, &ip_range, &requests);

        assert_eq!(results.len(), 3, "should have one result per request");
        for (req_idx, result) in results.iter().enumerate() {
            assert_eq!(
                result.len(),
                n,
                "request {req_idx}: output length should match combo count"
            );
            for (i, &v) in result.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "request {req_idx}, combo {i}: non-finite CFV: {v}"
                );
            }
        }
    }

    #[test]
    fn test_evaluate_boundaries_matches_individual() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RebelLeafEvaluator::from_shared(
            Arc::new(Mutex::new(model)),
            device,
        );

        let board = test_turn_board();
        let hands = SubgameHands::enumerate(&board);
        let n = hands.combos.len().min(20);
        let combos = &hands.combos[..n];
        let oop_range = vec![1.0 / n as f64; n];
        let ip_range = vec![1.0 / n as f64; n];

        let requests = vec![(100.0, 200.0, 0u8), (150.0, 175.0, 1u8)];

        // Batch evaluation.
        let batch_results =
            evaluator.evaluate_boundaries(combos, &board, &oop_range, &ip_range, &requests);

        // Individual evaluations.
        for (req_idx, &(pot, eff_stack, traverser)) in requests.iter().enumerate() {
            let individual =
                evaluator.evaluate(combos, &board, pot, eff_stack, &oop_range, &ip_range, traverser);

            for (i, (batch_v, ind_v)) in batch_results[req_idx]
                .iter()
                .zip(individual.iter())
                .enumerate()
            {
                assert!(
                    (batch_v - ind_v).abs() < 1e-4,
                    "request {req_idx}, combo {i}: batch={batch_v} vs individual={ind_v}"
                );
            }
        }
    }

    #[test]
    fn test_evaluate_boundaries_empty_requests() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RebelLeafEvaluator::new(model, device);

        let board = test_turn_board();
        let hands = SubgameHands::enumerate(&board);
        let n = hands.combos.len().min(10);
        let combos = &hands.combos[..n];
        let oop_range = vec![1.0 / n as f64; n];
        let ip_range = vec![1.0 / n as f64; n];

        let results = evaluator.evaluate_boundaries(combos, &board, &oop_range, &ip_range, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_can_box_as_dyn_leaf_evaluator() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RebelLeafEvaluator::new(model, device);
        let _boxed: Box<dyn LeafEvaluator> = Box::new(evaluator);
    }

    #[test]
    #[ignore]
    fn test_load_model_and_evaluate() {
        // Integration test: requires a trained model at a known path.
        // Run with: cargo test -p rebel leaf_evaluator::tests::test_load_model_and_evaluate -- --ignored
        todo!("Load a trained CfvNet model and evaluate a boundary");
    }
}
