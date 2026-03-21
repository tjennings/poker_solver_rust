//! Turn depth-boundary evaluator that averages the river CFV network
//! over all possible river cards.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use poker_solver_core::blueprint_v2::cfv_subgame_solver::LeafEvaluator;
use poker_solver_core::poker::{Card, Suit};
use range_solver::card::card_pair_to_index;

use crate::model::network::{CfvNet, DECK_SIZE, INPUT_SIZE, NUM_RANKS, OUTPUT_SIZE};

/// Convert an `rs_poker::core::Card` to a range-solver `u8` card.
///
/// Range-solver encoding: `card = 4 * rank + suit`
/// where rank Two=0..Ace=12, suit Club=0, Diamond=1, Heart=2, Spade=3.
pub fn rs_card_to_u8(card: Card) -> u8 {
    let rank = card.value as u8;
    let suit = match card.suit {
        Suit::Club => 0,
        Suit::Diamond => 1,
        Suit::Heart => 2,
        Suit::Spade => 3,
    };
    4 * rank + suit
}

/// Convert a range-solver `u8` card to an `rs_poker::core::Card`.
#[cfg(test)]
fn u8_to_rs_card(id: u8) -> Card {
    use poker_solver_core::poker::Value;
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

/// Evaluates turn depth boundaries by averaging river CFV network
/// predictions over all 48 possible river cards.
///
/// For each river card that doesn't conflict with the board or the
/// traverser's hand, the evaluator:
/// 1. Builds a 5-card river board
/// 2. Maps solver combo ranges to 1326-indexed vectors
/// 3. Runs a forward pass through the network
/// 4. Maps the 1326 outputs back to solver combo order
///
/// The final CFV for each combo is the average over all non-conflicting
/// river cards.
pub struct RiverNetEvaluator<B: Backend> {
    model: CfvNet<B>,
    device: B::Device,
}

impl<B: Backend> RiverNetEvaluator<B> {
    pub fn new(model: CfvNet<B>, device: B::Device) -> Self {
        Self { model, device }
    }
}

/// GPU-accelerated evaluator that shares a single model behind a mutex.
///
/// Batches all valid river card inputs into a single `[N, INPUT_SIZE]` tensor
/// for one forward pass per `evaluate()` call. Tracks cumulative mutex wait time.
pub struct SharedRiverNetEvaluator<B: Backend> {
    model: Arc<Mutex<CfvNet<B>>>,
    device: B::Device,
    /// Cumulative nanoseconds spent waiting for the mutex across all threads.
    pub wait_nanos: Arc<AtomicU64>,
    /// Number of mutex acquisitions across all threads.
    pub lock_count: Arc<AtomicU64>,
}

impl<B: Backend> SharedRiverNetEvaluator<B>
where
    B::Device: Clone,
{
    pub fn new(
        model: Arc<Mutex<CfvNet<B>>>,
        device: B::Device,
        wait_nanos: Arc<AtomicU64>,
        lock_count: Arc<AtomicU64>,
    ) -> Self {
        Self {
            model,
            device,
            wait_nanos,
            lock_count,
        }
    }
}

/// Build the input vector for a single river board evaluation.
///
/// Layout (2720 floats):
///   [0..1326)      — OOP range (1326 combo probabilities)
///   [1326..2652)   — IP range (1326 combo probabilities)
///   [2652..2704)   — board one-hot (52 elements)
///   [2704..2717)   — rank presence (13 elements)
///   [2717]         — pot / 400.0
///   [2718]         — effective_stack / 400.0
///   [2719]         — player indicator (0.0=OOP, 1.0=IP)
pub fn build_input(
    oop_1326: &[f32; OUTPUT_SIZE],
    ip_1326: &[f32; OUTPUT_SIZE],
    board_u8: &[u8; 5],
    pot: f64,
    effective_stack: f64,
    traverser: u8,
) -> Vec<f32> {
    let mut input = Vec::with_capacity(INPUT_SIZE);
    input.extend_from_slice(oop_1326);
    input.extend_from_slice(ip_1326);
    let mut board_onehot = [0.0_f32; DECK_SIZE];
    for &card in board_u8 {
        debug_assert!((card as usize) < DECK_SIZE, "card id {card} out of range");
        board_onehot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_onehot);
    let mut rank_presence = [0.0_f32; NUM_RANKS];
    for &card in board_u8 {
        debug_assert!((card as usize) < DECK_SIZE, "card id {card} out of range");
        rank_presence[(card / 4) as usize] = 1.0;
    }
    input.extend_from_slice(&rank_presence);
    input.push(pot as f32 / 400.0);
    input.push(effective_stack as f32 / 400.0);
    input.push(if traverser == 0 { 0.0 } else { 1.0 });
    debug_assert_eq!(input.len(), INPUT_SIZE);
    input
}

impl<B: Backend> LeafEvaluator for RiverNetEvaluator<B>
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
        assert_eq!(board.len(), 4, "RiverNetEvaluator requires a 4-card turn board");
        assert_eq!(combos.len(), oop_range.len());
        assert_eq!(combos.len(), ip_range.len());

        let num_combos = combos.len();

        // Pre-convert combos to u8 pairs and their 1326 indices.
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

        // Accumulate CFVs and counts per combo.
        let mut cfv_sum = vec![0.0_f64; num_combos];
        let mut cfv_count = vec![0_u32; num_combos];

        // Iterate over all 52 possible river cards.
        for river_u8 in 0u8..52 {
            // Skip cards already on the board.
            if board_u8.contains(&river_u8) {
                continue;
            }

            // Build the 5-card river board.
            let river_board_u8: [u8; 5] = [
                board_u8[0], board_u8[1], board_u8[2], board_u8[3], river_u8,
            ];

            // Map solver ranges to 1326-indexed arrays.
            // Only include combos that don't conflict with the river card.
            let mut oop_1326 = [0.0_f32; OUTPUT_SIZE];
            let mut ip_1326 = [0.0_f32; OUTPUT_SIZE];
            let mut valid_combo_mask = vec![false; num_combos];

            for (i, &idx) in combo_indices.iter().enumerate() {
                if combos_u8[i][0] == river_u8 || combos_u8[i][1] == river_u8 {
                    continue;
                }
                valid_combo_mask[i] = true;
                oop_1326[idx] = oop_range[i] as f32;
                ip_1326[idx] = ip_range[i] as f32;
            }

            // Build input and run forward pass.
            let input_vec = build_input(
                &oop_1326,
                &ip_1326,
                &river_board_u8,
                pot,
                effective_stack,
                traverser,
            );

            let data = TensorData::new(input_vec, [1, INPUT_SIZE]);
            let input_tensor = Tensor::<B, 2>::from_data(data, &self.device);
            let output = self.model.forward(input_tensor);

            // Extract output values.
            let out_data = output.into_data();
            let out_vec: Vec<f32> = out_data.to_vec().expect("output tensor conversion");

            // Map 1326-indexed outputs back to solver combo order.
            for (i, &idx) in combo_indices.iter().enumerate() {
                if valid_combo_mask[i] {
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
                    sum / f64::from(count)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

impl<B: Backend> LeafEvaluator for SharedRiverNetEvaluator<B>
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
        assert_eq!(
            board.len(),
            4,
            "SharedRiverNetEvaluator requires a 4-card turn board"
        );
        assert_eq!(combos.len(), oop_range.len());
        assert_eq!(combos.len(), ip_range.len());

        let num_combos = combos.len();

        let combos_u8: Vec<[u8; 2]> = combos
            .iter()
            .map(|c| [rs_card_to_u8(c[0]), rs_card_to_u8(c[1])])
            .collect();
        let combo_indices: Vec<usize> = combos_u8
            .iter()
            .map(|c| card_pair_to_index(c[0], c[1]))
            .collect();

        let board_u8: Vec<u8> = board.iter().map(|c| rs_card_to_u8(*c)).collect();

        // Build all river card inputs CPU-side.
        let mut inputs: Vec<f32> = Vec::new();
        let mut valid_combo_masks: Vec<Vec<bool>> = Vec::new();

        for river_u8 in 0u8..52 {
            if board_u8.contains(&river_u8) {
                continue;
            }

            let river_board_u8: [u8; 5] = [
                board_u8[0], board_u8[1], board_u8[2], board_u8[3], river_u8,
            ];

            let mut oop_1326 = [0.0_f32; OUTPUT_SIZE];
            let mut ip_1326 = [0.0_f32; OUTPUT_SIZE];
            let mut valid_combo_mask = vec![false; num_combos];

            for (i, &idx) in combo_indices.iter().enumerate() {
                if combos_u8[i][0] == river_u8 || combos_u8[i][1] == river_u8 {
                    continue;
                }
                valid_combo_mask[i] = true;
                oop_1326[idx] = oop_range[i] as f32;
                ip_1326[idx] = ip_range[i] as f32;
            }

            let input_vec = build_input(
                &oop_1326,
                &ip_1326,
                &river_board_u8,
                pot,
                effective_stack,
                traverser,
            );
            inputs.extend_from_slice(&input_vec);
            valid_combo_masks.push(valid_combo_mask);
        }

        let batch_size = valid_combo_masks.len();

        // Acquire mutex, run batched forward pass.
        let t0 = Instant::now();
        let model = self.model.lock().unwrap();
        let wait = t0.elapsed();
        self.wait_nanos
            .fetch_add(wait.as_nanos() as u64, Ordering::Relaxed);
        self.lock_count.fetch_add(1, Ordering::Relaxed);

        let data = TensorData::new(inputs, [batch_size, INPUT_SIZE]);
        let input_tensor = Tensor::<B, 2>::from_data(data, &self.device);
        let output = model.forward(input_tensor);

        drop(model); // release mutex before post-processing

        let out_data = output.into_data();
        let out_vec: Vec<f32> = out_data.to_vec().expect("output tensor conversion");

        // Post-process: average over river cards per combo.
        let mut cfv_sum = vec![0.0_f64; num_combos];
        let mut cfv_count = vec![0_u32; num_combos];

        for (river_idx, mask) in valid_combo_masks.iter().enumerate() {
            let row_start = river_idx * OUTPUT_SIZE;
            for (i, &idx) in combo_indices.iter().enumerate() {
                if mask[i] {
                    cfv_sum[i] += f64::from(out_vec[row_start + idx]);
                    cfv_count[i] += 1;
                }
            }
        }

        cfv_sum
            .iter()
            .zip(cfv_count.iter())
            .map(|(&sum, &count)| {
                if count > 0 {
                    sum / f64::from(count)
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn evaluate_boundaries(
        &self,
        combos: &[[Card; 2]],
        board: &[Card],
        oop_range: &[f64],
        ip_range: &[f64],
        requests: &[(f64, f64, u8)],
    ) -> Vec<Vec<f64>> {
        assert_eq!(board.len(), 4);
        assert_eq!(combos.len(), oop_range.len());
        assert_eq!(combos.len(), ip_range.len());
        if requests.is_empty() {
            return Vec::new();
        }

        let num_combos = combos.len();
        let combos_u8: Vec<[u8; 2]> = combos
            .iter()
            .map(|c| [rs_card_to_u8(c[0]), rs_card_to_u8(c[1])])
            .collect();
        let combo_indices: Vec<usize> = combos_u8
            .iter()
            .map(|c| card_pair_to_index(c[0], c[1]))
            .collect();
        let board_u8: Vec<u8> = board.iter().map(|c| rs_card_to_u8(*c)).collect();

        // Valid river cards (shared across all requests).
        let valid_rivers: Vec<u8> = (0u8..52)
            .filter(|r| !board_u8.contains(r))
            .collect();
        let rivers_per_request = valid_rivers.len();

        // Per-river combo validity masks (shared across requests).
        let river_valid_masks: Vec<Vec<bool>> = valid_rivers
            .iter()
            .map(|&river_u8| {
                combos_u8
                    .iter()
                    .map(|c| c[0] != river_u8 && c[1] != river_u8)
                    .collect()
            })
            .collect();

        // Build ALL inputs: requests × rivers → one big batch.
        let total_rows = requests.len() * rivers_per_request;
        let mut inputs: Vec<f32> = Vec::with_capacity(total_rows * INPUT_SIZE);

        for &(pot, effective_stack, traverser) in requests {
            for (ri, &river_u8) in valid_rivers.iter().enumerate() {
                let river_board_u8: [u8; 5] = [
                    board_u8[0], board_u8[1], board_u8[2], board_u8[3], river_u8,
                ];

                let mut oop_1326 = [0.0_f32; OUTPUT_SIZE];
                let mut ip_1326 = [0.0_f32; OUTPUT_SIZE];

                for (i, &idx) in combo_indices.iter().enumerate() {
                    if river_valid_masks[ri][i] {
                        oop_1326[idx] = oop_range[i] as f32;
                        ip_1326[idx] = ip_range[i] as f32;
                    }
                }

                let input_vec = build_input(
                    &oop_1326,
                    &ip_1326,
                    &river_board_u8,
                    pot,
                    effective_stack,
                    traverser,
                );
                inputs.extend_from_slice(&input_vec);
            }
        }

        // ONE mutex lock, ONE forward pass for all requests.
        let t0 = Instant::now();
        let model = self.model.lock().unwrap();
        let wait = t0.elapsed();
        self.wait_nanos
            .fetch_add(wait.as_nanos() as u64, Ordering::Relaxed);
        self.lock_count.fetch_add(1, Ordering::Relaxed);

        let data = TensorData::new(inputs, [total_rows, INPUT_SIZE]);
        let input_tensor = Tensor::<B, 2>::from_data(data, &self.device);
        let output = model.forward(input_tensor);

        drop(model);

        let out_data = output.into_data();
        let out_vec: Vec<f32> = out_data.to_vec().expect("output tensor conversion");

        // Scatter results: for each request, average over rivers.
        let mut results = Vec::with_capacity(requests.len());
        for req_idx in 0..requests.len() {
            let mut cfv_sum = vec![0.0_f64; num_combos];
            let mut cfv_count = vec![0_u32; num_combos];

            for (ri, mask) in river_valid_masks.iter().enumerate() {
                let row = req_idx * rivers_per_request + ri;
                let row_start = row * OUTPUT_SIZE;
                for (i, &idx) in combo_indices.iter().enumerate() {
                    if mask[i] {
                        cfv_sum[i] += f64::from(out_vec[row_start + idx]);
                        cfv_count[i] += 1;
                    }
                }
            }

            results.push(
                cfv_sum
                    .iter()
                    .zip(cfv_count.iter())
                    .map(|(&sum, &count)| {
                        if count > 0 {
                            sum / f64::from(count)
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            );
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use poker_solver_core::blueprint_v2::subgame_cfr::SubgameHands;
    use poker_solver_core::poker::Value;
    use std::sync::atomic::AtomicU64;
    use std::sync::{Arc, Mutex};

    type TestBackend = NdArray;

    fn test_board() -> Vec<Card> {
        vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
        ]
    }

    #[test]
    fn output_length_matches_combos() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RiverNetEvaluator::new(model, device);

        let board = test_board();
        let hands = SubgameHands::enumerate(&board);
        let n = hands.combos.len().min(20);
        let combos = &hands.combos[..n];

        let oop_range = vec![1.0 / n as f64; n];
        let ip_range = vec![1.0 / n as f64; n];

        let result = evaluator.evaluate(combos, &board, 100.0, 200.0, &oop_range, &ip_range, 0);

        assert_eq!(result.len(), n, "output length should match combo count");
    }

    #[test]
    fn all_values_finite() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RiverNetEvaluator::new(model, device);

        let board = test_board();
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
    fn can_box_as_dyn_leaf_evaluator() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RiverNetEvaluator::new(model, device);

        let _boxed: Box<dyn LeafEvaluator> = Box::new(evaluator);
    }

    #[test]
    fn both_traverser_positions() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RiverNetEvaluator::new(model, device);

        let board = test_board();
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

        // With a random model, OOP and IP results should differ due to player indicator.
        let diff: f64 = oop_result
            .iter()
            .zip(ip_result.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-10, "OOP and IP results should differ, diff={diff}");
    }

    #[test]
    fn rs_card_roundtrip() {
        for rank in 0u8..13 {
            for suit in 0u8..4 {
                let id = 4 * rank + suit;
                let card = u8_to_rs_card(id);
                let back = rs_card_to_u8(card);
                assert_eq!(id, back, "roundtrip failed for card {id}");
            }
        }
    }

    #[test]
    fn shared_evaluator_matches_sequential() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let wait_nanos = Arc::new(AtomicU64::new(0));
        let lock_count = Arc::new(AtomicU64::new(0));

        let shared = SharedRiverNetEvaluator::new(
            Arc::new(Mutex::new(model.clone())),
            device,
            wait_nanos.clone(),
            lock_count,
        );
        let sequential = RiverNetEvaluator::new(model, Default::default());

        let board = test_board();
        let hands = SubgameHands::enumerate(&board);
        let n = hands.combos.len().min(20);
        let combos = &hands.combos[..n];
        let oop_range = vec![1.0 / n as f64; n];
        let ip_range = vec![1.0 / n as f64; n];

        let shared_result =
            shared.evaluate(combos, &board, 100.0, 200.0, &oop_range, &ip_range, 0);
        let seq_result =
            sequential.evaluate(combos, &board, 100.0, 200.0, &oop_range, &ip_range, 0);

        assert_eq!(shared_result.len(), seq_result.len());
        for (i, (s, q)) in shared_result.iter().zip(seq_result.iter()).enumerate() {
            assert!(
                (s - q).abs() < 1e-4,
                "combo {i}: shared={s} vs sequential={q}, diff={}",
                (s - q).abs()
            );
        }
    }
}
