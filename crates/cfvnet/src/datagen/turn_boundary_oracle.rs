use crate::datagen::sampler::Situation;
use crate::datagen::solver::{SolveConfig, solve_situation};
use crate::datagen::storage::{NUM_COMBOS, TrainingRecord};
use crate::eval::boundary_evaluator::encode_boundary_inference_input;
use range_solver::card::index_to_card_pair;

/// A sampled turn boundary state that should be evaluated by averaging legal
/// river runouts.
#[derive(Debug, Clone)]
pub struct TurnBoundaryInput {
    pub board: [u8; 4],
    pub pot: f32,
    pub effective_stack: f32,
    pub player: u8,
    pub oop_range: [f32; NUM_COMBOS],
    pub ip_range: [f32; NUM_COMBOS],
}

/// A concrete river state requested by the turn-boundary oracle builder.
pub struct RiverRunoutInput<'a> {
    pub board: [u8; 5],
    pub pot: f32,
    pub effective_stack: f32,
    pub player: u8,
    pub oop_range: &'a [f32; NUM_COMBOS],
    pub ip_range: &'a [f32; NUM_COMBOS],
}

/// Evaluates one river runout and returns pot-relative CFVs for the requested
/// player perspective.
pub trait RiverRunoutOracle {
    fn evaluate(&self, input: RiverRunoutInput<'_>) -> Result<[f32; NUM_COMBOS], String>;
}

/// Minimal inference boundary for a river BoundaryNet. Production implements
/// this with ONNX/GPU evaluators; tests can use small deterministic fakes.
pub trait BoundaryNetInfer {
    fn infer_batch(&self, input: Vec<f32>, num_rows: usize) -> Result<Vec<f32>, String>;
}

#[cfg(feature = "gpu-turn-datagen")]
impl BoundaryNetInfer for crate::datagen::gpu_boundary_eval::GpuBoundaryEvaluator {
    fn infer_batch(&self, input: Vec<f32>, num_rows: usize) -> Result<Vec<f32>, String> {
        crate::datagen::gpu_boundary_eval::GpuBoundaryEvaluator::infer_batch(self, input, num_rows)
    }
}

impl<T: BoundaryNetInfer + ?Sized> BoundaryNetInfer for &T {
    fn infer_batch(&self, input: Vec<f32>, num_rows: usize) -> Result<Vec<f32>, String> {
        (*self).infer_batch(input, num_rows)
    }
}

/// River oracle backed by the existing river BoundaryNet contract.
///
/// BoundaryNet predicts normalized values (`chip_cfv / (pot + stack)`). The
/// turn-boundary dataset stores pot-relative CFVs, so this adapter converts the
/// model output back to `chip_cfv / pot` before returning it.
pub struct BoundaryNetRiverRunoutOracle<I> {
    inferer: I,
}

impl<I> BoundaryNetRiverRunoutOracle<I> {
    pub fn new(inferer: I) -> Self {
        Self { inferer }
    }

    pub fn into_inner(self) -> I {
        self.inferer
    }
}

impl<I: BoundaryNetInfer> RiverRunoutOracle for BoundaryNetRiverRunoutOracle<I> {
    fn evaluate(&self, input: RiverRunoutInput<'_>) -> Result<[f32; NUM_COMBOS], String> {
        validate_river_input(&input)?;

        let (oop_range, ip_range) =
            ranges_with_board_blockers_zeroed(input.oop_range, input.ip_range, &input.board);
        let row = encode_boundary_inference_input(
            &oop_range,
            &ip_range,
            &input.board,
            input.pot,
            input.effective_stack,
            input.player,
        );
        let normalized = self.inferer.infer_batch(row, 1)?;
        if normalized.len() != NUM_COMBOS {
            return Err(format!(
                "BoundaryNet returned {} values, expected {}",
                normalized.len(),
                NUM_COMBOS
            ));
        }

        let scale = (input.pot + input.effective_stack) / input.pot;
        let mut cfvs = [0.0_f32; NUM_COMBOS];
        for (out, value) in cfvs.iter_mut().zip(normalized.iter()) {
            *out = *value * scale;
        }
        Ok(cfvs)
    }
}

/// River oracle backed by exact range-solver river solves.
pub struct ExactRiverSolverOracle {
    config: SolveConfig,
}

impl ExactRiverSolverOracle {
    pub fn new(config: SolveConfig) -> Self {
        Self { config }
    }

    pub fn evaluate_both(
        &self,
        input: RiverRunoutInput<'_>,
    ) -> Result<([f32; NUM_COMBOS], [f32; NUM_COMBOS]), String> {
        let situation = river_input_to_situation(&input)?;
        let result = solve_situation(&situation, &self.config)?;
        Ok((result.oop_evs, result.ip_evs))
    }
}

impl RiverRunoutOracle for ExactRiverSolverOracle {
    fn evaluate(&self, input: RiverRunoutInput<'_>) -> Result<[f32; NUM_COMBOS], String> {
        let situation = river_input_to_situation(&input)?;
        let result = solve_situation(&situation, &self.config)?;
        match input.player {
            0 => Ok(result.oop_evs),
            1 => Ok(result.ip_evs),
            player => Err(format!("invalid player {}, expected 0 or 1", player)),
        }
    }
}

/// Build one turn-boundary training record by averaging oracle values over all
/// legal river cards.
pub fn build_turn_boundary_record<O: RiverRunoutOracle>(
    input: &TurnBoundaryInput,
    oracle: &O,
) -> Result<TrainingRecord, String> {
    validate_input(input)?;

    let mut sums = [0.0_f32; NUM_COMBOS];
    let mut counts = [0_u8; NUM_COMBOS];

    for river in remaining_river_cards(&input.board) {
        let mut board = [0_u8; 5];
        board[..4].copy_from_slice(&input.board);
        board[4] = river;

        let values = oracle.evaluate(RiverRunoutInput {
            board,
            pot: input.pot,
            effective_stack: input.effective_stack,
            player: input.player,
            oop_range: &input.oop_range,
            ip_range: &input.ip_range,
        })?;

        for idx in 0..NUM_COMBOS {
            if !combo_conflicts_with_card(idx, river)
                && !combo_conflicts_with_board(idx, &input.board)
            {
                sums[idx] += values[idx];
                counts[idx] += 1;
            }
        }
    }

    let mut cfvs = [0.0_f32; NUM_COMBOS];
    let mut valid_mask = [0_u8; NUM_COMBOS];
    for idx in 0..NUM_COMBOS {
        if counts[idx] > 0 {
            cfvs[idx] = sums[idx] / f32::from(counts[idx]);
            valid_mask[idx] = 1;
        }
    }

    let player_range = if input.player == 0 {
        &input.oop_range
    } else {
        &input.ip_range
    };
    let game_value = player_range
        .iter()
        .zip(cfvs.iter())
        .map(|(&reach, &cfv)| reach * cfv)
        .sum();

    Ok(TrainingRecord {
        board: input.board.to_vec(),
        pot: input.pot,
        effective_stack: input.effective_stack,
        player: input.player,
        game_value,
        oop_range: input.oop_range,
        ip_range: input.ip_range,
        cfvs,
        valid_mask,
    })
}

/// Build both player records for exact-river targets while sharing each river
/// solve. A river solve returns both players' EV arrays, so this avoids doing
/// the same 48 river games twice.
pub fn build_exact_turn_boundary_records(
    input: &TurnBoundaryInput,
    oracle: &ExactRiverSolverOracle,
) -> Result<[TrainingRecord; 2], String> {
    validate_input(input)?;

    let mut oop_sums = [0.0_f32; NUM_COMBOS];
    let mut ip_sums = [0.0_f32; NUM_COMBOS];
    let mut counts = [0_u8; NUM_COMBOS];

    for river in remaining_river_cards(&input.board) {
        let mut board = [0_u8; 5];
        board[..4].copy_from_slice(&input.board);
        board[4] = river;

        let (oop_values, ip_values) = oracle.evaluate_both(RiverRunoutInput {
            board,
            pot: input.pot,
            effective_stack: input.effective_stack,
            player: 0,
            oop_range: &input.oop_range,
            ip_range: &input.ip_range,
        })?;

        for idx in 0..NUM_COMBOS {
            if !combo_conflicts_with_card(idx, river)
                && !combo_conflicts_with_board(idx, &input.board)
            {
                oop_sums[idx] += oop_values[idx];
                ip_sums[idx] += ip_values[idx];
                counts[idx] += 1;
            }
        }
    }

    Ok([
        record_from_sums(input, 0, &oop_sums, &counts),
        record_from_sums(input, 1, &ip_sums, &counts),
    ])
}

fn record_from_sums(
    input: &TurnBoundaryInput,
    player: u8,
    sums: &[f32; NUM_COMBOS],
    counts: &[u8; NUM_COMBOS],
) -> TrainingRecord {
    let mut cfvs = [0.0_f32; NUM_COMBOS];
    let mut valid_mask = [0_u8; NUM_COMBOS];
    for idx in 0..NUM_COMBOS {
        if counts[idx] > 0 {
            cfvs[idx] = sums[idx] / f32::from(counts[idx]);
            valid_mask[idx] = 1;
        }
    }

    let player_range = if player == 0 {
        &input.oop_range
    } else {
        &input.ip_range
    };
    let game_value = player_range
        .iter()
        .zip(cfvs.iter())
        .map(|(&reach, &cfv)| reach * cfv)
        .sum();

    TrainingRecord {
        board: input.board.to_vec(),
        pot: input.pot,
        effective_stack: input.effective_stack,
        player,
        game_value,
        oop_range: input.oop_range,
        ip_range: input.ip_range,
        cfvs,
        valid_mask,
    }
}

pub fn remaining_river_cards(board: &[u8; 4]) -> Vec<u8> {
    (0_u8..52).filter(|card| !board.contains(card)).collect()
}

fn validate_input(input: &TurnBoundaryInput) -> Result<(), String> {
    if input.player > 1 {
        return Err(format!("invalid player {}, expected 0 or 1", input.player));
    }
    if input.pot <= 0.0 {
        return Err(format!("pot must be positive, got {}", input.pot));
    }
    if input.effective_stack < 0.0 {
        return Err(format!(
            "effective_stack must be non-negative, got {}",
            input.effective_stack
        ));
    }

    let mut seen = [false; 52];
    for &card in &input.board {
        if card >= 52 {
            return Err(format!("invalid board card {card}"));
        }
        if seen[card as usize] {
            return Err(format!("duplicate board card {card}"));
        }
        seen[card as usize] = true;
    }

    Ok(())
}

fn validate_river_input(input: &RiverRunoutInput<'_>) -> Result<(), String> {
    if input.player > 1 {
        return Err(format!("invalid player {}, expected 0 or 1", input.player));
    }
    if input.pot <= 0.0 {
        return Err(format!("pot must be positive, got {}", input.pot));
    }
    if input.effective_stack < 0.0 {
        return Err(format!(
            "effective_stack must be non-negative, got {}",
            input.effective_stack
        ));
    }

    let mut seen = [false; 52];
    for &card in &input.board {
        if card >= 52 {
            return Err(format!("invalid board card {card}"));
        }
        if seen[card as usize] {
            return Err(format!("duplicate board card {card}"));
        }
        seen[card as usize] = true;
    }

    Ok(())
}

fn river_input_to_situation(input: &RiverRunoutInput<'_>) -> Result<Situation, String> {
    validate_river_input(input)?;
    if input.effective_stack <= 0.0 {
        return Err(format!(
            "effective_stack must be positive for exact river solve, got {}",
            input.effective_stack
        ));
    }

    let pot = integer_chip_value(input.pot, "pot")?;
    let effective_stack = integer_chip_value(input.effective_stack, "effective_stack")?;
    let (oop_range, ip_range) =
        ranges_with_board_blockers_zeroed(input.oop_range, input.ip_range, &input.board);

    Ok(Situation {
        board: input.board,
        board_size: 5,
        pot,
        effective_stack,
        ranges: [oop_range, ip_range],
    })
}

fn integer_chip_value(value: f32, field: &str) -> Result<i32, String> {
    if !value.is_finite() {
        return Err(format!("{field} must be finite, got {value}"));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-3 {
        return Err(format!(
            "{field} must be an integer chip value for exact river solve, got {value}"
        ));
    }
    if rounded < i32::MIN as f32 || rounded > i32::MAX as f32 {
        return Err(format!("{field} is outside i32 range, got {value}"));
    }
    Ok(rounded as i32)
}

fn ranges_with_board_blockers_zeroed(
    oop_range: &[f32; NUM_COMBOS],
    ip_range: &[f32; NUM_COMBOS],
    board: &[u8; 5],
) -> ([f32; NUM_COMBOS], [f32; NUM_COMBOS]) {
    let mut oop = *oop_range;
    let mut ip = *ip_range;
    for idx in 0..NUM_COMBOS {
        let (c0, c1) = index_to_card_pair(idx);
        if board.contains(&c0) || board.contains(&c1) {
            oop[idx] = 0.0;
            ip[idx] = 0.0;
        }
    }
    (oop, ip)
}

fn combo_conflicts_with_board(idx: usize, board: &[u8; 4]) -> bool {
    let (c0, c1) = index_to_card_pair(idx);
    board.contains(&c0) || board.contains(&c1)
}

fn combo_conflicts_with_card(idx: usize, card: u8) -> bool {
    let (c0, c1) = index_to_card_pair(idx);
    c0 == card || c1 == card
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::network::{DECK_SIZE, INPUT_SIZE};
    use range_solver::card::card_pair_to_index;
    use std::cell::RefCell;

    struct ConstantOracle {
        value: f32,
        boards: RefCell<Vec<[u8; 5]>>,
    }

    impl ConstantOracle {
        fn new(value: f32) -> Self {
            Self {
                value,
                boards: RefCell::new(Vec::new()),
            }
        }
    }

    impl RiverRunoutOracle for ConstantOracle {
        fn evaluate(&self, input: RiverRunoutInput<'_>) -> Result<[f32; NUM_COMBOS], String> {
            self.boards.borrow_mut().push(input.board);
            Ok([self.value; NUM_COMBOS])
        }
    }

    struct CapturingInferer {
        output: RefCell<Vec<f32>>,
        calls: RefCell<Vec<(Vec<f32>, usize)>>,
    }

    impl CapturingInferer {
        fn new(output: Vec<f32>) -> Self {
            Self {
                output: RefCell::new(output),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl BoundaryNetInfer for CapturingInferer {
        fn infer_batch(&self, input: Vec<f32>, num_rows: usize) -> Result<Vec<f32>, String> {
            self.calls.borrow_mut().push((input, num_rows));
            Ok(self.output.borrow().clone())
        }
    }

    fn sample_input(player: u8) -> TurnBoundaryInput {
        let mut oop_range = [0.0_f32; NUM_COMBOS];
        let mut ip_range = [0.0_f32; NUM_COMBOS];
        oop_range[card_pair_to_index(8, 9)] = 1.0;
        ip_range[card_pair_to_index(10, 11)] = 1.0;
        TurnBoundaryInput {
            board: [0, 1, 2, 3],
            pot: 100.0,
            effective_stack: 200.0,
            player,
            oop_range,
            ip_range,
        }
    }

    #[test]
    fn remaining_river_cards_exclude_turn_board() {
        let cards = remaining_river_cards(&[0, 1, 2, 3]);
        assert_eq!(cards.len(), 48);
        assert!(!cards.contains(&0));
        assert!(!cards.contains(&1));
        assert!(cards.contains(&4));
        assert!(cards.contains(&51));
    }

    #[test]
    fn build_record_averages_legal_river_runouts() {
        let oracle = ConstantOracle::new(0.25);
        let input = sample_input(0);
        let rec = build_turn_boundary_record(&input, &oracle).unwrap();

        let live_combo = card_pair_to_index(8, 9);
        let board_blocked_combo = card_pair_to_index(0, 4);

        assert_eq!(oracle.boards.borrow().len(), 48);
        assert_eq!(rec.board, vec![0, 1, 2, 3]);
        assert_eq!(rec.player, 0);
        assert_eq!(rec.valid_mask[live_combo], 1);
        assert_eq!(rec.cfvs[live_combo], 0.25);
        assert_eq!(rec.valid_mask[board_blocked_combo], 0);
        assert_eq!(rec.cfvs[board_blocked_combo], 0.0);
        assert_eq!(rec.game_value, 0.25);
    }

    #[test]
    fn build_record_uses_requested_player_range_for_game_value() {
        let oracle = ConstantOracle::new(-0.5);
        let input = sample_input(1);
        let rec = build_turn_boundary_record(&input, &oracle).unwrap();

        assert_eq!(rec.player, 1);
        assert_eq!(rec.game_value, -0.5);
    }

    #[test]
    fn invalid_turn_board_is_rejected() {
        let oracle = ConstantOracle::new(0.0);
        let mut input = sample_input(0);
        input.board = [0, 1, 2, 2];

        let err = build_turn_boundary_record(&input, &oracle).unwrap_err();
        assert!(err.contains("duplicate board card"));
    }

    #[test]
    fn boundary_net_oracle_zeroes_board_blockers_and_returns_pot_relative_cfvs() {
        let inferer = CapturingInferer::new(vec![0.125; NUM_COMBOS]);
        let oracle = BoundaryNetRiverRunoutOracle::new(&inferer);
        let input = sample_input(1);

        let values = oracle
            .evaluate(RiverRunoutInput {
                board: [0, 1, 2, 3, 4],
                pot: 100.0,
                effective_stack: 300.0,
                player: input.player,
                oop_range: &input.oop_range,
                ip_range: &input.ip_range,
            })
            .unwrap();

        assert!((values[card_pair_to_index(8, 9)] - 0.5).abs() < 1e-6);

        let calls = inferer.calls.borrow();
        assert_eq!(calls.len(), 1);
        let (encoded, rows) = &calls[0];
        assert_eq!(*rows, 1);
        assert_eq!(encoded.len(), INPUT_SIZE);

        let blocked_combo = card_pair_to_index(0, 8);
        let live_combo = card_pair_to_index(8, 9);
        assert_eq!(encoded[blocked_combo], 0.0);
        assert_eq!(encoded[live_combo], 1.0);
        assert_eq!(encoded[NUM_COMBOS + blocked_combo], 0.0);
        assert_eq!(encoded[NUM_COMBOS + live_combo], 0.0);

        let board_offset = NUM_COMBOS * 2;
        assert_eq!(encoded[board_offset + 4], 1.0);
        let board_ones: f32 = encoded[board_offset..board_offset + DECK_SIZE].iter().sum();
        assert_eq!(board_ones, 5.0);
    }

    #[test]
    fn boundary_net_oracle_rejects_wrong_output_length() {
        let inferer = CapturingInferer::new(vec![0.0; NUM_COMBOS - 1]);
        let oracle = BoundaryNetRiverRunoutOracle::new(inferer);
        let input = sample_input(0);

        let err = oracle
            .evaluate(RiverRunoutInput {
                board: [0, 1, 2, 3, 4],
                pot: 100.0,
                effective_stack: 300.0,
                player: input.player,
                oop_range: &input.oop_range,
                ip_range: &input.ip_range,
            })
            .unwrap_err();

        assert!(err.contains("BoundaryNet returned 1325 values"));
    }

    #[test]
    fn exact_river_situation_zeroes_board_blockers() {
        let input = sample_input(0);
        let situation = river_input_to_situation(&RiverRunoutInput {
            board: [0, 1, 2, 3, 4],
            pot: 100.0,
            effective_stack: 300.0,
            player: input.player,
            oop_range: &input.oop_range,
            ip_range: &input.ip_range,
        })
        .unwrap();

        assert_eq!(situation.board_size, 5);
        assert_eq!(situation.pot, 100);
        assert_eq!(situation.effective_stack, 300);
        assert_eq!(situation.ranges[0][card_pair_to_index(0, 8)], 0.0);
        assert_eq!(situation.ranges[0][card_pair_to_index(8, 9)], 1.0);
    }

    #[test]
    fn exact_river_situation_requires_integer_chip_values() {
        let input = sample_input(0);
        let err = river_input_to_situation(&RiverRunoutInput {
            board: [0, 1, 2, 3, 4],
            pot: 100.25,
            effective_stack: 300.0,
            player: input.player,
            oop_range: &input.oop_range,
            ip_range: &input.ip_range,
        })
        .unwrap_err();

        assert!(err.contains("pot must be an integer chip value"));
    }
}
