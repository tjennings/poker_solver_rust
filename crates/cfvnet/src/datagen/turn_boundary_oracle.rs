use crate::datagen::storage::{TrainingRecord, NUM_COMBOS};
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
}
