use crate::datagen::storage::TrainingRecord;
use crate::model::network::{DECK_SIZE, INPUT_SIZE, NUM_RANKS};

/// A single training item for BoundaryNet with normalized EV targets.
#[derive(Debug, Clone)]
pub struct BoundaryItem {
    pub input: Vec<f32>,
    pub target: Vec<f32>,
    pub mask: Vec<f32>,
    pub range: Vec<f32>,
    pub game_value: f32,
}

/// Encode a TrainingRecord into a BoundaryItem with normalized pot/stack and targets.
pub fn encode_boundary_record(rec: &TrainingRecord) -> BoundaryItem {
    let total_stake = rec.pot + rec.effective_stack;
    let norm = if total_stake > 0.0 { total_stake } else { 1.0 };

    let mut input = Vec::with_capacity(INPUT_SIZE);
    input.extend_from_slice(&rec.oop_range);
    input.extend_from_slice(&rec.ip_range);

    let mut board_onehot = [0.0_f32; DECK_SIZE];
    for &card in &rec.board {
        debug_assert!((card as usize) < DECK_SIZE, "card id {card} out of range");
        board_onehot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_onehot);

    let mut rank_presence = [0.0_f32; NUM_RANKS];
    for &card in &rec.board {
        rank_presence[(card / 4) as usize] = 1.0;
    }
    input.extend_from_slice(&rank_presence);

    // Normalized pot and stack (key difference from CfvNet encoding)
    input.push(rec.pot / norm);
    input.push(rec.effective_stack / norm);
    input.push(f32::from(rec.player));

    debug_assert_eq!(input.len(), INPUT_SIZE);

    // Normalize targets: chip_ev / total_stake
    // chip_ev = cfv_pot_relative * pot
    let pot_over_norm = rec.pot / norm;
    let target: Vec<f32> = rec.cfvs.iter().map(|&cfv| cfv * pot_over_norm).collect();

    let mask: Vec<f32> = rec
        .valid_mask
        .iter()
        .map(|&v| if v != 0 { 1.0 } else { 0.0 })
        .collect();

    let range = if rec.player == 0 {
        rec.oop_range.to_vec()
    } else {
        rec.ip_range.to_vec()
    };

    // game_value = sum(range[i] * target[i]) — recompute from normalized targets
    // rather than scaling raw game_value (which is a weighted sum, not a single value).
    let game_value: f32 = range.iter().zip(target.iter()).map(|(&r, &t)| r * t).sum();

    BoundaryItem {
        input,
        target,
        mask,
        range,
        game_value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datagen::storage::TrainingRecord;
    use crate::model::network::{INPUT_SIZE, POT_INDEX};

    fn sample_record() -> TrainingRecord {
        let mut rec = TrainingRecord {
            board: vec![0, 4, 8, 12, 16],
            pot: 100.0,
            effective_stack: 150.0,
            player: 0,
            game_value: 0.05,
            oop_range: [0.0; 1326],
            ip_range: [0.0; 1326],
            cfvs: [0.0; 1326],
            valid_mask: [0; 1326],
        };
        rec.oop_range[0] = 0.5;
        rec.oop_range[1] = 0.5;
        rec.ip_range[100] = 1.0;
        rec.cfvs[0] = 0.3;
        rec.valid_mask[0] = 1;
        rec.valid_mask[1] = 1;
        rec.valid_mask[100] = 1;
        rec
    }

    #[test]
    fn encode_produces_correct_input_size() {
        let rec = sample_record();
        let item = encode_boundary_record(&rec);
        assert_eq!(item.input.len(), INPUT_SIZE);
    }

    #[test]
    fn encode_normalizes_pot_and_stack() {
        let rec = sample_record();
        let item = encode_boundary_record(&rec);
        let pot_idx = POT_INDEX;
        // pot=100, stack=150, total=250: pot/total=0.4, stack/total=0.6
        assert!(
            (item.input[pot_idx] - 0.4).abs() < 1e-6,
            "pot feature: {}",
            item.input[pot_idx]
        );
        assert!(
            (item.input[pot_idx + 1] - 0.6).abs() < 1e-6,
            "stack feature: {}",
            item.input[pot_idx + 1]
        );
    }

    #[test]
    fn encode_normalizes_target() {
        let rec = sample_record();
        let item = encode_boundary_record(&rec);
        // cfvs[0]=0.3 (pot-relative), chip_ev = 0.3 * pot = 0.3 * 100 = 30
        // normalized = 30 / 250 = 0.12
        assert!(
            (item.target[0] - 0.12).abs() < 1e-6,
            "target[0]: {}",
            item.target[0]
        );
    }

    #[test]
    fn encode_normalizes_game_value() {
        let rec = sample_record();
        let item = encode_boundary_record(&rec);
        // game_value = sum(range[i] * target[i])
        // range = oop_range = [0.5, 0.5, 0, ...]
        // target[0] = 0.3 * 100/250 = 0.12, target[1] = 0 * 100/250 = 0
        // game_value = 0.5 * 0.12 + 0.5 * 0.0 = 0.06
        assert!(
            (item.game_value - 0.06).abs() < 1e-6,
            "game_value: {}",
            item.game_value
        );
    }

    #[test]
    fn encode_selects_correct_player_range() {
        let mut rec = sample_record();
        rec.player = 1;
        let item = encode_boundary_record(&rec);
        assert!((item.range[100] - 1.0).abs() < 1e-6);
    }
}
