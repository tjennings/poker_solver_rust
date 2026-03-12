use std::io::BufReader;
use std::path::Path;

use crate::datagen::sampler::Situation;
use crate::datagen::storage::{read_record, TrainingRecord};
use crate::model::network::INPUT_SIZE;

/// Maximum number of board cards in the input encoding.
const MAX_BOARD_CARDS: usize = 5;

/// Sentinel value for missing board card positions (normalized).
const MISSING_CARD: f32 = -1.0;

/// A single training item with encoded input, target CFVs, mask, range, and game value.
#[derive(Debug, Clone)]
pub struct CfvItem {
    pub input: Vec<f32>,      // length INPUT_SIZE (2660)
    pub target: Vec<f32>,     // length OUTPUT_SIZE (1326) — target CFVs
    pub mask: Vec<f32>,       // length OUTPUT_SIZE — 1.0 valid, 0.0 masked
    pub range: Vec<f32>,      // length OUTPUT_SIZE — player's range for aux loss
    pub game_value: f32,      // scalar game value for aux loss
}

/// Dataset that loads binary training records from disk.
pub struct CfvDataset {
    records: Vec<TrainingRecord>,
}

impl CfvDataset {
    /// Load all records from a binary training file into memory.
    ///
    /// Records are self-describing (each starts with a board_size byte),
    /// so this reads until EOF.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("open dataset: {e}"))?;
        let mut reader = BufReader::new(file);

        let mut records = Vec::new();
        loop {
            match read_record(&mut reader) {
                Ok(rec) => records.push(rec),
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(format!("read: {e}")),
            }
        }

        Ok(Self { records })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Encode and return the training item at `idx`, or `None` if out of bounds.
    pub fn get(&self, idx: usize) -> Option<CfvItem> {
        self.records.get(idx).map(encode_record)
    }
}

/// Encode a [`Situation`] into a model input vector for inference.
///
/// Mirrors the encoding in [`encode_record`] but works from a `Situation`
/// rather than a raw `TrainingRecord`, making it usable at inference time
/// when no stored record exists.
pub fn encode_situation_for_inference(sit: &Situation, player: u8) -> Vec<f32> {
    let mut input = Vec::with_capacity(INPUT_SIZE);
    // OOP range (1326 floats)
    for &v in &sit.ranges[0] {
        input.push(v);
    }
    // IP range (1326 floats)
    for &v in &sit.ranges[1] {
        input.push(v);
    }
    // Board cards (5 floats, normalized to [0, 1])
    for &card in &sit.board {
        input.push(f32::from(card) / 51.0);
    }
    // Pot (normalized by max pot)
    input.push(sit.pot as f32 / 400.0);
    // Effective stack (normalized by max stack)
    input.push(sit.effective_stack as f32 / 400.0);
    // Player indicator (0.0 = OOP, 1.0 = IP)
    input.push(f32::from(player));
    debug_assert_eq!(input.len(), INPUT_SIZE);
    input
}

fn encode_record(rec: &TrainingRecord) -> CfvItem {
    let mut input = Vec::with_capacity(INPUT_SIZE);

    // OOP range (1326 floats)
    input.extend_from_slice(&rec.oop_range);
    // IP range (1326 floats)
    input.extend_from_slice(&rec.ip_range);
    // Board cards (up to MAX_BOARD_CARDS floats, normalized to [0, 1]).
    // Missing positions are filled with MISSING_CARD sentinel.
    for i in 0..MAX_BOARD_CARDS {
        if i < rec.board.len() {
            input.push(f32::from(rec.board[i]) / 51.0);
        } else {
            input.push(MISSING_CARD);
        }
    }
    // Pot (normalized by max pot)
    input.push(rec.pot / 400.0);
    // Effective stack (normalized by max stack)
    input.push(rec.effective_stack / 400.0);
    // Player indicator (0.0 = OOP, 1.0 = IP)
    input.push(f32::from(rec.player));

    debug_assert_eq!(input.len(), INPUT_SIZE);

    let target = rec.cfvs.to_vec();

    let mask: Vec<f32> = rec.valid_mask.iter()
        .map(|&v| if v != 0 { 1.0 } else { 0.0 })
        .collect();

    // Select the acting player's range for the auxiliary loss.
    let range = if rec.player == 0 {
        rec.oop_range.to_vec()
    } else {
        rec.ip_range.to_vec()
    };

    CfvItem {
        input,
        target,
        mask,
        range,
        game_value: rec.game_value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datagen::storage::write_record;
    use crate::model::network::OUTPUT_SIZE;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_test_data(n: usize) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..n {
            let mut rec = TrainingRecord {
                board: vec![0, 4, 8, 12, 16],
                pot: 100.0,
                effective_stack: 50.0,
                player: (i % 2) as u8,
                game_value: 0.1 * i as f32,
                oop_range: [0.0; 1326],
                ip_range: [0.0; 1326],
                cfvs: [0.0; 1326],
                valid_mask: [1; 1326],
            };
            rec.cfvs[0] = i as f32 * 0.01;
            rec.oop_range[0] = 0.5;
            rec.ip_range[0] = 0.3;
            write_record(&mut file, &rec).unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    fn dataset_length_correct() {
        let file = write_test_data(10);
        let dataset = CfvDataset::from_file(file.path()).unwrap();
        assert_eq!(dataset.len(), 10);
    }

    #[test]
    fn dataset_get_returns_valid_item() {
        let file = write_test_data(5);
        let dataset = CfvDataset::from_file(file.path()).unwrap();
        let item = dataset.get(0).unwrap();
        assert_eq!(item.input.len(), INPUT_SIZE);
        assert_eq!(item.target.len(), OUTPUT_SIZE);
        assert_eq!(item.mask.len(), OUTPUT_SIZE);
        assert_eq!(item.range.len(), OUTPUT_SIZE);
    }

    #[test]
    fn dataset_input_encoding_correct() {
        let file = write_test_data(1);
        let dataset = CfvDataset::from_file(file.path()).unwrap();
        let item = dataset.get(0).unwrap();

        // First element of OOP range should be 0.5
        assert!((item.input[0] - 0.5).abs() < 1e-6);
        // First element of IP range should be 0.3 (at index 1326)
        assert!((item.input[1326] - 0.3).abs() < 1e-6);
        // Board cards normalized: card 0 / 51 = 0.0
        assert!((item.input[2652]).abs() < 1e-6);
        // Board card 4 / 51.0
        assert!((item.input[2653] - 4.0 / 51.0).abs() < 1e-6);
        // Pot = 100 / 400 = 0.25
        assert!((item.input[2657] - 0.25).abs() < 1e-6);
        // Stack = 50 / 400 = 0.125
        assert!((item.input[2658] - 0.125).abs() < 1e-6);
        // Player = 0 (first record)
        assert!((item.input[2659]).abs() < 1e-6);

        // Total length
        assert_eq!(item.input.len(), 1326 + 1326 + 5 + 1 + 1 + 1);
    }

    #[test]
    fn dataset_player_range_selection() {
        let file = write_test_data(2);
        let dataset = CfvDataset::from_file(file.path()).unwrap();
        // Record 0: player=0, should use OOP range
        let item0 = dataset.get(0).unwrap();
        assert!(
            (item0.range[0] - 0.5).abs() < 1e-6,
            "player 0 should use OOP range"
        );
        // Record 1: player=1, should use IP range
        let item1 = dataset.get(1).unwrap();
        assert!(
            (item1.range[0] - 0.3).abs() < 1e-6,
            "player 1 should use IP range"
        );
    }
}
