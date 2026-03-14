use std::io::BufReader;
use std::path::Path;

use crate::datagen::sampler::Situation;
use crate::datagen::storage::{read_record, TrainingRecord};
use crate::model::network::{DECK_SIZE, INPUT_SIZE, NUM_RANKS};

/// A single training item with encoded input, target CFVs, mask, range, and game value.
#[derive(Debug, Clone)]
pub struct CfvItem {
    pub input: Vec<f32>,      // length INPUT_SIZE (2720)
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
    pub fn from_file(path: &Path, _board_cards: usize) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("open dataset {}: {e}", path.display()))?;
        let mut reader = BufReader::new(file);

        let mut records = Vec::new();
        loop {
            match read_record(&mut reader) {
                Ok(rec) => records.push(rec),
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(format!("read {}: {e}", path.display())),
            }
        }

        Ok(Self { records })
    }

    /// Load records from all files in a directory.
    ///
    /// Reads every file in the directory (non-recursive, skipping subdirectories),
    /// sorted by name for deterministic ordering. Returns an error if the directory
    /// is empty or cannot be read.
    pub fn from_dir(dir: &Path, board_cards: usize) -> Result<Self, String> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("read directory {}: {e}", dir.display()))?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if entry.file_type().ok()?.is_file() {
                    Some(entry.path())
                } else {
                    None
                }
            })
            .collect();

        if paths.is_empty() {
            return Err(format!("no files found in {}", dir.display()));
        }

        paths.sort();

        let mut records = Vec::new();
        for path in &paths {
            let ds = Self::from_file(path, board_cards)?;
            eprintln!("  {} — {} records", path.display(), ds.len());
            records.extend(ds.records);
        }

        Ok(Self { records })
    }

    /// Load from a path that may be a file or directory.
    ///
    /// If `path` is a directory, loads all files within it. If it's a file,
    /// loads that single file.
    pub fn from_path(path: &Path, board_cards: usize) -> Result<Self, String> {
        if path.is_dir() {
            Self::from_dir(path, board_cards)
        } else {
            Self::from_file(path, board_cards)
        }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Access the underlying records for bulk pre-encoding.
    pub(crate) fn records(&self) -> &[TrainingRecord] {
        &self.records
    }

    /// The input feature size (constant for all streets with one-hot board encoding).
    pub fn input_size(&self) -> usize {
        INPUT_SIZE
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
    // Board cards (52-element one-hot)
    let mut board_onehot = [0.0_f32; DECK_SIZE];
    for &card in sit.board_cards() {
        debug_assert!((card as usize) < DECK_SIZE, "card id {card} out of range");
        board_onehot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_onehot);
    // Rank presence (13-element binary vector)
    let mut rank_presence = [0.0_f32; NUM_RANKS];
    for &card in sit.board_cards() {
        debug_assert!((card as usize) < DECK_SIZE, "card id {card} out of range");
        rank_presence[(card / 4) as usize] = 1.0;
    }
    input.extend_from_slice(&rank_presence);
    // Pot (normalized by max pot)
    input.push(sit.pot as f32 / 400.0);
    // Effective stack (normalized by max stack)
    input.push(sit.effective_stack as f32 / 400.0);
    // Player indicator (0.0 = OOP, 1.0 = IP)
    input.push(f32::from(player));
    debug_assert_eq!(input.len(), INPUT_SIZE);
    input
}

pub(crate) fn encode_record(rec: &TrainingRecord) -> CfvItem {
    let mut input = Vec::with_capacity(INPUT_SIZE);

    // OOP range (1326 floats)
    input.extend_from_slice(&rec.oop_range);
    // IP range (1326 floats)
    input.extend_from_slice(&rec.ip_range);
    // Board cards (52-element one-hot)
    let mut board_onehot = [0.0_f32; DECK_SIZE];
    for &card in &rec.board {
        debug_assert!((card as usize) < DECK_SIZE, "card id {card} out of range");
        board_onehot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_onehot);
    // Rank presence (13-element binary vector)
    let mut rank_presence = [0.0_f32; NUM_RANKS];
    for &card in &rec.board {
        debug_assert!((card as usize) < DECK_SIZE, "card id {card} out of range");
        rank_presence[(card / 4) as usize] = 1.0;
    }
    input.extend_from_slice(&rank_presence);
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
    use crate::model::network::{OUTPUT_SIZE, POT_INDEX};
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
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
        assert_eq!(dataset.len(), 10);
    }

    #[test]
    fn dataset_get_returns_valid_item() {
        let file = write_test_data(5);
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
        let item = dataset.get(0).unwrap();
        assert_eq!(item.input.len(), INPUT_SIZE);
        assert_eq!(item.target.len(), OUTPUT_SIZE);
        assert_eq!(item.mask.len(), OUTPUT_SIZE);
        assert_eq!(item.range.len(), OUTPUT_SIZE);
    }

    #[test]
    fn dataset_input_size_method() {
        let file = write_test_data(1);
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
        assert_eq!(dataset.input_size(), INPUT_SIZE);

        // Turn dataset also returns the same INPUT_SIZE now.
        let dataset4 = CfvDataset::from_file(file.path(), 4).unwrap();
        assert_eq!(dataset4.input_size(), INPUT_SIZE);
    }

    #[test]
    fn dataset_input_encoding_correct() {
        let file = write_test_data(1);
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
        let item = dataset.get(0).unwrap();

        // First element of OOP range should be 0.5
        assert!((item.input[0] - 0.5).abs() < 1e-6);
        // First element of IP range should be 0.3 (at index 1326)
        assert!((item.input[1326] - 0.3).abs() < 1e-6);
        // Board one-hot: card 0 should be 1.0 at index 2652+0
        assert!((item.input[2652] - 1.0).abs() < 1e-6, "card 0 one-hot");
        // Card 4 should be 1.0 at index 2652+4
        assert!((item.input[2652 + 4] - 1.0).abs() < 1e-6, "card 4 one-hot");
        // Card 8 should be 1.0 at index 2652+8
        assert!((item.input[2652 + 8] - 1.0).abs() < 1e-6, "card 8 one-hot");
        // Card 12 should be 1.0 at index 2652+12
        assert!((item.input[2652 + 12] - 1.0).abs() < 1e-6, "card 12 one-hot");
        // Card 16 should be 1.0 at index 2652+16
        assert!((item.input[2652 + 16] - 1.0).abs() < 1e-6, "card 16 one-hot");
        // A non-board card should be 0.0
        assert!((item.input[2652 + 1]).abs() < 1e-6, "non-board card should be 0");
        // Rank presence vector starts at 2652 + 52 = 2704
        // Board cards are 0, 4, 8, 12, 16 → ranks 0/4=0, 4/4=1, 8/4=2, 12/4=3, 16/4=4
        let rp_start = 2652 + 52; // 2704
        assert!((item.input[rp_start] - 1.0).abs() < 1e-6, "rank 0 present");
        assert!((item.input[rp_start + 1] - 1.0).abs() < 1e-6, "rank 1 present");
        assert!((item.input[rp_start + 2] - 1.0).abs() < 1e-6, "rank 2 present");
        assert!((item.input[rp_start + 3] - 1.0).abs() < 1e-6, "rank 3 present");
        assert!((item.input[rp_start + 4] - 1.0).abs() < 1e-6, "rank 4 present");
        assert!((item.input[rp_start + 5]).abs() < 1e-6, "rank 5 absent");
        assert!((item.input[rp_start + 12]).abs() < 1e-6, "rank 12 absent");
        // Pot at POT_INDEX = 2717: 100 / 400 = 0.25
        assert!((item.input[POT_INDEX] - 0.25).abs() < 1e-6);
        // Stack at POT_INDEX + 1: 50 / 400 = 0.125
        assert!((item.input[POT_INDEX + 1] - 0.125).abs() < 1e-6);
        // Player at POT_INDEX + 2: 0 (first record)
        assert!((item.input[POT_INDEX + 2]).abs() < 1e-6);

        // Total length
        assert_eq!(item.input.len(), INPUT_SIZE);
    }

    #[test]
    fn dataset_turn_encoding_uses_onehot() {
        let file = write_test_data(1);
        // Turn dataset: board has 5 cards but record stores all of them.
        // The one-hot encodes all board cards regardless of board_cards param.
        let dataset = CfvDataset::from_file(file.path(), 4).unwrap();
        let item = dataset.get(0).unwrap();

        // Input size is always INPUT_SIZE (2720)
        assert_eq!(item.input.len(), INPUT_SIZE);
        // Board one-hot: all 5 board cards are encoded (entire rec.board).
        assert!((item.input[2652] - 1.0).abs() < 1e-6, "card 0 one-hot");
        assert!((item.input[2652 + 4] - 1.0).abs() < 1e-6, "card 4 one-hot");
        assert!((item.input[2652 + 8] - 1.0).abs() < 1e-6, "card 8 one-hot");
        assert!((item.input[2652 + 12] - 1.0).abs() < 1e-6, "card 12 one-hot");
        assert!((item.input[2652 + 16] - 1.0).abs() < 1e-6, "card 16 one-hot");
        // Pot at POT_INDEX = 2717
        assert!((item.input[POT_INDEX] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn dataset_player_range_selection() {
        let file = write_test_data(2);
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
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

    fn write_test_data_to(path: &Path, n: usize) {
        let mut file = std::fs::File::create(path).unwrap();
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
    }

    #[test]
    fn from_dir_loads_all_files() {
        let dir = tempfile::tempdir().unwrap();
        write_test_data_to(&dir.path().join("a.bin"), 5);
        write_test_data_to(&dir.path().join("b.bin"), 3);
        let dataset = CfvDataset::from_dir(dir.path(), 5).unwrap();
        assert_eq!(dataset.len(), 8);
    }

    #[test]
    fn from_dir_sorts_by_name() {
        let dir = tempfile::tempdir().unwrap();
        write_test_data_to(&dir.path().join("b.bin"), 2);
        write_test_data_to(&dir.path().join("a.bin"), 3);
        let dataset = CfvDataset::from_dir(dir.path(), 5).unwrap();
        // a.bin (3 records) loaded first, then b.bin (2 records) = 5 total
        assert_eq!(dataset.len(), 5);
    }

    #[test]
    fn from_dir_empty_directory_errors() {
        let dir = tempfile::tempdir().unwrap();
        let result = CfvDataset::from_dir(dir.path(), 5);
        assert!(result.is_err());
    }

    #[test]
    fn from_path_detects_file() {
        let file = write_test_data(4);
        let dataset = CfvDataset::from_path(file.path(), 5).unwrap();
        assert_eq!(dataset.len(), 4);
    }

    #[test]
    fn from_path_detects_directory() {
        let dir = tempfile::tempdir().unwrap();
        write_test_data_to(&dir.path().join("data.bin"), 7);
        let dataset = CfvDataset::from_path(dir.path(), 5).unwrap();
        assert_eq!(dataset.len(), 7);
    }
}
