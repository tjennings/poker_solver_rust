use std::io::BufReader;
use std::path::Path;

use crate::datagen::sampler::Situation;
use crate::datagen::storage::{read_record, TrainingRecord};
use crate::model::network::{input_size, INPUT_SIZE};

/// A single training item with encoded input and dual-player targets.
#[derive(Debug, Clone)]
pub struct CfvItem {
    pub input: Vec<f32>,       // length INPUT_SIZE (2706)
    pub oop_target: Vec<f32>,  // length NUM_COMBOS (1326) — OOP target CFVs
    pub ip_target: Vec<f32>,   // length NUM_COMBOS (1326) — IP target CFVs
    pub mask: Vec<f32>,        // length NUM_COMBOS — 1.0 valid, 0.0 masked
    pub oop_range: Vec<f32>,   // length NUM_COMBOS — OOP range weights
    pub ip_range: Vec<f32>,    // length NUM_COMBOS — IP range weights
    pub pot: f32,              // raw pot size for pot-weighted loss
}

/// Dataset that loads binary training records from disk.
pub struct CfvDataset {
    records: Vec<TrainingRecord>,
    board_cards: usize,
}

impl CfvDataset {
    /// Load all records from a binary training file into memory.
    ///
    /// Records are self-describing (each starts with a board_size byte),
    /// so this reads until EOF.
    pub fn from_file(path: &Path, board_cards: usize) -> Result<Self, String> {
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

        Ok(Self { records, board_cards })
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

        Ok(Self { records, board_cards })
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

    /// The number of board cards used for encoding.
    pub(crate) fn board_cards(&self) -> usize {
        self.board_cards
    }

    /// Access the underlying records for bulk pre-encoding.
    pub(crate) fn records(&self) -> &[TrainingRecord] {
        &self.records
    }

    /// The input feature size for this dataset's board card count.
    pub fn input_size(&self) -> usize {
        input_size(self.board_cards)
    }

    /// Encode and return the training item at `idx`, or `None` if out of bounds.
    pub fn get(&self, idx: usize) -> Option<CfvItem> {
        self.records.get(idx).map(|rec| encode_record(rec, self.board_cards))
    }
}

/// Encode a [`Situation`] into a model input vector for inference.
///
/// Mirrors the encoding in [`encode_record`] but works from a `Situation`
/// rather than a raw `TrainingRecord`, making it usable at inference time
/// when no stored record exists.
///
/// Layout: `[OOP_range(1326), IP_range(1326), board_one_hot(52), pot(1), stack(1)]`
pub fn encode_situation_for_inference(sit: &Situation) -> Vec<f32> {
    let in_size = INPUT_SIZE;
    let mut input = Vec::with_capacity(in_size);
    // OOP range (1326 floats)
    input.extend_from_slice(&sit.ranges[0]);
    // IP range (1326 floats)
    input.extend_from_slice(&sit.ranges[1]);
    // Board cards: 52-dim one-hot
    let mut board_one_hot = [0.0f32; 52];
    for &card in sit.board_cards() {
        board_one_hot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_one_hot);
    // Pot (normalized by max pot)
    input.push(sit.pot as f32 / 400.0);
    // Effective stack (normalized by max stack)
    input.push(sit.effective_stack as f32 / 400.0);
    debug_assert_eq!(input.len(), in_size);
    input
}

pub(crate) fn encode_record(rec: &TrainingRecord, board_cards: usize) -> CfvItem {
    let in_size = INPUT_SIZE;
    let mut input = Vec::with_capacity(in_size);

    // OOP range (1326 floats)
    input.extend_from_slice(&rec.oop_range);
    // IP range (1326 floats)
    input.extend_from_slice(&rec.ip_range);
    // Board cards: 52-dim one-hot
    let mut board_one_hot = [0.0f32; 52];
    for &card in &rec.board[..board_cards] {
        board_one_hot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_one_hot);
    // Pot (normalized by max pot)
    input.push(rec.pot / 400.0);
    // Effective stack (normalized by max stack)
    input.push(rec.effective_stack / 400.0);

    debug_assert_eq!(input.len(), in_size);

    // Both players' CFVs directly from the record.
    let oop_target = rec.oop_cfvs.to_vec();
    let ip_target = rec.ip_cfvs.to_vec();

    let mask: Vec<f32> = rec.valid_mask.iter()
        .map(|&v| if v != 0 { 1.0 } else { 0.0 })
        .collect();

    CfvItem {
        input,
        oop_target,
        ip_target,
        mask,
        oop_range: rec.oop_range.to_vec(),
        ip_range: rec.ip_range.to_vec(),
        pot: rec.pot,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datagen::storage::write_record;
    use crate::model::network::{INPUT_SIZE, NUM_COMBOS};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_record(i: usize) -> TrainingRecord {
        let mut rec = TrainingRecord {
            board: vec![0, 4, 8, 12, 16],
            pot: 100.0,
            effective_stack: 50.0,
            oop_range: [0.0; 1326],
            ip_range: [0.0; 1326],
            oop_cfvs: [0.0; 1326],
            ip_cfvs: [0.0; 1326],
            valid_mask: [1; 1326],
        };
        rec.oop_cfvs[0] = i as f32 * 0.01;
        rec.ip_cfvs[0] = -(i as f32) * 0.01;
        rec.oop_range[0] = 0.5;
        rec.ip_range[0] = 0.3;
        rec
    }

    fn write_test_data(n: usize) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..n {
            let rec = make_record(i);
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
        assert_eq!(item.oop_target.len(), NUM_COMBOS);
        assert_eq!(item.ip_target.len(), NUM_COMBOS);
        assert_eq!(item.mask.len(), NUM_COMBOS);
        assert_eq!(item.oop_range.len(), NUM_COMBOS);
        assert_eq!(item.ip_range.len(), NUM_COMBOS);
    }

    #[test]
    fn dataset_input_size_method() {
        let file = write_test_data(1);
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
        assert_eq!(dataset.input_size(), INPUT_SIZE);

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

        // Board one-hot: cards are [0, 4, 8, 12, 16]
        let board_start = 2652;
        assert!((item.input[board_start] - 1.0).abs() < 1e-6);
        assert!((item.input[board_start + 4] - 1.0).abs() < 1e-6);
        assert!((item.input[board_start + 8] - 1.0).abs() < 1e-6);
        assert!((item.input[board_start + 12] - 1.0).abs() < 1e-6);
        assert!((item.input[board_start + 16] - 1.0).abs() < 1e-6);
        // Non-board cards should be 0
        assert!((item.input[board_start + 1]).abs() < 1e-6);
        assert!((item.input[board_start + 51]).abs() < 1e-6);

        // Pot = 100 / 400 = 0.25, at index 2652 + 52 = 2704
        assert!((item.input[2704] - 0.25).abs() < 1e-6);
        // Stack = 50 / 400 = 0.125, at index 2705
        assert!((item.input[2705] - 0.125).abs() < 1e-6);

        assert_eq!(item.input.len(), INPUT_SIZE);
        assert_eq!(item.input.len(), 2706);
    }

    #[test]
    fn dataset_turn_encoding_same_input_size() {
        let file = write_test_data(1);
        let dataset = CfvDataset::from_file(file.path(), 4).unwrap();
        let item = dataset.get(0).unwrap();

        assert_eq!(item.input.len(), INPUT_SIZE);

        let board_start = 2652;
        assert!((item.input[board_start] - 1.0).abs() < 1e-6);
        assert!((item.input[board_start + 4] - 1.0).abs() < 1e-6);
        assert!((item.input[board_start + 8] - 1.0).abs() < 1e-6);
        assert!((item.input[board_start + 12] - 1.0).abs() < 1e-6);
        // Card 16 NOT encoded (only 4 board cards)
        assert!((item.input[board_start + 16]).abs() < 1e-6);

        assert!((item.input[2704] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn dataset_dual_target_from_record() {
        let file = write_test_data(3);
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();

        let item1 = dataset.get(1).unwrap();
        assert!((item1.oop_target[0] - 0.01).abs() < 1e-6);
        assert!((item1.ip_target[0] - (-0.01)).abs() < 1e-6);

        let item2 = dataset.get(2).unwrap();
        assert!((item2.oop_target[0] - 0.02).abs() < 1e-6);
        assert!((item2.ip_target[0] - (-0.02)).abs() < 1e-6);
    }

    #[test]
    fn dataset_range_fields() {
        let file = write_test_data(2);
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();

        let item0 = dataset.get(0).unwrap();
        assert!((item0.oop_range[0] - 0.5).abs() < 1e-6);
        assert!((item0.ip_range[0] - 0.3).abs() < 1e-6);

        let item1 = dataset.get(1).unwrap();
        assert!((item1.oop_range[0] - 0.5).abs() < 1e-6);
        assert!((item1.ip_range[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn no_player_indicator_in_input() {
        let file = write_test_data(1);
        let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
        let item = dataset.get(0).unwrap();

        assert_eq!(item.input.len(), INPUT_SIZE);
        assert_eq!(INPUT_SIZE, 2706);
    }

    fn write_test_data_to(path: &Path, n: usize) {
        let mut file = std::fs::File::create(path).unwrap();
        for i in 0..n {
            let rec = make_record(i);
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
