# One-Hot Board Encoding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace the variable-length `card / 51.0` scalar board encoding with a fixed 52-element one-hot binary vector across all cfvnet encoding and inference paths.

**Architecture:** Pure domain-layer change — modify the encoding functions (`input_size`, `encode_record`, `encode_situation_for_inference`, `build_input`, `predict_with_model`) and update all call sites and tests. No new files, no new dependencies.

**Tech Stack:** Rust, burn (neural network framework)

---

### Task 1: Change `input_size` to a constant `INPUT_SIZE`

**Files:**
- Modify: `crates/cfvnet/src/model/network.rs`

**Step 1: Update the constant and function**

Replace lines 14-19 in `network.rs`:

```rust
/// Number of hole-card combinations (52 choose 2).
pub const NUM_COMBOS: usize = 1326;
/// One counterfactual value per combo.
pub const OUTPUT_SIZE: usize = NUM_COMBOS;

/// Compute input feature size for a given number of board cards.
///
/// Layout: OOP range (1326) + IP range (1326) + board cards + pot + stack + player indicator.
pub fn input_size(board_cards: usize) -> usize {
    NUM_COMBOS + NUM_COMBOS + board_cards + 1 + 1 + 1
}
```

With:

```rust
/// Number of hole-card combinations (52 choose 2).
pub const NUM_COMBOS: usize = 1326;
/// One counterfactual value per combo.
pub const OUTPUT_SIZE: usize = NUM_COMBOS;
/// Number of cards in a standard deck.
pub const DECK_SIZE: usize = 52;

/// Fixed input feature size.
///
/// Layout: OOP range (1326) + IP range (1326) + one-hot board (52) + pot + stack + player indicator.
pub const INPUT_SIZE: usize = NUM_COMBOS + NUM_COMBOS + DECK_SIZE + 1 + 1 + 1; // 2707

/// Pot feature index within the encoded input vector.
pub const POT_INDEX: usize = NUM_COMBOS + NUM_COMBOS + DECK_SIZE; // 2704
```

**Step 2: Update the tests in `network.rs`**

Replace the test module (lines 87-155) with:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn input_size_is_2707() {
        assert_eq!(INPUT_SIZE, 2707);
    }

    #[test]
    fn pot_index_is_2704() {
        assert_eq!(POT_INDEX, 2704);
    }

    #[test]
    fn model_output_shape() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 7, 500, INPUT_SIZE);
        let input = Tensor::<TestBackend, 2>::zeros([1, INPUT_SIZE], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, OUTPUT_SIZE]);
    }

    #[test]
    fn model_batch_forward() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 7, 500, INPUT_SIZE);
        let batch_size = 4;
        let input =
            Tensor::<TestBackend, 2>::zeros([batch_size, INPUT_SIZE], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [batch_size, OUTPUT_SIZE]);
    }

    #[test]
    fn model_output_changes_with_input() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 7, 500, INPUT_SIZE);
        let input1 = Tensor::<TestBackend, 2>::zeros([1, INPUT_SIZE], &device);
        let input2 = Tensor::<TestBackend, 2>::ones([1, INPUT_SIZE], &device);
        let out1 = model.forward(input1);
        let out2 = model.forward(input2);
        let diff: f32 = (out1 - out2).abs().sum().into_scalar();
        assert!(
            diff > 1e-6,
            "outputs should differ for different inputs, diff={diff}"
        );
    }
}
```

**Step 3: Run tests to verify `network.rs` changes compile and pass**

Run: `cargo test -p cfvnet model::network -- --nocapture 2>&1 | tail -20`
Expected: All network tests pass (some downstream will fail — that's expected)

**Step 4: Commit**

```bash
git add crates/cfvnet/src/model/network.rs
git commit -m "refactor(cfvnet): replace input_size() with constant INPUT_SIZE for one-hot board encoding"
```

---

### Task 2: Update `encode_record` and `encode_situation_for_inference` in `dataset.rs`

**Files:**
- Modify: `crates/cfvnet/src/model/dataset.rs`

**Step 1: Update imports and encoding functions**

In `dataset.rs`, replace the import on line 6:
```rust
use crate::model::network::input_size;
```
With:
```rust
use crate::model::network::{DECK_SIZE, INPUT_SIZE};
```

Replace `encode_situation_for_inference` (lines 126-150):
```rust
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
    // One-hot board vector (52 floats)
    let mut board_onehot = [0.0_f32; DECK_SIZE];
    for &card in sit.board_cards() {
        board_onehot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_onehot);
    // Pot (normalized by max pot)
    input.push(sit.pot as f32 / 400.0);
    // Effective stack (normalized by max stack)
    input.push(sit.effective_stack as f32 / 400.0);
    // Player indicator (0.0 = OOP, 1.0 = IP)
    input.push(f32::from(player));
    debug_assert_eq!(input.len(), INPUT_SIZE);
    input
}
```

Replace `encode_record` (lines 152-193):
```rust
pub(crate) fn encode_record(rec: &TrainingRecord) -> CfvItem {
    let mut input = Vec::with_capacity(INPUT_SIZE);

    // OOP range (1326 floats)
    input.extend_from_slice(&rec.oop_range);
    // IP range (1326 floats)
    input.extend_from_slice(&rec.ip_range);
    // One-hot board vector (52 floats)
    let mut board_onehot = [0.0_f32; DECK_SIZE];
    for &card in &rec.board {
        board_onehot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_onehot);
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
```

Note: `encode_record` no longer takes `board_cards` parameter — the one-hot encodes all cards in `rec.board` regardless of count.

**Step 2: Update `CfvDataset` methods that pass `board_cards` to `encode_record`**

Update `input_size` method (line 111-113):
```rust
    pub fn input_size(&self) -> usize {
        INPUT_SIZE
    }
```

Update `get` method (line 116-118):
```rust
    pub fn get(&self, idx: usize) -> Option<CfvItem> {
        self.records.get(idx).map(|rec| encode_record(rec))
    }
```

**Step 3: Update tests in `dataset.rs`**

Replace the entire test module (lines 195-374) with:

```rust
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
        assert_eq!(dataset.input_size(), 2707);

        let dataset4 = CfvDataset::from_file(file.path(), 4).unwrap();
        assert_eq!(dataset4.input_size(), 2707);
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

        // One-hot board: cards 0, 4, 8, 12, 16 should be 1.0
        let board_start = 2 * 1326; // 2652
        for i in 0..52 {
            let expected = if [0, 4, 8, 12, 16].contains(&i) { 1.0 } else { 0.0 };
            assert!(
                (item.input[board_start + i] - expected).abs() < 1e-6,
                "board one-hot at card {i}: expected {expected}, got {}",
                item.input[board_start + i]
            );
        }

        // Pot = 100 / 400 = 0.25 (at POT_INDEX = 2704)
        assert!((item.input[POT_INDEX] - 0.25).abs() < 1e-6);
        // Stack = 50 / 400 = 0.125
        assert!((item.input[POT_INDEX + 1] - 0.125).abs() < 1e-6);
        // Player = 0 (first record)
        assert!((item.input[POT_INDEX + 2]).abs() < 1e-6);

        // Total length
        assert_eq!(item.input.len(), INPUT_SIZE);
    }

    #[test]
    fn dataset_turn_encoding_same_size() {
        // With one-hot, turn and river use the same input size
        let file = write_test_data(1);
        let dataset = CfvDataset::from_file(file.path(), 4).unwrap();
        let item = dataset.get(0).unwrap();

        assert_eq!(item.input.len(), INPUT_SIZE);
        // Board cards: only first 4 should be set in one-hot (cards 0, 4, 8, 12)
        // Card 16 is in the record but board_cards=4, so it's still encoded
        // because encode_record uses rec.board (all cards in record)
        let board_start = 2 * 1326;
        assert!((item.input[board_start + 0] - 1.0).abs() < 1e-6); // card 0
        assert!((item.input[board_start + 4] - 1.0).abs() < 1e-6); // card 4
        assert!((item.input[board_start + 8] - 1.0).abs() < 1e-6); // card 8
        assert!((item.input[board_start + 12] - 1.0).abs() < 1e-6); // card 12
        // Pot at POT_INDEX
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
```

**Step 4: Run tests**

Run: `cargo test -p cfvnet model::dataset -- --nocapture 2>&1 | tail -20`
Expected: All dataset tests pass

**Step 5: Commit**

```bash
git add crates/cfvnet/src/model/dataset.rs
git commit -m "feat(cfvnet): encode board as 52-element one-hot vector in dataset"
```

---

### Task 3: Update `training.rs` — remove `board_cards` from encoding calls

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs`

**Step 1: Update imports**

Replace line 19:
```rust
use crate::model::network::{CfvNet, OUTPUT_SIZE, input_size};
```
With:
```rust
use crate::model::network::{CfvNet, INPUT_SIZE, OUTPUT_SIZE};
```

**Step 2: Update `PreEncoded::from_records`**

Replace the method signature and body (lines 65-90):
```rust
    fn from_records(records: &[TrainingRecord]) -> Self {
        let n = records.len();

        let items: Vec<_> = records
            .iter()
            .map(|rec| encode_record(rec))
            .collect();

        // Flatten into contiguous arrays.
        let mut input = Vec::with_capacity(n * INPUT_SIZE);
        let mut target = Vec::with_capacity(n * OUTPUT_SIZE);
        let mut mask = Vec::with_capacity(n * OUTPUT_SIZE);
        let mut range = Vec::with_capacity(n * OUTPUT_SIZE);
        let mut game_value = Vec::with_capacity(n);

        for item in &items {
            input.extend_from_slice(&item.input);
            target.extend_from_slice(&item.target);
            mask.extend_from_slice(&item.mask);
            range.extend_from_slice(&item.range);
            game_value.push(item.game_value);
        }

        Self { input, target, mask, range, game_value, in_size: INPUT_SIZE, len: n }
    }
```

**Step 3: Update `load_validation_set`**

Replace lines 408-422:
```rust
fn load_validation_set(
    files: &[PathBuf],
    val_count: usize,
) -> Option<PreEncoded> {
    if val_count == 0 {
        return None;
    }
    eprintln!("Loading {val_count} validation records...");
    let mut val_reader = StreamingReader::new(files.to_vec());
    let val_records = val_reader.read_chunk(val_count);
    let actual_val = val_records.len();
    eprintln!("Loaded {actual_val} validation records");
    Some(PreEncoded::from_records(&val_records))
}
```

**Step 4: Update `spawn_dataloader_thread`**

Replace the signature (lines 431-436) to remove `board_cards`:
```rust
fn spawn_dataloader_thread(
    files: &[PathBuf],
    config: &TrainConfig,
    val_count: usize,
) -> (mpsc::Receiver<PreEncoded>, Vec<std::thread::JoinHandle<()>>) {
```

Inside the encoder thread closure (line 526), change:
```rust
                let encoded = PreEncoded::from_records(&records, board_cards);
```
To:
```rust
                let encoded = PreEncoded::from_records(&records);
```

**Step 5: Update `train` function**

Replace the signature (lines 543-549) to remove `board_cards`:
```rust
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    data_path: &Path,
    config: &TrainConfig,
    output_dir: Option<&std::path::Path>,
) -> TrainResult {
```

Inside the function body:
- Replace `let in_size = input_size(board_cards);` (line 550) with `let in_size = INPUT_SIZE;`
- Replace the model creation (line 551) with: `let model = CfvNet::<B>::new(device, config.hidden_layers, config.hidden_size, in_size);`
- Replace `count_total_records(&files, board_cards)` (line 562) with `count_total_records(&files, 5)` — board_size is still needed for file size arithmetic since the binary format is board-size-dependent. Actually, we need to keep board_cards for this. Add it back as a parameter or read from the first record. The simplest approach: keep `board_cards` only for `count_total_records` and pass `5` as default for river (the most common case). But actually, the board_size is encoded in each binary record. Let me check...

Actually — `count_total_records` uses `record_size(board_size)` to compute how many records fit in a file. The binary format still stores a variable number of board cards. We need `board_cards` for this calculation only.

Better approach: pass `board_cards` to `train()` only for the record counting, but don't pass it to encoding. Update signature:

```rust
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    data_path: &Path,
    board_cards: usize,
    config: &TrainConfig,
    output_dir: Option<&std::path::Path>,
) -> TrainResult {
```

Keep `board_cards` in the signature for `count_total_records` and `spawn_dataloader_thread` (the reader needs to know the record size for seeking), but remove it from all encoding paths.

So the changes to `train` are:
- Line 550: `let in_size = INPUT_SIZE;` (was `input_size(board_cards)`)
- Line 562: keep `count_total_records(&files, board_cards)` as-is
- Line 572: `let val_encoded = load_validation_set(&files, val_count);` (drop `board_cards`)
- Lines 591-596: `spawn_dataloader_thread(&files, config, val_count)` (drop `board_cards`)

And in `spawn_dataloader_thread`, the encoder closure changes to `PreEncoded::from_records(&records)`.

**Step 6: Update tests in `training.rs`**

All test calls to `train` keep `board_cards` (5) since the binary record format is unchanged. Only the encoding no longer uses it.

The test calls like:
```rust
let result = train::<B>(&device, file.path(), 5, &config, None);
```
Stay the same since `board_cards` is still in the signature for record counting.

The `spawn_dataloader_thread` test calls remove `board_cards`:
```rust
// Before:
let (rx, handles) = spawn_dataloader_thread(&[empty_path], &config, 0, 5);
// After:
let (rx, handles) = spawn_dataloader_thread(&[empty_path], &config, 0);
```

Wait, actually `spawn_dataloader_thread` still needs to know the val_count to skip validation records in the reader. Let me reconsider. The reader reads raw binary records — it doesn't need board_cards for reading (the record format is self-describing with board_size as the first byte). So we can drop `board_cards` from `spawn_dataloader_thread`.

**Step 7: Run tests**

Run: `cargo test -p cfvnet model::training -- --nocapture 2>&1 | tail -30`
Expected: All training tests pass

**Step 8: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "refactor(cfvnet): remove board_cards from encoding in training pipeline"
```

---

### Task 4: Update `river_net_evaluator.rs`

**Files:**
- Modify: `crates/cfvnet/src/eval/river_net_evaluator.rs`

**Step 1: Update imports and `build_input`**

Replace line 10:
```rust
use crate::model::network::{CfvNet, OUTPUT_SIZE, input_size};
```
With:
```rust
use crate::model::network::{CfvNet, DECK_SIZE, INPUT_SIZE, OUTPUT_SIZE};
```

Replace the `build_input` function (lines 67-96):
```rust
/// Build the input vector for a single river board evaluation.
///
/// Layout (2707 floats):
///   [0..1326)     — OOP range (1326 combo probabilities)
///   [1326..2652)  — IP range (1326 combo probabilities)
///   [2652..2704)  — one-hot board vector (52 binary floats)
///   [2704]        — pot / 400.0
///   [2705]        — effective_stack / 400.0
///   [2706]        — player indicator (0.0=OOP, 1.0=IP)
fn build_input(
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
        board_onehot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_onehot);
    input.push(pot as f32 / 400.0);
    input.push(effective_stack as f32 / 400.0);
    input.push(if traverser == 0 { 0.0 } else { 1.0 });
    debug_assert_eq!(input.len(), INPUT_SIZE);
    input
}
```

**Step 2: Update `evaluate` method**

In the `evaluate` method (line 117), replace:
```rust
let in_size = input_size(5);
```
With:
```rust
let in_size = INPUT_SIZE;
```

And update the tensor creation (line 174):
```rust
let data = TensorData::new(input_vec, [1, in_size]);
```
(This stays the same since `in_size` is now `INPUT_SIZE`.)

**Step 3: Update tests**

Replace all `input_size(5)` calls in tests with `INPUT_SIZE`. There are 4 occurrences in the test module (lines 227, 247, 269, 279).

**Step 4: Run tests**

Run: `cargo test -p cfvnet eval::river_net_evaluator -- --nocapture 2>&1 | tail -20`
Expected: All river_net_evaluator tests pass

**Step 5: Commit**

```bash
git add crates/cfvnet/src/eval/river_net_evaluator.rs
git commit -m "feat(cfvnet): use one-hot board encoding in river net evaluator"
```

---

### Task 5: Update `compare_turn.rs`

**Files:**
- Modify: `crates/cfvnet/src/eval/compare_turn.rs`

**Step 1: Update imports**

Replace line 28:
```rust
use crate::model::network::{CfvNet, input_size};
```
With:
```rust
use crate::model::network::{CfvNet, DECK_SIZE, INPUT_SIZE};
```

**Step 2: Update `predict_with_model`**

Replace lines 130-147:
```rust
fn predict_with_model(
    model: &CfvNet<B>,
    device: &<B as burn::tensor::backend::Backend>::Device,
    sit: &Situation,
    traverser: u8,
) -> Vec<f32> {
    let mut input = Vec::with_capacity(INPUT_SIZE);
    input.extend_from_slice(&sit.ranges[0]);
    input.extend_from_slice(&sit.ranges[1]);
    let mut board_onehot = [0.0_f32; DECK_SIZE];
    for &card in sit.board_cards() {
        board_onehot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_onehot);
    input.push(sit.pot as f32 / 400.0);
    input.push(sit.effective_stack as f32 / 400.0);
    input.push(f32::from(traverser));
    debug_assert_eq!(input.len(), INPUT_SIZE);

    let data = TensorData::new(input, [1, INPUT_SIZE]);
    let input_tensor = Tensor::<B, 2>::from_data(data, device);
    let output = model.forward(input_tensor);
    output.into_data().to_vec::<f32>().expect("output tensor conversion")
}
```

**Step 3: Update model creation calls**

Replace all `input_size(4)` and `input_size(5)` with `INPUT_SIZE` throughout the file. There are occurrences at lines 131, 223, 233, 303, 491, 494, 519, 537, 540, 575.

For `run_turn_comparison_net` (lines 223, 233):
```rust
    let turn_in_size = INPUT_SIZE;
    // ... and ...
    let river_in_size = INPUT_SIZE;
```

For `run_turn_comparison_exact` (line 303):
```rust
    let turn_in_size = INPUT_SIZE;
```

And similarly in all test functions.

**Step 4: Run tests**

Run: `cargo test -p cfvnet eval::compare_turn -- --nocapture 2>&1 | tail -20`
Expected: All compare_turn tests pass

**Step 5: Commit**

```bash
git add crates/cfvnet/src/eval/compare_turn.rs
git commit -m "feat(cfvnet): use one-hot board encoding in turn comparison"
```

---

### Task 6: Update `main.rs`

**Files:**
- Modify: `crates/cfvnet/src/main.rs`

**Step 1: Update `cmd_train`**

The call to `train` at lines 260, 268, 277 already passes `board_cards`. Keep those as-is since the `train` signature still accepts `board_cards` for record counting.

**Step 2: Update `cmd_evaluate`**

Replace line 464:
```rust
    use cfvnet::model::network::{CfvNet, input_size};
```
With:
```rust
    use cfvnet::model::network::{CfvNet, INPUT_SIZE, POT_INDEX};
```

Replace line 468:
```rust
    let in_size = input_size(board_cards);
```
With:
```rust
    let in_size = INPUT_SIZE;
```

Replace line 506 (pot offset):
```rust
        let pot = item.input[2 * 1326 + board_cards] * 400.0;
```
With:
```rust
        let pot = item.input[POT_INDEX] * 400.0;
```

**Step 3: Update `cmd_compare`**

Replace line 539:
```rust
    use cfvnet::model::network::{CfvNet, input_size};
```
With:
```rust
    use cfvnet::model::network::{CfvNet, INPUT_SIZE};
```

Replace line 543:
```rust
    let in_size = input_size(board_cards);
```
With:
```rust
    let in_size = INPUT_SIZE;
```

**Step 4: Run the full test suite**

Run: `cargo test -p cfvnet 2>&1 | tail -30`
Expected: All cfvnet tests pass

**Step 5: Commit**

```bash
git add crates/cfvnet/src/main.rs
git commit -m "feat(cfvnet): use INPUT_SIZE and POT_INDEX constants in CLI commands"
```

---

### Task 7: Run full test suite and clippy

**Step 1: Run all tests**

Run: `cargo test 2>&1 | tail -30`
Expected: All tests pass across all crates

**Step 2: Run clippy**

Run: `cargo clippy -p cfvnet 2>&1 | tail -30`
Expected: No warnings or errors

**Step 3: Clean up any unused imports**

If `board_cards_for_street` is no longer used in some files, or if `input_size` imports remain, remove them.

Check for unused `board_cards` variables that were only used for encoding (not for record counting or file operations).

**Step 4: Final commit if cleanup needed**

```bash
git add -A
git commit -m "chore(cfvnet): clean up unused imports after one-hot encoding change"
```
