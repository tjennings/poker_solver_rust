# CFVnet Architecture Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace scalar board encoding with 52-dim one-hot and add hard zero-sum constraint network matching Supremus architecture.

**Architecture:** Two changes to the CFVnet MLP: (1) board cards encoded as a 52-dim binary vector instead of N scalars, (2) network outputs both players' CFVs (2×1326) with a differentiable zero-sum correction layer. Training records change to store both players per situation. The aux loss is removed; zero-sum is enforced architecturally.

**Tech Stack:** Rust, burn (tensor framework), bytemuck (binary serialization), range-solver (DCFR)

---

### Task 1: Update TrainingRecord format

**Files:**
- Modify: `crates/cfvnet/src/datagen/storage.rs`

**Step 1: Write the failing test**

Add to `storage.rs` tests:

```rust
#[test]
fn record_v2_roundtrip() {
    let mut oop_cfvs = [0.0f32; 1326];
    let mut ip_cfvs = [0.0f32; 1326];
    oop_cfvs[0] = 0.123;
    ip_cfvs[100] = -0.456;

    let rec = TrainingRecord {
        board: vec![0, 4, 8, 12, 16],
        pot: 100.0,
        effective_stack: 50.0,
        oop_range: [0.0; 1326],
        ip_range: [0.0; 1326],
        oop_cfvs,
        ip_cfvs,
        valid_mask: [1; 1326],
    };

    let mut file = NamedTempFile::new().unwrap();
    write_record(&mut file, &rec).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();

    let loaded = read_record(&mut file).unwrap();
    assert_eq!(rec.board, loaded.board);
    assert_eq!(rec.pot, loaded.pot);
    assert_eq!(rec.oop_cfvs[0], loaded.oop_cfvs[0]);
    assert_eq!(rec.ip_cfvs[100], loaded.ip_cfvs[100]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet record_v2_roundtrip`
Expected: FAIL — `TrainingRecord` doesn't have `oop_cfvs`/`ip_cfvs` fields yet.

**Step 3: Update TrainingRecord struct and serialization**

In `storage.rs`, replace the struct and functions:

```rust
pub const fn record_size(board_size: usize) -> usize {
    1 // board_size prefix byte
    + board_size
    + 4  // pot: f32
    + 4  // effective_stack: f32
    + NUM_COMBOS * 4  // oop_range
    + NUM_COMBOS * 4  // ip_range
    + NUM_COMBOS * 4  // oop_cfvs
    + NUM_COMBOS * 4  // ip_cfvs
    + NUM_COMBOS      // valid_mask
}

pub struct TrainingRecord {
    pub board: Vec<u8>,
    pub pot: f32,
    pub effective_stack: f32,
    pub oop_range: [f32; NUM_COMBOS],
    pub ip_range: [f32; NUM_COMBOS],
    pub oop_cfvs: [f32; NUM_COMBOS],
    pub ip_cfvs: [f32; NUM_COMBOS],
    pub valid_mask: [u8; NUM_COMBOS],
}
```

Remove `player: u8` and `game_value: f32` fields. Replace `cfvs` with `oop_cfvs` and `ip_cfvs`.

Update `write_record`: remove player/game_value writes, write both cfv arrays.
Update `read_record`: remove player/game_value reads, read both cfv arrays.

**Step 4: Fix all existing tests in storage.rs**

Update `sample_record()` helper, `round_trip_single_record`, `round_trip_multiple_records`, `record_size_is_consistent`, etc. to use new fields. Remove any assertions on `player` or `game_value`.

**Step 5: Run tests to verify they pass**

Run: `cargo test -p cfvnet storage`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/cfvnet/src/datagen/storage.rs
git commit -m "feat: update TrainingRecord to store both players' CFVs"
```

---

### Task 2: Update datagen to write one record per situation

**Files:**
- Modify: `crates/cfvnet/src/datagen/generate.rs`

**Step 1: Update write_chunk to produce one record per situation**

Replace the two-record write (OOP + IP) with a single record:

```rust
fn write_chunk(
    situations: &[super::sampler::Situation],
    results: Vec<Option<SolveResult>>,
    writer: &mut BufWriter<std::fs::File>,
) -> Result<(), String> {
    for (sit, result) in situations.iter().zip(results) {
        let result = match result {
            Some(r) => r,
            None => continue,
        };

        let valid_mask = bool_mask_to_u8(&result.valid_mask);
        let board_vec = sit.board_cards().to_vec();

        let rec = TrainingRecord {
            board: board_vec,
            pot: sit.pot as f32,
            effective_stack: sit.effective_stack as f32,
            oop_range: sit.ranges[0],
            ip_range: sit.ranges[1],
            oop_cfvs: result.oop_evs,
            ip_cfvs: result.ip_evs,
            valid_mask,
        };
        write_record(writer, &rec).map_err(|e| format!("write: {e}"))?;
    }
    Ok(())
}
```

Update the doc comment on `generate_training_data` from "writing two records (OOP + IP)" to "writing one record per solved situation".

**Step 2: Fix generate.rs tests**

In `generate_small_batch`: change assertion from `count <= 8` (4 sits × 2 players) to `count <= 4` (4 sits × 1 record). Remove the `count % 2 == 0` assertion. Remove `rec.player` assertion.

**Step 3: Run tests**

Run: `cargo test -p cfvnet generate`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/cfvnet/src/datagen/generate.rs
git commit -m "feat: write one record per situation with both players' CFVs"
```

---

### Task 3: 52-dim one-hot board encoding

**Files:**
- Modify: `crates/cfvnet/src/model/network.rs`
- Modify: `crates/cfvnet/src/model/dataset.rs`

**Step 1: Write failing test for one-hot encoding**

Add to `dataset.rs` tests:

```rust
#[test]
fn dataset_board_one_hot_encoding() {
    let file = write_test_data(1);
    let dataset = CfvDataset::from_file(file.path(), 5).unwrap();
    let item = dataset.get(0).unwrap();

    // Board cards are [0, 4, 8, 12, 16] in test data.
    // One-hot: 52 dims starting at index 2652.
    let board_start = 2652;
    for i in 0..52 {
        let expected = if [0, 4, 8, 12, 16].contains(&i) { 1.0 } else { 0.0 };
        assert!(
            (item.input[board_start + i] - expected).abs() < 1e-6,
            "board one-hot[{i}] = {}, expected {expected}",
            item.input[board_start + i]
        );
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet dataset_board_one_hot`
Expected: FAIL — encoding still uses scalar card_id / 51.0

**Step 3: Update input_size in network.rs**

```rust
/// Input feature size.
///
/// Layout: OOP range (1326) + IP range (1326) + board (52) + pot (1) + stack (1) = 2706
pub const INPUT_SIZE: usize = NUM_COMBOS + NUM_COMBOS + 52 + 1 + 1;

/// Compute input feature size.
///
/// Board is always encoded as a 52-dim one-hot regardless of street.
pub fn input_size(_board_cards: usize) -> usize {
    INPUT_SIZE
}
```

Note: `board_cards` parameter is kept for API compatibility but ignored. The input is always 2706. Remove the player indicator (1) — it's gone with zero-sum.

Update existing tests: `input_size_correct_for_river` → assert 2706, `input_size_correct_for_turn` → assert 2706, `input_size_correct_for_flop` → assert 2706. Update `RIVER_INPUT` constant to 2706.

**Step 4: Update encode_record in dataset.rs**

Replace the board encoding section:

```rust
// Board cards: 52-dim one-hot
let mut board_one_hot = [0.0f32; 52];
for &card in &rec.board[..board_cards] {
    board_one_hot[card as usize] = 1.0;
}
input.extend_from_slice(&board_one_hot);
```

Remove the player indicator push. The encoding is now:
`[OOP_range(1326), IP_range(1326), board_one_hot(52), pot(1), stack(1)]`

**Step 5: Update encode_situation_for_inference**

Same change — replace scalar card encoding with one-hot, remove player parameter:

```rust
pub fn encode_situation_for_inference(sit: &Situation) -> Vec<f32> {
    let in_size = input_size(sit.board_size);
    let mut input = Vec::with_capacity(in_size);
    for &v in &sit.ranges[0] { input.push(v); }
    for &v in &sit.ranges[1] { input.push(v); }
    // 52-dim one-hot board
    let mut board_one_hot = [0.0f32; 52];
    for &card in sit.board_cards() {
        board_one_hot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_one_hot);
    input.push(sit.pot as f32 / 400.0);
    input.push(sit.effective_stack as f32 / 400.0);
    debug_assert_eq!(input.len(), in_size);
    input
}
```

**Step 6: Update CfvItem — remove game_value, add both targets**

```rust
pub struct CfvItem {
    pub input: Vec<f32>,
    pub oop_target: Vec<f32>,
    pub ip_target: Vec<f32>,
    pub mask: Vec<f32>,
    pub oop_range: Vec<f32>,
    pub ip_range: Vec<f32>,
    pub pot: f32,
}
```

Update `encode_record` to populate new fields from the updated `TrainingRecord`.

**Step 7: Fix all dataset.rs tests**

Update `dataset_input_encoding_correct` for new layout (one-hot at 2652, pot at 2704, stack at 2705). Update `dataset_input_size_method` (both 4 and 5 card → 2706). Update `dataset_turn_encoding_uses_4_board_cards` for one-hot. Remove `dataset_player_range_selection` (no player field). Update `dataset_get_returns_valid_item` for new CfvItem fields.

**Step 8: Run all tests**

Run: `cargo test -p cfvnet dataset`
Expected: PASS

**Step 9: Commit**

```bash
git add crates/cfvnet/src/model/network.rs crates/cfvnet/src/model/dataset.rs
git commit -m "feat: 52-dim one-hot board encoding, remove player indicator"
```

---

### Task 4: Zero-sum constraint network

**Files:**
- Modify: `crates/cfvnet/src/model/network.rs`

**Step 1: Write the failing test**

Add to `network.rs` tests:

```rust
#[test]
fn zero_sum_holds_after_forward() {
    let device = Default::default();
    let model = CfvNet::<TestBackend>::new(&device, 2, 64, INPUT_SIZE);

    // Random-ish ranges that sum to ~1
    let mut range_oop_data = vec![0.0f32; NUM_COMBOS];
    let mut range_ip_data = vec![0.0f32; NUM_COMBOS];
    for i in 0..100 {
        range_oop_data[i] = 0.01;
        range_ip_data[i] = 0.01;
    }

    let input = Tensor::<TestBackend, 2>::zeros([1, INPUT_SIZE], &device);
    let range_oop = Tensor::<TestBackend, 2>::from_floats([range_oop_data.as_slice()], &device);
    let range_ip = Tensor::<TestBackend, 2>::from_floats([range_ip_data.as_slice()], &device);

    let output = model.forward(input, range_oop.clone(), range_ip.clone());
    assert_eq!(output.dims(), [1, 2 * NUM_COMBOS]);

    // Split and check zero-sum
    let oop_cfv = output.clone().narrow(1, 0, NUM_COMBOS);
    let ip_cfv = output.narrow(1, NUM_COMBOS, NUM_COMBOS);
    let gv_oop: f32 = (oop_cfv * range_oop).sum().into_scalar();
    let gv_ip: f32 = (ip_cfv * range_ip).sum().into_scalar();
    let error = (gv_oop + gv_ip).abs();
    assert!(error < 1e-5, "zero-sum violated: gv_oop={gv_oop}, gv_ip={gv_ip}, error={error}");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet zero_sum_holds`
Expected: FAIL — `forward` doesn't take range arguments yet.

**Step 3: Implement the zero-sum network**

Update `CfvNet` in `network.rs`:

```rust
pub const DUAL_OUTPUT_SIZE: usize = 2 * NUM_COMBOS;

#[derive(Module, Debug)]
pub struct CfvNet<B: Backend> {
    hidden: Vec<HiddenBlock<B>>,
    output: Linear<B>,
}

impl<B: Backend> CfvNet<B> {
    pub fn new(device: &B::Device, num_layers: usize, hidden_size: usize, in_size: usize) -> Self {
        assert!(num_layers > 0, "need at least one hidden layer");
        let mut hidden = Vec::with_capacity(num_layers);
        hidden.push(HiddenBlock::new(device, in_size, hidden_size));
        for _ in 1..num_layers {
            hidden.push(HiddenBlock::new(device, hidden_size, hidden_size));
        }
        let output = LinearConfig::new(hidden_size, DUAL_OUTPUT_SIZE).init(device);
        Self { hidden, output }
    }

    /// Forward pass with zero-sum correction.
    ///
    /// Returns `[batch, 2652]` — first 1326 are OOP CFVs, last 1326 are IP CFVs.
    /// The correction ensures `dot(range_oop, cfv_oop) + dot(range_ip, cfv_ip) == 0`.
    pub fn forward(
        &self,
        mut x: Tensor<B, 2>,
        range_oop: Tensor<B, 2>,
        range_ip: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        for block in &self.hidden {
            x = block.forward(x);
        }
        let raw = self.output.forward(x);

        // Split into OOP and IP halves.
        let oop_raw = raw.clone().narrow(1, 0, NUM_COMBOS);
        let ip_raw = raw.narrow(1, NUM_COMBOS, NUM_COMBOS);

        // Compute game values: gv = sum(range * cfv) per sample.
        let gv_oop: Tensor<B, 1> = (oop_raw.clone() * range_oop.clone()).sum_dim(1).squeeze(1);
        let gv_ip: Tensor<B, 1> = (ip_raw.clone() * range_ip.clone()).sum_dim(1).squeeze(1);

        // Zero-sum error: should be 0 in a zero-sum game.
        let error: Tensor<B, 1> = (gv_oop + gv_ip).div_scalar(2.0);

        // Correction: subtract error / sum(range) from each player's CFVs.
        let sum_oop: Tensor<B, 1> = range_oop.clone().sum_dim(1).squeeze::<1>(1).clamp_min(1e-8);
        let sum_ip: Tensor<B, 1> = range_ip.clone().sum_dim(1).squeeze::<1>(1).clamp_min(1e-8);

        let corr_oop: Tensor<B, 2> = (error.clone() / sum_oop).unsqueeze_dim(1);
        let corr_ip: Tensor<B, 2> = (error / sum_ip).unsqueeze_dim(1);

        let oop_corrected = oop_raw - corr_oop;
        let ip_corrected = ip_raw - corr_ip;

        Tensor::cat(vec![oop_corrected, ip_corrected], 1)
    }
}
```

**Step 4: Update existing network tests**

All existing tests that call `model.forward(input)` need to pass `range_oop` and `range_ip`. For tests that don't care about zero-sum, pass zero ranges. Update `RIVER_INPUT` to `INPUT_SIZE` (2706). Update output shape assertions from `OUTPUT_SIZE` (1326) to `DUAL_OUTPUT_SIZE` (2652).

**Step 5: Run tests**

Run: `cargo test -p cfvnet network`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/cfvnet/src/model/network.rs
git commit -m "feat: zero-sum constraint network with dual-player output"
```

---

### Task 5: Update loss function

**Files:**
- Modify: `crates/cfvnet/src/model/loss.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn dual_player_loss_sums_both() {
    let device = Default::default();
    // pred and target: [batch=1, 2652]
    let pred = Tensor::<B, 2>::zeros([1, 2652], &device);
    let mut target_data = vec![0.0f32; 2652];
    target_data[0] = 0.5;       // OOP error
    target_data[1326] = 0.5;    // IP error
    let target = Tensor::<B, 2>::from_data(TensorData::new(target_data, [1, 2652]), &device);
    let mask = Tensor::<B, 2>::ones([1, 1326], &device);
    let loss = cfvnet_loss(pred, target, mask, 1.0, None);
    let val: f32 = loss.into_scalar();
    assert!(val > 0.0, "dual loss should be positive");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet dual_player_loss`
Expected: FAIL — `cfvnet_loss` signature doesn't match yet.

**Step 3: Simplify cfvnet_loss**

```rust
/// Combined CFVnet loss for dual-player output.
///
/// `pred` and `target` are `[batch, 2652]` — first 1326 OOP, last 1326 IP.
/// `mask` is `[batch, 1326]` — same mask applies to both players.
/// Returns sum of Huber losses for both players.
pub fn cfvnet_loss<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    huber_delta: f64,
    pot_weight: Option<Tensor<B, 1>>,
) -> Tensor<B, 1> {
    let pred_oop = pred.clone().narrow(1, 0, 1326);
    let pred_ip = pred.narrow(1, 1326, 1326);
    let tgt_oop = target.clone().narrow(1, 0, 1326);
    let tgt_ip = target.narrow(1, 1326, 1326);

    let loss_oop = masked_huber_loss(
        pred_oop, tgt_oop, mask.clone(), huber_delta, pot_weight.clone(),
    );
    let loss_ip = masked_huber_loss(
        pred_ip, tgt_ip, mask, huber_delta, pot_weight,
    );
    loss_oop + loss_ip
}
```

Remove `aux_game_value_loss` function. Remove `range`, `game_value`, `aux_weight` parameters from `cfvnet_loss`.

**Step 4: Update existing loss tests**

Remove `aux_loss_*` tests. Update `combined_loss_includes_both_terms` to use new signature. Keep `masked_huber_loss` tests unchanged (they don't depend on cfvnet_loss). Keep pot-weighted tests.

**Step 5: Run tests**

Run: `cargo test -p cfvnet loss`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/cfvnet/src/model/loss.rs
git commit -m "feat: dual-player loss, remove aux game value loss"
```

---

### Task 6: Update training loop

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs`

**Step 1: Update PreEncoded, ChunkTensors, MiniBatch**

Replace `range`, `game_value` fields with `oop_target`, `ip_target`, `oop_range`, `ip_range`:

`PreEncoded` fields:
```rust
struct PreEncoded {
    input: Vec<f32>,
    oop_target: Vec<f32>,
    ip_target: Vec<f32>,
    mask: Vec<f32>,
    oop_range: Vec<f32>,
    ip_range: Vec<f32>,
    pot: Vec<f32>,
    in_size: usize,
    len: usize,
}
```

`ChunkTensors` fields:
```rust
struct ChunkTensors<B: Backend> {
    input: Tensor<B, 2>,
    oop_target: Tensor<B, 2>,
    ip_target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    oop_range: Tensor<B, 2>,
    ip_range: Tensor<B, 2>,
    pot: Tensor<B, 1>,
    len: usize,
}
```

`MiniBatch` — same field changes.

Update `from_records`, `into_tensors`, `to_tensors`, `chunk_tensors`, `index_select`, `slice_batch` to thread the new fields through.

**Step 2: Update TrainConfig**

Remove `aux_loss_weight` field.

**Step 3: Update training loop forward pass**

In the training loop, change:
```rust
let pred = model.forward(batch.input);
```
to:
```rust
// Concatenate targets for loss
let target = Tensor::cat(vec![batch.oop_target, batch.ip_target], 1);
let pred = model.forward(batch.input, batch.oop_range, batch.ip_range);
```

Update `cfvnet_loss` call to new signature (remove `range`, `game_value`, `aux_loss_weight`).

**Step 4: Update compute_val_loss similarly**

Same changes for the validation loss computation.

**Step 5: Fix training.rs tests**

Update all `TrainConfig` constructions to remove `aux_loss_weight`. Update test helper `write_test_data` to create records with new format. Update any assertions.

**Step 6: Run tests**

Run: `cargo test -p cfvnet training`
Expected: PASS

**Step 7: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "feat: update training loop for dual-player zero-sum network"
```

---

### Task 7: Update config (remove aux_loss_weight)

**Files:**
- Modify: `crates/cfvnet/src/config.rs`

**Step 1: Remove aux_loss_weight from TrainingConfig**

Remove the `aux_loss_weight` field, its serde default function, and the Default impl entry.

**Step 2: Update config tests if any reference it**

Check and fix `parse_full_config` test.

**Step 3: Run tests**

Run: `cargo test -p cfvnet config`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/cfvnet/src/config.rs
git commit -m "config: remove aux_loss_weight (replaced by zero-sum constraint)"
```

---

### Task 8: Update CLI and evaluation commands

**Files:**
- Modify: `crates/cfvnet/src/main.rs`
- Modify: `crates/cfvnet/src/eval/compare.rs`

**Step 1: Update cmd_compare in main.rs**

The predict_fn closure currently calls `encode_situation_for_inference(sit, 0)` and `model.forward(input)`. Update to:

```rust
let summary = run_comparison(&cfg.game, &cfg.datagen, num_spots, cfg.datagen.seed, |sit, _solve_result| {
    let input_data = encode_situation_for_inference(sit);
    let input = Tensor::<B, 2>::from_data(
        TensorData::new(input_data, [1, in_size]),
        &device,
    );

    // Build range tensors for zero-sum correction
    let range_oop = Tensor::<B, 2>::from_data(
        TensorData::new(sit.ranges[0].to_vec(), [1, 1326]),
        &device,
    );
    let range_ip = Tensor::<B, 2>::from_data(
        TensorData::new(sit.ranges[1].to_vec(), [1, 1326]),
        &device,
    );

    let output = model.forward(input, range_oop, range_ip);
    let all_cfvs = output.into_data().to_vec::<f32>().unwrap();
    // Return OOP CFVs (first 1326)
    all_cfvs[..1326].to_vec()
});
```

Remove `encode_situation_for_inference` player parameter at all call sites.

**Step 2: Update cmd_train to remove aux_loss_weight**

Remove `aux_loss_weight: cfg.training.aux_loss_weight` from the `TrainConfig` construction.

**Step 3: Update cmd_compare_net and cmd_compare_exact similarly**

Same pattern — update forward call to include ranges, slice output.

**Step 4: Run cargo check**

Run: `cargo check -p cfvnet`
Expected: No errors

**Step 5: Commit**

```bash
git add crates/cfvnet/src/main.rs crates/cfvnet/src/eval/compare.rs
git commit -m "feat: update CLI commands for dual-player zero-sum network"
```

---

### Task 9: Update river_net_evaluator (turn datagen)

**Files:**
- Modify: `crates/cfvnet/src/eval/river_net_evaluator.rs`

**Step 1: Update forward call**

The evaluator runs the river model for each possible river card during turn solving. Update the forward call to pass range tensors and extract the correct player's CFVs from the dual output.

Find the `model.forward(input_tensor)` call and update:
```rust
// Build range tensors
let range_oop = Tensor::<B, 2>::from_data(
    TensorData::new(oop_range_vec.clone(), [1, 1326]),
    &self.device,
);
let range_ip = Tensor::<B, 2>::from_data(
    TensorData::new(ip_range_vec.clone(), [1, 1326]),
    &self.device,
);
let output = self.model.forward(input_tensor, range_oop, range_ip);
// Extract OOP or IP CFVs based on player
let all_cfvs = output.into_data().to_vec::<f32>().unwrap();
let cfvs = if player == 0 {
    &all_cfvs[..1326]
} else {
    &all_cfvs[1326..]
};
```

**Step 2: Run cargo check**

Run: `cargo check -p cfvnet`
Expected: No errors

**Step 3: Commit**

```bash
git add crates/cfvnet/src/eval/river_net_evaluator.rs
git commit -m "feat: update river_net_evaluator for dual-player network"
```

---

### Task 10: Update integration test

**Files:**
- Modify: `crates/cfvnet/tests/integration_test.rs`

**Step 1: Update the integration test**

Update `full_pipeline_smoke_test`:
- Remove `pot_weighted_loss` and `aux_loss_weight` from `TrainConfig` (if they were there)
- Add `pot_weighted_loss: false` if still needed
- Verify the test generates data with the new record format and trains successfully

```rust
let train_config = cfvnet::model::training::TrainConfig {
    hidden_layers: 2,
    hidden_size: 32,
    batch_size: 8,
    epochs: 10,
    learning_rate: 0.001,
    lr_min: 0.001,
    huber_delta: 1.0,
    validation_split: 0.0,
    checkpoint_every_n_epochs: 0,
    gpu_chunk_size: 100,
    epochs_per_chunk: 1,
    prefetch_chunks: 1,
    pot_weighted_loss: false,
};
```

**Step 2: Run the integration test**

Run: `cargo test -p cfvnet --test integration_test`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/cfvnet/tests/integration_test.rs
git commit -m "test: update integration test for new architecture"
```

---

### Task 11: Update sample configs and docs

**Files:**
- Modify: `sample_configurations/river_cfvnet.yaml`
- Modify: `docs/training.md`

**Step 1: Remove aux_loss_weight from sample config**

Remove the `aux_loss_weight: 1.0` line from `sample_configurations/river_cfvnet.yaml` if present.

**Step 2: Update docs/training.md**

Note the breaking change: new record format requires data regeneration. Update any references to player indicator or aux loss.

**Step 3: Commit**

```bash
git add sample_configurations/river_cfvnet.yaml docs/training.md
git commit -m "docs: update config and training docs for architecture changes"
```

---

### Task 12: Full test suite verification

**Step 1: Run complete test suite**

Run: `cargo test`
Expected: All tests pass

**Step 2: Run clippy**

Run: `cargo clippy`
Expected: No new warnings

**Step 3: Verify test suite runs in < 1 minute**

Time the test run. If it exceeds 1 minute, investigate and fix.
