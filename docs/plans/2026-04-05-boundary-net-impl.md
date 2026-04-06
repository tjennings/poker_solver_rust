# BoundaryNet Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add a `BoundaryNet` model to the cfvnet crate that outputs normalized EVs for use as a depth-boundary evaluator in the range-solver, then wire it into the Tauri explorer for turn solving.

**Architecture:** `BoundaryNet` is a sibling to `CfvNet` in `cfvnet::model`, sharing `HiddenBlock`. It uses normalized pot/stack inputs (`pot/(pot+stack)`, `stack/(pot+stack)`) and outputs normalized EVs (chip EV / total stake). A new `boundary_evaluator` module in `cfvnet::eval` adapts it for the range-solver's `evaluate_internal()` boundary path. The existing datagen and storage are reused — target conversion happens at encode time.

**Tech Stack:** Rust, burn (ML framework), range-solver crate, Tauri

**Design doc:** `docs/plans/2026-04-05-boundary-net-design.md`

---

### Task 1: BoundaryNet model struct

**Files:**
- Create: `crates/cfvnet/src/model/boundary_net.rs`
- Modify: `crates/cfvnet/src/model/mod.rs`

**Step 1: Write the failing test**

In `crates/cfvnet/src/model/boundary_net.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn boundary_net_output_shape() {
        let device = Default::default();
        let model = BoundaryNet::<TestBackend>::new(&device, 2, 64);
        let input = Tensor::<TestBackend, 2>::zeros([1, INPUT_SIZE], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, OUTPUT_SIZE]);
    }

    #[test]
    fn boundary_net_batch_forward() {
        let device = Default::default();
        let model = BoundaryNet::<TestBackend>::new(&device, 2, 64);
        let input = Tensor::<TestBackend, 2>::zeros([8, INPUT_SIZE], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [8, OUTPUT_SIZE]);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet boundary_net`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

`crates/cfvnet/src/model/boundary_net.rs`:

```rust
use burn::{
    module::Module,
    nn::Linear,
    nn::LinearConfig,
    tensor::{backend::Backend, Tensor},
};

use super::network::{HiddenBlock, INPUT_SIZE, OUTPUT_SIZE};

/// Boundary value network for depth-bounded range solving.
///
/// Same architecture as CfvNet (MLP with BatchNorm + PReLU hidden blocks),
/// but outputs normalized EVs: `chip_ev / (pot + effective_stack)`.
///
/// Input encoding differs: pot and stack are encoded as fractions of
/// `(pot + effective_stack)` rather than divided by 400.
#[derive(Module, Debug)]
pub struct BoundaryNet<B: Backend> {
    hidden: Vec<HiddenBlock<B>>,
    output: Linear<B>,
}

impl<B: Backend> BoundaryNet<B> {
    /// Build a new network with `num_layers` hidden blocks of width `hidden_size`.
    pub fn new(device: &B::Device, num_layers: usize, hidden_size: usize) -> Self {
        assert!(num_layers > 0, "need at least one hidden layer");

        let mut hidden = Vec::with_capacity(num_layers);
        hidden.push(HiddenBlock::new(device, INPUT_SIZE, hidden_size));
        for _ in 1..num_layers {
            hidden.push(HiddenBlock::new(device, hidden_size, hidden_size));
        }

        let output = LinearConfig::new(hidden_size, OUTPUT_SIZE).init(device);

        Self { hidden, output }
    }

    /// Forward pass: returns `[batch, OUTPUT_SIZE]` normalized EVs.
    pub fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        for block in &self.hidden {
            x = block.forward(x);
        }
        self.output.forward(x)
    }
}
```

Note: `HiddenBlock` is currently private in `network.rs`. Make it `pub(crate)`:

In `crates/cfvnet/src/model/network.rs`, change:
```rust
struct HiddenBlock<B: Backend> {
```
to:
```rust
pub(crate) struct HiddenBlock<B: Backend> {
```

Also make `HiddenBlock::new` and `HiddenBlock::forward` `pub(crate)`.

Add to `crates/cfvnet/src/model/mod.rs`:
```rust
pub mod boundary_net;
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p cfvnet boundary_net`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/src/model/boundary_net.rs crates/cfvnet/src/model/mod.rs crates/cfvnet/src/model/network.rs
git commit -m "feat(cfvnet): add BoundaryNet model struct"
```

---

### Task 2: BoundaryNet dataset encoding

**Files:**
- Create: `crates/cfvnet/src/model/boundary_dataset.rs`
- Modify: `crates/cfvnet/src/model/mod.rs`

**Step 1: Write the failing test**

In `crates/cfvnet/src/model/boundary_dataset.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::datagen::storage::TrainingRecord;
    use crate::model::network::{INPUT_SIZE, OUTPUT_SIZE};

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
        rec.cfvs[0] = 0.3; // pot-relative
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
        // pot=100, stack=150, total=250
        // pot_feature = 100/250 = 0.4
        // stack_feature = 150/250 = 0.6
        let pot_idx = POT_INDEX;
        assert!((item.input[pot_idx] - 0.4).abs() < 1e-6, "pot feature: {}", item.input[pot_idx]);
        assert!((item.input[pot_idx + 1] - 0.6).abs() < 1e-6, "stack feature: {}", item.input[pot_idx + 1]);
    }

    #[test]
    fn encode_normalizes_target() {
        let rec = sample_record();
        let item = encode_boundary_record(&rec);
        // cfv[0] = 0.3 (pot-relative), pot=100, total=250
        // target = 0.3 * 100 / 250 = 0.12
        assert!((item.target[0] - 0.12).abs() < 1e-6, "target[0]: {}", item.target[0]);
    }

    #[test]
    fn encode_normalizes_game_value() {
        let rec = sample_record();
        let item = encode_boundary_record(&rec);
        // game_value=0.05 (pot-relative), pot=100, total=250
        // normalized = 0.05 * 100 / 250 = 0.02
        assert!((item.game_value - 0.02).abs() < 1e-6, "game_value: {}", item.game_value);
    }

    #[test]
    fn encode_selects_correct_player_range() {
        let mut rec = sample_record();
        rec.player = 1;
        let item = encode_boundary_record(&rec);
        // Player 1 = IP, should use ip_range
        assert!((item.range[100] - 1.0).abs() < 1e-6);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet boundary_dataset`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

`crates/cfvnet/src/model/boundary_dataset.rs`:

```rust
use crate::datagen::storage::TrainingRecord;
use crate::model::network::{DECK_SIZE, INPUT_SIZE, NUM_RANKS, OUTPUT_SIZE, POT_INDEX};

/// A single training item for BoundaryNet with normalized EV targets.
#[derive(Debug, Clone)]
pub struct BoundaryItem {
    pub input: Vec<f32>,      // length INPUT_SIZE (2720)
    pub target: Vec<f32>,     // length OUTPUT_SIZE (1326) — normalized EVs
    pub mask: Vec<f32>,       // length OUTPUT_SIZE — 1.0 valid, 0.0 masked
    pub range: Vec<f32>,      // length OUTPUT_SIZE — player's range for aux loss
    pub game_value: f32,      // normalized game value for aux loss
}

/// Encode a TrainingRecord into a BoundaryItem with normalized pot/stack and targets.
pub fn encode_boundary_record(rec: &TrainingRecord) -> BoundaryItem {
    let total_stake = rec.pot + rec.effective_stack;
    // Guard against division by zero (shouldn't happen with valid data).
    let norm = if total_stake > 0.0 { total_stake } else { 1.0 };

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
        rank_presence[(card / 4) as usize] = 1.0;
    }
    input.extend_from_slice(&rank_presence);
    // Normalized pot and stack
    input.push(rec.pot / norm);
    input.push(rec.effective_stack / norm);
    // Player indicator
    input.push(f32::from(rec.player));

    debug_assert_eq!(input.len(), INPUT_SIZE);

    // Normalize targets: chip_ev / total_stake
    // chip_ev = cfv_pot_relative * pot
    let pot_over_norm = rec.pot / norm;
    let target: Vec<f32> = rec.cfvs.iter().map(|&cfv| cfv * pot_over_norm).collect();

    let mask: Vec<f32> = rec.valid_mask.iter()
        .map(|&v| if v != 0 { 1.0 } else { 0.0 })
        .collect();

    let range = if rec.player == 0 {
        rec.oop_range.to_vec()
    } else {
        rec.ip_range.to_vec()
    };

    let game_value = rec.game_value * pot_over_norm;

    BoundaryItem {
        input,
        target,
        mask,
        range,
        game_value,
    }
}
```

Add to `crates/cfvnet/src/model/mod.rs`:
```rust
pub mod boundary_dataset;
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p cfvnet boundary_dataset`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/src/model/boundary_dataset.rs crates/cfvnet/src/model/mod.rs
git commit -m "feat(cfvnet): add BoundaryNet dataset encoding with normalized EVs"
```

---

### Task 3: BoundaryNet training loop

**Files:**
- Create: `crates/cfvnet/src/model/boundary_training.rs`
- Modify: `crates/cfvnet/src/model/mod.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use crate::datagen::storage::{write_record, TrainingRecord};
    use std::io::Write;
    use tempfile::NamedTempFile;

    type B = Autodiff<NdArray>;

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
            for j in 0..10 {
                rec.cfvs[j] = (i as f32 + j as f32) * 0.01;
                rec.oop_range[j] = 0.1;
                rec.ip_range[j] = 0.1;
            }
            write_record(&mut file, &rec).unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    fn boundary_training_reduces_loss() {
        let file = write_test_data(16);
        let device = Default::default();
        let config = BoundaryTrainConfig {
            hidden_layers: 2,
            hidden_size: 64,
            batch_size: 16,
            epochs: 200,
            learning_rate: 0.001,
            lr_min: 0.001,
            huber_delta: 1.0,
            aux_loss_weight: 0.0,
            validation_split: 0.0,
            checkpoint_every_n_epochs: 0,
            shuffle_buffer_size: 100,
            prefetch_depth: 2,
            encoder_threads: 2,
        };
        let result = train_boundary::<B>(&device, file.path(), 5, &config, None);
        assert!(
            result.final_train_loss < 0.05,
            "should overfit small data, got loss {}",
            result.final_train_loss
        );
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet boundary_training`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

The training loop is structurally identical to `training.rs`, but uses `BoundaryNet` and `encode_boundary_record`. Rather than duplicating 500+ lines, refactor the shared parts:

1. Create `boundary_training.rs` that mirrors `training.rs` but imports `BoundaryNet` and `encode_boundary_record`
2. Replace `CfvNet` with `BoundaryNet` throughout
3. Replace `encode_record` with `encode_boundary_record`
4. The loss function (`cfvnet_loss`) is identical — it operates on whatever normalized space the targets are in

The `BoundaryTrainConfig` struct is identical to `TrainConfig`. Consider using a type alias or shared struct, but for now keep them separate for clarity.

Key differences from `training.rs`:
- Uses `BoundaryNet::new(device, config.hidden_layers, config.hidden_size)` (no `in_size` param)
- `PreEncoded::from_records` calls `encode_boundary_record` instead of `encode_record`
- Validation metrics print normalized MAE in addition to loss

**Step 4: Run tests to verify they pass**

Run: `cargo test -p cfvnet boundary_training`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/src/model/boundary_training.rs crates/cfvnet/src/model/mod.rs
git commit -m "feat(cfvnet): add BoundaryNet training loop"
```

---

### Task 4: Validation metrics — normalized MAE and per-SPR breakdown

**Files:**
- Modify: `crates/cfvnet/src/model/boundary_training.rs`
- Modify: `crates/cfvnet/src/eval/metrics.rs` (add normalized MAE metric)

**Step 1: Write the failing test**

In `crates/cfvnet/src/eval/metrics.rs`:

```rust
#[test]
fn normalized_mae_basic() {
    let pred = vec![0.1, 0.2, 0.0];
    let target = vec![0.12, 0.18, 0.0];
    let mask = vec![true, true, false];
    let mae = compute_normalized_mae(&pred, &target, &mask);
    // |0.1-0.12| + |0.2-0.18| = 0.02 + 0.02 = 0.04, /2 = 0.02
    assert!((mae - 0.02).abs() < 1e-6);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet normalized_mae`
Expected: FAIL

**Step 3: Implement `compute_normalized_mae` in `metrics.rs`**

```rust
/// Compute mean absolute error on normalized (pot-relative or stake-relative) values.
///
/// Only counts entries where mask is true. Returns 0.0 if no valid entries.
pub fn compute_normalized_mae(pred: &[f32], target: &[f32], mask: &[bool]) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0u64;
    for i in 0..pred.len().min(target.len()) {
        if mask.get(i).copied().unwrap_or(false) {
            sum += (pred[i] as f64 - target[i] as f64).abs();
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}
```

Then add normalized MAE reporting to `boundary_training.rs` validation pass — compute MAE alongside the loss at each epoch boundary.

**Step 4: Run tests**

Run: `cargo test -p cfvnet normalized_mae`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/src/eval/metrics.rs crates/cfvnet/src/model/boundary_training.rs
git commit -m "feat(cfvnet): add normalized MAE validation metric for BoundaryNet"
```

---

### Task 5: CLI commands — `train-boundary` and `eval-boundary`

**Files:**
- Modify: `crates/cfvnet/src/main.rs`

**Step 1: Add `TrainBoundary` and `EvalBoundary` variants to `Commands` enum**

```rust
/// Train the BoundaryNet model (normalized EV output for range-solver integration)
TrainBoundary {
    #[arg(short, long)]
    config: PathBuf,
    #[arg(short, long)]
    data: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long, default_value_t = default_backend())]
    backend: String,
},
/// Evaluate BoundaryNet on held-out data
EvalBoundary {
    #[arg(short, long)]
    model: PathBuf,
    #[arg(short, long)]
    data: PathBuf,
},
```

**Step 2: Implement `cmd_train_boundary` and `cmd_eval_boundary`**

`cmd_train_boundary` mirrors `cmd_train` but uses `BoundaryNet` and `boundary_training::train_boundary`.

`cmd_eval_boundary` mirrors `cmd_evaluate` but:
- Uses `BoundaryNet` for inference
- Reports normalized MAE instead of pot-relative MAE
- Breaks down MAE by SPR bucket

**Step 3: Wire into `main()` match arm**

**Step 4: Run**

Run: `cargo build -p cfvnet`
Expected: compiles

**Step 5: Commit**

```bash
git add crates/cfvnet/src/main.rs
git commit -m "feat(cfvnet): add train-boundary and eval-boundary CLI commands"
```

---

### Task 6: Boundary evaluator for range-solver

**Files:**
- Create: `crates/cfvnet/src/eval/boundary_evaluator.rs`
- Modify: `crates/cfvnet/src/eval/mod.rs`

This is the integration layer that adapts BoundaryNet for the range-solver's `evaluate_internal()` depth-boundary path.

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denormalize_produces_chip_ev() {
        let normalized = vec![0.12_f32, -0.04, 0.0];
        let pot = 100.0;
        let effective_stack = 150.0;
        let chip_ev = denormalize_ev(&normalized, pot, effective_stack);
        // 0.12 * 250 = 30.0
        assert!((chip_ev[0] - 30.0).abs() < 1e-4);
        // -0.04 * 250 = -10.0
        assert!((chip_ev[1] - (-10.0)).abs() < 1e-4);
        assert!((chip_ev[2] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn encode_boundary_input_from_ranges() {
        let oop_range = vec![0.5; 1326];
        let ip_range = vec![0.5; 1326];
        let board = vec![0u8, 4, 8, 12, 16];
        let pot = 100.0_f32;
        let effective_stack = 150.0_f32;
        let player = 0u8;
        let input = encode_boundary_inference_input(
            &oop_range, &ip_range, &board, pot, effective_stack, player,
        );
        assert_eq!(input.len(), INPUT_SIZE);
        // pot feature = 100/250 = 0.4
        assert!((input[POT_INDEX] - 0.4).abs() < 1e-6);
        // stack feature = 150/250 = 0.6
        assert!((input[POT_INDEX + 1] - 0.6).abs() < 1e-6);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet boundary_evaluator`
Expected: FAIL

**Step 3: Implement**

```rust
//! Boundary evaluator that adapts BoundaryNet for range-solver leaf evaluation.
//!
//! At depth-boundary nodes in the range-solver, this evaluator:
//! 1. Normalizes cfreach into ranges
//! 2. Encodes inputs with normalized pot/stack
//! 3. Runs BoundaryNet forward pass → normalized EVs
//! 4. Denormalizes to chip EVs: `chip_ev = normalized_ev * (pot + eff_stack)`

use crate::model::network::{DECK_SIZE, INPUT_SIZE, NUM_RANKS, OUTPUT_SIZE, POT_INDEX};

/// Encode inputs for boundary inference from raw game state.
pub fn encode_boundary_inference_input(
    oop_range: &[f32],
    ip_range: &[f32],
    board: &[u8],
    pot: f32,
    effective_stack: f32,
    player: u8,
) -> Vec<f32> {
    let total = pot + effective_stack;
    let norm = if total > 0.0 { total } else { 1.0 };

    let mut input = Vec::with_capacity(INPUT_SIZE);
    input.extend_from_slice(oop_range);
    input.extend_from_slice(ip_range);

    let mut board_onehot = [0.0_f32; DECK_SIZE];
    for &card in board {
        board_onehot[card as usize] = 1.0;
    }
    input.extend_from_slice(&board_onehot);

    let mut rank_presence = [0.0_f32; NUM_RANKS];
    for &card in board {
        rank_presence[(card / 4) as usize] = 1.0;
    }
    input.extend_from_slice(&rank_presence);

    input.push(pot / norm);
    input.push(effective_stack / norm);
    input.push(f32::from(player));

    debug_assert_eq!(input.len(), INPUT_SIZE);
    input
}

/// Convert normalized EVs back to chip EVs.
pub fn denormalize_ev(normalized: &[f32], pot: f32, effective_stack: f32) -> Vec<f32> {
    let total = pot + effective_stack;
    normalized.iter().map(|&v| v * total).collect()
}
```

Add to `crates/cfvnet/src/eval/mod.rs`:
```rust
pub mod boundary_evaluator;
```

**Step 4: Run tests**

Run: `cargo test -p cfvnet boundary_evaluator`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/src/eval/boundary_evaluator.rs crates/cfvnet/src/eval/mod.rs
git commit -m "feat(cfvnet): add boundary evaluator encoding and denormalization"
```

---

### Task 7: Integrate boundary evaluator with range-solver's PostFlopGame

**Files:**
- Modify: `crates/range-solver/src/game/evaluation.rs` (or the appropriate game impl)
- Modify: `crates/cfvnet/src/eval/boundary_evaluator.rs` (add full evaluator struct)

This task wires the BoundaryNet into the range-solver so it can be used as a leaf evaluator during turn solving. The existing depth-boundary code path in `evaluate_internal()` already handles boundary CFVs — we need to provide a `BoundaryEvaluator` struct that:

1. Holds a loaded `BoundaryNet` model
2. Accepts a game state (board, pot, stack, ranges, player)
3. Returns chip EVs per combo

**Step 1: Add `BoundaryEvaluator` struct that wraps model + forward pass**

```rust
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use burn::tensor::{Tensor, TensorData};
use crate::model::boundary_net::BoundaryNet;
use crate::model::network::INPUT_SIZE;

type B = NdArray;

pub struct BoundaryEvaluator {
    model: BoundaryNet<B>,
    device: <B as burn::tensor::backend::Backend>::Device,
}

impl BoundaryEvaluator {
    /// Load a trained BoundaryNet from a model directory.
    pub fn load(model_dir: &std::path::Path, hidden_layers: usize, hidden_size: usize) -> Result<Self, String> {
        let device = Default::default();
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        let model_path = if model_dir.is_dir() {
            model_dir.join("model")
        } else {
            model_dir.to_path_buf()
        };
        let model = BoundaryNet::<B>::new(&device, hidden_layers, hidden_size)
            .load_file(&model_path, &recorder, &device)
            .map_err(|e| format!("failed to load BoundaryNet: {e}"))?;
        Ok(Self { model, device })
    }

    /// Evaluate a single river boundary: returns chip EVs per combo (1326).
    pub fn evaluate(
        &self,
        oop_range: &[f32],
        ip_range: &[f32],
        board: &[u8],
        pot: f32,
        effective_stack: f32,
        player: u8,
    ) -> Vec<f32> {
        let input_data = encode_boundary_inference_input(
            oop_range, ip_range, board, pot, effective_stack, player,
        );
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(input_data, [1, INPUT_SIZE]),
            &self.device,
        );
        let normalized = self.model.forward(input);
        let normalized_vec: Vec<f32> = normalized.into_data().to_vec::<f32>().unwrap();
        denormalize_ev(&normalized_vec, pot, effective_stack)
    }

    /// Batch evaluate multiple river boundaries (e.g., 48 river cards).
    pub fn evaluate_batch(
        &self,
        inputs: &[Vec<f32>],
        pots: &[f32],
        effective_stacks: &[f32],
    ) -> Vec<Vec<f32>> {
        let n = inputs.len();
        if n == 0 {
            return vec![];
        }
        let flat: Vec<f32> = inputs.iter().flat_map(|v| v.iter().copied()).collect();
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(flat, [n, INPUT_SIZE]),
            &self.device,
        );
        let normalized = self.model.forward(input);
        let all_vals: Vec<f32> = normalized.into_data().to_vec::<f32>().unwrap();

        (0..n)
            .map(|i| {
                let start = i * 1326;
                let end = start + 1326;
                let total = pots[i] + effective_stacks[i];
                all_vals[start..end].iter().map(|&v| v * total).collect()
            })
            .collect()
    }
}
```

**Step 2: Write test for `BoundaryEvaluator`**

```rust
#[test]
fn evaluator_returns_correct_length() {
    // This test requires a saved model — use a small model trained in a temp dir.
    // For unit testing, test the encode/denormalize paths instead.
    // Integration test can be added later with a real model.
}
```

**Step 3: Commit**

```bash
git add crates/cfvnet/src/eval/boundary_evaluator.rs
git commit -m "feat(cfvnet): add BoundaryEvaluator struct with batch inference"
```

---

### Task 8: Tauri integration — wire boundary evaluator into explorer

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs`
- Potentially modify the Tauri commands that invoke range-solver

This task makes the BoundaryNet available in the explorer UI so users can solve turn spots with the network at river boundaries.

**Step 1: Understand current turn solving path in Tauri**

Read `crates/tauri-app/src/exploration.rs` to find where range-solver is invoked for turn solving and how leaf evaluators are passed in.

**Step 2: Add config option for boundary model path**

The existing `GameConfig` already has `river_model_path`. Reuse this field or add a `boundary_model_path` field. When set, the explorer loads a `BoundaryEvaluator` and passes it to the range-solver.

**Step 3: Wire the evaluator into the solve call**

When the user triggers a turn solve in the explorer:
1. If a boundary model is loaded, create a `BoundaryEvaluator`
2. Pass it to the range-solver as a leaf evaluator
3. The solver uses it at river boundary nodes instead of full-depth solving

**Step 4: Test manually**

Train a small boundary model, configure its path, and verify turn solving works in the explorer.

**Step 5: Commit**

```bash
git commit -m "feat(tauri): wire BoundaryNet into explorer for turn solving"
```

---

### Task 9: Post-training evaluation command with exploitability comparison

**Files:**
- Modify: `crates/cfvnet/src/main.rs`
- Create or modify: `crates/cfvnet/src/eval/boundary_compare.rs`

**Step 1: Add `CompareBoundary` CLI command**

Solves N reference river situations with:
- Full-depth range-solver (ground truth)
- Range-solver with BoundaryNet at leaves

Compares exploitability between the two.

**Step 2: Implement comparison logic**

For each reference situation:
1. Build a PostFlopGame for the river spot
2. Solve with full depth → get exploitability
3. Solve with boundary evaluator → get exploitability
4. Record the delta

Report: mean/max exploitability delta, per-SPR breakdown.

**Step 3: Commit**

```bash
git commit -m "feat(cfvnet): add compare-boundary command for exploitability validation"
```

---

### Task 10: Full workspace build and test verification

**Step 1: Build entire workspace**

Run: `cargo build`
Expected: clean build

**Step 2: Run all tests**

Run: `cargo test`
Expected: all pass, < 1 minute

**Step 3: Run clippy**

Run: `cargo clippy`
Expected: no warnings in changed code

**Step 4: Update architecture docs**

Update `docs/architecture.md` and `docs/training.md` with BoundaryNet information per CLAUDE.md instructions.

**Step 5: Final commit**

```bash
git commit -m "docs: update architecture and training docs for BoundaryNet"
```
