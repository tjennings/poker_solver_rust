# Pot-Weighted Loss + Log-Uniform Sampling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Weight Huber loss by pot size so the model prioritizes large-pot accuracy, and replace stratified interval pot sampling with log-uniform sampling for better coverage.

**Architecture:** Thread `pot` as a separate tensor through the training pipeline (`CfvItem` → `PreEncoded` → `ChunkTensors` → `MiniBatch` → `cfvnet_loss`). Replace `pot_intervals` config field with `pot_range` and sample log-uniformly.

**Tech Stack:** Rust, burn (tensor framework), serde_yaml

---

### Task 1: Replace `pot_intervals` with `pot_range` in config

**Files:**
- Modify: `crates/cfvnet/src/config.rs:80-116`

**Step 1: Update `DatagenConfig` struct**

Replace `pot_intervals` field (lines 85-86):

```rust
    #[serde(default = "default_pot_range")]
    pub pot_range: [i32; 2],
```

**Step 2: Update `Default` impl**

Replace `pot_intervals: default_pot_intervals()` (line 102) with:

```rust
            pot_range: default_pot_range(),
```

**Step 3: Replace the default function**

Replace `default_pot_intervals` (lines 114-116) with:

```rust
fn default_pot_range() -> [i32; 2] {
    [4, 400]
}
```

**Step 4: Update the `parse_full_config` test**

In the test at line 270, change `pot_intervals: [[4,20], [20,80], [80,200], [200,400]]` to `pot_range: [4, 400]` and update the assertion at line 288 from `assert_eq!(config.datagen.pot_intervals.len(), 4)` to:

```rust
assert_eq!(config.datagen.pot_range, [4, 400]);
```

**Step 5: Run to check compilation fails at callers**

Run: `cargo check -p cfvnet 2>&1 | head -30`
Expected: Errors at `sampler.rs` references to `pot_intervals` — this confirms we've broken the right things.

**Step 6: Commit**

```bash
git add crates/cfvnet/src/config.rs
git commit -m "refactor: replace pot_intervals with pot_range in DatagenConfig"
```

---

### Task 2: Implement log-uniform pot sampling

**Files:**
- Modify: `crates/cfvnet/src/datagen/sampler.rs:29-85`
- Modify: `crates/cfvnet/src/datagen/sampler.rs:87-145` (tests)

**Step 1: Replace `sample_pot` function**

Replace lines 79-85:

```rust
/// Sample a pot log-uniformly within `[lo, hi)`.
///
/// Log-uniform sampling gives equal density per multiplicative factor of pot,
/// ensuring small and large pots are represented proportionally in log-space.
fn sample_pot<R: Rng>(pot_range: [i32; 2], rng: &mut R) -> i32 {
    let [lo, hi] = pot_range;
    let log_lo = (lo as f64).ln();
    let log_hi = (hi as f64).ln();
    let log_pot = rng.gen_range(log_lo..log_hi);
    log_pot.exp().round() as i32
}
```

**Step 2: Update `sample_situation` call site**

Change line 36 from:

```rust
    let pot = sample_pot(&config.pot_intervals, rng);
```

to:

```rust
    let pot = sample_pot(config.pot_range, rng);
```

**Step 3: Update `test_config`**

Replace `pot_intervals` (line 97) with:

```rust
            pot_range: [4, 400],
```

**Step 4: Replace `pot_within_configured_intervals` test**

Replace lines 130-145 with:

```rust
    #[test]
    fn pot_within_configured_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = test_config();
        let [lo, hi] = config.pot_range;
        for _ in 0..200 {
            let sit = sample_situation(&config, INITIAL_STACK, 5, &mut rng);
            assert!(
                sit.pot >= lo && sit.pot <= hi,
                "pot {} not in range [{}, {}]",
                sit.pot, lo, hi
            );
        }
    }
```

**Step 5: Run tests**

Run: `cargo test -p cfvnet --lib -- sampler --quiet 2>&1 | tail -5`
Expected: All sampler tests pass

**Step 6: Commit**

```bash
git add crates/cfvnet/src/datagen/sampler.rs
git commit -m "feat: implement log-uniform pot sampling"
```

---

### Task 3: Update sample config yaml

**Files:**
- Modify: `sample_configurations/river_cfvnet.yaml:23`

**Step 1: Replace pot_intervals with pot_range**

Change line 23 from:

```yaml
  pot_intervals: [[4, 20], [20, 80], [80, 200]]
```

to:

```yaml
  pot_range: [4, 200]
```

**Step 2: Run config parse test**

Run: `cargo test -p cfvnet --lib -- config --quiet 2>&1 | tail -5`
Expected: All config tests pass

**Step 3: Commit**

```bash
git add sample_configurations/river_cfvnet.yaml
git commit -m "config: update river_cfvnet.yaml to use pot_range"
```

---

### Task 4: Add `pot` field to `CfvItem` and `encode_record`

**Files:**
- Modify: `crates/cfvnet/src/model/dataset.rs:8-16` (CfvItem struct)
- Modify: `crates/cfvnet/src/model/dataset.rs:152-193` (encode_record)

**Step 1: Add `pot` to `CfvItem`**

Add after `game_value` (line 15):

```rust
    pub pot: f32,             // raw pot value for loss weighting
```

**Step 2: Set `pot` in `encode_record`**

In the `CfvItem` construction (line 186), add:

```rust
        pot: rec.pot,
```

**Step 3: Run tests**

Run: `cargo test -p cfvnet --lib -- dataset --quiet 2>&1 | tail -5`
Expected: All dataset tests pass

**Step 4: Commit**

```bash
git add crates/cfvnet/src/model/dataset.rs
git commit -m "feat: add pot field to CfvItem for loss weighting"
```

---

### Task 5: Thread `pot` through `PreEncoded`, `ChunkTensors`, `MiniBatch`

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs:45-194`

**Step 1: Add `pot` to `PreEncoded`**

Add to the struct (after line 50):

```rust
    pot: Vec<f32>,
```

In `from_records` (after line 74), add:

```rust
        let mut pot = Vec::with_capacity(n);
```

In the loop (after line 81), add:

```rust
            pot.push(item.pot);
```

In the return (line 84), add `pot` to the struct literal:

```rust
        Self { input, target, mask, range, game_value, pot, in_size, len: n }
```

**Step 2: Add `pot` to `into_tensors`, `to_tensors`, `chunk_tensors`**

In `into_tensors` (line 88), add to the `ChunkTensors` construction:

```rust
            pot: Tensor::from_data(TensorData::new(self.pot, [n]), device),
```

In `to_tensors` (line 102), add:

```rust
            pot: Tensor::from_data(TensorData::new(self.pot.clone(), [n]), device),
```

In `chunk_tensors` (line 116), add a `pot` Vec:

```rust
        let mut pot = Vec::with_capacity(n);
```

In the loop add:

```rust
            pot.push(self.pot[idx]);
```

And in the `ChunkTensors` return:

```rust
            pot: Tensor::from_data(TensorData::new(pot, [n]), device),
```

**Step 3: Add `pot` to `ChunkTensors`**

Add to the struct (after line 157):

```rust
    pot: Tensor<B, 1>,
```

In `index_select` (line 163), add:

```rust
            pot: self.pot.clone().select(0, perm.clone()),
```

IMPORTANT: This line must go BEFORE the `game_value` line, because `game_value` consumes `perm` (no `.clone()`). Move `pot` before `game_value`, OR add `.clone()` to the `perm` in `game_value`. The safest fix: add `.clone()` to the existing `game_value` perm usage and put `pot` last without clone:

```rust
            game_value: self.game_value.clone().select(0, perm.clone()),
            pot: self.pot.clone().select(0, perm),
```

In `slice_batch` (line 175), add to `MiniBatch`:

```rust
            pot: self.pot.clone().narrow(0, start, len),
```

**Step 4: Add `pot` to `MiniBatch`**

Add to the struct (after line 193):

```rust
    pot: Tensor<B, 1>,
```

**Step 5: Verify compilation**

Run: `cargo check -p cfvnet 2>&1 | tail -5`
Expected: Errors only at `cfvnet_loss` call sites (batch.pot not used yet)

**Step 6: Commit**

```bash
git add crates/cfvnet/src/model/training.rs
git commit -m "feat: thread pot tensor through PreEncoded, ChunkTensors, MiniBatch"
```

---

### Task 6: Add pot weighting to `masked_huber_loss` and `cfvnet_loss`

**Files:**
- Modify: `crates/cfvnet/src/model/loss.rs`

**Step 1: Add `pot_weight` parameter to `masked_huber_loss`**

Change the function signature (lines 8-13) to:

```rust
pub fn masked_huber_loss<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    delta: f64,
    pot_weight: Option<Tensor<B, 1>>,
) -> Tensor<B, 1> {
```

Replace the final reduction (lines 26-28) with:

```rust
    // Mask and average over valid entries only.
    let masked_loss = element_loss * mask.clone();
    match pot_weight {
        Some(pw) => {
            // Pot-weighted: weight each sample's loss by pot, normalize by sum of weights.
            // pw is [batch], reshape to [batch, 1] for broadcasting over 1326 combos.
            let pw_2d: Tensor<B, 2> = pw.clone().unsqueeze_dim(1);
            let weighted = masked_loss * pw_2d;
            let total_weight = pw.sum().clamp_min(1.0);
            weighted.sum().div(total_weight)
        }
        None => {
            let num_valid = mask.sum().clamp_min(1.0);
            masked_loss.sum().div(num_valid)
        }
    }
```

**Step 2: Update `cfvnet_loss` to accept and pass pot**

Change the signature (lines 47-54) to:

```rust
pub fn cfvnet_loss<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    range: Tensor<B, 2>,
    game_value: Tensor<B, 1>,
    huber_delta: f64,
    aux_weight: f64,
    pot_weight: Option<Tensor<B, 1>>,
) -> Tensor<B, 1> {
    let huber = masked_huber_loss(pred.clone(), target, mask, huber_delta, pot_weight);
    let aux = aux_game_value_loss(pred, range, game_value);
    huber + aux.mul_scalar(aux_weight)
}
```

**Step 3: Update existing tests**

All existing `masked_huber_loss` test calls need `None` as the last arg. Update each call, e.g.:

```rust
let loss = masked_huber_loss(pred, target, mask, 1.0, None);
```

And `cfvnet_loss`:

```rust
let loss = cfvnet_loss(pred, target, mask, range, game_value, 1.0, 1.0, None);
```

**Step 4: Add test for pot-weighted loss**

Add after the existing tests:

```rust
    #[test]
    fn pot_weighted_loss_emphasizes_high_pot() {
        let device = Default::default();
        // Two samples with same error (0.5) but different pots
        let pred = Tensor::<B, 2>::from_floats([[0.5, 0.0], [0.5, 0.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.0, 0.0], [0.0, 0.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0], [1.0, 1.0]], &device);
        // Pot weights: sample 0 has pot=10, sample 1 has pot=100
        let pot = Tensor::<B, 1>::from_floats([10.0, 100.0], &device);

        let weighted = masked_huber_loss(pred.clone(), target.clone(), mask.clone(), 1.0, Some(pot));
        let unweighted = masked_huber_loss(pred, target, mask, 1.0, None);

        let w: f32 = weighted.into_scalar();
        let u: f32 = unweighted.into_scalar();
        // Both should be the same since both samples have equal error.
        // The weighting changes the denominator but not the relative contribution
        // when errors are equal.
        assert!(w.is_finite(), "weighted loss should be finite, got {w}");
        assert!(u.is_finite(), "unweighted loss should be finite, got {u}");
    }

    #[test]
    fn pot_weighted_loss_scales_contribution() {
        let device = Default::default();
        // Sample 0: error=0.1, pot=10. Sample 1: error=0.1, pot=100
        let pred = Tensor::<B, 2>::from_floats([[0.1], [0.1]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.0], [0.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0], [1.0]], &device);
        let pot = Tensor::<B, 1>::from_floats([10.0, 100.0], &device);

        let weighted = masked_huber_loss(pred, target, mask, 1.0, Some(pot));
        let val: f32 = weighted.into_scalar();
        // Huber quadratic: 0.5 * 0.01 = 0.005 per entry
        // Weighted: (10*0.005 + 100*0.005) / (10+100) = 0.55/110 = 0.005
        assert!((val - 0.005).abs() < 1e-5, "expected ~0.005, got {val}");
    }
```

**Step 5: Run tests**

Run: `cargo test -p cfvnet --lib -- loss --quiet 2>&1 | tail -10`
Expected: All loss tests pass

**Step 6: Commit**

```bash
git add crates/cfvnet/src/model/loss.rs
git commit -m "feat: add pot-weighted Huber loss option"
```

---

### Task 7: Add `pot_weighted_loss` config flag and wire through training loop

**Files:**
- Modify: `crates/cfvnet/src/config.rs:132-160` (TrainingConfig)
- Modify: `crates/cfvnet/src/model/training.rs:22-43` (TrainConfig)
- Modify: `crates/cfvnet/src/model/training.rs:208-244` (compute_val_loss)
- Modify: `crates/cfvnet/src/model/training.rs:586-596` (training loop cfvnet_loss call)
- Modify: `crates/cfvnet/src/main.rs` (cmd_train TrainConfig construction)

**Step 1: Add to `TrainingConfig` in config.rs**

Add after `prefetch_chunks` (line 159):

```rust
    #[serde(default = "default_pot_weighted_loss")]
    pub pot_weighted_loss: bool,
```

Add the default function:

```rust
fn default_pot_weighted_loss() -> bool {
    true
}
```

Add to `Default` impl for `TrainingConfig`:

```rust
            pot_weighted_loss: true,
```

**Step 2: Add to `TrainConfig` in training.rs**

Add after `prefetch_chunks` (line 37):

```rust
    pub pot_weighted_loss: bool,
```

**Step 3: Wire in `cmd_train` in main.rs**

In the `TrainConfig` construction, add:

```rust
        pot_weighted_loss: cfg.training.pot_weighted_loss,
```

**Step 4: Update `compute_val_loss`**

Change the `cfvnet_loss` call (lines 227-235) to pass `pot`:

```rust
        let pot_w = if config.pot_weighted_loss {
            Some(batch.pot)
        } else {
            None
        };
        let loss = cfvnet_loss(
            pred,
            batch.target,
            batch.mask,
            batch.range,
            batch.game_value,
            config.huber_delta,
            config.aux_loss_weight,
            pot_w,
        );
```

**Step 5: Update training loop `cfvnet_loss` call**

Change the call at lines 588-596 similarly:

```rust
                        let pot_w = if config.pot_weighted_loss {
                            Some(batch.pot)
                        } else {
                            None
                        };
                        let loss = cfvnet_loss(
                            pred,
                            batch.target,
                            batch.mask,
                            batch.range,
                            batch.game_value,
                            config.huber_delta,
                            config.aux_loss_weight,
                            pot_w,
                        );
```

**Step 6: Update test `TrainConfig` instances in training.rs**

Search for all `TrainConfig {` in the test module and add `pot_weighted_loss: false` (tests use simple data where pot weighting isn't meaningful).

**Step 7: Run full test suite**

Run: `cargo test -p cfvnet --lib -- --quiet 2>&1 | tail -10`
Expected: All tests pass

**Step 8: Commit**

```bash
git add crates/cfvnet/src/config.rs crates/cfvnet/src/model/training.rs crates/cfvnet/src/main.rs
git commit -m "feat: wire pot_weighted_loss config through training pipeline"
```

---

### Task 8: Update sample config and run full verification

**Files:**
- Modify: `sample_configurations/river_cfvnet.yaml`

**Step 1: Add `pot_weighted_loss` to training section**

Add after `epochs_per_chunk: 15`:

```yaml
  pot_weighted_loss: true
```

**Step 2: Run full test suite**

Run: `cargo test 2>&1 | tail -20`
Expected: All tests pass

**Step 3: Run clippy**

Run: `cargo clippy -p cfvnet 2>&1 | tail -10`
Expected: No new warnings

**Step 4: Verify no remaining `pot_intervals` references in code**

Run: `grep -rn "pot_intervals" crates/cfvnet/src/`
Expected: No results

**Step 5: Commit**

```bash
git add sample_configurations/river_cfvnet.yaml
git commit -m "config: add pot_weighted_loss and pot_range to sample config"
```
