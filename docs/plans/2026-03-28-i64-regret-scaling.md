# i64 Regrets with ×100,000 Scaling — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Switch regret/prediction/baseline storage from AtomicI32 to AtomicI64 with ×100,000 fixed-point scaling to restore sub-chip precision lost in the scaling refactor.

**Architecture:** All three atomic buffers (regrets, predictions, baselines) become `Vec<AtomicI64>`. A single constant `REGRET_SCALE = 100_000.0` controls the fixed-point conversion. Scaling happens at the boundary: writes multiply by REGRET_SCALE, reads divide. The `CfrOptimizer` trait signature updates to `&[AtomicI64]`. Serialization changes from `Vec<i32>` to `Vec<i64>` (breaks old snapshots).

**Tech Stack:** Rust, std::sync::atomic::AtomicI64, rayon (parallel discount), bincode (serialization)

---

### Task 1: Storage buffers — AtomicI32 → AtomicI64

**Files:**
- Modify: `crates/core/src/blueprint_v2/storage.rs`

**Step 1: Write failing tests for the new types**

In the existing `#[cfg(test)]` module, add these tests:

```rust
#[test]
fn regret_scale_constant_exists() {
    // REGRET_SCALE should be 100_000.0
    assert!((super::REGRET_SCALE - 100_000.0).abs() < f64::EPSILON);
}

#[timed_test]
fn add_regret_i64_round_trip() {
    let tree = toy_tree();
    let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);
    let node_idx = tree
        .nodes
        .iter()
        .position(|n| matches!(n, GameNode::Decision { .. }))
        .expect("need a decision node") as u32;
    // Add a delta of 0.5 chips (scaled: 50_000)
    storage.add_regret(node_idx, 0, 0, 50_000);
    let raw = storage.regrets[storage.slot_index(node_idx, 0, 0)].load(Ordering::Relaxed);
    assert_eq!(raw, 50_000, "raw i64 should be 50_000 (0.5 chips × 100k)");
}

#[timed_test]
fn add_regret_i64_no_overflow() {
    let tree = toy_tree();
    let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);
    let node_idx = tree
        .nodes
        .iter()
        .position(|n| matches!(n, GameNode::Decision { .. }))
        .expect("need a decision node") as u32;
    // Accumulate large values that would overflow i32 but not i64
    let big = 5_000_000_000_i64; // 5 billion, exceeds i32::MAX
    storage.add_regret(node_idx, 0, 0, big);
    storage.add_regret(node_idx, 0, 0, big);
    let raw = storage.regrets[storage.slot_index(node_idx, 0, 0)].load(Ordering::Relaxed);
    assert_eq!(raw, 10_000_000_000, "should accumulate to 10 billion without overflow");
}
```

**Step 2: Run tests — should fail (REGRET_SCALE doesn't exist, types mismatch)**

Run: `cargo test -p poker-solver-core add_regret_i64 -- --nocapture`

**Step 3: Make the changes**

In `storage.rs`:

1. Replace constants:
```rust
// DELETE these two lines:
pub const REGRET_FLOOR: i32 = -310_000_000;
pub const REGRET_CAP: i32 = 310_000_000;

// ADD:
/// Fixed-point scaling factor for regret/prediction/baseline storage.
/// Multiply chip-unit values by this before storing as i64.
/// Divide by this when reading back to f64.
pub const REGRET_SCALE: f64 = 100_000.0;
```

2. Change struct fields:
```rust
pub struct BlueprintStorage {
    /// Cumulative regrets: one `AtomicI64` per (decision node, bucket, action).
    /// Stored as fixed-point: chip_value × REGRET_SCALE.
    pub regrets: Vec<AtomicI64>,
    /// Strategy sums: one `AtomicI64` per (decision node, bucket, action).
    pub strategy_sums: Vec<AtomicI64>,
    /// Optional baseline buffer for VR-MCCFR variance reduction.
    /// Stored as fixed-point: chip_value × REGRET_SCALE.
    pub(crate) baselines: Option<Vec<AtomicI64>>,
    /// Optional prediction buffer for SAPCFR+.
    /// Stored as fixed-point: chip_value × REGRET_SCALE.
    pub(crate) predictions: Option<Vec<AtomicI64>>,
    // ... rest unchanged
}
```

3. Update `new()`: change `AtomicI32::new(0)` → `AtomicI64::new(0)` for regrets
4. Update `new_with_baselines()`: change `AtomicI32::new(0)` → `AtomicI64::new(0)` for baselines
5. Update `enable_predictions()`: change `AtomicI32::new(0)` → `AtomicI64::new(0)` for predictions

6. Update `add_regret()`:
```rust
#[inline]
pub fn add_regret(&self, node_idx: u32, bucket: u16, action: usize, delta: i64) {
    let nl = &self.layout[node_idx as usize];
    let idx = Self::slot_offset(nl, bucket) + action;
    self.regrets[idx].fetch_add(delta, Ordering::Relaxed);
}
```
(Remove all clamping logic — i64 won't overflow in practice.)

7. Update `get_regret()` return type:
```rust
pub fn get_regret(&self, node_idx: u32, bucket: u16, action: usize) -> i64 {
    let nl = &self.layout[node_idx as usize];
    let idx = Self::slot_offset(nl, bucket) + action;
    self.regrets[idx].load(Ordering::Relaxed)
}
```

8. Update `get_baseline()` — scale on read:
```rust
pub fn get_baseline(&self, node_idx: u32, bucket: u16, action: usize) -> f64 {
    self.baselines
        .as_ref()
        .map(|b| {
            b[self.slot_index(node_idx, bucket, action)].load(Ordering::Relaxed) as f64
                / REGRET_SCALE
        })
        .unwrap_or(0.0)
}
```

9. Update `update_baseline()` — scale on write:
```rust
pub fn update_baseline(
    &self, node_idx: u32, bucket: u16, action: usize, value: f64, alpha: f64,
) {
    if let Some(ref b) = self.baselines {
        let idx = self.slot_index(node_idx, bucket, action);
        let old = b[idx].load(Ordering::Relaxed) as f64 / REGRET_SCALE;
        let new_val = old * (1.0 - alpha) + value * alpha;
        b[idx].store((new_val * REGRET_SCALE) as i64, Ordering::Relaxed);
    }
}
```

10. Update `get_prediction()`:
```rust
pub fn get_prediction(&self, node_idx: u32, bucket: u16, action: usize) -> i64 {
    self.predictions
        .as_ref()
        .map_or(0, |p| p[self.slot_index(node_idx, bucket, action)].load(Ordering::Relaxed))
}
```

11. Update `set_prediction()`:
```rust
pub fn set_prediction(&self, node_idx: u32, bucket: u16, action: usize, value: i64) {
    if let Some(ref p) = self.predictions {
        p[self.slot_index(node_idx, bucket, action)].store(value, Ordering::Relaxed);
    }
}
```

12. Update `save_regrets()` serialization:
```rust
let regrets_plain: Vec<i64> = self
    .regrets
    .iter()
    .map(|a| a.load(Ordering::Relaxed))
    .collect();
// ...
let payload = (&self.bucket_counts, &regrets_plain, &sums_plain);
```

13. Update `load_regrets()` deserialization:
```rust
let (stored_counts, regrets_plain, sums_plain): ([u16; 4], Vec<i64>, Vec<i64>) =
    bincode::deserialize_from(reader)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
```

14. Update `humanize_bytes` print in `new()`: change `total * 4` to `total * 8` for regret bytes.

15. Update ALL existing tests in the `#[cfg(test)]` module: change `AtomicI32` → `AtomicI64`, `i32` → `i64`, `f64::from(v)` → `v as f64` where loading from AtomicI64. For baseline tests, values need to be scaled by REGRET_SCALE. For the old clamping tests, DELETE them (clamping is removed).

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core -- storage`
Expected: All storage tests pass

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/storage.rs
git commit -m "feat: switch regret/prediction/baseline storage to AtomicI64 with ×100k scaling"
```

---

### Task 2: CfrOptimizer trait — AtomicI32 → AtomicI64

**Files:**
- Modify: `crates/core/src/cfr/optimizer.rs`

**Step 1: Write failing test**

```rust
#[test]
fn dcfr_discount_i64_preserves_precision() {
    let regrets = vec![AtomicI64::new(50_000), AtomicI64::new(-30_000)]; // 0.5 and -0.3 chips
    let strategy_sums = vec![AtomicI64::new(100)];
    let opt = DcfrOptimizer { alpha: 1.5, beta: 0.5, gamma: 2.0 };
    opt.apply_discount(&regrets, &strategy_sums, None, 10);
    let r0 = regrets[0].load(Ordering::Relaxed);
    assert!(r0 > 0 && r0 < 50_000, "positive regret should be discounted, got {r0}");
    let r1 = regrets[1].load(Ordering::Relaxed);
    assert!(r1 < 0, "negative regret should still be negative, got {r1}");
}

#[test]
fn sapcfr_discount_i64_floors_negative() {
    let regrets = vec![AtomicI64::new(50_000), AtomicI64::new(-30_000)];
    let strategy_sums = vec![AtomicI64::new(100)];
    let opt = SapcfrPlusOptimizer { alpha: 1.5, gamma: 2.0, eta: 0.5 };
    opt.apply_discount(&regrets, &strategy_sums, None, 10);
    let r1 = regrets[1].load(Ordering::Relaxed);
    assert_eq!(r1, 0, "SAPCFR+ should floor negative regrets to 0, got {r1}");
}

#[test]
fn sapcfr_current_strategy_i64_with_predictions() {
    // Two actions: regrets 300k and 100k (3.0 and 1.0 chips scaled)
    // Predictions: 50k and -50k (0.5 and -0.5 chips scaled)
    let regrets = vec![AtomicI64::new(300_000), AtomicI64::new(100_000)];
    let preds = vec![AtomicI64::new(50_000), AtomicI64::new(-50_000)];
    let opt = SapcfrPlusOptimizer { alpha: 1.5, gamma: 2.0, eta: 0.5 };
    let mut out = [0.0; 2];
    opt.current_strategy(&regrets, Some(&preds), 0, 2, &mut out);
    // predicted = R + eta * v:
    //   action 0: 300000 + 0.5 * 50000 = 325000
    //   action 1: 100000 + 0.5 * (-50000) = 75000
    // strategy = [325000/400000, 75000/400000] = [0.8125, 0.1875]
    assert!((out[0] - 0.8125).abs() < 0.001, "expected ~0.8125, got {}", out[0]);
    assert!((out[1] - 0.1875).abs() < 0.001, "expected ~0.1875, got {}", out[1]);
}
```

**Step 2: Run tests — should fail (type mismatch)**

Run: `cargo test -p poker-solver-core -- optimizer`

**Step 3: Update the trait and implementations**

Change every `&[AtomicI32]` to `&[AtomicI64]` in:
- `CfrOptimizer` trait: `apply_discount` and `current_strategy` signatures
- `DcfrOptimizer::apply_discount`: change `f64::from(v)` → `v as f64`, `as i32` → `as i64`
- `DcfrOptimizer::current_strategy`: change `f64::from(r)` → `r as f64`
- `SapcfrPlusOptimizer::apply_discount`: change `f64::from(v)` → `v as f64`, `as i32` → `as i64`, `.max(0)` → `.max(0)`
- `SapcfrPlusOptimizer::current_strategy`: change `f64::from(r)` → `r as f64`, `f64::from(v)` → `v as f64`

Update ALL existing tests: `AtomicI32` → `AtomicI64` everywhere, adjust test values if needed.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core -- optimizer`
Expected: All pass

**Step 5: Commit**

```bash
git add crates/core/src/cfr/optimizer.rs
git commit -m "feat: update CfrOptimizer trait to AtomicI64 regrets"
```

---

### Task 3: MCCFR traversal — scale deltas on write

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`

**Step 1: Write failing test**

In mccfr.rs tests, add:

```rust
#[timed_test]
fn regret_delta_uses_scale_factor() {
    use super::super::storage::REGRET_SCALE;
    // After a traversal, regrets should be scaled by REGRET_SCALE.
    // Build a minimal game and run one iteration.
    let tree = super::tests::build_preflop_tree();
    let storage = BlueprintStorage::new(&tree, [169, 50, 50, 50]);
    // ... (use existing test setup pattern from the file)
    // After traversal, check that regret values are in the scaled range
    // (i.e., a 1-chip delta should produce ~100_000 in storage, not ~1).
    let any_nonzero = storage.regrets.iter().any(|r| r.load(Ordering::Relaxed).abs() > 1000);
    assert!(any_nonzero, "regrets should be scaled by REGRET_SCALE, not raw chip units");
}
```

Note: Use the existing test infrastructure in mccfr.rs. The exact test depends on what helpers exist. The key assertion is that stored regret magnitudes are ~REGRET_SCALE × chip_delta, not raw chip values.

**Step 2: Run test — should fail (deltas still stored as raw)**

**Step 3: Update mccfr.rs**

At `crates/core/src/blueprint_v2/mccfr.rs`, around line 834:

```rust
// BEFORE:
let delta = av - node_value;
let delta_i32 = delta as i32;
storage.add_regret(node_idx, bucket, a, delta_i32);
storage.set_prediction(node_idx, bucket, a, delta_i32);

// AFTER:
use super::storage::REGRET_SCALE;
let delta = av - node_value;
let delta_scaled = (delta * REGRET_SCALE) as i64;
storage.add_regret(node_idx, bucket, a, delta_scaled);
storage.set_prediction(node_idx, bucket, a, delta_scaled);
```

Also update any other place in `mccfr.rs` that calls `add_regret` or `set_prediction` with raw values.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core -- mccfr`
Expected: All pass

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/mccfr.rs
git commit -m "feat: scale regret deltas by REGRET_SCALE (×100k) before storage"
```

---

### Task 4: Trainer — scale-aware display and prune threshold

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

**Step 1: Write failing test**

```rust
#[timed_test]
fn prune_threshold_is_scaled() {
    use super::super::storage::REGRET_SCALE;
    // A config threshold of -5 chips should match scaled regrets of -500_000
    let mut config = test_config();
    config.training.prune_threshold = -5;
    let mut trainer = BlueprintTrainer::new(config).unwrap();
    // Store a regret of -3 chips (scaled: -300_000) — should be ABOVE threshold
    // Store a regret of -6 chips (scaled: -600_000) — should be BELOW threshold
    if trainer.storage.regrets.len() >= 2 {
        trainer.storage.regrets[0].store((-3.0 * REGRET_SCALE) as i64, Ordering::Relaxed);
        trainer.storage.regrets[1].store((-6.0 * REGRET_SCALE) as i64, Ordering::Relaxed);
        let frac = trainer.prune_fraction();
        // Only 1 of 2 entries is below -5 chips, so fraction ≈ depends on total entries
        // Just verify it's > 0 and < 1
        assert!(frac > 0.0 && frac < 1.0, "prune_fraction should detect scaled threshold, got {frac}");
    }
}

#[timed_test]
fn min_max_regret_in_chip_units() {
    use super::super::storage::REGRET_SCALE;
    let config = test_config();
    let mut trainer = BlueprintTrainer::new(config).unwrap();
    if trainer.storage.regrets.len() >= 2 {
        trainer.storage.regrets[0].store((10.0 * REGRET_SCALE) as i64, Ordering::Relaxed);
        trainer.storage.regrets[1].store((-3.0 * REGRET_SCALE) as i64, Ordering::Relaxed);
        let max = trainer.max_regret();
        let min = trainer.min_regret();
        assert!((max - 10.0).abs() < 0.01, "max_regret should be in chip units, got {max}");
        assert!((min - (-3.0)).abs() < 0.01, "min_regret should be in chip units, got {min}");
    }
}
```

**Step 2: Run tests — should fail**

Run: `cargo test -p poker-solver-core prune_threshold_is_scaled min_max_regret_in_chip_units`

**Step 3: Update trainer.rs**

1. `min_regret()` — divide by REGRET_SCALE:
```rust
pub fn min_regret(&self) -> f64 {
    let min_raw = self.storage.regrets.iter()
        .map(|atom| atom.load(Ordering::Relaxed))
        .min()
        .unwrap_or(0);
    min_raw as f64 / storage::REGRET_SCALE
}
```

2. `max_regret()` — same pattern, divide by REGRET_SCALE.

3. `avg_pos_regret()` — load as i64, accumulate in f64, divide final result by REGRET_SCALE.

4. `mean_positive_regret()` — same: load as i64, convert to f64, divide by REGRET_SCALE.

5. `prune_fraction()` — scale the threshold:
```rust
pub fn prune_fraction(&self) -> f64 {
    let threshold = (self.config.training.prune_threshold as f64 * storage::REGRET_SCALE) as i64;
    // ... rest same but comparing i64 values
}
```

6. Prune threshold in batch loop (~line 517):
```rust
let threshold = (self.config.training.prune_threshold as f64 * storage::REGRET_SCALE) as i64;
```

This also requires updating the `prune_threshold` parameter type passed to `traverse_traverser` and related functions from `i32` to `i64`. Check the function signatures in `mccfr.rs` for `prune_threshold` parameter and update accordingly.

7. Inline DCFR fallback (~line 783-786): change `f64::from(v)` → `v as f64`, `as i32` → `as i64`.

8. Update ALL existing trainer tests that store raw regret values: multiply stored values by REGRET_SCALE.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core -- trainer`
Expected: All pass

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/trainer.rs crates/core/src/blueprint_v2/mccfr.rs
git commit -m "feat: scale-aware prune threshold and display in trainer"
```

---

### Task 5: Baseline scaling test

**Files:**
- Modify: `crates/core/src/blueprint_v2/storage.rs` (tests only)

**Step 1: Write tests for baseline round-trip with scaling**

```rust
#[timed_test]
fn baseline_ema_scaled_round_trip() {
    let tree = toy_tree();
    let storage = BlueprintStorage::new_with_baselines(&tree, [50, 50, 50, 50], true);
    let node_idx = tree.nodes.iter()
        .position(|n| matches!(n, GameNode::Decision { .. }))
        .expect("need a decision node") as u32;
    // Write a baseline of 5.0 chips with alpha=1.0 (full replace)
    storage.update_baseline(node_idx, 0, 0, 5.0, 1.0);
    let b = storage.get_baseline(node_idx, 0, 0);
    assert!((b - 5.0).abs() < 0.01, "expected ~5.0 chips, got {b}");

    // EMA update: old=5.0, new value=10.0, alpha=0.5 → result = 7.5
    storage.update_baseline(node_idx, 0, 0, 10.0, 0.5);
    let b = storage.get_baseline(node_idx, 0, 0);
    assert!((b - 7.5).abs() < 0.01, "expected ~7.5 chips after EMA, got {b}");
}

#[timed_test]
fn baseline_small_values_not_truncated() {
    let tree = toy_tree();
    let storage = BlueprintStorage::new_with_baselines(&tree, [50, 50, 50, 50], true);
    let node_idx = tree.nodes.iter()
        .position(|n| matches!(n, GameNode::Decision { .. }))
        .expect("need a decision node") as u32;
    // Write a small baseline of 0.003 chips — this was the bug with i32 no-scaling
    storage.update_baseline(node_idx, 0, 0, 0.003, 1.0);
    let b = storage.get_baseline(node_idx, 0, 0);
    assert!((b - 0.003).abs() < 0.0001, "small baseline should survive scaling, got {b}");
}
```

**Step 2: Run tests**

Run: `cargo test -p poker-solver-core baseline_ema_scaled baseline_small_values`
Expected: PASS (if Task 1 was done correctly)

**Step 3: Commit**

```bash
git add crates/core/src/blueprint_v2/storage.rs
git commit -m "test: verify baseline EMA round-trip with i64 ×100k scaling"
```

---

### Task 6: E2E test and full suite

**Files:**
- Modify: `crates/core/tests/blueprint_v2_e2e.rs` (if needed for type changes)

**Step 1: Run the full test suite**

Run: `cargo test`
Expected: All pass. Fix any remaining i32/i64 type mismatches in e2e tests or other crates.

**Step 2: Run clippy**

Run: `cargo clippy -p poker-solver-core -p poker-solver-trainer`
Expected: No new warnings.

**Step 3: Final commit**

```bash
git add -A
git commit -m "fix: resolve remaining i32→i64 type mismatches across crates"
```
