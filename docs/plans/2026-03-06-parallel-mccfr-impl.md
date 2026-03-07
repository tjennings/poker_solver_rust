# Parallel Blueprint V2 MCCFR — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Parallelize Blueprint V2 MCCFR training to use all CPU cores via atomic shared buffers and Rayon batched execution.

**Architecture:** Replace `Vec<i32>` / `Vec<i64>` storage with `Vec<AtomicI32>` / `Vec<AtomicI64>`, change `traverse_external` from `&mut` to `&` (shared reference), batch 200 deals per iteration and process them via `rayon::par_iter` with thread-local RNGs.

**Tech Stack:** Rust, `std::sync::atomic` (`AtomicI32`, `AtomicI64`, `Ordering::Relaxed`), `rayon` (already in deps), `rand::rngs::SmallRng`

---

## Context

**Crate:** `poker-solver-core` at `crates/core/`

**Key files being modified:**
- `crates/core/src/blueprint_v2/storage.rs` — flat-buffer regret/strategy storage
- `crates/core/src/blueprint_v2/mccfr.rs` — external-sampling MCCFR traversal
- `crates/core/src/blueprint_v2/trainer.rs` — training loop driver
- `crates/core/src/blueprint_v2/config.rs` — training config with `batch_size`
- `crates/core/src/blueprint_v2/bundle.rs` — snapshot serialization (reads storage)

**Key files that read storage (will need minor updates):**
- `crates/trainer/src/blueprint_tui_scenarios.rs` — TUI reads `storage.bucket_counts` and calls `storage.current_strategy()`, `storage.average_strategy()`, `storage.num_actions()`. These read-only methods will still work after the atomic conversion (they return owned `Vec`, not slices).

**Build & test command:**
```bash
cargo test -p poker-solver-core --lib -- blueprint_v2  # core tests only
cargo test -p poker-solver-trainer                      # trainer tests
cargo clippy -p poker-solver-core -p poker-solver-trainer  # lint
```

**Important:** `rayon` is already in `Cargo.toml` for `poker-solver-core`. `rand` is already a dependency (version 0.9.2). `SmallRng` is in `rand::rngs::SmallRng`.

---

### Task 1: Add `batch_size` to TrainingConfig

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs:72-100`

This is a standalone config change. Nothing else depends on it yet.

**Step 1: Add field and default function**

In `config.rs`, add to `TrainingConfig` struct (after line 99, before the closing `}`):

```rust
/// Number of iterations per parallel batch.
#[serde(default = "default_batch_size")]
pub batch_size: u64,
```

Add the default function (after `default_print_every` around line 145):

```rust
const fn default_batch_size() -> u64 {
    200
}
```

**Step 2: Fix compile errors**

Every place that constructs a `TrainingConfig` directly needs the new field. There are two places:

In `config.rs` tests (`test_serialize_round_trip`, around line 268):
```rust
batch_size: 200,
```

In `trainer.rs` tests (`toy_config`, around line 467):
```rust
batch_size: 200,
```

**Step 3: Add a deserialize test**

In `config.rs` test `test_deserialize_toy_config`, the YAML doesn't include `batch_size`, so the default should kick in. Add this assertion (around line 235):

```rust
assert_eq!(cfg.training.batch_size, default_batch_size());
```

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core --lib -- blueprint_v2::config
```
Expected: all config tests pass.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/config.rs crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat(blueprint_v2): add batch_size to TrainingConfig (default 200)"
```

---

### Task 2: Convert BlueprintStorage to Atomic Types

**Files:**
- Modify: `crates/core/src/blueprint_v2/storage.rs` (full rewrite of struct + methods)

This is the core refactor. Change `Vec<i32>` → `Vec<AtomicI32>` and `Vec<i64>` → `Vec<AtomicI64>`. Replace slice-returning methods with atomic accessors. All existing tests must be updated to use the new API.

**Step 1: Update imports and struct definition**

At the top of `storage.rs`, add:
```rust
use std::sync::atomic::{AtomicI32, AtomicI64, Ordering};
```

Change the struct fields (lines 19-23):
```rust
pub struct BlueprintStorage {
    /// Cumulative regrets: one `AtomicI32` per (decision node, bucket, action).
    pub regrets: Vec<AtomicI32>,
    /// Strategy sums: one `AtomicI64` per (decision node, bucket, action).
    pub strategy_sums: Vec<AtomicI64>,
    /// Number of buckets per street `[preflop, flop, turn, river]`.
    pub bucket_counts: [u16; 4],
    /// Per-node layout metadata.
    layout: Vec<NodeLayout>,
}
```

**Step 2: Update `new()` constructor**

Change the vec initialization (around line 68):
```rust
Self {
    regrets: (0..total).map(|_| AtomicI32::new(0)).collect(),
    strategy_sums: (0..total).map(|_| AtomicI64::new(0)).collect(),
    bucket_counts,
    layout,
}
```

**Step 3: Replace slice accessor methods with atomic ones**

Remove `get_regrets()`, `get_regrets_mut()`, `get_strategy_sums()`, `get_strategy_sums_mut()`. Replace with:

```rust
/// Read a single regret value atomically.
#[inline]
#[must_use]
pub fn get_regret(&self, node_idx: u32, bucket: u16, action: usize) -> i32 {
    let nl = &self.layout[node_idx as usize];
    let idx = Self::slot_offset(nl, bucket) + action;
    self.regrets[idx].load(Ordering::Relaxed)
}

/// Add a delta to a single regret value atomically.
#[inline]
pub fn add_regret(&self, node_idx: u32, bucket: u16, action: usize, delta: i32) {
    let nl = &self.layout[node_idx as usize];
    let idx = Self::slot_offset(nl, bucket) + action;
    self.regrets[idx].fetch_add(delta, Ordering::Relaxed);
}

/// Read a single strategy sum value atomically.
#[inline]
#[must_use]
pub fn get_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize) -> i64 {
    let nl = &self.layout[node_idx as usize];
    let idx = Self::slot_offset(nl, bucket) + action;
    self.strategy_sums[idx].load(Ordering::Relaxed)
}

/// Add a delta to a single strategy sum value atomically.
#[inline]
pub fn add_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize, delta: i64) {
    let nl = &self.layout[node_idx as usize];
    let idx = Self::slot_offset(nl, bucket) + action;
    self.strategy_sums[idx].fetch_add(delta, Ordering::Relaxed);
}

/// Read all regrets for a (node, bucket) into a caller-supplied buffer.
/// Returns the number of actions written.
#[inline]
pub fn get_regrets_into(&self, node_idx: u32, bucket: u16, out: &mut [i32]) -> usize {
    let nl = &self.layout[node_idx as usize];
    let n = nl.num_actions as usize;
    let start = Self::slot_offset(nl, bucket);
    for i in 0..n {
        out[i] = self.regrets[start + i].load(Ordering::Relaxed);
    }
    n
}
```

**Step 4: Update `current_strategy_into()`**

Replace the body to use atomic loads instead of slice access:

```rust
#[inline]
pub fn current_strategy_into(&self, node_idx: u32, bucket: u16, out: &mut [f64]) {
    let nl = &self.layout[node_idx as usize];
    let num_actions = nl.num_actions as usize;
    debug_assert!(
        out.len() >= num_actions,
        "buffer too small: {} < {num_actions}",
        out.len()
    );
    let out = &mut out[..num_actions];
    let start = Self::slot_offset(nl, bucket);

    let mut positive_sum = 0.0_f64;
    for i in 0..num_actions {
        let r = self.regrets[start + i].load(Ordering::Relaxed).max(0);
        out[i] = f64::from(r);
        positive_sum += out[i];
    }

    if positive_sum > 0.0 {
        for o in out.iter_mut() {
            *o /= positive_sum;
        }
    } else {
        let u = 1.0 / num_actions as f64;
        out.fill(u);
    }
}
```

**Step 5: Update `current_strategy()` and `average_strategy()`**

```rust
#[must_use]
pub fn current_strategy(&self, node_idx: u32, bucket: u16) -> Vec<f64> {
    let nl = &self.layout[node_idx as usize];
    let num_actions = nl.num_actions as usize;
    let start = Self::slot_offset(nl, bucket);

    let positive_sum: f64 = (0..num_actions)
        .map(|i| f64::from(self.regrets[start + i].load(Ordering::Relaxed).max(0)))
        .sum();

    if positive_sum > 0.0 {
        (0..num_actions)
            .map(|i| f64::from(self.regrets[start + i].load(Ordering::Relaxed).max(0)) / positive_sum)
            .collect()
    } else {
        vec![1.0 / num_actions as f64; num_actions]
    }
}

#[must_use]
pub fn average_strategy(&self, node_idx: u32, bucket: u16) -> Vec<f64> {
    let nl = &self.layout[node_idx as usize];
    let num_actions = nl.num_actions as usize;
    let start = Self::slot_offset(nl, bucket);

    let total: f64 = (0..num_actions)
        .map(|i| self.strategy_sums[start + i].load(Ordering::Relaxed) as f64)
        .sum();

    if total > 0.0 {
        (0..num_actions)
            .map(|i| self.strategy_sums[start + i].load(Ordering::Relaxed) as f64 / total)
            .collect()
    } else {
        vec![1.0 / num_actions as f64; num_actions]
    }
}
```

**Step 6: Update serialization (`save_regrets` and `load_regrets`)**

```rust
pub fn save_regrets(&self, path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    let regrets_plain: Vec<i32> = self.regrets.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    let sums_plain: Vec<i64> = self.strategy_sums.iter().map(|a| a.load(Ordering::Relaxed)).collect();

    let payload = (&self.bucket_counts, &regrets_plain, &sums_plain);
    bincode::serialize_into(&mut writer, &payload)
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    writer.flush()
}

pub fn load_regrets(
    path: &Path,
    tree: &GameTree,
    bucket_counts: [u16; 4],
) -> std::io::Result<Self> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let (stored_counts, regrets_plain, sums_plain): ([u16; 4], Vec<i32>, Vec<i64>) =
        bincode::deserialize_from(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    if stored_counts != bucket_counts {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bucket count mismatch: file has {stored_counts:?}, expected {bucket_counts:?}"),
        ));
    }

    let mut storage = Self::new(tree, bucket_counts);

    if regrets_plain.len() != storage.regrets.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("regret buffer length mismatch: file has {}, expected {}", regrets_plain.len(), storage.regrets.len()),
        ));
    }
    if sums_plain.len() != storage.strategy_sums.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("strategy_sums buffer length mismatch: file has {}, expected {}", sums_plain.len(), storage.strategy_sums.len()),
        ));
    }

    for (atom, &val) in storage.regrets.iter().zip(regrets_plain.iter()) {
        atom.store(val, Ordering::Relaxed);
    }
    for (atom, &val) in storage.strategy_sums.iter().zip(sums_plain.iter()) {
        atom.store(val, Ordering::Relaxed);
    }

    Ok(storage)
}
```

**Step 7: Update storage tests**

All tests that used `get_regrets_mut()` or `get_strategy_sums_mut()` must switch to `add_regret()` / `add_strategy_sum()`. Tests that used `get_regrets()` must use `get_regret()` or `current_strategy()`.

Key test changes:
- `regret_update_changes_strategy`: use `add_regret(&storage, node, 0, 0, 1000)` instead of slice mutation
- `average_strategy_proportional`: use `add_strategy_sum(&storage, node, 0, 0, 300)` etc.
- `save_load_round_trip`: use `add_regret` / `get_regret` and `add_strategy_sum` / `get_strategy_sum`
- `different_buckets_are_independent`: use atomic accessors

Also, `storage.regrets.iter().all(|&r| r == 0)` → `storage.regrets.iter().all(|r| r.load(Ordering::Relaxed) == 0)` (in mccfr and trainer tests).

**Step 8: Run tests**

```bash
cargo test -p poker-solver-core --lib -- blueprint_v2::storage
```
Expected: all storage tests pass.

**Step 9: Commit**

```bash
git add crates/core/src/blueprint_v2/storage.rs
git commit -m "feat(blueprint_v2): convert BlueprintStorage to atomic types

Vec<i32> → Vec<AtomicI32>, Vec<i64> → Vec<AtomicI64>.
Slice accessors replaced with atomic get/add methods.
Serialization collects to plain vecs for bincode compat."
```

---

### Task 3: Update MCCFR Traversal for Atomic Storage

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`

Change `traverse_external` from `&mut BlueprintStorage` to `&BlueprintStorage`. Update regret/strategy_sum writes to use atomic add methods. Update pruning reads to use atomic accessors.

**Step 1: Change `traverse_external` signature**

Line 110, change `storage: &mut BlueprintStorage` to `storage: &BlueprintStorage`.

Do the same for `traverse_traverser` (line 223) and `traverse_opponent` (line 288).

**Step 2: Update `traverse_traverser` regret and strategy writes**

Replace the regret update block (around lines 270-274):
```rust
// Before:
// let regrets = storage.get_regrets_mut(node_idx, bucket);
// for a in 0..num_actions {
//     let delta = action_values[a] - node_value;
//     regrets[a] += (delta * 1000.0) as i32;
// }

// After:
for a in 0..num_actions {
    let delta = action_values[a] - node_value;
    storage.add_regret(node_idx, bucket, a, (delta * 1000.0) as i32);
}
```

Replace the strategy sum update block (around lines 277-280):
```rust
// Before:
// let strategy_sums = storage.get_strategy_sums_mut(node_idx, bucket);
// for a in 0..num_actions {
//     strategy_sums[a] += (strategy[a] * 1000.0) as i64;
// }

// After:
for a in 0..num_actions {
    storage.add_strategy_sum(node_idx, bucket, a, (strategy[a] * 1000.0) as i64);
}
```

**Step 3: Update pruning check**

Replace the pruning check (around lines 245-251):
```rust
// Before:
// if prune {
//     let regrets = storage.get_regrets(node_idx, bucket);
//     if regrets[a] < prune_threshold {
//         continue;
//     }
// }

// After:
if prune {
    if storage.get_regret(node_idx, bucket, a) < prune_threshold {
        continue;
    }
}
```

**Step 4: Fix all test call sites**

In `mccfr.rs` tests, change all `&mut storage` to `&storage` in `traverse_external` calls. Also update assertions that read storage:

```rust
// Before: storage.regrets.iter().all(|&r| r == 0)
// After:
storage.regrets.iter().all(|r| r.load(Ordering::Relaxed) == 0)

// Before: storage.regrets.iter().any(|&r| r != 0)
// After:
storage.regrets.iter().any(|r| r.load(Ordering::Relaxed) != 0)

// Before: storage.strategy_sums.iter().any(|&s| s != 0)
// After:
storage.strategy_sums.iter().any(|s| s.load(Ordering::Relaxed) != 0)
```

Also in `traverse_with_pruning` test, change the regret seeding:
```rust
// Before: for r in storage.regrets.iter_mut().step_by(3) { *r = -400_000_000; }
// After:
for r in storage.regrets.iter().step_by(3) {
    r.store(-400_000_000, Ordering::Relaxed);
}
```

Add the import at the top of the test module:
```rust
use std::sync::atomic::Ordering;
```

**Step 5: Run tests**

```bash
cargo test -p poker-solver-core --lib -- blueprint_v2::mccfr
```
Expected: all mccfr tests pass.

**Step 6: Commit**

```bash
git add crates/core/src/blueprint_v2/mccfr.rs
git commit -m "feat(blueprint_v2): change traverse_external to shared reference

&mut BlueprintStorage → &BlueprintStorage. Regret and strategy sum
updates use atomic fetch_add. Pruning reads use atomic load."
```

---

### Task 4: Update Trainer for Atomic Storage and Batched Parallel Loop

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

This is the main payoff task. Convert the single-threaded training loop to batched parallel execution. Update LCFR discount and mean_positive_regret to use atomic accessors.

**Step 1: Add imports**

At the top of `trainer.rs`, add:
```rust
use std::sync::atomic::Ordering;
use rand::rngs::SmallRng;
use rayon::prelude::*;
```

**Step 2: Update `train()` method**

Replace the existing loop body (lines 178-224) with the batched parallel version:

```rust
pub fn train(&mut self) -> Result<(), Box<dyn Error>> {
    let batch_size = self.config.training.batch_size;

    while !self.should_stop() {
        // Honour pause requests from the TUI.
        while self.paused.load(Ordering::Relaxed) {
            if self.quit_requested.load(Ordering::Relaxed) {
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        // Calculate how many iterations remain (respect iteration limit).
        let remaining = self.config.training.iterations
            .map(|max| max.saturating_sub(self.iterations))
            .unwrap_or(batch_size);
        let this_batch = batch_size.min(remaining);
        if this_batch == 0 {
            break;
        }

        // 1. Generate batch of deals (sequential, fast).
        let deals: Vec<Deal> = (0..this_batch)
            .map(|_| self.sample_deal())
            .collect();

        let prune = self.should_prune();
        let threshold = self.config.training.prune_threshold;

        // 2. Parallel traversal.
        let tree = &self.tree;
        let storage = &self.storage;
        let buckets = &self.buckets;

        deals.par_iter().for_each(|deal| {
            let mut rng = SmallRng::from_entropy();

            traverse_external(
                tree, storage, buckets, deal,
                0, tree.root, prune, threshold, &mut rng,
            );
            traverse_external(
                tree, storage, buckets, deal,
                1, tree.root, prune, threshold, &mut rng,
            );
        });

        // 3. Sequential: update counters and check timed actions.
        self.iterations += this_batch;
        self.shared_iterations
            .store(self.iterations, Ordering::Relaxed);
        self.check_timed_actions()?;
    }
    Ok(())
}
```

**Step 3: Update `apply_lcfr_discount()`**

Replace the body (around lines 346-359):

```rust
fn apply_lcfr_discount(&self) {
    let elapsed_min = self.elapsed_minutes();
    let interval = self.config.training.lcfr_discount_interval.max(1);
    let t = elapsed_min / interval;
    let d = t as f64 / (t as f64 + 1.0);

    for atom in &self.storage.regrets {
        let v = atom.load(Ordering::Relaxed);
        atom.store((v as f64 * d) as i32, Ordering::Relaxed);
    }
    for atom in &self.storage.strategy_sums {
        let v = atom.load(Ordering::Relaxed);
        atom.store((v as f64 * d) as i64, Ordering::Relaxed);
    }
}
```

Note: `apply_lcfr_discount` changes from `&mut self` to `&self` for the storage access, but it still needs `&mut self` to update `self.last_discount_time`. Keep it as `&mut self`.

Actually, the method body accesses `self.last_discount_time` via the caller in `check_timed_actions`, so keep the full method as-is but change only the loop body. The `self.last_discount_time = elapsed_min;` stays.

```rust
fn apply_lcfr_discount(&mut self) {
    let elapsed_min = self.elapsed_minutes();
    let interval = self.config.training.lcfr_discount_interval.max(1);
    let t = elapsed_min / interval;
    let d = t as f64 / (t as f64 + 1.0);

    for atom in &self.storage.regrets {
        let v = atom.load(Ordering::Relaxed);
        atom.store((v as f64 * d) as i32, Ordering::Relaxed);
    }
    for atom in &self.storage.strategy_sums {
        let v = atom.load(Ordering::Relaxed);
        atom.store((v as f64 * d) as i64, Ordering::Relaxed);
    }

    self.last_discount_time = elapsed_min;
}
```

**Step 4: Update `mean_positive_regret()`**

```rust
pub fn mean_positive_regret(&self) -> f64 {
    let (sum, count) = self
        .storage
        .regrets
        .iter()
        .fold((0.0_f64, 0_u64), |(s, c), atom| {
            let r = atom.load(Ordering::Relaxed);
            if r > 0 {
                (s + f64::from(r), c + 1)
            } else {
                (s, c)
            }
        });
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}
```

**Step 5: Update `should_prune()` — no change needed**

The method uses `self.rng` and `self.elapsed_minutes()`, both of which are on the coordinator. The prune decision is made once per batch (not per deal), which is fine — all deals in a batch share the same prune flag.

**Step 6: Fix trainer tests**

In `train_updates_storage` and other tests that check storage state:

```rust
// Before: trainer.storage.regrets.iter().all(|&r| r == 0)
// After:
use std::sync::atomic::Ordering;
trainer.storage.regrets.iter().all(|r| r.load(Ordering::Relaxed) == 0)

// Before: trainer.storage.regrets.iter().any(|&r| r != 0)
// After:
trainer.storage.regrets.iter().any(|r| r.load(Ordering::Relaxed) != 0)
```

In `lcfr_discount_at_t_zero`:
```rust
// Before:
// trainer.storage.regrets[0] = 1000;
// trainer.storage.strategy_sums[0] = 2000;
// After:
trainer.storage.regrets[0].store(1000, Ordering::Relaxed);
trainer.storage.strategy_sums[0].store(2000, Ordering::Relaxed);

// Before: assert_eq!(trainer.storage.regrets[0], 0);
// After:
assert_eq!(trainer.storage.regrets[0].load(Ordering::Relaxed), 0);
assert_eq!(trainer.storage.strategy_sums[0].load(Ordering::Relaxed), 0);
```

In `mean_positive_regret_initially_zero`:
```rust
// Before:
// trainer.storage.regrets[0] = 100;
// trainer.storage.regrets[1] = -50;
// trainer.storage.regrets[2] = 200;
// After:
trainer.storage.regrets[0].store(100, Ordering::Relaxed);
trainer.storage.regrets[1].store(-50, Ordering::Relaxed);
trainer.storage.regrets[2].store(200, Ordering::Relaxed);
```

**Step 7: Add a new test for batched iteration counting**

```rust
#[test]
fn train_batch_iterations() {
    let mut config = toy_config();
    config.training.iterations = Some(50);
    config.training.batch_size = 10;
    let mut trainer = BlueprintTrainer::new(config);
    trainer.train().expect("training should complete");
    assert_eq!(trainer.iterations, 50);
}
```

**Step 8: Run tests**

```bash
cargo test -p poker-solver-core --lib -- blueprint_v2::trainer
```
Expected: all trainer tests pass.

**Step 9: Commit**

```bash
git add crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat(blueprint_v2): batched parallel MCCFR training with Rayon

Training loop generates batches of deals and processes them in parallel
via rayon::par_iter. Each worker uses SmallRng::from_entropy() for
opponent sampling. LCFR discount runs between batches. Batch size
configurable (default 200)."
```

---

### Task 5: Update Bundle and Snapshot for Atomic Storage

**Files:**
- Modify: `crates/core/src/blueprint_v2/bundle.rs`

The `BlueprintV2Strategy::from_storage()` method calls `storage.average_strategy()` which already returns `Vec<f64>` — this works unchanged with atomic storage. However, `save_snapshot` calls `storage.save_regrets()` which was updated in Task 2. The bundle code should just compile.

**Step 1: Verify compilation**

```bash
cargo check -p poker-solver-core
```

If there are any remaining compile errors in `bundle.rs`, they'll be due to test code that directly accesses `storage.regrets` or `storage.strategy_sums` as plain values. Fix these using atomic loads/stores.

**Step 2: Run bundle tests**

```bash
cargo test -p poker-solver-core --lib -- blueprint_v2::bundle
```
Expected: all bundle tests pass without changes (they use high-level APIs).

**Step 3: Verify full crate compilation and tests**

```bash
cargo test -p poker-solver-core --lib -- blueprint_v2
cargo clippy -p poker-solver-core
```

Fix any clippy warnings (e.g., unnecessary `&` on atomic operations, or `Ordering` import suggestions).

**Step 4: Commit (only if changes were needed)**

```bash
git add crates/core/src/blueprint_v2/
git commit -m "fix(blueprint_v2): update bundle for atomic storage compatibility"
```

---

### Task 6: Update Trainer Crate (TUI Scenarios) and Integration Tests

**Files:**
- Modify: `crates/trainer/src/blueprint_tui_scenarios.rs` (if needed)
- Modify: `crates/trainer/tests/blueprint_tui_integration.rs` (if needed)

The TUI scenarios module calls `storage.current_strategy()`, `storage.average_strategy()`, `storage.num_actions()`, and reads `storage.bucket_counts`. All of these still work with atomic storage since they return owned values.

**Step 1: Verify trainer crate compiles**

```bash
cargo check -p poker-solver-trainer
```

Fix any issues. The most likely problem is in integration tests that construct `BlueprintStorage` or access `storage.regrets` directly.

**Step 2: Run all trainer tests**

```bash
cargo test -p poker-solver-trainer
```

**Step 3: Run the full test suite**

```bash
cargo test
```

Ensure everything passes. Fix any remaining atomic access issues in test code.

**Step 4: Run clippy on everything**

```bash
cargo clippy
```

Fix any warnings.

**Step 5: Commit**

```bash
git add -A
git commit -m "fix: update trainer crate for atomic BlueprintStorage compatibility"
```

---

### Task 7: Add Parallel Training Integration Test

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs` (add test at end of test module)

**Step 1: Write a test that verifies parallel batches produce results**

Add to the `tests` module in `trainer.rs`:

```rust
#[test]
fn parallel_batch_produces_regret_updates() {
    let mut config = toy_config();
    config.training.iterations = Some(200);
    config.training.batch_size = 50;
    let mut trainer = BlueprintTrainer::new(config);
    trainer.train().expect("training should complete");

    assert_eq!(trainer.iterations, 200);
    assert!(
        trainer.storage.regrets.iter().any(|r| r.load(Ordering::Relaxed) != 0),
        "parallel training should update regrets"
    );
    assert!(
        trainer.storage.strategy_sums.iter().any(|s| s.load(Ordering::Relaxed) != 0),
        "parallel training should update strategy sums"
    );
}

#[test]
fn batch_size_larger_than_iterations() {
    let mut config = toy_config();
    config.training.iterations = Some(10);
    config.training.batch_size = 200;  // batch > total iterations
    let mut trainer = BlueprintTrainer::new(config);
    trainer.train().expect("training should complete");
    assert_eq!(trainer.iterations, 10);
}
```

**Step 2: Run the new tests**

```bash
cargo test -p poker-solver-core --lib -- blueprint_v2::trainer::tests::parallel_batch
cargo test -p poker-solver-core --lib -- blueprint_v2::trainer::tests::batch_size_larger
```
Expected: PASS.

**Step 3: Run the full test suite one final time**

```bash
cargo test
```

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_v2/trainer.rs
git commit -m "test(blueprint_v2): add parallel batch training integration tests"
```

---

### Task 8: Update Sample Config and Documentation

**Files:**
- Modify: `sample_configurations/blueprint_v2_with_tui.yaml`
- Modify: `docs/training.md`

**Step 1: Add `batch_size` to sample config**

In `blueprint_v2_with_tui.yaml`, add to the `training:` section:
```yaml
  batch_size: 200
```

**Step 2: Update training docs**

In `docs/training.md`, in the Blueprint V2 training section, add a note about parallelism:

> **Parallel Training**: Blueprint V2 automatically uses all available CPU cores. Each batch of `batch_size` deals (default: 200) is processed in parallel using Rayon's thread pool. LCFR discount and snapshots run between batches. Set `RAYON_NUM_THREADS=N` to limit core usage.

**Step 3: Commit**

```bash
git add sample_configurations/blueprint_v2_with_tui.yaml docs/training.md
git commit -m "docs: document parallel MCCFR training and batch_size config"
```
