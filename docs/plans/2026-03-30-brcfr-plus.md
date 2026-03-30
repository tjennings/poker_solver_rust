# BRCFR+ Optimizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add a BRCFR+ optimizer that augments DCFR+ with periodic best-response prediction passes, decaying linearly between refresh intervals.

**Architecture:** BRCFR+ layers a BR prediction signal on top of the existing DCFR+ optimizer. The prediction buffer (already in `BlueprintStorage`) is populated by periodic BR tree traversals instead of per-iteration MCCFR writes. A `predictions_locked` flag on storage prevents MCCFR from overwriting the BR signal. The trainer manages the warmup/interval/decay lifecycle and sets `optimizer.decay` before each MCCFR batch.

**Tech Stack:** Rust, existing `CfrOptimizer` trait, `BlueprintStorage` prediction buffer, `traverse_best_response`

**Design doc:** `docs/plans/2026-03-30-brcfr-plus-design.md`

---

### Task 1: Add `BrcfrPlusOptimizer` struct and `current_strategy`

**Files:**
- Modify: `crates/core/src/cfr/optimizer.rs:213` (after `SapcfrPlusOptimizer` impl, before `#[cfg(test)]`)

**Step 1: Write the failing tests**

Add these tests at the end of the `mod tests` block in `optimizer.rs` (before the closing `}`):

```rust
// Tests for BrcfrPlusOptimizer

#[test]
fn brcfr_current_strategy_with_decay() {
    use super::*;
    let opt = BrcfrPlusOptimizer {
        alpha: 1.5,
        gamma: 2.0,
        eta: 0.6,
        decay: 0.5,
    };
    let regrets = vec![AtomicI64::new(200), AtomicI64::new(100), AtomicI64::new(0)];
    let preds = vec![AtomicI64::new(100), AtomicI64::new(-300), AtomicI64::new(50)];
    let mut out = [0.0; 3];
    opt.current_strategy(&regrets, Some(&preds), 0, 3, &mut out);
    // R_tilde[0] = max(0, 200 + 0.6 * 0.5 * 100) = max(0, 230) = 230
    // R_tilde[1] = max(0, 100 + 0.6 * 0.5 * (-300)) = max(0, 10) = 10
    // R_tilde[2] = max(0, 0 + 0.6 * 0.5 * 50) = max(0, 15) = 15
    // sum = 255
    assert!((out[0] - 230.0 / 255.0).abs() < 0.01, "action 0: got {}", out[0]);
    assert!((out[1] - 10.0 / 255.0).abs() < 0.01, "action 1: got {}", out[1]);
    assert!((out[2] - 15.0 / 255.0).abs() < 0.01, "action 2: got {}", out[2]);
}

#[test]
fn brcfr_decay_zero_matches_dcfr_plus() {
    use super::*;
    let brcfr = BrcfrPlusOptimizer {
        alpha: 1.5,
        gamma: 2.0,
        eta: 0.6,
        decay: 0.0,
    };
    let regrets = vec![AtomicI64::new(300), AtomicI64::new(100), AtomicI64::new(-50)];
    let preds = vec![AtomicI64::new(999), AtomicI64::new(-999), AtomicI64::new(999)];
    let mut out_brcfr = [0.0; 3];
    brcfr.current_strategy(&regrets, Some(&preds), 0, 3, &mut out_brcfr);
    // With decay=0, predictions are ignored => standard RM+: 300/400, 100/400, 0
    assert!((out_brcfr[0] - 0.75).abs() < 0.01, "got {}", out_brcfr[0]);
    assert!((out_brcfr[1] - 0.25).abs() < 0.01, "got {}", out_brcfr[1]);
    assert!((out_brcfr[2] - 0.0).abs() < 0.01, "got {}", out_brcfr[2]);
}

#[test]
fn brcfr_discount_floors_negative_regrets() {
    use super::*;
    let opt = BrcfrPlusOptimizer {
        alpha: 1.5,
        gamma: 2.0,
        eta: 0.6,
        decay: 1.0,
    };
    let regrets = vec![AtomicI64::new(1000), AtomicI64::new(-500)];
    let strats = vec![AtomicI64::new(2000)];
    opt.apply_discount(&regrets, &strats, None, 10);
    // Negative regrets are floored to 0 (RM+ style).
    let r1 = regrets[1].load(Ordering::Relaxed);
    assert_eq!(r1, 0, "negative regrets floored to 0, got {r1}");
    // Positive regrets are discounted normally.
    let r0 = regrets[0].load(Ordering::Relaxed);
    assert!(r0 > 960 && r0 < 980, "positive regret discounted: got {r0}");
}

#[test]
fn brcfr_name() {
    use super::*;
    let opt = BrcfrPlusOptimizer {
        alpha: 1.5,
        gamma: 2.0,
        eta: 0.6,
        decay: 0.0,
    };
    assert_eq!(opt.name(), "brcfr+");
}

#[test]
fn brcfr_needs_predictions_true() {
    use super::*;
    let opt = BrcfrPlusOptimizer {
        alpha: 1.5,
        gamma: 2.0,
        eta: 0.6,
        decay: 0.0,
    };
    assert!(opt.needs_predictions());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core -- brcfr_ --no-run 2>&1 | head -5`
Expected: compilation error — `BrcfrPlusOptimizer` not found

**Step 3: Write the implementation**

Add after line 213 in `optimizer.rs` (after the `SapcfrPlusOptimizer` impl block, before `#[cfg(test)]`):

```rust
/// BRCFR+ optimizer: DCFR+ augmented with periodic best-response predictions.
///
/// Discounting is identical to SAPCFR+ (RM+ style: floor negatives to 0).
/// Strategy computation uses BR-derived predictions with configurable decay:
///   `R_tilde = max(0, R + eta * decay * v_br)`
///
/// The `decay` field is set by the trainer based on how many iterations
/// have elapsed since the last BR pass. At `decay = 0`, this is pure DCFR+.
pub struct BrcfrPlusOptimizer {
    /// Positive regret discount exponent.
    pub alpha: f64,
    /// Strategy sum discount exponent.
    pub gamma: f64,
    /// BR prediction weight.
    pub eta: f64,
    /// Current decay factor (0.0 to 1.0), set by trainer.
    pub decay: f64,
}

impl CfrOptimizer for BrcfrPlusOptimizer {
    fn apply_discount(
        &self,
        regrets: &[AtomicI64],
        strategy_sums: &[AtomicI64],
        _predictions: Option<&[AtomicI64]>,
        iteration: u64,
    ) {
        let tf = iteration as f64;
        let t_alpha = tf.powf(self.alpha);
        let d_pos = t_alpha / (t_alpha + 1.0);
        let d_strat = (tf / (tf + 1.0)).powf(self.gamma);

        // Discount regrets: floor negative to 0 (RM+ style).
        regrets.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            let discounted = (v as f64 * d_pos) as i64;
            atom.store(discounted.max(0), Ordering::Relaxed);
        });

        // Discount strategy sums.
        strategy_sums.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            atom.store((v as f64 * d_strat) as i64, Ordering::Relaxed);
        });
    }

    fn current_strategy(
        &self,
        regrets: &[AtomicI64],
        predictions: Option<&[AtomicI64]>,
        offset: usize,
        num_actions: usize,
        out: &mut [f64],
    ) {
        let eta_decay = self.eta * self.decay;
        let mut sum = 0.0_f64;
        for a in 0..num_actions {
            let r = regrets[offset + a].load(Ordering::Relaxed);
            let v = predictions.map_or(0, |p| p[offset + a].load(Ordering::Relaxed));
            let predicted = r as f64 + eta_decay * v as f64;
            let clamped = predicted.max(0.0);
            out[a] = clamped;
            sum += clamped;
        }
        if sum > 0.0 {
            for v in &mut out[..num_actions] {
                *v /= sum;
            }
        } else {
            let uniform = 1.0 / num_actions as f64;
            for v in &mut out[..num_actions] {
                *v = uniform;
            }
        }
    }

    fn name(&self) -> &'static str {
        "brcfr+"
    }

    fn needs_predictions(&self) -> bool {
        true
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core -- brcfr_`
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add crates/core/src/cfr/optimizer.rs
git commit -m "feat: add BrcfrPlusOptimizer with decay-weighted BR predictions"
```

---

### Task 2: Add `predictions_locked` flag to `BlueprintStorage`

The MCCFR traversal writes per-iteration predictions at `mccfr.rs:917`. BRCFR+ needs to prevent this from overwriting the BR signal. Add a lock flag that `set_prediction` checks.

**Files:**
- Modify: `crates/core/src/blueprint_v2/storage.rs:42-66` (struct definition)
- Modify: `crates/core/src/blueprint_v2/storage.rs:253-260` (`set_prediction`)

**Step 1: Write the failing test**

Add in the `mod tests` block of `storage.rs`:

```rust
#[test]
fn predictions_locked_prevents_writes() {
    let tree = test_tree();
    let mut storage = BlueprintStorage::new(&tree, [2, 2, 2, 2]);
    storage.enable_predictions();
    // Write a prediction value.
    storage.set_prediction(0, 0, 0, 42);
    assert_eq!(storage.get_prediction(0, 0, 0), 42);
    // Lock predictions.
    storage.lock_predictions();
    // Write should be silently ignored.
    storage.set_prediction(0, 0, 0, 999);
    assert_eq!(storage.get_prediction(0, 0, 0), 42, "locked buffer should not change");
    // Unlock.
    storage.unlock_predictions();
    storage.set_prediction(0, 0, 0, 999);
    assert_eq!(storage.get_prediction(0, 0, 0), 999, "unlocked buffer should update");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core -- predictions_locked --no-run 2>&1 | head -5`
Expected: compilation error — `lock_predictions` not found

**Step 3: Implement**

Add field to `BlueprintStorage` struct (after `predictions` field, around line 55):

```rust
    /// When true, `set_prediction` becomes a no-op. Used by BRCFR+ to
    /// protect BR-derived predictions from being overwritten by MCCFR.
    predictions_locked: bool,
```

Initialize it to `false` in `BlueprintStorage::new` (find the struct initialization block).

Add methods after `set_prediction` (around line 260):

```rust
    /// Lock the prediction buffer so `set_prediction` becomes a no-op.
    pub fn lock_predictions(&mut self) {
        self.predictions_locked = true;
    }

    /// Unlock the prediction buffer so `set_prediction` writes again.
    pub fn unlock_predictions(&mut self) {
        self.predictions_locked = false;
    }
```

Modify `set_prediction` to check the flag:

```rust
    pub fn set_prediction(&self, node_idx: u32, bucket: u16, action: usize, value: i64) {
        if self.predictions_locked {
            return;
        }
        if let Some(ref p) = self.predictions {
            p[self.slot_index(node_idx, bucket, action)].store(value, Ordering::Relaxed);
        }
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core -- predictions_locked`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/storage.rs
git commit -m "feat: add predictions_locked flag to BlueprintStorage"
```

---

### Task 3: Add `predict_storage` parameter to `traverse_best_response`

Modify the existing BR traversal to optionally write per-infoset BR regrets into the prediction buffer.

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs:786-842` (`traverse_best_response`)
- Modify: `crates/core/src/blueprint_v2/trainer.rs:707-711` (callers in `compute_exploitability`)

**Step 1: Write the failing test**

Add in the `traverse_best_response` test section of `mccfr.rs` (around line 2524):

```rust
#[test]
fn best_response_writes_predictions_when_storage_provided() {
    let (tree, storage, precomputed) = make_test_game();
    // Clone storage for prediction target (or enable predictions on existing).
    // Enable predictions buffer.
    let mut predict_store = BlueprintStorage::new(&tree, storage.bucket_counts);
    predict_store.enable_predictions();
    // Set some non-zero strategy sums on predict_store so average_strategy works.
    // Copy strategy sums from the solved storage.
    for i in 0..storage.strategy_sums.len() {
        predict_store.strategy_sums[i].store(
            storage.strategy_sums[i].load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    let br0 = traverse_best_response(
        &tree, &storage, &precomputed, 0, tree.root, 0.0, 0.0, Some(&predict_store),
    );
    // BR value should be the same as without predictions.
    let br0_plain = traverse_best_response(
        &tree, &storage, &precomputed, 0, tree.root, 0.0, 0.0, None,
    );
    assert!((br0 - br0_plain).abs() < 1e-10, "BR value should match: {} vs {}", br0, br0_plain);

    // Predictions should have been written (at least some non-zero).
    let preds = predict_store.predictions.as_ref().expect("predictions enabled");
    let any_nonzero = preds.iter().any(|a| a.load(Ordering::Relaxed) != 0);
    assert!(any_nonzero, "predictions should be populated by BR traversal");
}
```

Note: this test references `make_test_game` — use whatever existing test helper creates a tree + storage + precomputed deal in the file. Check the existing `best_response_returns_finite` test for the pattern.

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core -- best_response_writes_predictions --no-run 2>&1 | head -5`
Expected: compilation error — wrong number of arguments to `traverse_best_response`

**Step 3: Implement**

Update the function signature at `mccfr.rs:786`:

```rust
pub fn traverse_best_response(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    rake_rate: f64,
    rake_cap: f64,
    predict_storage: Option<&BlueprintStorage>,
) -> f64 {
```

Update all recursive calls within the function (lines ~801, ~818, ~832) to pass through `predict_storage`.

At the traverser decision node (around line 814-825), after computing `best` (the max action value), add the prediction write:

```rust
            if player == traverser {
                // BR player: take max over all actions.
                let mut best = f64::NEG_INFINITY;
                let mut action_values = [0.0f64; 16]; // max actions
                for (a, &child_idx) in children.iter().enumerate() {
                    let v = traverse_best_response(
                        tree, storage, deal, traverser, child_idx, rake_rate, rake_cap,
                        predict_storage,
                    );
                    action_values[a] = v;
                    if v > best {
                        best = v;
                    }
                }
                // Write BR-derived regrets to prediction buffer.
                if let Some(ps) = predict_storage {
                    for (a, _) in children.iter().enumerate() {
                        let delta = action_values[a] - best;
                        let delta_scaled = (delta * super::storage::REGRET_SCALE) as i64;
                        ps.set_prediction(node_idx, bucket, a, delta_scaled);
                    }
                }
                best
```

Update all callers of `traverse_best_response` in `trainer.rs` (`compute_exploitability`, lines ~707-711) to pass `None`:

```rust
                let br0 = traverse_best_response(
                    tree, storage, &deal, 0, tree.root, rake_rate, rake_cap, None,
                );
                let br1 = traverse_best_response(
                    tree, storage, &deal, 1, tree.root, rake_rate, rake_cap, None,
                );
```

Also update any test callers in `mccfr.rs` that call `traverse_best_response` — add `None` as the last argument to all existing calls (around lines 2533-2534, 2557, 2571-2572, 2590-2591).

**Step 4: Run full test suite to verify**

Run: `cargo test -p poker-solver-core`
Expected: all tests PASS (existing tests unchanged via `None`, new test passes)

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/mccfr.rs crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat: add predict_storage parameter to traverse_best_response"
```

---

### Task 4: Add BRCFR+ config fields

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs:211` (after `sapcfr_eta`)

**Step 1: Write the failing test**

Add in the config tests section:

```rust
#[test]
fn test_optimizer_brcfr_plus_config() {
    let yaml = r#"
game:
  name: test
  players: 2
  stack_depth: 200
  small_blind: 1
  big_blind: 2
training:
  optimizer: "brcfr+"
  brcfr_eta: 0.7
  brcfr_warmup_iterations: 300000000
  brcfr_interval: 100000000
  dcfr_alpha: 1.5
  dcfr_gamma: 2.0
"#;
    let cfg: BlueprintConfig =
        serde_yaml::from_str(yaml).expect("failed to parse brcfr+ config");
    assert_eq!(cfg.training.optimizer, "brcfr+");
    assert!((cfg.training.brcfr_eta - 0.7).abs() < f64::EPSILON);
    assert_eq!(cfg.training.brcfr_warmup_iterations, 300_000_000);
    assert_eq!(cfg.training.brcfr_interval, 100_000_000);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core -- test_optimizer_brcfr_plus_config --no-run 2>&1 | head -5`
Expected: compilation error — `brcfr_eta` field not found

**Step 3: Implement**

Add after `sapcfr_eta` (around line 211) in `TrainingConfig`:

```rust
    /// BRCFR+ prediction weight (0 = no prediction, higher = stronger BR nudge).
    #[serde(default = "default_brcfr_eta")]
    pub brcfr_eta: f64,
    /// BRCFR+ warmup: pure DCFR+ iterations before first BR pass.
    #[serde(default)]
    pub brcfr_warmup_iterations: u64,
    /// BRCFR+ interval: iterations between BR prediction passes (after warmup).
    #[serde(default = "default_brcfr_interval")]
    pub brcfr_interval: u64,
```

Add default functions near the other defaults (around line 345):

```rust
fn default_brcfr_eta() -> f64 {
    0.6
}

fn default_brcfr_interval() -> u64 {
    100_000_000
}
```

Add the fields to the `TrainingConfig::default_for_test` or any manual construction sites (search for `sapcfr_eta:` in the file to find them, around line 1386 and line 504):

```rust
brcfr_eta: 0.6,
brcfr_warmup_iterations: 0,
brcfr_interval: 100_000_000,
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core -- test_optimizer_brcfr`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/config.rs
git commit -m "feat: add BRCFR+ config fields (eta, warmup, interval)"
```

---

### Task 5: Wire `BrcfrPlusOptimizer` into trainer construction

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs:31` (imports)
- Modify: `crates/core/src/blueprint_v2/trainer.rs:261-278` (optimizer construction)
- Modify: `crates/core/src/blueprint_v2/trainer.rs:462-478` (reload path)

**Step 1: Write the failing test**

Add in trainer tests:

```rust
#[test]
fn trainer_creates_brcfr_optimizer() {
    let mut config = test_config();
    config.training.optimizer = "brcfr+".to_string();
    config.training.brcfr_eta = 0.7;
    let trainer = BlueprintTrainer::new(config).expect("trainer should be created");
    let opt = trainer.storage.optimizer.as_ref().expect("optimizer should be set");
    assert_eq!(opt.name(), "brcfr+");
    assert!(trainer.storage.predictions.is_some(), "predictions should be enabled");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core -- trainer_creates_brcfr --no-run 2>&1 | head -5`
Expected: FAIL — "brcfr+" falls through to default DCFR

**Step 3: Implement**

Add import at line 31:

```rust
use crate::cfr::optimizer::{BrcfrPlusOptimizer, CfrOptimizer, DcfrOptimizer, SapcfrPlusOptimizer};
```

Update optimizer construction (around line 262). Replace the if/else with a match-like chain:

```rust
        let optimizer: Arc<dyn CfrOptimizer> = if config.training.optimizer == "brcfr+" {
            Arc::new(BrcfrPlusOptimizer {
                alpha: config.training.dcfr_alpha,
                gamma: config.training.dcfr_gamma,
                eta: config.training.brcfr_eta,
                decay: 0.0, // starts at 0 (warmup)
            })
        } else if config.training.optimizer == "sapcfr+" {
            Arc::new(SapcfrPlusOptimizer {
                alpha: config.training.dcfr_alpha,
                gamma: config.training.dcfr_gamma,
                eta: config.training.sapcfr_eta,
            })
        } else {
            Arc::new(DcfrOptimizer {
                alpha: config.training.dcfr_alpha,
                beta: config.training.dcfr_beta,
                gamma: config.training.dcfr_gamma,
            })
        };
```

Apply the same pattern to the reload path at line 462.

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core -- trainer_creates_brcfr`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat: wire BrcfrPlusOptimizer into trainer construction"
```

---

### Task 6: Implement BR prediction pass and decay lifecycle in trainer

This is the core integration: the trainer runs periodic BR passes, populates the prediction buffer, locks it, and updates the optimizer's decay factor.

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs` (trainer struct fields, `check_timed_actions`, train loop)

**Step 1: Write the failing test**

```rust
#[test]
fn brcfr_runs_br_pass_after_warmup() {
    let mut config = test_config();
    config.training.optimizer = "brcfr+".to_string();
    config.training.brcfr_eta = 0.6;
    config.training.brcfr_warmup_iterations = 100;
    config.training.brcfr_interval = 50;
    config.training.iterations = Some(200);
    config.training.exploitability_samples = 100;
    let mut trainer = BlueprintTrainer::new(config).expect("trainer");
    trainer.train().expect("training should complete");
    // After training 200 iters with warmup=100, interval=50, we should have
    // had at least 2 BR passes (at iter 100 and 150).
    // Predictions should be populated.
    let preds = trainer.storage.predictions.as_ref().expect("predictions enabled");
    let any_nonzero = preds.iter().any(|a| a.load(std::sync::atomic::Ordering::Relaxed) != 0);
    assert!(any_nonzero, "predictions should be populated after BR pass");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core -- brcfr_runs_br_pass --no-run 2>&1 | head -5`
Expected: compiles but predictions are all zeros (no BR pass logic yet)

**Step 3: Implement**

Add trainer state fields to the `BlueprintTrainer` struct:

```rust
    /// Iteration of the most recent BR prediction pass (BRCFR+).
    last_br_iteration: u64,
```

Initialize `last_br_iteration: 0` in `BlueprintTrainer::new`.

Add a new method `run_br_prediction_pass`:

```rust
    /// Run a BR prediction pass: traverse both players, write per-infoset
    /// BR-derived regrets into the prediction buffer. Returns exploitability
    /// in mbb/hand as a free side-effect.
    fn run_br_prediction_pass(&mut self) -> f64 {
        let n = self.config.training.exploitability_samples.max(1);
        let tree = &self.tree;
        let storage = &self.storage;
        let buckets_ref = &self.buckets;
        let rake_rate = self.config.game.rake_rate;
        let rake_cap = self.config.game.rake_cap;
        let big_blind = self.config.game.big_blind;

        // Unlock predictions so BR traversal can write.
        self.storage.unlock_predictions();

        let (sum_p0, sum_p1): (f64, f64) = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut rng = SmallRng::seed_from_u64(i.wrapping_mul(0x9E3779B97F4A7C15));
                let deal = sample_deal_with_rng(&mut rng);
                let buckets = buckets_ref.precompute_buckets(&deal);
                let deal = DealWithBuckets { deal, buckets };

                let br0 = traverse_best_response(
                    tree, storage, &deal, 0, tree.root, rake_rate, rake_cap,
                    Some(storage),
                );
                let br1 = traverse_best_response(
                    tree, storage, &deal, 1, tree.root, rake_rate, rake_cap,
                    Some(storage),
                );
                (br0, br1)
            })
            .reduce(|| (0.0, 0.0), |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1));

        // Lock predictions to prevent MCCFR from overwriting.
        self.storage.lock_predictions();
        self.last_br_iteration = self.iterations;

        let n_f = n as f64;
        (sum_p0 / n_f + sum_p1 / n_f) / 2.0 / big_blind * 1000.0
    }
```

Add BR pass check in `check_timed_actions` (after the exploitability section, around line 861). Note: this uses iteration-based triggers, not time-based:

```rust
        // BRCFR+ BR prediction pass (iteration-based).
        if self.config.training.optimizer == "brcfr+" {
            let warmup = self.config.training.brcfr_warmup_iterations;
            let interval = self.config.training.brcfr_interval.max(1);

            if self.iterations >= warmup {
                let should_br = if self.last_br_iteration < warmup {
                    true // First pass right at warmup boundary
                } else {
                    self.iterations >= self.last_br_iteration + interval
                };

                if should_br {
                    let exploit = self.run_br_prediction_pass();
                    if !self.tui_active {
                        eprintln!("  BRCFR+ BR pass: exploitability = {exploit:.2} mbb/hand");
                    }
                    if let Some(ref cb) = self.on_exploitability {
                        cb(exploit);
                    }
                }
            }

            // Update decay on the optimizer.
            let decay = if self.iterations < warmup || self.last_br_iteration == 0 {
                0.0
            } else {
                let elapsed = self.iterations.saturating_sub(self.last_br_iteration) as f64;
                (1.0 - elapsed / interval as f64).max(0.0)
            };
            // Update the optimizer's decay field.
            // Safety: we need interior mutability. Use unsafe to cast or
            // add a set_decay method via UnsafeCell/AtomicF64.
            // Simplest: store decay as AtomicU64 bits on the optimizer.
            // Alternative: make decay an AtomicU64 on the trainer and pass
            // it to current_strategy_into via a new parameter.
        }
```

**Important design note for the implementer:** The `CfrOptimizer` trait uses `&self`, so mutating `decay` requires interior mutability. The simplest approach is to make `decay` an `std::sync::atomic::AtomicU64` storing `f64` bits via `f64::to_bits`/`f64::from_bits`. Add helper methods:

```rust
pub struct BrcfrPlusOptimizer {
    pub alpha: f64,
    pub gamma: f64,
    pub eta: f64,
    decay_bits: AtomicU64,
}

impl BrcfrPlusOptimizer {
    pub fn new(alpha: f64, gamma: f64, eta: f64) -> Self {
        Self { alpha, gamma, eta, decay_bits: AtomicU64::new(0.0_f64.to_bits()) }
    }

    pub fn set_decay(&self, decay: f64) {
        self.decay_bits.store(decay.to_bits(), Ordering::Relaxed);
    }

    fn decay(&self) -> f64 {
        f64::from_bits(self.decay_bits.load(Ordering::Relaxed))
    }
}
```

Update `current_strategy` to use `self.decay()` instead of `self.decay`.

Update the Task 1 tests and struct construction accordingly (use `BrcfrPlusOptimizer::new(...)` + `set_decay(...)` instead of struct literal with `decay` field).

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core -- brcfr_`
Expected: all BRCFR+ tests PASS

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/trainer.rs crates/core/src/cfr/optimizer.rs
git commit -m "feat: implement BR prediction pass and decay lifecycle in trainer"
```

---

### Task 7: Update sample config and docs

**Files:**
- Modify: `sample_configurations/blueprint_v2_200bkt_sapcfr.yaml` (add commented BRCFR+ example)
- Modify: `docs/training.md` (document new optimizer)

**Step 1: Add BRCFR+ example to sample config**

In `blueprint_v2_200bkt_sapcfr.yaml`, add after the commented SAPCFR+ block:

```yaml
  # BRCFR+ optimizer with periodic best-response prediction
  # optimizer: "brcfr+"
  # brcfr_eta: 0.6
  # brcfr_warmup_iterations: 300000000
  # brcfr_interval: 100000000
  # dcfr_alpha: 1.5
  # dcfr_gamma: 2.0
```

**Step 2: Update training.md**

Add a section describing the BRCFR+ optimizer under the existing optimizer documentation, covering the three config fields and the decay lifecycle.

**Step 3: Commit**

```bash
git add sample_configurations/ docs/training.md
git commit -m "docs: add BRCFR+ optimizer documentation and sample config"
```

---

### Task 8: E2E convergence test

**Files:**
- Modify: `crates/core/tests/blueprint_v2_e2e.rs`

**Step 1: Write the test**

```rust
#[test]
fn brcfr_plus_converges_small_game() {
    // Build a small config (Kuhn-like or minimal hold'em).
    let mut config = minimal_test_config();
    config.training.optimizer = "brcfr+".to_string();
    config.training.brcfr_eta = 0.6;
    config.training.brcfr_warmup_iterations = 500;
    config.training.brcfr_interval = 200;
    config.training.iterations = Some(2000);
    config.training.exploitability_samples = 50;

    let mut trainer = BlueprintTrainer::new(config).expect("trainer");
    trainer.train().expect("brcfr+ training should complete");

    let exploit = trainer.compute_exploitability();
    // Should converge to reasonable exploitability for a small game.
    assert!(exploit < 500.0, "exploitability should decrease: got {exploit}");
}
```

Adapt the test helper (`minimal_test_config`) to match existing E2E test patterns in the file.

**Step 2: Run test**

Run: `cargo test -p poker-solver-core --test blueprint_v2_e2e -- brcfr_plus_converges`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/core/tests/blueprint_v2_e2e.rs
git commit -m "test: add BRCFR+ E2E convergence test"
```
