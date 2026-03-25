# Pluggable Optimizer Framework + VR-MCCFR Baselines — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace hardcoded DCFR discounting with a pluggable optimizer trait (first impl: SAPCFR+), and add variance-reducing baselines to MCCFR opponent traversal.

**Architecture:** New `CfrOptimizer` trait in `crates/core/src/cfr/optimizer.rs` with `DcfrOptimizer` and `SapcfrPlusOptimizer` implementations. Trainer holds `Box<dyn CfrOptimizer>` selected by config. Baselines are an `Option<Vec<AtomicI32>>` buffer in `BlueprintStorage` used during opponent traversal. Both workstreams are independent and can be implemented in parallel.

**Tech Stack:** Rust, rayon (parallel iteration), AtomicI32/AtomicI64 (lock-free concurrent storage)

---

## Workstream A: Pluggable Optimizer Framework + SAPCFR+

### Task 1: Define CfrOptimizer trait and DcfrOptimizer

**Files:**
- Create: `crates/core/src/cfr/optimizer.rs`
- Modify: `crates/core/src/cfr/mod.rs` (add `pub mod optimizer;`)

**Step 1: Create the optimizer trait and DCFR implementation**

The trait must match how the trainer currently interacts with regrets. Currently:
- `storage.current_strategy_into()` (line 166 in storage.rs) reads regrets → strategy
- `storage.add_regret()` (line 110) accumulates instantaneous regrets during traversal
- `apply_lcfr_discount()` (line 718 in trainer.rs) does bulk discount of regrets + strategy sums

The optimizer doesn't replace the per-traversal `add_regret()` calls — those accumulate raw instantaneous regrets as before. The optimizer replaces:
1. **How regrets are discounted** (replaces `apply_lcfr_discount`)
2. **How strategy is derived from regrets** (replaces `current_strategy_into` for SAPCFR+ which uses predicted regrets)

```rust
// crates/core/src/cfr/optimizer.rs

use std::sync::atomic::{AtomicI32, AtomicI64, Ordering};
use rayon::prelude::*;

/// Trait for CFR regret/strategy update rules.
///
/// The optimizer is called periodically by the trainer to apply discounting
/// and compute strategies. Raw instantaneous regrets are accumulated in
/// `BlueprintStorage.regrets` by the traversal code (unchanged).
pub trait CfrOptimizer: Send + Sync {
    /// Apply discount to accumulated regrets and strategy sums.
    /// Called every `lcfr_discount_interval` iterations.
    fn apply_discount(
        &self,
        regrets: &[AtomicI32],
        strategy_sums: &[AtomicI64],
        iteration: u64,
    );

    /// Compute current strategy at (node, bucket) into `out`.
    /// Default: standard regret matching on the regret buffer.
    /// SAPCFR+ overrides this to use predicted regrets.
    fn current_strategy(
        &self,
        regrets: &[AtomicI32],
        offset: usize,
        num_actions: usize,
        out: &mut [f64],
    );

    /// Name for logging.
    fn name(&self) -> &str;
}
```

**Step 2: Implement DcfrOptimizer wrapping the existing logic**

Port the code from `trainer.rs:718-745` into `DcfrOptimizer::apply_discount`:

```rust
pub struct DcfrOptimizer {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl CfrOptimizer for DcfrOptimizer {
    fn apply_discount(
        &self,
        regrets: &[AtomicI32],
        strategy_sums: &[AtomicI64],
        iteration: u64,
    ) {
        let tf = iteration as f64;
        let t_alpha = tf.powf(self.alpha);
        let t_beta = tf.powf(self.beta);
        let d_pos = t_alpha / (t_alpha + 1.0);
        let d_neg = t_beta / (t_beta + 1.0);
        let d_strat = (tf / (tf + 1.0)).powf(self.gamma);

        regrets.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            let d = if v >= 0 { d_pos } else { d_neg };
            atom.store((f64::from(v) * d) as i32, Ordering::Relaxed);
        });

        strategy_sums.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            atom.store((v as f64 * d_strat) as i64, Ordering::Relaxed);
        });
    }

    fn current_strategy(
        &self,
        regrets: &[AtomicI32],
        offset: usize,
        num_actions: usize,
        out: &mut [f64],
    ) {
        // Standard regret matching: normalize positive regrets.
        // Port from storage.rs:166-190
        let mut sum = 0.0f64;
        for a in 0..num_actions {
            let r = regrets[offset + a].load(Ordering::Relaxed);
            let v = if r > 0 { f64::from(r) } else { 0.0 };
            out[a] = v;
            sum += v;
        }
        if sum > 0.0 {
            for v in out[..num_actions].iter_mut() {
                *v /= sum;
            }
        } else {
            let uniform = 1.0 / num_actions as f64;
            for v in out[..num_actions].iter_mut() {
                *v = uniform;
            }
        }
    }

    fn name(&self) -> &str { "dcfr" }
}
```

**Step 3: Add module export**

In `crates/core/src/cfr/mod.rs`, add:
```rust
pub mod optimizer;
pub use optimizer::CfrOptimizer;
```

**Step 4: Write test for DcfrOptimizer**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dcfr_discount_positive_regrets() {
        let opt = DcfrOptimizer { alpha: 1.5, beta: 0.0, gamma: 2.0 };
        let regrets = vec![AtomicI32::new(1000), AtomicI32::new(-500)];
        let strats = vec![AtomicI64::new(2000)];
        opt.apply_discount(&regrets, &strats, 10);
        // After discount at t=10: d_pos = 10^1.5 / (10^1.5 + 1) ≈ 0.969
        let r0 = regrets[0].load(Ordering::Relaxed);
        assert!(r0 > 960 && r0 < 980, "positive regret discounted: {r0}");
        // d_neg = 10^0 / (10^0 + 1) = 0.5
        let r1 = regrets[1].load(Ordering::Relaxed);
        assert_eq!(r1, -250, "negative regret halved");
    }

    #[test]
    fn dcfr_current_strategy_regret_matching() {
        let opt = DcfrOptimizer { alpha: 1.5, beta: 0.0, gamma: 2.0 };
        let regrets = vec![
            AtomicI32::new(300),  // action 0
            AtomicI32::new(100),  // action 1
            AtomicI32::new(-50),  // action 2 (negative, excluded)
        ];
        let mut out = [0.0; 3];
        opt.current_strategy(&regrets, 0, 3, &mut out);
        assert!((out[0] - 0.75).abs() < 0.01);
        assert!((out[1] - 0.25).abs() < 0.01);
        assert!((out[2] - 0.0).abs() < 0.01);
    }
}
```

**Step 5: Build and test**

```bash
cargo test -p poker-solver-core -- cfr::optimizer
```

**Step 6: Commit**

```
git commit -m "feat: CfrOptimizer trait and DcfrOptimizer implementation"
```

---

### Task 2: Add prediction buffer to BlueprintStorage

**Files:**
- Modify: `crates/core/src/blueprint_v2/storage.rs:32-97`

**Step 1: Add optional prediction buffer**

Add a `predictions: Option<Vec<AtomicI32>>` field to `BlueprintStorage`. This stores the previous iteration's instantaneous regret per slot, used by SAPCFR+ to predict the next regret.

In the struct definition (line ~34):
```rust
pub struct BlueprintStorage {
    pub(crate) regrets: Vec<AtomicI32>,
    pub(crate) strategy_sums: Vec<AtomicI64>,
    pub(crate) predictions: Option<Vec<AtomicI32>>,  // NEW
    // ... rest unchanged
}
```

In `new()` constructor, initialize based on a `use_predictions: bool` parameter:
```rust
predictions: if use_predictions {
    Some((0..total_slots).map(|_| AtomicI32::new(0)).collect())
} else {
    None
},
```

**Step 2: Add prediction accessors**

```rust
pub fn get_prediction(&self, node_idx: u32, bucket: u16, action: usize) -> i32 {
    self.predictions.as_ref()
        .map(|p| p[self.slot_index(node_idx, bucket, action)].load(Ordering::Relaxed))
        .unwrap_or(0)
}

pub fn set_prediction(&self, node_idx: u32, bucket: u16, action: usize, value: i32) {
    if let Some(ref p) = self.predictions {
        p[self.slot_index(node_idx, bucket, action)].store(value, Ordering::Relaxed);
    }
}
```

Note: `slot_index` should extract the index computation from `get_regret`. If it's inline, factor it out:
```rust
fn slot_index(&self, node_idx: u32, bucket: u16, action: usize) -> usize {
    let layout = &self.layout[node_idx as usize];
    layout.offset + bucket as usize * layout.num_actions + action
}
```

**Step 3: Update callers of `BlueprintStorage::new()`**

Search for all places `BlueprintStorage::new()` is called and add `false` for `use_predictions` (preserving existing behavior). The trainer will pass `true` when SAPCFR+ is selected.

**Step 4: Build and test**

```bash
cargo build -p poker-solver-core
cargo test -p poker-solver-core
```

**Step 5: Commit**

```
git commit -m "feat: add optional prediction buffer to BlueprintStorage"
```

---

### Task 3: Implement SapcfrPlusOptimizer

**Files:**
- Modify: `crates/core/src/cfr/optimizer.rs`

**Step 1: Implement SAPCFR+**

The SAPCFR+ optimizer needs access to the prediction buffer. Since predictions live in `BlueprintStorage`, the optimizer methods receive them via the storage reference.

However, the current trait takes `&[AtomicI32]` for regrets. For SAPCFR+, we also need predictions. Extend the trait to pass predictions:

```rust
pub trait CfrOptimizer: Send + Sync {
    fn apply_discount(
        &self,
        regrets: &[AtomicI32],
        strategy_sums: &[AtomicI64],
        predictions: Option<&[AtomicI32]>,  // NEW
        iteration: u64,
    );

    fn current_strategy(
        &self,
        regrets: &[AtomicI32],
        predictions: Option<&[AtomicI32]>,  // NEW
        offset: usize,
        num_actions: usize,
        out: &mut [f64],
    );

    fn name(&self) -> &str;

    fn needs_predictions(&self) -> bool { false }
}
```

Update `DcfrOptimizer` to ignore predictions.

```rust
pub struct SapcfrPlusOptimizer {
    pub alpha: f64,    // positive regret discount exponent
    pub gamma: f64,    // strategy sum discount exponent
    pub eta: f64,      // prediction step size (0..1, default 0.5)
}

impl CfrOptimizer for SapcfrPlusOptimizer {
    fn apply_discount(
        &self,
        regrets: &[AtomicI32],
        strategy_sums: &[AtomicI64],
        predictions: Option<&[AtomicI32]>,
        iteration: u64,
    ) {
        let tf = iteration as f64;
        let t_alpha = tf.powf(self.alpha);
        let d_pos = t_alpha / (t_alpha + 1.0);
        let d_strat = (tf / (tf + 1.0)).powf(self.gamma);

        // Discount regrets (floor negative to 0 = RM+ style).
        regrets.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            let discounted = (f64::from(v) * d_pos) as i32;
            atom.store(discounted.max(0), Ordering::Relaxed);  // RM+ floor
        });

        // Discount strategy sums.
        strategy_sums.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            atom.store((v as f64 * d_strat) as i64, Ordering::Relaxed);
        });

        // Predictions are updated per-slot in the traversal (add_regret stores
        // the instantaneous regret as the prediction for next iteration).
        // No bulk discount needed for predictions.
    }

    fn current_strategy(
        &self,
        regrets: &[AtomicI32],
        predictions: Option<&[AtomicI32]>,
        offset: usize,
        num_actions: usize,
        out: &mut [f64],
    ) {
        // SAPCFR+: strategy from predicted regrets = R + eta * v
        let mut sum = 0.0f64;
        for a in 0..num_actions {
            let r = regrets[offset + a].load(Ordering::Relaxed);
            let v = predictions
                .map(|p| p[offset + a].load(Ordering::Relaxed))
                .unwrap_or(0);
            // R_tilde = max(0, R + eta * v)
            let predicted = f64::from(r) + self.eta * f64::from(v);
            let clamped = predicted.max(0.0);
            out[a] = clamped;
            sum += clamped;
        }
        if sum > 0.0 {
            for v in out[..num_actions].iter_mut() {
                *v /= sum;
            }
        } else {
            let uniform = 1.0 / num_actions as f64;
            for v in out[..num_actions].iter_mut() {
                *v = uniform;
            }
        }
    }

    fn name(&self) -> &str { "sapcfr+" }

    fn needs_predictions(&self) -> bool { true }
}
```

**Step 2: Update prediction buffer during traversal**

In `traverse_traverser` (mccfr.rs, ~line 823-831), after computing instantaneous regret `delta`, store it as the prediction:

```rust
// After: storage.add_regret(node_idx, bucket, a, (delta * 1000.0) as i32);
// Add:   storage.set_prediction(node_idx, bucket, a, (delta * 1000.0) as i32);
```

This sets the prediction for the NEXT iteration to be the current instantaneous regret.

**Step 3: Write tests**

```rust
#[test]
fn sapcfr_current_strategy_uses_predictions() {
    let opt = SapcfrPlusOptimizer { alpha: 1.5, gamma: 2.0, eta: 0.5 };
    let regrets = vec![AtomicI32::new(200), AtomicI32::new(100), AtomicI32::new(0)];
    let preds = vec![AtomicI32::new(100), AtomicI32::new(-50), AtomicI32::new(50)];
    let mut out = [0.0; 3];
    opt.current_strategy(&regrets, Some(&preds), 0, 3, &mut out);
    // R_tilde[0] = max(0, 200 + 0.5*100) = 250
    // R_tilde[1] = max(0, 100 + 0.5*(-50)) = 75
    // R_tilde[2] = max(0, 0 + 0.5*50) = 25
    // sum = 350
    assert!((out[0] - 250.0/350.0).abs() < 0.01);
    assert!((out[1] - 75.0/350.0).abs() < 0.01);
    assert!((out[2] - 25.0/350.0).abs() < 0.01);
}

#[test]
fn sapcfr_discount_floors_negative_regrets() {
    let opt = SapcfrPlusOptimizer { alpha: 1.5, gamma: 2.0, eta: 0.5 };
    let regrets = vec![AtomicI32::new(1000), AtomicI32::new(-500)];
    let strats = vec![AtomicI64::new(2000)];
    opt.apply_discount(&regrets, &strats, None, 10);
    let r1 = regrets[1].load(Ordering::Relaxed);
    assert_eq!(r1, 0, "negative regrets floored to 0");
}
```

**Step 4: Build, test, commit**

```bash
cargo test -p poker-solver-core -- cfr::optimizer
git commit -m "feat: SapcfrPlusOptimizer with prediction-based strategy"
```

---

### Task 4: Wire optimizer into trainer

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs:435-544, 618-627, 718-745`
- Modify: `crates/core/src/blueprint_v2/config.rs:136-206`

**Step 1: Add optimizer config**

In `TrainingConfig` (config.rs ~line 136), add:

```rust
/// Optimizer variant: "dcfr" (default), "sapcfr+", "lcfr", "cfr+"
#[serde(default = "default_optimizer")]
pub optimizer: String,

/// SAPCFR+ prediction step size (0 = no prediction, 1 = full PCFR+)
#[serde(default = "default_sapcfr_eta")]
pub sapcfr_eta: f64,
```

With defaults:
```rust
fn default_optimizer() -> String { "dcfr".to_string() }
fn default_sapcfr_eta() -> f64 { 0.5 }
```

**Step 2: Create optimizer in trainer initialization**

In the trainer's constructor or `train()` setup, create the optimizer based on config:

```rust
let optimizer: Box<dyn CfrOptimizer> = match self.config.training.optimizer.as_str() {
    "sapcfr+" => Box::new(SapcfrPlusOptimizer {
        alpha: self.config.training.dcfr_alpha,
        gamma: self.config.training.dcfr_gamma,
        eta: self.config.training.sapcfr_eta,
    }),
    "lcfr" => Box::new(DcfrOptimizer { alpha: 1.0, beta: 1.0, gamma: 1.0 }),
    "cfr+" => Box::new(DcfrOptimizer { alpha: 0.0, beta: -1e9, gamma: 0.0 }),
    _ => Box::new(DcfrOptimizer {
        alpha: self.config.training.dcfr_alpha,
        beta: self.config.training.dcfr_beta,
        gamma: self.config.training.dcfr_gamma,
    }),
};
```

Also enable prediction buffer if optimizer needs it:
```rust
let use_predictions = optimizer.needs_predictions();
// Pass to BlueprintStorage::new()
```

**Step 3: Replace `apply_lcfr_discount` with optimizer call**

In `check_timed_actions()` (~line 618), replace the direct discount with:

```rust
self.optimizer.apply_discount(
    &self.storage.regrets,
    &self.storage.strategy_sums,
    self.storage.predictions.as_deref(),
    t,
);
```

**Step 4: Replace `current_strategy_into` with optimizer call**

In `traverse_traverser` (mccfr.rs ~line 794), replace:
```rust
storage.current_strategy_into(node_idx, bucket, &mut strategy_buf[..num_actions])
```

With a call through the optimizer. Since the traversal functions receive `storage` but not the optimizer, add the optimizer as a parameter to the traversal functions, or store it in a thread-local, or pass it through the traversal context.

The cleanest approach: add `optimizer: &dyn CfrOptimizer` as a parameter to `traverse_external`, `traverse_traverser`, and `traverse_opponent`. Then at line 794:

```rust
let offset = storage.slot_offset(node_idx, bucket);
optimizer.current_strategy(
    &storage.regrets,
    storage.predictions.as_deref(),
    offset,
    num_actions,
    &mut strategy_buf[..num_actions],
);
```

**Step 5: Build and test full pipeline**

```bash
cargo build -p poker-solver-core
cargo test -p poker-solver-core
cargo test -p poker-solver-trainer
```

**Step 6: Commit**

```
git commit -m "feat: wire CfrOptimizer into trainer, configurable via optimizer field"
```

---

### Task 5: Update prediction buffer in traversal

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs:823-836`

**Step 1: Store instantaneous regret as prediction**

After the existing `add_regret` call at line ~830, add prediction update:

```rust
for (a, &av) in action_values.iter().enumerate().take(num_actions) {
    if pruned[a] { continue; }
    let delta = av - node_value;
    let delta_i32 = (delta * 1000.0) as i32;
    storage.add_regret(node_idx, bucket, a, delta_i32);
    storage.set_prediction(node_idx, bucket, a, delta_i32);  // NEW
}
```

This is a no-op when predictions are `None` (DCFR mode).

**Step 2: Build and test**

```bash
cargo test -p poker-solver-core
```

**Step 3: Commit**

```
git commit -m "feat: store instantaneous regret as SAPCFR+ prediction"
```

---

## Workstream C: VR-MCCFR Baselines

### Task 6: Add baseline buffer to BlueprintStorage

**Files:**
- Modify: `crates/core/src/blueprint_v2/storage.rs`
- Modify: `crates/core/src/blueprint_v2/config.rs`

**Step 1: Add baseline buffer**

Add `baselines: Option<Vec<AtomicI32>>` to `BlueprintStorage`. Same layout as regrets — one i32 per (node, bucket, action) slot. Stores running average of sampled counterfactual values × 1000.

```rust
pub(crate) baselines: Option<Vec<AtomicI32>>,
```

Add accessor methods:
```rust
pub fn get_baseline(&self, node_idx: u32, bucket: u16, action: usize) -> f64 {
    self.baselines.as_ref()
        .map(|b| f64::from(b[self.slot_index(node_idx, bucket, action)].load(Ordering::Relaxed)) / 1000.0)
        .unwrap_or(0.0)
}

/// Update baseline with exponential moving average.
/// alpha controls the learning rate (e.g. 0.01 for slow averaging).
pub fn update_baseline(&self, node_idx: u32, bucket: u16, action: usize, value: f64, alpha: f64) {
    if let Some(ref b) = self.baselines {
        let idx = self.slot_index(node_idx, bucket, action);
        let old = f64::from(b[idx].load(Ordering::Relaxed)) / 1000.0;
        let new_val = old * (1.0 - alpha) + value * alpha;
        b[idx].store((new_val * 1000.0) as i32, Ordering::Relaxed);
    }
}
```

**Step 2: Add config flag**

In `TrainingConfig`, add:
```rust
/// Enable variance-reducing baselines for MCCFR.
#[serde(default)]
pub use_baselines: bool,

/// Baseline EMA learning rate.
#[serde(default = "default_baseline_alpha")]
pub baseline_alpha: f64,
```

Default: `fn default_baseline_alpha() -> f64 { 0.01 }`

**Step 3: Build and test**

```bash
cargo build -p poker-solver-core
```

**Step 4: Commit**

```
git commit -m "feat: add optional baseline buffer to BlueprintStorage"
```

---

### Task 7: Implement baseline-corrected opponent traversal

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs:858-898`

**Step 1: Modify `traverse_opponent` to use baselines**

The current opponent traversal (lines ~861-898):
1. Gets opponent's strategy
2. Samples one action
3. Recurses on that action
4. Returns the sampled value

With baselines, change to:
1. Get strategy
2. Sample action `a_sampled`
3. Recurse on `a_sampled` → get `v_sampled`
4. Compute baseline-corrected value:
   ```
   v = sum_a(strategy[a] * baseline[a]) + (v_sampled - baseline[a_sampled]) / strategy[a_sampled]
   ```
   But this reformulation is equivalent to:
   ```
   v = sum_a(strategy[a] * baseline[a])                    // baseline estimate
     + (v_sampled - baseline[a_sampled]) / strategy[a_sampled]  // importance-weighted deviation
   ```
   Simplifying (for external sampling where we already weight by 1/π):
   ```
   // The traverser's value at this opponent node:
   v = v_sampled  // standard path (already correct for traverser)
   ```

Actually, for **external sampling MCCFR**, the baseline applies differently. The traverser visits ALL their own actions but samples ONE opponent action. The counterfactual value computation at traverser nodes doesn't use importance weighting — it's the opponent sampling that creates variance.

The correct VR-MCCFR integration for external sampling (Schmid 2019, Section 4.2):

At the opponent node, instead of just recursing on the sampled action:

```rust
fn traverse_opponent_with_baselines(/* ... */) -> f64 {
    let strategy = /* get opponent strategy */;
    let a_sampled = /* sample from strategy */;

    // Recurse on sampled action
    let v_sampled = traverse_external(/* child of a_sampled */);

    // Update baseline for sampled action
    storage.update_baseline(node_idx, bucket, a_sampled, v_sampled, baseline_alpha);

    // Baseline-corrected value:
    // v = Σ_a π(a) * b(a)  +  (v_sampled - b(a_sampled))
    let mut v = 0.0;
    for a in 0..num_actions {
        v += strategy[a] * storage.get_baseline(node_idx, bucket, a);
    }
    v += v_sampled - storage.get_baseline(node_idx, bucket, a_sampled);

    v
}
```

This is unbiased because `E[v] = Σ_a π(a) * b(a) + Σ_a π(a) * (Q(a) - b(a)) / π(a) * π(a) = Σ_a π(a) * Q(a)` (the standard expected value). But the variance is reduced because the `(v_sampled - b(a))` term is small when the baseline is accurate.

**Step 2: Gate behind config flag**

```rust
if storage.baselines.is_some() {
    // Use baseline-corrected traversal
} else {
    // Original traversal
}
```

Or better: the baseline accessors return 0.0 when baselines are None, so the corrected formula degenerates to the standard formula when all baselines are 0. No branching needed — just always use the corrected formula.

**Step 3: Write test**

```rust
#[test]
fn baseline_corrected_value_is_unbiased() {
    // When baseline == actual value, correction term is 0.
    // When baseline == 0, formula reduces to standard sampling.
    // Test both cases.
}
```

**Step 4: Build and run training test**

```bash
cargo test -p poker-solver-core
# Also run a short training to verify no crash:
cargo run -p poker-solver-trainer --release -- train-blueprint -c sample_configurations/blueprint_v2_1kbkt.yaml
```
(Kill after a few iterations to verify it starts.)

**Step 5: Commit**

```
git commit -m "feat: VR-MCCFR baseline-corrected opponent traversal"
```

---

### Task 8: Integration test — convergence harness comparison

**Files:**
- Modify: `crates/convergence-harness/` (if it exists and is set up for comparing variants)

**Step 1: Run convergence comparison**

Run the convergence harness with:
1. DCFR (baseline, existing)
2. SAPCFR+ (new optimizer)
3. DCFR + baselines (new VR-MCCFR)
4. SAPCFR+ + baselines (both)

Compare exploitability curves over iterations.

**Step 2: Document results**

Record convergence rates and final exploitability in the training docs.

**Step 3: Commit**

```
git commit -m "test: convergence comparison of optimizer variants"
```

---

### Task 9: Update documentation

**Files:**
- Modify: `docs/training.md`
- Modify: `docs/architecture.md`

**Step 1: Document new config options**

```yaml
training:
  optimizer: "sapcfr+"     # Options: "dcfr" (default), "sapcfr+", "lcfr", "cfr+"
  sapcfr_eta: 0.5          # SAPCFR+ prediction step size
  use_baselines: true      # Enable VR-MCCFR variance-reducing baselines
  baseline_alpha: 0.01     # Baseline EMA learning rate
```

**Step 2: Commit**

```
git commit -m "docs: document optimizer and baseline config options"
```
