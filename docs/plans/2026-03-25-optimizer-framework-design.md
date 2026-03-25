# Pluggable Optimizer Framework + VR-MCCFR Baselines — Design

**Date:** 2026-03-25

## Problem

The solver's DCFR discounting is hardcoded in `apply_lcfr_discount()` with fixed α/β/γ. There's no way to experiment with alternative update rules without modifying the training loop. Additionally, MCCFR convergence is bottlenecked by sampling variance — no discount scheme change addresses this.

## Two Workstreams

### Workstream A: Pluggable Optimizer Framework + SAPCFR+

**Goal:** Replace the hardcoded DCFR discount with a pluggable optimizer trait. First implementation: SAPCFR+ (Simplified Asymmetric Predictive CFR+).

#### Architecture

A new `CfrOptimizer` trait defines the interface:

```rust
pub trait CfrOptimizer: Send + Sync {
    /// Called after each MCCFR traversal iteration with the raw instantaneous regret.
    /// Updates internal state (cumulative regrets, predictions, etc.)
    fn update_regrets(&self, storage: &BlueprintStorage, iteration: u64);

    /// Returns the current strategy at (node, bucket) via regret matching on
    /// the optimizer's internal regret state.
    fn current_strategy(&self, node_idx: u32, bucket: u16, out: &mut [f64]);

    /// Accumulate strategy sums (for average strategy computation).
    fn update_strategy_sums(&self, storage: &BlueprintStorage, iteration: u64);

    /// Name for logging/config.
    fn name(&self) -> &str;
}
```

Implementations:
1. **DcfrOptimizer** — wraps current `apply_lcfr_discount` logic (α/β/γ polynomial discount)
2. **SapcfrPlusOptimizer** — SAPCFR+ with prediction buffer and asymmetric step sizes

The trainer holds a `Box<dyn CfrOptimizer>` selected by config. The training loop calls `optimizer.update_regrets()` instead of the inline discount code.

#### SAPCFR+ Mechanics

SAPCFR+ (Simplified Asymmetric Predictive CFR+) extends PCFR+ with:

1. **Prediction buffer** — stores previous iteration's instantaneous regret as prediction for next
2. **Asymmetric step sizes** — prediction step is scaled by a factor `eta < 1.0` to dampen bad predictions
3. **Regret flooring** — cumulative regrets floored to 0 (RM+ style)
4. **DCFR discount** — applied to cumulative regrets before adding prediction (PDCFR+ style)

Per-iteration update at each (infoset, action):
```
r_t = instantaneous_regret                    // from MCCFR traversal
R_t = max(0, R_{t-1} * discount + r_t)        // discounted accumulation + floor
v_{t+1} = r_t                                  // prediction = current regret
R_tilde_{t+1} = max(0, R_t * discount + eta * v_{t+1})  // predicted + scaled
strategy = normalize(R_tilde)                  // regret matching on predicted regrets
```

Where `eta` (default ~0.5) is the asymmetric step size that dampens predictions.

**Storage:** +1 buffer of `AtomicI32` (same size as regret buffer) for the prediction vector. Total: ~1.1 GB extra at 286M slots.

**Config:**
```yaml
training:
  optimizer: "sapcfr+"    # or "dcfr", "lcfr", "cfr+"
  sapcfr_eta: 0.5         # prediction step size (0 = no prediction, 1 = full PCFR+)
  dcfr_alpha: 1.5         # discount exponent (shared with DCFR)
  dcfr_gamma: 2.0         # strategy sum discount
```

### Workstream C: VR-MCCFR Baselines

**Goal:** Add state-action baselines to the MCCFR traversal to reduce sampling variance by ~1000×.

#### Mechanism (Schmid et al., AAAI 2019)

The key insight: in external-sampling MCCFR, the sampled counterfactual value has high variance because it depends on which opponent action was sampled. A baseline subtracts the expected contribution of unsampled actions.

At each opponent decision node, instead of:
```
v(h) = v(h·a_sampled) / π_opp(a_sampled)   // standard importance-sampled value
```

Use:
```
v(h) = b(h) + (v(h·a_sampled) - b(h·a_sampled)) / π_opp(a_sampled)
```

Where `b(h, a)` is the baseline — a running average of previously observed values at (infoset, action). The baseline absorbs the expected value, leaving only the deviation to be importance-weighted.

**Storage:** One `f32` per (decision node, bucket, action) — same layout as regrets. ~1.1 GB at 286M slots.

**Compute:** One table lookup and one running-average update per opponent action during traversal. Negligible.

**Integration:** Modify `traverse_opponent` in `mccfr.rs` to use baseline-corrected sampling.

## Files to Change

**Workstream A:**
- Create: `crates/core/src/cfr/optimizer.rs` — trait + DcfrOptimizer + SapcfrPlusOptimizer
- Modify: `crates/core/src/blueprint_v2/storage.rs` — add prediction buffer
- Modify: `crates/core/src/blueprint_v2/trainer.rs` — use optimizer trait instead of inline discount
- Modify: `crates/core/src/blueprint_v2/config.rs` — add optimizer config

**Workstream C:**
- Modify: `crates/core/src/blueprint_v2/storage.rs` — add baseline buffer
- Modify: `crates/core/src/blueprint_v2/mccfr.rs` — baseline-corrected opponent traversal
- Modify: `crates/core/src/blueprint_v2/config.rs` — add baseline config flag
