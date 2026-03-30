# BRCFR+ Optimizer Design

## Summary

BRCFR+ is a novel CFR optimizer that augments DCFR+ with periodic best-response prediction passes. After a configurable warmup period, a full BR traversal is run every N iterations for both players. The BR-derived per-infoset regrets are stored in the existing prediction buffer and used as a predictive signal in the strategy computation, decaying linearly to zero over the BR refresh interval.

When the prediction signal is zero (during warmup or at the end of a decay interval), BRCFR+ behaves identically to DCFR+.

## Motivation

- **CFR-BR** (Johanson 2012) shows that having one player always play BR converges to the least exploitable strategy in an abstraction.
- **SAPCFR+** uses the previous iteration's instantaneous regret as a cheap but noisy predictive signal.
- **BRCFR+** interpolates: periodic BR passes provide a high-quality predictive signal at manageable cost. The decay schedule ensures the signal fades naturally, avoiding stale corrections.

## Optimizer

New `BrcfrPlusOptimizer` in `optimizer.rs`. Identical to `DcfrPlusOptimizer` for discounting:

- Positive regrets discounted by `t^alpha / (t^alpha + 1)`
- Negative regrets floored to 0 (RM+ style)
- Strategy sums discounted by `(t / (t + 1))^gamma`
- `dcfr_epoch_cap` respected

Adds BR prediction:

```rust
pub struct BrcfrPlusOptimizer {
    pub alpha: f64,     // positive regret discount exponent
    pub gamma: f64,     // strategy sum discount exponent
    pub eta: f64,       // BR prediction weight
    pub decay: f64,     // current decay factor, set by trainer
}
```

Strategy computation:

```
effective_regret(a) = max(0, R(a) + eta * decay * v_br(a))
strategy(a) = effective_regret(a) / sum(effective_regret)
```

When `decay = 0`, this reduces to exactly DCFR+.

## BR Prediction Pass

Modify `traverse_best_response` to accept an optional prediction target:

```rust
pub fn traverse_best_response(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    rake_rate: f64,
    rake_cap: f64,
    predict_storage: Option<&BlueprintStorage>,  // NEW
) -> f64
```

At traverser decision nodes, when `predict_storage` is `Some`:

```
for each action a:
    delta = action_value[a] - best_response_value
    predict_storage.set_prediction(node_idx, bucket, a, (delta * REGRET_SCALE) as i64)
```

This writes the BR-derived regret signal (how much worse each action is vs the best response action) into the prediction buffer.

Existing callers (`compute_exploitability`) pass `None` — zero behavior change.

## Trainer Integration

### Config

```yaml
optimizer: "brcfr+"
dcfr_alpha: 1.5
dcfr_gamma: 2.0
dcfr_epoch_cap: 40
brcfr_eta: 0.6
brcfr_warmup_iterations: 300000000
brcfr_interval: 100000000
```

### Trainer State

```rust
last_br_iteration: u64,
br_active: bool,
```

### Lifecycle

1. **Warmup** (`iterations < brcfr_warmup_iterations`): `decay = 0`, pure DCFR+. No BR passes. Prediction buffer is all zeros.

2. **BR pass** (at warmup boundary and every `brcfr_interval` thereafter): Sample N deals (reuse `exploitability_samples`), traverse both players with `predict_storage = Some(&storage)`. Record `last_br_iteration`. Exploitability is obtained for free from the same traversal.

3. **Between passes**: Trainer computes decay and sets it on the optimizer before each MCCFR batch:
   ```
   decay = max(0.0, 1.0 - (iterations - last_br_iteration) / brcfr_interval)
   ```

4. **MCCFR traversal**: The per-iteration `set_prediction` call at `mccfr.rs:917` is disabled when using BRCFR+ to prevent overwriting the BR predictions with noisy per-sample signals.

### Schedule Diagram

```
iteration 0          warmup         warmup+N      warmup+2N
    |--- pure DCFR ----|--- DCFR+BR ---|--- DCFR+BR ---|
                       BR₁             BR₂             BR₃
                       v_br = full     v_br = full     v_br = full
                       ↘ decay → 0     ↘ decay → 0     ↘ decay → 0
```

## Shared Prediction Buffer

One shared `Vec<AtomicI64>` buffer (already exists in `BlueprintStorage`). Both players' BR signals coexist without conflict since decision nodes are player-specific. No structural changes to storage.

## Testing

1. **Unit: optimizer** — verify `current_strategy` for known regrets/predictions/decay. Verify `decay = 0` matches DCFR+.
2. **Unit: traverse_best_response** — verify `None` is backward-compatible. Verify `Some` populates buffer with expected deltas.
3. **E2E** — small game (Kuhn), run BRCFR+ for short training, verify exploitability decreases. Compare convergence vs plain DCFR+.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Which player gets BR'd | Both, every pass | Cheap relative to 100M MCCFR iters, keeps both buffers fresh |
| How to collect per-infoset signals | Modify `traverse_best_response` with `predict_storage` param | No duplication, backward-compatible via `None` |
| Where decay lives | Optimizer field, set by trainer | Buffer stays pristine, decay is just a view |
| Prediction buffer | Single shared buffer | Decision nodes are player-specific, no conflict |
| Base behavior | DCFR+ (all params preserved) | BRCFR+ is a strict superset; `decay=0` is identical to DCFR+ |
