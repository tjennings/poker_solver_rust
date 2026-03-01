# Regret-Based Pruning for Exhaustive Postflop CFR

**Date:** 2026-02-28
**Status:** Approved
**Scope:** `postflop_exhaustive.rs` only (MCCFR deferred)

## Background

Regret-Based Pruning (Brown & Sandholm, NeurIPS 2015) skips subtree traversal for actions whose cumulative regret is negative. Since regret matching already assigns zero probability to these actions, their reach is zero and regret updates contribute nothing. Skipping them avoids wasted computation.

Expected speedup: ~3-5x per-iteration after warmup (60-70% of actions pruned at converged nodes).

## Design

### Pruning Logic (hero decision nodes only)

At each hero decision node in `exhaustive_cfr_traverse`:

1. **Guard:** If iteration < `prune_warmup`, skip all pruning logic.
2. **Guard:** If `iteration % prune_explore_freq == 0`, skip pruning (exploration pass).
3. **Check:** If any action has positive regret (`snapshot[start + i] > 0.0` for any i in 0..num_actions):
   - For each action where `snapshot[start + i] < 0.0`: skip subtree, set `action_values[i] = 0.0`.
   - For actions with non-negative regret: traverse normally.
4. **Else (all regrets non-positive):** Traverse all actions (uniform strategy, no pruning).

### Regret Floor

After DCFR discounting, clamp all cumulative regrets to `>= -regret_floor`. This bounds the "unprune delay" — how many iterations it takes for a pruned action to recover. Applied in the iteration loop after `discount_regrets()`.

### Config Fields

Add to `PostflopModelConfig`:

```rust
/// Iterations before regret-based pruning activates. Default: 200.
pub prune_warmup: usize,
/// Explore all actions every N iterations (disables pruning). Default: 20.
pub prune_explore_freq: usize,
/// Magnitude of negative regret clamp. Default: 1_000_000.0.
pub regret_floor: f64,
```

All fields have `#[serde(default)]` with sensible defaults. Existing configs work unchanged.

### Files Changed

| File | Change |
|-|-|
| `postflop_model.rs` | Add 3 config fields with serde defaults |
| `postflop_exhaustive.rs` | Pruning in hero branch + regret floor after discounting |
| `postflop_abstraction.rs` | Thread config to traversal (if not already available) |

### What This Does NOT Do

- No changes to `postflop_mccfr.rs` (deferred)
- No changes to `DcfrParams` struct
- No changes to preflop solver
