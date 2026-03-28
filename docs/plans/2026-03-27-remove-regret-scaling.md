# Remove ×1000 Regret Scaling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Remove the ×1000 fixed-point scaling from regret storage and add Pluribus-style regret clamping, giving ~10.7M iterations of headroom between discounts (up from ~10K).

**Architecture:** The ×1000 scaling is applied at write time in `mccfr.rs` and reversed at read time in `storage.rs`, `trainer.rs`, and `mccfr.rs`. Remove all scaling from regret deltas and baselines/predictions (which share the regret convention). Keep ×1000 for strategy sums and EV tracking (they're `AtomicI64` with ample headroom and need fractional precision). Add regret clamping at ±310M in `add_regret()`.

**Tech Stack:** Rust, `AtomicI32`, poker-solver-core, poker-solver-trainer

---

## Task 1: Add regret clamping and remove scaling from storage.rs

**Files:**
- Modify: `crates/core/src/blueprint_v2/storage.rs`

**Step 1: Write failing test for regret clamping**

Add to the existing test module in `storage.rs`:

```rust
#[test]
fn add_regret_clamps_at_positive_limit() {
    // Build a minimal storage with 1 node, 1 bucket, 1 action
    let tree = /* use existing test helper or build_toy_tree() */;
    let storage = BlueprintStorage::new(&tree, [1, 1, 1, 1]);
    storage.regrets[0].store(300_000_000, Ordering::Relaxed);
    storage.add_regret(0, 0, 0, 100_000_000);
    // Should clamp to 310M, not overflow or panic
    assert!(storage.regrets[0].load(Ordering::Relaxed) <= 310_000_000);
}

#[test]
fn add_regret_clamps_at_negative_limit() {
    let tree = /* same */;
    let storage = BlueprintStorage::new(&tree, [1, 1, 1, 1]);
    storage.regrets[0].store(-300_000_000, Ordering::Relaxed);
    storage.add_regret(0, 0, 0, -100_000_000);
    assert!(storage.regrets[0].load(Ordering::Relaxed) >= -310_000_000);
}
```

**Step 2: Run tests to verify they fail**

```bash
cargo test -p poker-solver-core add_regret_clamps -- --nocapture
```
Expected: FAIL (current code panics on overflow)

**Step 3: Implement regret clamping in `add_regret()`**

Replace the overflow panic in `add_regret()` (lines 184-196) with clamping:

```rust
const REGRET_FLOOR: i32 = -310_000_000;
const REGRET_CAP: i32 = 310_000_000;

pub fn add_regret(&self, node_idx: u32, bucket: u16, action: usize, delta: i32) {
    let nl = &self.layout[node_idx as usize];
    let idx = Self::slot_offset(nl, bucket) + action;
    let old = self.regrets[idx].fetch_add(delta, Ordering::Relaxed);
    let new = old.wrapping_add(delta);
    // Clamp on overflow or exceeding bounds (Pluribus-style regret floor/cap)
    if new > REGRET_CAP || (delta > 0 && new < old) {
        self.regrets[idx].store(REGRET_CAP, Ordering::Relaxed);
    } else if new < REGRET_FLOOR || (delta < 0 && new > old) {
        self.regrets[idx].store(REGRET_FLOOR, Ordering::Relaxed);
    }
}
```

**Step 4: Remove ×1000 from `get_baseline()` and `update_baseline()`**

In `get_baseline()` (line 144): remove `/ 1000.0`
In `update_baseline()` (line 165): remove `/ 1000.0`; (line 167): remove `* 1000.0`

Before:
```rust
f64::from(b[...].load(Ordering::Relaxed)) / 1000.0
```
After:
```rust
f64::from(b[...].load(Ordering::Relaxed))
```

Before:
```rust
let old = f64::from(b[idx].load(Ordering::Relaxed)) / 1000.0;
let new_val = old * (1.0 - alpha) + value * alpha;
b[idx].store((new_val * 1000.0) as i32, Ordering::Relaxed);
```
After:
```rust
let old = f64::from(b[idx].load(Ordering::Relaxed));
let new_val = old * (1.0 - alpha) + value * alpha;
b[idx].store(new_val as i32, Ordering::Relaxed);
```

**Step 5: Run tests**

```bash
cargo test -p poker-solver-core
```

**Step 6: Commit**

```
feat: add regret clamping at ±310M, remove ×1000 from baselines
```

---

## Task 2: Remove ×1000 from regret writes in mccfr.rs

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`

**Step 1: Remove regret scaling at write site**

Line 836 — change:
```rust
let delta_i32 = (delta * 1000.0) as i32;
```
To:
```rust
let delta_i32 = delta as i32;
```

Update the comment at line 828-829:
```rust
// Update regrets: delta = action_value - node_value (integer chip units).
// Skip pruned actions. Also store as prediction for SAPCFR+.
```

**Step 2: Keep strategy sum scaling (already i64)**

Lines 842-843 — **no change**. Strategy sums stay at `(s * 1000.0) as i64` because strategy probabilities are [0,1] and need fractional precision.

**Step 3: Keep EV tracking scaling (already i64)**

Lines 419, 429, 567, 576 — **no change**. EV tracking stays at `* 1000.0` / `/ 1000.0` because EVs are fractional chip values and the buffer is `AtomicI64`.

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core
```

**Step 5: Commit**

```
feat: remove ×1000 scaling from regret deltas in MCCFR
```

---

## Task 3: Remove ×1000 from regret reads in trainer.rs

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

**Step 1: Update `min_regret()` (line 854)**

Remove `/ 1000.0`:
```rust
f64::from(min_raw)
```

Update doc comment (line 843-844): remove "divided by the ×1000 scaling factor used in storage."

**Step 2: Update `max_regret()` (line 868)**

Remove `/ 1000.0`:
```rust
f64::from(max_raw)
```

Update doc comment (line 857-858).

**Step 3: Update `avg_pos_regret()` (line 892)**

Change:
```rust
sum / count as f64 / 1000.0 / self.iterations as f64
```
To:
```rust
sum / count as f64 / self.iterations as f64
```

**Step 4: Update `prune_fraction()` (lines 901-902)**

Change:
```rust
// Config is in BB units; stored regrets are ×1000.
let threshold = self.config.training.prune_threshold.saturating_mul(1000);
```
To:
```rust
// Config prune_threshold is in chip units, matching stored regrets directly.
let threshold = self.config.training.prune_threshold;
```

**Step 5: Update prune threshold in training loop (lines 511-512)**

Change:
```rust
// Config prune_threshold is in chip units; stored regrets are ×1000.
let threshold = self.config.training.prune_threshold.saturating_mul(1000);
```
To:
```rust
// Config prune_threshold is in chip units, matching stored regrets directly.
let threshold = self.config.training.prune_threshold;
```

**Step 6: Update tests (lines 1348-1419)**

The tests use literal `1000` to represent "1.0 chips in ×1000 format". Now `1` represents 1 chip.

`lcfr_discount_at_t_zero` (line 1348):
```rust
trainer.storage.regrets[0].store(1, Ordering::Relaxed);
trainer.storage.strategy_sums[0].store(2000, Ordering::Relaxed); // strategy sums keep ×1000
```

`dcfr_epoch_cap_limits_discount` (line 1372):
```rust
trainer.storage.regrets[0].store(1000, Ordering::Relaxed); // 1000 chips
// d_pos = 5/6 → expected = (1000 * 5/6) as i32 = 833
let expected_regret = (1000.0 * 5.0 / 6.0) as i32; // still 833 — unchanged, just different units
```

Actually these tests are discount-formula tests — the regret values are arbitrary integers. The discount multiplier is the same regardless of scaling. **These tests don't need to change** because the discount formula operates on raw i32 values and doesn't know about the scaling convention.

**Step 7: Run tests**

```bash
cargo test -p poker-solver-core
```

**Step 8: Commit**

```
feat: remove ×1000 from regret reads in trainer diagnostics and pruning
```

---

## Task 4: Update config docs and TUI sparklines

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs`
- Modify: `crates/trainer/src/blueprint_tui.rs`

**Step 1: Update `prune_threshold` doc comment**

In `config.rs` (lines 157-158), change:
```rust
/// Regret threshold below which actions are pruned (in BB units;
/// internally scaled by ×1000 to match stored regret precision).
```
To:
```rust
/// Regret threshold below which actions are pruned (in chip units,
/// matching stored regret values directly).
```

**Step 2: Update TUI sparkline comments**

In `blueprint_tui.rs` (lines 51-56), update comments:
```rust
// Max positive regret sparkline data
max_regret_history: Vec<u64>,
// Max negative regret sparkline data
min_regret_history: Vec<u64>,
// Avg positive regret sparkline data
avg_pos_regret_history: Vec<u64>,
```

The TUI sparkline code calls `trainer.max_regret()`, `trainer.min_regret()`, `trainer.avg_pos_regret()` which now return unscaled values. The TUI then applies its own `* 1000.0` for sparkline integer resolution (lines 153, 171, 205). This is fine — the TUI's `* 1000.0` is for sparkline display resolution, independent of storage scaling.

However, verify the sparkline display labels still make sense. The values shown to the user should now be in chip units directly (no longer "×1000 internally"). Check lines 436, 452 where sparkline values are converted back for display.

**Step 3: Run full test suite**

```bash
cargo test
```

**Step 4: Commit**

```
chore: update config docs and TUI comments for unscaled regret storage
```

---

## Task 5: Verify serialization and document the break

**Files:**
- Modify: `crates/core/src/blueprint_v2/storage.rs` (add version comment to save/load)

**Step 1: Verify serialization functions**

Read `save_regrets()` and `load_regrets()` in `storage.rs`. They serialize `Vec<i32>` directly via bincode — no scaling applied during save/load. The format is unchanged; only the semantic interpretation of values changes (old files had values ×1000 larger).

**Step 2: Add a comment documenting the break**

At the top of `save_regrets()`:
```rust
/// Save regret and strategy sum buffers to disk.
///
/// Regrets are stored as raw chip-unit i32 values (no scaling factor).
/// Files saved before the scaling removal (pre-2026-03-27) have regrets
/// ×1000 larger and are NOT compatible — discard old checkpoints.
```

**Step 3: Run full test suite**

```bash
cargo test
time cargo test  # verify under 1 minute
```

**Step 4: Final build check**

```bash
cargo build --release 2>&1 | grep "^warning\|^error" | head -10
```

**Step 5: Commit**

```
feat: remove ×1000 regret scaling — Pluribus-style integer chip units

Regret deltas are now stored as raw integer chip values instead of
×1000 fixed-point. This gives ~10.7M iterations of i32 headroom
between discounts (up from ~10K), enabling Pluribus-style long
discount intervals. Regret clamping at ±310M prevents overflow.

Strategy sums and EV tracking keep ×1000 scaling (AtomicI64).

BREAKING: Old .bin checkpoint files are incompatible (regrets 1000× larger).
```
