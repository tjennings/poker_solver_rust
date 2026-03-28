# Remove ×1000 Regret Scaling + Add Regret Floor — Design

**Date:** 2026-03-27

## Problem

Regrets are stored as `AtomicI32` with a ×1000 fixed-point scaling factor. With a 200-chip stack, max delta per iteration is 200,000 (200 × 1000), giving only ~10,700 iterations of worst-case headroom before `i32` overflow. This forces frequent discounting (`lcfr_discount_interval`) and prevents running long warmup phases like Pluribus (which discounted only every 10 minutes of wall-clock time).

## Solution

Remove the ×1000 scaling entirely (Pluribus-style raw integer chip units) and add Pluribus-style regret clamping at ±310,000,000.

### What changes

1. **Remove `* 1000.0` scaling on writes:**
   - `mccfr.rs:836` — regret deltas: `(delta * 1000.0) as i32` → `delta as i32`
   - `mccfr.rs:843` — strategy sums: `(s * 1000.0) as i64` → `s as i64`... wait, strategy probs are [0,1], so raw cast truncates to 0. Strategy sums need different handling — see below.
   - `mccfr.rs:419,567` — EV tracking: `(ev * 1000.0) as i64` → `ev as i64`... same issue for fractional EVs.

2. **Remove `/ 1000.0` scaling on reads:**
   - `storage.rs:145,165,167` — strategy computation and baseline reads
   - `trainer.rs:511,854,868,892,901,999,1018,1020` — diagnostic reporting, prune threshold comparison

3. **Add regret clamp in `add_regret()`:**
   - Floor at `-310_000_000` and cap at `310_000_000`
   - Replace the existing overflow panic with silent clamping

4. **Update config interpretation:**
   - `prune_threshold` — currently documented as "chip units, internally scaled ×1000". Now just chip units directly.

5. **Serialization break:**
   - Old `.bin` checkpoint files have regrets 1000× larger than new format
   - No migration tool needed — old checkpoints are disposable training state

### Strategy sums and EV tracking

Strategy probabilities are in [0.0, 1.0] and EV values can be fractional. Removing scaling entirely would lose all sub-integer precision. Two options:

- **Keep ×1000 for strategy sums and EVs only** — regrets go unscaled, strategy sums keep ×1000. This is asymmetric but each buffer has independent semantics.
- **Use a smaller scale (×100) for strategy sums** — reduces precision slightly but stays consistent.

**Decision:** Keep strategy sums at ×1000 (they're already `AtomicI64` with massive headroom). Only remove scaling from regret deltas (which are chip-unit differences where sub-chip precision is noise). EV tracking keeps ×1000 since it's also `AtomicI64`.

### Headroom after change

| Metric | Before (×1000) | After (no scaling) |
|--------|----------------|-------------------|
| Max delta/iter | 200,000 | 200 |
| i32 headroom | ~10,700 iters | ~10,700,000 iters |
| With 310M clamp | N/A | ~1,550,000 iters before clamp |
| Practical limit | Must discount every ~10K iters | Can run millions between discounts |

### What does NOT change

- `AtomicI32` for regrets (no type change)
- `AtomicI64` for strategy sums
- ×1000 scaling for strategy sums (already i64, plenty of headroom)
- ×1000 scaling for EV tracking (already i64)
- Discount logic in optimizers (type-agnostic float multiplication)
- Baseline/prediction buffers follow regret convention (no scaling)
- All public APIs

### Files affected

- `crates/core/src/blueprint_v2/mccfr.rs` — remove `* 1000.0` on regret writes, keep for strategy sums
- `crates/core/src/blueprint_v2/storage.rs` — add regret clamping in `add_regret()`, update reads
- `crates/core/src/blueprint_v2/trainer.rs` — update diagnostic reads that divide by 1000.0
- `crates/core/src/blueprint_v2/config.rs` — update prune_threshold docs
- `crates/core/src/cfr/optimizer.rs` — verify no scaling assumptions (should be clean)
- `crates/trainer/src/blueprint_tui.rs` — update sparkline scaling comments
- Tests throughout that use literal regret values (e.g., `1000` meaning 1.0 chips → now just `1`)

### Risk

- **Precision loss in regrets:** Sub-chip precision gone. For a solver using bucket abstraction, regret precision is already dominated by abstraction error and MCCFR sampling variance. Non-issue.
- **Serialization break:** Old checkpoints incompatible. Acceptable — they're training state, not final strategies.
- **Strategy sum scaling stays:** Asymmetric but correct — strategy probs need fractional precision, regret deltas don't.
