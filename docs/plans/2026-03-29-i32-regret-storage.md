# i32 Regret Storage Migration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Migrate regret, strategy sum, baseline, and prediction storage from AtomicI64 to AtomicI32, halving memory usage. Drop REGRET_SCALE from 100,000 to 1,000.

**Architecture:** Change the atomic type in BlueprintStorage's flat buffers, update REGRET_SCALE, propagate i32 through the CfrOptimizer trait, MCCFR traversal, trainer, and serialization. EV trackers (FullTreeEvTracker, ScenarioEvTracker) stay i64 — they're small and not the bottleneck.

**Tech Stack:** Rust, AtomicI32, bincode

**Breaking change:** Snapshot binary format changes from i64 to i32. Old snapshots won't load. A migration helper is provided for converting existing snapshots.

---

### Task 1: Change REGRET_SCALE and storage buffer types

**Files:**
- Modify: `crates/core/src/blueprint_v2/storage.rs`

**Changes:**

1. Change `REGRET_SCALE` constant (line 27):
   ```rust
   // Was: pub(super) const REGRET_SCALE: f64 = 100_000.0;
   pub(super) const REGRET_SCALE: f64 = 1_000.0;
   ```

2. Change all 4 buffer fields from `AtomicI64` to `AtomicI32` (lines 45-55):
   ```rust
   pub regrets: Vec<AtomicI32>,
   pub strategy_sums: Vec<AtomicI32>,
   pub(crate) baselines: Option<Vec<AtomicI32>>,
   pub(crate) predictions: Option<Vec<AtomicI32>>,
   ```

3. Update `regret_floor` field type: `i64` → `i32`

4. Update all initializers: `AtomicI64::new(0)` → `AtomicI32::new(0)` (lines 117, 118, 137, 240)

5. Update all accessor return types and parameter types:
   - `get_regret` → returns `i32`
   - `add_regret` → takes `delta: i32`, uses `i32` arithmetic, clamp to `i32` floor
   - `get_strategy_sum` → returns `i32`
   - `add_strategy_sum` → takes `delta: i32`
   - `get_prediction` → returns `i32`
   - `set_prediction` → takes `i32`
   - `get_baseline` / `update_baseline` — internal loads/stores become i32, public API stays f64

6. Update `current_strategy` / `current_strategy_into`: load as i32, cast to f64

7. Update `average_strategy`: load strategy_sums as i32, cast to f64

8. Update `snapshot_strategy_sums`: returns `Vec<i32>` (was `Vec<i64>`)

9. Update `strategy_delta`: takes `prev_sums: &[i32]`

10. Update memory size calculation (line 106-107):
    ```rust
    let regret_bytes = total * 4;   // was * 8
    let strategy_bytes = total * 4; // was * 8
    ```

**Verification:** `cargo test -p poker-solver-core storage::tests` — update test expectations for i32 types.

---

### Task 2: Update CfrOptimizer trait and implementations

**Files:**
- Modify: `crates/core/src/cfr/optimizer.rs`

**Changes:**

1. `CfrOptimizer` trait signatures (lines 26-47): change all `&[AtomicI64]` to `&[AtomicI32]`

2. `DcfrOptimizer::apply_discount` (lines 85-94): load/store as i32
   ```rust
   let v = atom.load(Ordering::Relaxed);          // i32
   atom.store((v as f64 * d) as i32, ...);        // cast back to i32
   ```

3. `DcfrOptimizer::current_strategy` (lines 108-109): load as i32, cast to f64

4. `SapcfrPlusOptimizer::apply_discount` (lines 163-173): same i64→i32 changes
   - Floor negative regrets: `discounted.max(0)` stays the same, just i32

5. `SapcfrPlusOptimizer::current_strategy` (lines 187-189): load i32 regrets/predictions, cast to f64

**Verification:** `cargo test -p poker-solver-core cfr::` — all optimizer tests pass.

---

### Task 3: Update MCCFR traversal cast sites

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`

**Changes:**

1. Regret update (line 852):
   ```rust
   // Was: let delta_scaled = (delta * super::storage::REGRET_SCALE) as i64;
   let delta_scaled = (delta * super::storage::REGRET_SCALE) as i32;
   ```

2. Strategy sum update (line 859):
   ```rust
   // Was: storage.add_strategy_sum(node_idx, bucket, a, (s * 1000.0) as i64);
   storage.add_strategy_sum(node_idx, bucket, a, (s * 1000.0) as i32);
   ```

3. Prediction store (line 854): already passes `delta_scaled`, type follows from above

4. Prune threshold comparison (line 827): `get_regret` now returns i32, `prune_threshold` must be i32

5. `prune_threshold` parameter type in `traverse_external`, `traverse_traverser`, `traverse_opponent`: `i64` → `i32`

6. **Do NOT change** FullTreeEvTracker or ScenarioEvTracker — keep i64 for EV accumulators

**Verification:** `cargo test -p poker-solver-core mccfr::` — all traversal tests pass.

---

### Task 4: Update trainer and TUI audit

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`
- Modify: `crates/trainer/src/blueprint_tui_audit.rs`

**Changes in trainer.rs:**

1. Prune threshold computation (line 517):
   ```rust
   // Was: let threshold = i64::from(self.config.training.prune_threshold) * REGRET_SCALE as i64;
   let threshold = (i64::from(self.config.training.prune_threshold) * REGRET_SCALE as i64) as i32;
   ```
   Note: config `prune_threshold` stays `i32` in config (it's in chip units). The scaling produces a value that must fit i32 — with REGRET_SCALE=1000 and threshold=-200, result is -200,000 which fits.

2. Regret floor wiring (line 241):
   ```rust
   // Was: storage.regret_floor = floor * super::storage::REGRET_SCALE as i64;
   storage.regret_floor = (floor * super::storage::REGRET_SCALE as i64) as i32;
   ```

3. Inline LCFR discount fallback (lines 782-790): load/store as i32

4. Regret statistics methods (lines 861, 875, 899): cast from i32 to f64

**Changes in blueprint_tui_audit.rs:**

1. Line 250: `get_regret()` returns i32 now, division by REGRET_SCALE stays the same
2. Lines 410-411, 427-429: `add_regret` / `add_strategy_sum` calls — pass i32 values

**Verification:** `cargo test -p poker-solver-core trainer::` and `cargo build -p poker-solver-trainer`

---

### Task 5: Update serialization format

**Files:**
- Modify: `crates/core/src/blueprint_v2/storage.rs` (save_regrets, load_regrets)

**Changes:**

1. `save_regrets` (lines 505-521): collect as `Vec<i32>`, serialize tuple `(&[u16; 4], &Vec<i32>, &Vec<i32>)`

2. `load_regrets` (lines 529-582): deserialize as `([u16; 4], Vec<i32>, Vec<i32>)`, store i32 values

3. Add `load_regrets_legacy` method that reads the old `([u16; 4], Vec<i64>, Vec<i64>)` format and converts to i32 by dividing each value by 100 (old_scale / new_scale = 100,000 / 1,000). This enables one-time migration of existing snapshots.

4. In `load_regrets`, try the new i32 format first. If deserialization fails (wrong size), fall back to `load_regrets_legacy`. Log a warning when falling back.

**Verification:** Create a test that saves with new format, loads, and verifies round-trip.

---

### Task 6: Update convergence harness and remaining crates

**Files:**
- Modify: `crates/convergence-harness/src/solvers/mccfr.rs`

**Changes:**
1. Update `traverse_external` calls: `prune_threshold` type i64 → i32
2. Any direct storage access patterns

**Verification:** `cargo build -p convergence-harness`

---

### Task 7: Full workspace build, test, and config update

1. `cargo build` — full workspace compiles
2. `cargo test` — all tests pass in < 1 minute
3. Update `sample_configurations/blueprint_v2_200bkt_sapcfr.yaml`:
   ```yaml
   regret_floor: -310000  # -310,000 chips × 1,000 scale = -310,000,000 internal (matches Pluribus)
   ```
   (Note: config value changes because REGRET_SCALE changed. -3100 old × 100 = -310,000 new.)
4. Update `docs/architecture.md` if it mentions REGRET_SCALE or storage format

---

## Migration Notes

- **Old snapshots**: The legacy loader divides old i64 values by 100 to convert to the new i32 scale. Precision loss is negligible (equivalent to rounding to nearest 0.001 chips).
- **resume: true** will auto-detect and convert old snapshots on first load.
- **Memory impact**: For 1000-bucket model, snapshot drops from ~28 GB to ~14 GB. Training memory follows.
- **No config changes needed** for `prune_threshold` or `prune_streets` — those are in chip units, unchanged.
- **`regret_floor` config value changes**: old `-3100` (× 100K) → new `-310000` (× 1K) = same internal -310M.
