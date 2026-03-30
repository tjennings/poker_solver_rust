# i64 Regrets with ×100,000 Scaling

**Date**: 2026-03-28
**Status**: Approved

## Problem

The scaling refactor (commit 7815fe8) removed ×1000 regret scaling, leaving `delta_i32 = delta as i32`. With 1/2 blinds, sub-chip regret deltas (e.g. 0.3 chips) truncate to 0, making the solver blind to most strategic signal. Result: AA has half expected EV, trash hands never fold.

## Solution

Switch regret/prediction/baseline buffers from `AtomicI32` to `AtomicI64` with ×100,000 fixed-point scaling. Provides 0.00001 chip precision with 460 billion iteration headroom before overflow.

## Changes

### 1. storage.rs
- `regrets: Vec<AtomicI32>` → `Vec<AtomicI64>`
- `predictions: Option<Vec<AtomicI32>>` → `Option<Vec<AtomicI64>>`
- `baselines: Option<Vec<AtomicI32>>` → `Option<Vec<AtomicI64>>`
- Add `pub const REGRET_SCALE: f64 = 100_000.0`
- Update `add_regret`, `get_regret`, `get_baseline`, `update_baseline`, `get_prediction`, `set_prediction` — scale on write, unscale on read
- Remove `REGRET_FLOOR`/`REGRET_CAP` (i64 makes clamping unnecessary)
- Update serialization: 8 bytes per regret/prediction/baseline slot
- Tests at every scaling boundary

### 2. mccfr.rs
- `delta as i32` → `(delta * REGRET_SCALE) as i64`
- Baseline reads/writes go through scaled accessors (already do)

### 3. optimizer.rs
- `CfrOptimizer` trait: `&[AtomicI32]` → `&[AtomicI64]`
- Loads produce i64, convert to f64 for computation

### 4. trainer.rs
- `min_regret`/`max_regret`/`avg_positive_regret`/`prune_fraction`: i32→i64 loads, divide by REGRET_SCALE for display
- Prune threshold scaled by REGRET_SCALE at use site

### 5. Snapshot compatibility
- Old i32 snapshots incompatible — discard old checkpoints
- Document in save/load comments

## Not changing
- `strategy_sums` (already i64)
- Config format (`prune_threshold` stays in chip units)
- REGRET_SCALE constant lives in storage.rs; all scaling goes through accessors

## Testing requirements
- Test `add_regret` round-trip: write delta, read back, verify scaling precision
- Test `get_baseline`/`update_baseline` EMA with scaled values
- Test `get_prediction`/`set_prediction` round-trip
- Test optimizer reads scaled i64 regrets correctly
- Test prune threshold scaling
- Test DCFR discounting on i64 values
- Test serialization/deserialization of i64 buffers
