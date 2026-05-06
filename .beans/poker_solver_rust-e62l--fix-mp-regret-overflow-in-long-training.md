---
# poker_solver_rust-e62l
title: Fix MP regret overflow in long training
status: completed
type: bug
priority: critical
created_at: 2026-05-06T13:23:38Z
updated_at: 2026-05-06T13:34:57Z
---

Latest 6-max MP trainer run has correct open actions but still saturates positive regret after roughly 1.5B iterations. Current MpStorage stores regrets as AtomicI16 with saturating adds, so long-running training can clamp the regret signal at i16::MAX. Investigate and implement a durable fix for billion-iteration MP runs.

## Implementation Plan

- [x] Convert MP regret storage from AtomicI16 to AtomicI32
- [x] Update regret delta, pruning threshold, DCFR discount, and tests for i32 regrets
- [x] Update MP TUI regret telemetry scans for AtomicI32
- [x] Run focused storage/trainer/TUI tests
- [x] Close bean and commit implementation

## Implementation checklist

- [x] Convert MP regret storage/API from i16 to i32
- [x] Update MCCFR regret delta and prune-threshold plumbing
- [x] Update trainer DCFR discount path
- [x] Update MP TUI regret telemetry
- [x] Update/add focused regression tests
- [x] Run focused test filters

## Summary of Changes

Converted blueprint_mp cumulative regret storage from AtomicI16 to AtomicI32 so long 6-max runs no longer clamp at i16::MAX. Updated MCCFR regret deltas and prune-threshold plumbing to use i32, updated DCFR discounting for i32 regrets, and updated MP TUI regret telemetry scans/tests for AtomicI32. Added regressions proving regret storage and telemetry can exceed i16::MAX without saturating.

## Verification

- cargo test -p poker-solver-core blueprint_mp::storage -- --nocapture
- cargo test -p poker-solver-core blueprint_mp::mccfr -- --nocapture
- cargo test -p poker-solver-core blueprint_mp::trainer -- --nocapture
- cargo test -p poker-solver-trainer mp_tui -- --nocapture

## Documentation

Updated architecture/training docs to note that blueprint_mp cumulative regrets now use 32-bit atomics and to list the simplified 6-max MP sample config.
