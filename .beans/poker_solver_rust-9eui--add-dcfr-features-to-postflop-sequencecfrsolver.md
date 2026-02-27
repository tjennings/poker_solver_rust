---
# poker_solver_rust-9eui
title: Add DCFR features to postflop SequenceCfrSolver
status: completed
type: feature
priority: normal
created_at: 2026-02-26T23:40:24Z
updated_at: 2026-02-26T23:57:30Z
---

Port alternating updates, exploration, warmup, and CFR variant selection from preflop solver to sequence-form CFR solver.

## Tasks

- [x] Step 1: Add `Linear` variant to shared `CfrVariant`
- [x] Step 2: Expand `SequenceCfrConfig` with new fields
- [x] Step 3: Add `player_map` to solver
- [x] Step 4: Switch `run_iteration` to alternating updates
- [x] Step 5: Modify `forward_pass` for exploration
- [x] Step 6: Modify `backward_pass` for alternating + exploration + variant weighting
- [x] Step 7: Implement variant-aware discounting
- [x] Step 8: Update `train_streaming`
- [x] Step 9: Unify trainer's `CfrVariant`
- [x] Step 10: Tests

## Summary of Changes

- Added `Linear` variant to shared `CfrVariant` enum in `preflop/config.rs`
- Expanded `SequenceCfrConfig` with `cfr_variant`, `dcfr_warmup`, `exploration` fields
- Added `player_map` to solver for per-player discounting
- Switched `run_iteration` and `train_streaming` to alternating player updates
- Modified `forward_pass` with ε-greedy exploration on hero strategy
- Modified `backward_pass` for hero-only regret/strategy updates with iteration weighting
- Implemented variant-aware discounting: pre-discount (DCFR/Linear) + post-update (CFR+)
- Unified trainer's local `CfrVariant` with core import; added `dcfr_warmup` and `seq_exploration` params
- Added 5 new tests covering all variants and features
- Updated `docs/training.md` with CFR variant documentation
