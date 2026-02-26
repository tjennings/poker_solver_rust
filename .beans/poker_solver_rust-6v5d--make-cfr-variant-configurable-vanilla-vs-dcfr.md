---
# poker_solver_rust-6v5d
title: Make CFR variant configurable (Vanilla vs DCFR)
status: completed
type: feature
priority: normal
created_at: 2026-02-25T22:44:34Z
updated_at: 2026-02-25T22:52:28Z
---

Add a cfr_variant enum to PreflopConfig that controls whether the solver uses Vanilla CFR (no discounting, no iteration weighting) or DCFR (current behavior).

## Tasks
- [x] Add CfrVariant enum + config field to config.rs
- [x] Conditionally apply DCFR behavior in solver.rs
- [x] Add unit test for Vanilla mode
- [x] Update sample YAML config with cfr_variant option
- [x] Run cargo test to verify

## Summary of Changes

- Added `CfrVariant` enum (Vanilla/Dcfr) to `config.rs` with serde support and Default (Dcfr)
- Added `cfr_variant` field to `PreflopConfig` with `#[serde(default)]`
- Solver conditionally applies DCFR discounting (skipped for Vanilla)
- `traverse_hero` uses weight=1.0 (Vanilla) vs weight=iteration (DCFR/LCFR)
- `avg_positive_regret` uses cumulative regret_sum (Vanilla) vs instantaneous regret (DCFR)
- 6 new tests: 3 config deserialization, 3 solver behavior
- Updated sample YAML config with `cfr_variant` option
- All 25 preflop tests passing, clippy clean, trainer compiles
