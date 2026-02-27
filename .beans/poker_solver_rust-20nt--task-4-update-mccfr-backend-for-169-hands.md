---
# poker_solver_rust-20nt
title: 'Task 4: Update MCCFR backend for 169 hands'
status: completed
type: task
priority: normal
created_at: 2026-02-26T05:58:06Z
updated_at: 2026-02-26T06:22:31Z
parent: poker_solver_rust-fga5
---

Replace FlopBucketMap with combo_map, use 169-hand indexing

## Summary of Changes

- Deleted `FlopBucketMap` struct entirely
- Changed `build_mccfr` return type from `(StreetBuckets, PostflopValues)` to `PostflopValues`
- Replaced all `FlopBucketMap` usage with `build_combo_map()` from `postflop_hands`
- Updated `sample_deal` to take `combo_map: &[Vec<(Card, Card)>]`
- Updated `mccfr_solve_one_flop` and `mccfr_extract_values` signatures
- All tests updated to use `build_combo_map` and `NUM_CANONICAL_HANDS`
- Removed all EHS clustering code from the MCCFR pipeline
