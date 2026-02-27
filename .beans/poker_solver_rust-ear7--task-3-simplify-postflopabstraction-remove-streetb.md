---
# poker_solver_rust-ear7
title: 'Task 3: Simplify PostflopAbstraction — remove StreetBuckets'
status: completed
type: task
priority: normal
created_at: 2026-02-26T05:58:06Z
updated_at: 2026-02-26T06:22:31Z
parent: poker_solver_rust-fga5
---

Remove StreetBuckets, BucketEquity, simplify FlopStage and BuildPhase

## Summary of Changes

- Removed `buckets`, `street_equity`, `transitions` fields from `PostflopAbstraction`
- Removed `FlopStage::Bucketing`, `BuildPhase::ExtractingEv`, `BuildPhase::Rebucketing`
- Removed `postflop_terminal_value` function
- Simplified `compute_hand_avg_values` to take only `PostflopValues`
- Simplified `build_from_cached` signature (4 args instead of 7)
- Updated `build()` to use `NUM_CANONICAL_HANDS` and `postflop_hands` imports
- Updated all tests to use 169-hand indexing
