---
# poker_solver_rust-9w6y
title: Add card-removal weighting to postflop exhaustive solver
status: completed
type: bug
priority: normal
created_at: 2026-03-05T15:34:56Z
updated_at: 2026-03-05T16:38:58Z
---

The postflop exhaustive CFR solver passes reach_opp=1.0 for all canonical hand pairs, ignoring how many concrete combo pairs each matchup represents. This distorts strategies by overweighting rare matchups. Fix: add compute_weight_table, thread weights through solver, and weight exploitability computation.


## Implementation

- [x] Add `compute_weight_table` function
- [x] Add `weight_table` field to `PostflopCfrCtx`
- [x] Update `traverse_pair` to use `reach_opp: w` instead of `1.0`
- [x] Add `weight_table` field to `SolveLoopCtx` and thread through
- [x] Update `compute_exploitability` to weight BR values by combo count
- [x] Update `exhaustive_solve_one_flop` signature
- [x] Update `solve_and_extract_flop` in `BuildCtx`
- [x] Update all test call sites
- [x] Add `weight_table_basic_properties` test
- [x] Add `weight_table_agrees_with_equity_table_validity` test (slow/ignored)
- [x] Increase timeout for borderline debug-mode tests
- [x] All tests pass, clippy clean

## Summary of Changes

Added card-removal weighting to the postflop exhaustive CFR solver. The `compute_weight_table` function counts non-overlapping concrete combo pairs per canonical (hero, opp) matchup for a given flop. This weight is threaded through as `reach_opp` in CFR traversal and used for weighted averaging in exploitability computation. Previously all matchups contributed equally (reach_opp=1.0), overweighting rare matchups and underweighting common ones.
