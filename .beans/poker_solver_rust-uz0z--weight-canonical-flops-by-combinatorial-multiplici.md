---
# poker_solver_rust-uz0z
title: Weight canonical flops by combinatorial multiplicity in postflop EV aggregation
status: completed
type: bug
priority: normal
created_at: 2026-02-27T15:41:32Z
updated_at: 2026-02-27T15:49:54Z
---

compute_hand_avg_values() averages per-flop EVs with equal weight across all canonical flops. But canonical flops have vastly different multiplicities (e.g. rainbow=24, monotone=4). This overweights monotone boards ~3x and underweights rainbow ~0.5x, systematically inflating the suitedness premium in preflop strategy.

Fix: weight each flop's EV contribution by its CanonicalFlop.weight() in compute_hand_avg_values().

## Tasks
- [x] Thread flop weights into PostflopValues or compute_hand_avg_values
- [x] Update compute_hand_avg_values to use weighted average
- [x] Add test verifying weighted vs unweighted aggregation differs
- [x] Verify existing tests pass


## Summary of Changes

- Added `flop_weight_map()` and `lookup_flop_weights()` to `flops.rs` for looking up canonical flop combinatorial weights
- Updated `compute_hand_avg_values()` in `postflop_abstraction.rs` to accept `&[u16]` flop weights and use weighted averaging
- Updated all 3 call sites: `PostflopAbstraction::build()`, `PostflopBundle::into_abstraction()`, and test helper
- Added tests: `compute_hand_avg_values_uses_flop_weights`, `lookup_flop_weights_returns_correct_values`, `lookup_flop_weights_all_flops_sum_to_22100`
