---
# poker_solver_rust-mn4k
title: Cross-validation tests for blueprint_mp N-player module
status: completed
type: task
priority: normal
created_at: 2026-04-08T15:12:18Z
updated_at: 2026-04-08T15:17:54Z
---

Write integration tests verifying that blueprint_mp produces reasonable results on 2-player configs.

## Tests
- [x] tree_structure_reasonable -- verifies Decision, Chance, Terminal nodes present; root is Decision; fold and showdown terminals exist
- [x] training_reduces_exploitability_direction -- trains 500+500 iters, verifies BR doesn't explode
- [x] three_player_convergence_smoke -- trains 200 iters of 3-player, verifies iteration count
- [x] three_player_convergence_smoke_strategy_nonzero -- verifies strategy sums populated after 3p training
- [x] payoffs_zero_sum_at_all_terminals -- 2p: contributions sum equals pot at every terminal
- [x] payoffs_zero_sum_at_all_terminals_3p -- 3p: same pot-sum check

## Summary of Changes
Added crates/core/tests/blueprint_mp_validation.rs with 6 integration tests validating tree structure, training convergence, and terminal payoff correctness for the blueprint_mp N-player module.
