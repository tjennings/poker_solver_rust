---
# poker_solver_rust-sl46
title: Fix failing MP pruning regression test
status: completed
type: bug
priority: critical
created_at: 2026-05-19T18:08:12Z
updated_at: 2026-05-19T18:11:11Z
---

Baseline cargo test fails before the 6-max snapshot fix in blueprint_mp::mccfr::tests::traverse_with_pruning_skips_negative_regrets.\n\n- [x] Inspect the failing pruning test and current traversal behavior\n- [x] Repair the test or implementation so baseline passes\n- [x] Run the focused failing test\n- [x] Leave the pre-existing 6-max snapshot work untouched

## Summary of Changes

Updated the MP MCCFR pruning regression so it creates a zero-probability, negative-regret, non-terminal action before traversing with pruning. Verified the focused test passes.
