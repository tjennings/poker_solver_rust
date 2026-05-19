---
# poker_solver_rust-sl46
title: Fix failing MP pruning regression test
status: in-progress
type: bug
priority: critical
created_at: 2026-05-19T18:08:12Z
updated_at: 2026-05-19T18:08:12Z
---

Baseline cargo test fails before the 6-max snapshot fix in blueprint_mp::mccfr::tests::traverse_with_pruning_skips_negative_regrets.\n\n- [ ] Inspect the failing pruning test and current traversal behavior\n- [ ] Repair the test or implementation so baseline passes\n- [ ] Run the focused failing test\n- [ ] Resume the 6-max snapshot investigation
