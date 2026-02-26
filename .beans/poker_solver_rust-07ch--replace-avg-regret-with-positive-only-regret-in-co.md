---
# poker_solver_rust-07ch
title: Replace avg_regret with positive-only regret in convergence.rs
status: completed
type: task
priority: normal
created_at: 2026-02-25T16:48:34Z
updated_at: 2026-02-25T16:51:30Z
---

Replace the body of `avg_regret()` in `crates/core/src/cfr/convergence.rs` to sum only positive regrets instead of absolute values. Update doc comment. Add test for negative regret filtering.

## Todo
- [ ] Update avg_regret function body and doc comment
- [ ] Update existing tests if needed
- [ ] Add avg_regret_ignores_negative_regrets test
- [ ] Verify tests pass
