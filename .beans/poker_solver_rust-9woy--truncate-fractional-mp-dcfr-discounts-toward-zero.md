---
# poker_solver_rust-9woy
title: Truncate fractional MP DCFR discounts toward zero
status: in-progress
type: bug
priority: high
created_at: 2026-08-04T18:00:16Z
updated_at: 2026-08-04T18:18:54Z
blocked_by:
    - poker_solver_rust-jyth
---

Make integer regret discounting symmetric and truncating toward zero. A discounted regret with absolute value below 1.0 must store as 0, for both negative and positive inputs. Remove MP round-to-nearest behavior that leaves -1 sticky under the default 0.5 negative discount.

- [x] Pass clean-worktree and under-one-minute baseline test gates
- [x] Research and specify rounding semantics across blueprint trainers
- [ ] Implement symmetric truncation in MP eager and lazy sparse discount paths
- [ ] Add regression tests for positive and negative fractional results
- [ ] Review implementation and repair findings
- [ ] Pass focused and full under-one-minute verification
- [ ] Update documentation if the persisted regret semantics warrant it
