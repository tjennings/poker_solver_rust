---
# poker_solver_rust-9woy
title: Truncate fractional MP DCFR discounts toward zero
status: completed
type: bug
priority: high
created_at: 2026-08-04T18:00:16Z
updated_at: 2026-08-04T18:37:01Z
blocked_by:
    - poker_solver_rust-jyth
---

Make integer regret discounting symmetric and truncating toward zero. A discounted regret with absolute value below 1.0 must store as 0, for both negative and positive inputs. Remove MP round-to-nearest behavior that leaves -1 sticky under the default 0.5 negative discount.

- [x] Pass clean-worktree and under-one-minute baseline test gates
- [x] Research and specify rounding semantics across blueprint trainers
- [x] Implement symmetric truncation in MP eager and lazy sparse discount paths
- [x] Add regression tests for positive and negative fractional results
- [x] Review implementation and repair findings
- [x] Pass focused and full under-one-minute verification
- [x] Update documentation if the persisted regret semantics warrant it

## Summary of Changes

Added a shared bounded MP signed-regret discount conversion that truncates toward zero and wired both eager and lazy/sparse trainers through it. Added symmetric regression coverage including ±1×0.5→0 and updated prior nearest-rounding expectations. Left HU fixed-point and strategy-sum discounting unchanged. Updated architecture and training documentation. Independent review found no code defects; its Markdown-table finding was repaired and re-reviewed clean. Verification: formatting and core check passed; 326/326 MP tests passed in 3.70s; the worker full suite passed in 36.1s. Subsequent manager full-suite attempts encountered an unrelated zero-CPU test-binary shutdown flake after completed harnesses, while all changed-path tests remained green.
