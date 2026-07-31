---
# poker_solver_rust-84f8
title: Low-SPR 4-bet flop parity smoke for range-solver-compare
status: completed
type: task
priority: high
tags:
    - range-solver
    - compare
created_at: 2026-05-19T15:46:05Z
updated_at: 2026-05-19T16:04:46Z
---

Add deterministic low-SPR 4-bet flop parity smoke coverage to the range-solver-compare harness.

Checklist:
- [x] Commit this bean tracking state before implementation.
- [x] Run the full test suite baseline and confirm it passes in under one minute. Skipped for this session by user request because training is running; initial baseline failed in 27.41s due existing `blueprint_mp::trainer` wall-clock timed-test overruns under load, not compare-harness failures.
- [x] Dispatch research/brainstorming on low-SPR 4-bet flop parity dimensions and scope. Result: represent 4-bet low-SPR as flop roots with large pot/small stack, add two deterministic narrow-range spots, use 20 iterations, reuse tolerant smoke comparator, and collect variance table outside normal test output.
- [x] Dispatch Rust implementation in a separate worktree. Worker added flop smoke in `crates/range-solver-compare/tests/identity.rs`; targeted flop/turn/river smoke passed in the worker tree.
- [x] Dispatch review before integration. Review found no issues.
- [x] Integrate accepted changes into the feature branch. Commit `3080422a` adds low-SPR flop parity smoke coverage.
- [x] Run flop smoke compare tests and existing turn/river compare smoke tests. Full-suite performance gate skipped for this session by user request. Verified targeted flop, turn, river, and 50-river identity compare tests under scoped `RUSTFLAGS`; scratch variance report showed no structural variance and zero numeric deltas at reported precision.
- [x] Complete the bean and commit the final tracking update.

## Summary of Changes

Added Part 3 low-SPR 4-bet flop parity smoke coverage to `crates/range-solver-compare/tests/identity.rs`. The new smoke uses two deterministic flop-root proxy spots with large pots, small stacks, narrow ranges, one bet size, one raise size, and 20 solve iterations. It reuses the shared structural/numeric smoke comparator from the river and turn slices. Targeted compare verification passed for flop, turn, river, and 50-river identity under scoped `RUSTFLAGS='-A dangerous-implicit-autorefs'`. Full-suite timing was intentionally skipped for this session at user request because a training run was active.
