---
# poker_solver_rust-ryjw
title: Sync MP preflop side worktree leftovers
status: completed
type: task
priority: normal
created_at: 2026-05-06T13:10:22Z
updated_at: 2026-05-06T13:12:06Z
---

Bring the remaining useful side-worktree changes onto main: MP trainer timed-test guard updates and the simplified 6-max MP config file. Avoid overwriting main's newer unopened-preflop implementation with older side-worktree variants.

## Summary of Changes

Synced the remaining useful side-worktree leftovers onto main without overwriting main's newer unopened-preflop implementation. Added the two MP trainer timed-test guard updates from the side worktree and committed the real simplified 6-max MP config file on main instead of the side worktree's local symlink.

## Verification

- cargo test -p poker-solver-core blueprint_mp::trainer::tests::train_result_tracks_iterations -- --nocapture
- cargo test -p poker-solver-core blueprint_mp::trainer::tests::dcfr_discount_handles_negative_regrets -- --nocapture
