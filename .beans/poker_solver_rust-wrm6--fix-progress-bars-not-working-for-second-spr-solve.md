---
# poker_solver_rust-wrm6
title: Fix progress bars not working for second SPR solve
status: completed
type: bug
priority: normal
created_at: 2026-02-28T03:29:59Z
updated_at: 2026-02-28T03:30:34Z
---

The progress bars (phase bar + flop bars) don't display/update properly for the 2nd+ SPR iteration in build_postflop_with_progress. Root cause: phase_bar position/length not reset between SPR iterations, and last_refresh not reset in cleanup.

## Summary of Changes\n\nFixed two issues in `build_postflop_with_progress` in `crates/trainer/src/main.rs`:\n1. Reset `phase_bar` position and length to 0 at the start of each SPR iteration to prevent stale bar state from corrupting indicatif's MultiProgress rendering.\n2. Reset `last_refresh` timestamp in the cleanup block between SPR iterations so flop bars appear with correct timing for subsequent solves.
