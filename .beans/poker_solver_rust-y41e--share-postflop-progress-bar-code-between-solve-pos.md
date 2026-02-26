---
# poker_solver_rust-y41e
title: Share postflop progress bar code between solve-postflop and solve-preflop
status: completed
type: task
priority: normal
created_at: 2026-02-26T17:40:54Z
updated_at: 2026-02-26T17:44:17Z
---

Extract the multi-progress bar infrastructure (FlopSlotData, FlopBarState, refresh_flop_slots, BuildPhase callback, cleanup) into a shared function. Both run_solve_postflop and run_solve_preflop should use it instead of duplicating progress code.

## Tasks
- [x] Extract shared progress structs and functions into a reusable helper
- [x] Update run_solve_postflop to use the shared helper
- [x] Update run_solve_preflop to use the shared helper
- [x] Verify compilation and tests pass

## Summary of Changes

Extracted `build_postflop_with_progress()` shared function in `crates/trainer/src/main.rs` that encapsulates:
- MultiProgress setup with per-flop sub-bars (FlopSlotData, FlopBarState, refresh_flop_slots)
- BuildPhase callback handling (FlopProgress, MccfrFlopsCompleted, fallback spinner)
- Bar cleanup and completion message

Both `run_solve_postflop` and `run_solve_preflop` now call this shared function instead of duplicating the progress infrastructure.
