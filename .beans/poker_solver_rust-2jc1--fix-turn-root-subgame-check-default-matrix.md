---
# poker_solver_rust-2jc1
title: Fix turn-root subgame check default matrix
status: completed
type: bug
priority: high
created_at: 2026-05-05T15:18:40Z
updated_at: 2026-05-05T15:45:00Z
---

After solving a subgame at the turn root, clicking BB Check advances to SB but the Subgame tab shows a default/blueprint-looking matrix instead of the solved child matrix.

## Report

User still reproduces on local main after edb8c66c: solve at turn root, BB check, SB shows default matrix.

## TODOs

- [x] Reproduce why the live BB Check path misses solved child matrix overlay.
- [x] Fix source-specific solved matrix selection for turn-root action navigation.
- [x] Add regression coverage for the exact turn root BB Check -> SB child case.
- [x] Run targeted and full verification.
- [x] Merge to local main for testing.

## Summary of Changes

Solved-node matrices now use the range solver's normalized reach weights at the current navigated node instead of reusing the root initial range for every cached node. This makes the SB matrix after BB Check represent the post-check solved state rather than a default/root-range view.

Added regression coverage that locks the root Check action to zero reach, navigates to the child, and verifies the child matrix reflects that post-action reach.

## Verification

- `cargo test -p poker-solver-tauri build_solve_matrix_at_current_uses_navigated_reach_weights`
- `cargo test -p poker-solver-tauri game_session::tests::`
- `cargo test -p poker-solver-tauri`
- `/usr/bin/time -p cargo test` passed warm in 54.74s
