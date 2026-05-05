---
# poker_solver_rust-g781
title: Fix SB child matrix after root subgame solve
status: done
type: bug
priority: high
created_at: 2026-05-05T18:01:56Z
updated_at: 2026-05-05T18:42:00Z
---

After solving a turn subgame at the street root, navigating BB Check to the SB action node still shows the default matrix instead of the solved subgame matrix for that game state.

## TODOs

- [x] Reproduce or encode the BB-check/SB default-matrix fallback case.
- [x] Trace subgame/exact/blueprint matrix source selection after street-root solve.
- [x] Fix solved matrix lookup so child states use solved representative matrices.
- [x] Add regression coverage for solved root -> BB Check -> SB matrix.
- [x] Run targeted and full verification.
- [ ] Merge to local main.

## Summary of Changes

- Added a backend regression for solving a turn root, taking BB Check, and serving the solved SB child matrix from the subgame cache.
- Changed the Game Explorer action handler to refresh the selected strategy source after each action, so blueprint, subgame, and exact re-select the representative matrix for the new game state instead of trusting a stale/default play response.

## Verification

- `cargo test -p poker-solver-tauri solve_then_bb_check_serves_solved_sb_child_matrix`
- `cargo test -p poker-solver-tauri game_session::tests::`
- `npm run build`
- `/usr/bin/time -p cargo test` passed warm in 55.35s
