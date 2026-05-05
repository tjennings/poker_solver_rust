---
# poker_solver_rust-mi3q
title: Fix solved matrix cell rendering after reach weighting
status: done
type: bug
priority: high
created_at: 2026-05-05T16:01:19Z
updated_at: 2026-05-05T16:23:00Z
---

After the solved-node matrix fix, matrix cells render as large solid green blocks/columns. The backend now sends normalized reach weights through the existing cell weight field, which the frontend uses for visual reach/availability masking.

## Report

User screenshot 2026-05-05 11:00 shows Subgame solved turn matrix with broken cell rendering after commit ea090047.

## TODOs

- [x] Separate action aggregation reach weighting from the UI display weight field.
- [x] Preserve solved child matrices while restoring sane matrix cell rendering.
- [x] Add regression coverage for solved matrix display weight scale.
- [x] Run targeted and full verification.
- [ ] Merge to local main.

## Resolution

`build_solve_matrix_at_current` now uses normalized range-solver weights only for action/EV aggregation and keeps raw reach weights for the matrix display field and combo detail weights. This preserves the solved child matrix behavior from the previous fix while keeping frontend cell reach rendering bounded.

## Verification

- `cargo test -p poker-solver-tauri game_session::tests::build_solve_matrix_at_current_`
- `cargo test -p poker-solver-tauri game_session::tests::`
- `/usr/bin/time -p cargo test` passed warm in 50.60s
