---
# poker_solver_rust-ski0
title: Harden solved matrix navigation across street actions
status: completed
type: bug
priority: critical
created_at: 2026-05-05T20:36:53Z
updated_at: 2026-05-05T20:49:08Z
---

Ensure solved subgame/exact representative matrices stay attached for every action navigation node within the solved street subtree, not just the first child transition.

- [x] Add broad regression coverage for navigating multiple actions after a street-root solve
- [x] Fix any cache/path miss found by the regression
- [x] Verify focused Tauri tests on main

## Notes

- Added subgame and exact regressions that navigate root, check branch, SB bet branches, BB response branches, sibling BB bet branches, and backtracking within the same turn street.
- The broader regression confirmed the existing solved-cache navigation covers those paths when the session has the real chance/card history; no production path change was needed beyond the prior turn-root exact solve behavior.

## Summary of Changes

- Merged the regression coverage to local main.
- Verified `cargo test -p poker-solver-tauri solved_matrix_navigation_covers_turn_street -- --nocapture` on main.
