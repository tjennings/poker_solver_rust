---
# poker_solver_rust-8rc4
title: Investigate Tauri UI subgame still showing old boundary behavior
status: in-progress
type: bug
priority: critical
created_at: 2026-05-05T20:01:00Z
updated_at: 2026-05-05T20:08:18Z
---

User still sees original Tauri UI behavior after main merge: solving subgame at turn root, BB checks, SB shows default/incorrect subgame matrix and odd all-in/fold response persists. Verify whether Tauri frontend/backend uses fixed solve root and boundary stack/exact-subtree path, and repair if needed.



## 2026-05-05 investigation
- Current main Tauri backend routes game_solve through build_solve_game_with_root and exact_subtree boundary setup.
- Focused backend regression solve_then_bb_check_serves_solved_sb_child_matrix passes on main.
- Devserver against the real bundle confirmed Tauri API selects exact-subtree mode when frontend sends river exact_subtree.
- During boundary setup / iteration 0, source=subgame can still show the blueprint/default root snapshot until the solve advances; after completion it should serve the solved cache.
- If the installed/running Tauri UI still shows old completed behavior, likely stale binary/process or a remaining UI/cache miss needing fresh runtime traces.
