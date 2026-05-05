---
# poker_solver_rust-8rc4
title: Investigate Tauri UI subgame still showing old boundary behavior
status: in-progress
type: bug
priority: critical
created_at: 2026-05-05T20:01:00Z
updated_at: 2026-05-05T20:16:30Z
---

User still sees original Tauri UI behavior after main merge: solving subgame at turn root, BB checks, SB shows default/incorrect subgame matrix and odd all-in/fold response persists. Verify whether Tauri frontend/backend uses fixed solve root and boundary stack/exact-subtree path, and repair if needed.



## 2026-05-05 investigation
- Current main Tauri backend routes game_solve through build_solve_game_with_root and exact_subtree boundary setup.
- Focused backend regression solve_then_bb_check_serves_solved_sb_child_matrix passes on main.
- Devserver against the real bundle confirmed Tauri API selects exact-subtree mode when frontend sends river exact_subtree.
- During boundary setup / iteration 0, source=subgame can still show the blueprint/default root snapshot until the solve advances; after completion it should serve the solved cache.
- If the installed/running Tauri UI still shows old completed behavior, likely stale binary/process or a remaining UI/cache miss needing fresh runtime traces.



## 2026-05-05 follow-up
- User confirmed the completed Tauri solve still shows the bad BB response matrix after solving root spot `sb:2bb,bb:10bb,sb:22bb,bb:call|Ks8d3c|bb:check,sb:15bb,bb:call|Js` and navigating `bb:check,sb:24bb`.
- CLI compare at the BB response confirms AA/KK/nuts call correctly, but many medium hands still overjam versus exact.
- SB response after BB all-in matches exact, so the remaining issue is the call-to-river boundary value, not all-in response folding.
- Suspected root cause: exact_subtree builds non-all-in downstream subtree with effective_stack = pot/2 + remaining_stack instead of stack-behind only.
