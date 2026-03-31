---
# poker_solver_rust-zrf6
title: Remove legacy postflop_solve_street code path
status: todo
type: task
created_at: 2026-03-31T01:18:07Z
updated_at: 2026-03-31T01:18:07Z
---

The postflop.rs solve path (postflop_solve_street / postflop_solve_street_core / solve_depth_limited) is unused — the explorer's solve button goes through game_session::game_solve_core instead. Remove:

- postflop_solve_street Tauri command registration in main.rs
- postflop_solve_street_core and postflop_solve_street_impl in postflop.rs
- solve_depth_limited function in postflop.rs
- PostflopState solve-related fields (solve_complete, solve_start, etc.) if only used by this path
- Related dead config (PostflopConfig solve fields)

Keep: build_subgame_solver, seed_solver_with_blueprint, CbvContext, RolloutLeafEvaluator — these are used by game_session.rs.
