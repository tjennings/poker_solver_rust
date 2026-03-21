---
# poker_solver_rust-i261
title: 'fix(rebel): pot-relative CFVs, unnormalized game_value, pot guard'
status: in-progress
type: bug
created_at: 2026-03-21T03:48:06Z
updated_at: 2026-03-21T03:48:06Z
---

Three critical correctness fixes in rebel solver.rs: (1) divide raw EVs by pot for pot-relative CFVs, (2) remove total_reach normalization from weighted_game_value, (3) add pot <= 0 guard. Must match cfvnet reference.
