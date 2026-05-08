---
# poker_solver_rust-uvx8
title: Make direct CFVNet a boundary evaluator option
status: completed
type: task
priority: normal
created_at: 2026-05-08T19:22:04Z
updated_at: 2026-05-08T19:25:42Z
parent: poker_solver_rust-fp06
---

Adjust Explorer settings so direct turn-boundary CFVNet is a distinct boundary evaluator option instead of a secondary model-kind selector. Keep legacy CFVNet available until the direct path is confirmed working.\n\n- [x] Replace model-kind selector with distinct boundary mode option\n- [x] Map direct option to cfvnet inference_mode=direct\n- [x] Preserve legacy cfvnet option as river_enumerated_turn\n- [x] Update tests and docs\n- [x] Verify frontend build/tests and relevant Rust tests

## Summary of Changes

Replaced the secondary model-kind selector with a distinct Direct CFVNet boundary evaluator option. The frontend now maps Direct CFVNet to backend cfvnet inference_mode=direct, maps legacy CFVNet to river_enumerated_turn, keeps shared model-path picking, and updates tests/docs to describe the temporary legacy option.
