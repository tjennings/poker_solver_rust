---
# poker_solver_rust-uvx8
title: Make direct CFVNet a boundary evaluator option
status: in-progress
type: task
created_at: 2026-05-08T19:22:04Z
updated_at: 2026-05-08T19:22:04Z
parent: poker_solver_rust-fp06
---

Adjust Explorer settings so direct turn-boundary CFVNet is a distinct boundary evaluator option instead of a secondary model-kind selector. Keep legacy CFVNet available until the direct path is confirmed working.\n\n- [ ] Replace model-kind selector with distinct boundary mode option\n- [ ] Map direct option to cfvnet inference_mode=direct\n- [ ] Preserve legacy cfvnet option as river_enumerated_turn\n- [ ] Update tests and docs\n- [ ] Verify frontend build/tests and relevant Rust tests
