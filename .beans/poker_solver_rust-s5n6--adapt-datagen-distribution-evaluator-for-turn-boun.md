---
# poker_solver_rust-s5n6
title: Adapt datagen distribution evaluator for turn-boundary records
status: in-progress
type: task
priority: high
created_at: 2026-05-06T02:20:33Z
updated_at: 2026-05-06T02:23:45Z
---

Ensure the existing cfvnet data distribution diagnostics work for the current turn_boundary datagen output, including SPR/pot coverage and boundary-specific metadata where available.\n\n- [x] Audit existing distribution/evaluator command and current turn_boundary record format\n- [x] Identify whether current command already works on new datagen output\n- [ ] Implement compatibility/reporting fixes if needed\n- [ ] Verify with tests and/or a small generated dataset
