---
# poker_solver_rust-uioh
title: Audit Blueprint MP preflop action legality
status: todo
type: bug
priority: high
created_at: 2026-05-20T13:27:33Z
updated_at: 2026-05-20T13:27:33Z
parent: poker_solver_rust-kiqt
---

Verify legal action sets for all key preflop public states before debugging CFR updates.

## Subtasks

- [ ] Assert unopened non-blind positions with allow_preflop_limp=false expose fold plus configured opens, not limp/call
- [ ] Assert SB/BB blind-posted unopened behavior matches intended blind rules
- [ ] Assert responses after an open include fold, call when legal, configured raises, and all-in when intended
- [ ] Assert action labels and sizes match config lead=[2bb] and raise rows
- [ ] Cross-reference existing bean poker_solver_rust-3t3n and update/close it if resolved by this work
- [ ] Add targeted tests for UTG, HJ, CO, BTN, SB, and BB preflop paths
