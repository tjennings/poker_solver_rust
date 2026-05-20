---
# poker_solver_rust-r1je
title: Explain max_flop_players impact on Blueprint MP preflop calls
status: todo
type: task
priority: high
created_at: 2026-05-20T13:27:44Z
updated_at: 2026-05-20T13:27:44Z
parent: poker_solver_rust-kiqt
---

Quantify which suspicious BTN/call spots are expected consequences of max_flop_players=3 versus actual solver bugs.

## Subtasks

- [ ] Enumerate preflop paths where call is removed by max_flop_players=3
- [ ] Distinguish non-closing calls from closing calls in the cap logic
- [ ] Produce examples where BTN cannot call by design and must fold or raise
- [ ] Verify unopened open frequencies are unaffected by this cap
- [ ] Decide whether TUI/diagnostics should annotate cap-removed calls
- [ ] Update training docs/config comments if current behavior is surprising but intended
