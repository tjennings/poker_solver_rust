---
# poker_solver_rust-bw9a
title: Trace suspect Blueprint MP infoset regret updates
status: todo
type: bug
priority: high
created_at: 2026-05-20T13:27:50Z
updated_at: 2026-05-20T13:27:50Z
parent: poker_solver_rust-kiqt
---

Instrument or otherwise inspect one suspicious preflop infoset to determine whether action values and regret deltas favor implausible folds.

## Subtasks

- [ ] Choose one primary suspect infoset, e.g. BTN unopened A5s or UTG AJs
- [ ] Run a tiny deterministic training or unit-level traversal with fixed seed/deal stream
- [ ] Log action values, node value, regret deltas, current strategy, average strategy, and terminal payoffs
- [ ] Verify fold EV, open EV, call EV, raise EV signs and magnitudes
- [ ] If fold is incorrectly preferred, audit payoff/blind accounting and apply_action sequence
- [ ] If open is preferred but average stays folded, audit strategy-sum accumulation and discounting
