---
# poker_solver_rust-kiqt
title: Audit Blueprint MP suspicious preflop strategies
status: in-progress
type: epic
priority: high
created_at: 2026-05-20T13:27:20Z
updated_at: 2026-05-20T13:27:20Z
---

Coordinate the fast audit/debug pass for suspicious Blueprint MP preflop output in the 6-max lazy_sparse trainer, especially suited Ax and BTN spots appearing to fold when they should open, call, or raise.

## Scope

- [ ] Distinguish raw stored strategy from TUI/display artifacts
- [ ] Audit preflop action legality
- [ ] Compare eager and lazy action generation
- [ ] Explain max_flop_players effects
- [ ] Trace one suspect infoset regret update
- [ ] Verify preflop bucket/key identity
