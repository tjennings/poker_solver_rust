---
# poker_solver_rust-7axe
title: Audit fold EV convention — should be relative to voluntary investment
status: todo
type: task
created_at: 2026-03-27T15:49:59Z
updated_at: 2026-03-27T15:49:59Z
---

Current EV tracking reports fold EVs relative to starting stack (including blinds). BB folding preflop shows -1BB instead of 0. Need to verify the convention matches commercial solvers (PioSOLVER, GTO Wizard) and fix if needed.

## TODO
- [ ] Check how PioSOLVER/GTO Wizard report fold EVs
- [ ] Check if this affects CFR regret computation (not just display)
- [ ] If regrets use the same convention, verify it doesn't distort strategy
- [ ] Decide: relative to starting stack vs relative to voluntary investment
- [ ] Fix EV tracker accumulation or display conversion accordingly

Note: fix chip unit unification first (docs/plans/2026-03-27-chip-units-impl.md), then address this.
