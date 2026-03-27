---
# poker_solver_rust-7axe
title: Audit fold EV convention — should be relative to voluntary investment
status: todo
type: task
priority: normal
created_at: 2026-03-27T15:49:59Z
updated_at: 2026-03-27T15:52:21Z
---

Terminal payoff formulas are already correct:
- Fold/Lose: EV = -invested (stacks[t] - starting_stack)
- Win: EV = pot - invested
- Tie: EV = pot/2 - invested

This matches standard convention. Blinds are an investment cost.
BB folding preflop = -2 chips (-1BB) which is correct.

Still need to verify:
- [ ] Per-node EV display: should show EV at that decision point (strategy-weighted across all continuations), not just fold EV
- [ ] Check if CFR regrets use the same payoff convention (they should)
- [ ] Confirm display matches PioSOLVER/GTO Wizard
