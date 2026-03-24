---
# poker_solver_rust-elst
title: Convergence Validation Harness
status: in-progress
type: epic
created_at: 2026-03-24T22:12:20Z
updated_at: 2026-03-24T22:12:20Z
---

# Convergence Validation Harness

Validate CFR algorithm convergence against an exact baseline. A small, tractable game ("Flop Poker") is solved exhaustively once to produce a golden baseline. Future algorithms run on the same game and are evaluated against that baseline.

## Game: "Flop Poker"
- Board: QhJdTh, all combos (~1176/player), 20bb effective, 2bb pot
- Bet sizes: 67% pot, all-in via thresholds
- Full street transitions: flop → turn → river

## Phases
1. **Exact baseline** — range-solver DCFR produces golden strategy, combo EVs, convergence curve
2. **Blueprint comparison** — run MCCFR with bucketing on same game, compare against baseline
3. **Algorithm variants** — test vanilla CFR, CFR+, Linear CFR, PCFR+, new algorithms
4. **Parameterized games** — different boards, stacks, bet structures

## Design & Plan
- `docs/plans/2026-03-24-convergence-harness-design.md`
- `docs/plans/2026-03-24-convergence-harness-impl.md`
