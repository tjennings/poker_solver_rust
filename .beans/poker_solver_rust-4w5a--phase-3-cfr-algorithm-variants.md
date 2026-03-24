---
# poker_solver_rust-4w5a
title: 'Phase 3: CFR algorithm variants'
status: todo
type: feature
created_at: 2026-03-24T22:12:50Z
updated_at: 2026-03-24T22:12:50Z
parent: poker_solver_rust-elst
blocked_by:
    - poker_solver_rust-ly7w
---

Test convergence of different CFR algorithm variants on the same Flop Poker game.

## Algorithms to Test
- [ ] Vanilla CFR (no discounting)
- [ ] CFR+ (regrets floored to zero, linear strategy weighting)
- [ ] Linear CFR (DCFR with alpha=beta=gamma=1.0, Pluribus-style)
- [ ] DCFR with different alpha/beta/gamma parameters
- [ ] PCFR+ (predictive regret matching) — if implemented
- [ ] Any new algorithms

## Deliverables
- [ ] Adapter per algorithm implementing ConvergenceSolver trait
- [ ] Convergence rate comparison across variants (exploitability vs iterations, log-log)
- [ ] Wall-clock time comparison
- [ ] Parameterize range-solver DCFR params (currently hardcoded)

## Blocked By
Phase 2 (MCCFR comparison provides the infrastructure for running multiple solvers)
