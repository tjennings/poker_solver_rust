---
# poker_solver_rust-6frv
title: 'Phase 4: Parameterized games'
status: todo
type: feature
created_at: 2026-03-24T22:12:57Z
updated_at: 2026-03-24T22:12:57Z
parent: poker_solver_rust-elst
blocked_by:
    - poker_solver_rust-4w5a
---

Extend the convergence harness to support different game configurations.

## Scope
- [ ] Different boards (dry, wet, monotone, paired)
- [ ] Different stack depths (10bb, 20bb, 50bb, 100bb)
- [ ] Different bet structures (single size, multiple sizes)
- [ ] Different pot sizes / SPR ratios
- [ ] Deck trimming for larger games (remove low cards to fit in memory)
- [ ] Ablation: vary bucket counts and plot exploitability vs abstraction granularity

## Blocked By
Phase 3 (algorithm variants establishes the full comparison framework)
