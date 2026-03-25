---
# poker_solver_rust-ly7w
title: 'Phase 2: Blueprint MCCFR comparison'
status: in-progress
type: feature
priority: normal
created_at: 2026-03-24T22:12:40Z
updated_at: 2026-03-25T03:04:12Z
parent: poker_solver_rust-elst
---

Run blueprint MCCFR solver on the Flop Poker game and compare against the exact baseline.

## Key Work
- [ ] Run clustering pipeline on Flop Poker to produce buckets (or use 169 canonical hands as simple first pass)
- [ ] MCCFR solver adapter implementing ConvergenceSolver trait
- [ ] Bucket-to-combo strategy lifting (map bucket-level strategy back to individual combos)
- [ ] Compute exploitability of MCCFR strategy in the full (unabstracted) game
- [ ] run-solver --solver mccfr CLI subcommand
- [ ] Compare report: exploitability gap, L1 distance, combo EV diff vs baseline

## Open Questions
- Can we bypass full clustering and use 169 canonical preflop hand indices as buckets for a simple first pass?
- How to build the blueprint game tree for postflop-only (no preflop)?
- What bucket counts to test? (100, 200, 500, 1000?)

## Blocked By
Phase 1 (complete)
