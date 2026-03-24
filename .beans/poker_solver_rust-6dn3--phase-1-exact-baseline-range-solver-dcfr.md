---
# poker_solver_rust-6dn3
title: 'Phase 1: Exact baseline (range-solver DCFR)'
status: completed
type: feature
created_at: 2026-03-24T22:12:31Z
updated_at: 2026-03-24T22:12:31Z
parent: poker_solver_rust-elst
---

Build convergence-harness crate with exact DCFR baseline generation.

## Deliverables
- [x] FlopPokerConfig + configurable game definition
- [x] ConvergenceSolver trait for pluggable algorithms
- [x] ExhaustiveSolver adapter wrapping range-solver
- [x] Baseline serialization (JSON + CSV + bincode)
- [x] Tree walker for strategy + combo EV extraction
- [x] Convergence loop with dense-early sampling
- [x] Comparison metrics (L1 strategy distance, combo EV diff)
- [x] Reporter (terminal, CSV, JSON, human summary)
- [x] CLI: generate-baseline, compare subcommands
- [x] Colored 13x13 SB strategy matrix on exit
- [x] 110 tests, clippy clean

## Summary of Changes
Implemented as crate `convergence-harness` on branch `feat/convergence-harness`, merged to main.
