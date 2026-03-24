---
# poker_solver_rust-8k0l
title: Convergence validation harness
status: completed
type: feature
priority: normal
created_at: 2026-03-24T16:55:29Z
updated_at: 2026-03-24T18:01:32Z
---

## Summary of Changes

Built the `convergence-harness` crate (Phase 1) with:
- **FlopPokerGame** configurable game definition (QhJdTh, 1176 combos, 20bb/2bb default)
- **ConvergenceSolver** trait for pluggable CFR algorithms
- **ExhaustiveSolver** adapter wrapping range-solver DCFR
- **Baseline** serialization (JSON summary, CSV convergence, bincode strategy/EVs)
- **Tree walker** extracts strategy and combo EVs at every decision node
- **Convergence loop** drives solver with dense-early/sparse-later sampling
- **Comparison metrics**: L1 strategy distance, combo EV diff
- **Reporter**: terminal, CSV, JSON, and human-readable summary output
- **CLI**: `generate-baseline` and `compare` subcommands
- **75 tests**, all passing, clippy clean

Branch: `feat/convergence-harness` (10 commits)
