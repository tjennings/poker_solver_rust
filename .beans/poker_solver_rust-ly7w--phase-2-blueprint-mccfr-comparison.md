---
# poker_solver_rust-ly7w
title: 'Phase 2: Blueprint MCCFR comparison'
status: completed
type: feature
priority: normal
created_at: 2026-03-24T22:12:40Z
updated_at: 2026-03-25T04:05:15Z
parent: poker_solver_rust-elst
---

## Summary of Changes

Added MCCFR solver adapter to the convergence harness (Phase 2):

- **MccfrSolver**: wraps BlueprintTrainer, custom training loop calling traverse_external() directly
- **Fixed-flop deals**: always QhJdTh, random hole cards/turn/river
- **169 canonical buckets**: preflop hand index for all streets (pipeline validation)
- **Strategy lifting**: bucket→combo mapping via CanonicalHand
- **Exploitability computation**: inject MCCFR strategy into range-solver via lock_current_strategy, compute best-response Nash gap
- **run-solver CLI**: \`--solver mccfr --iterations N --checkpoints 1K,10K,...\`
- **Metrics**: strategy delta, avg positive regret
- **Limitation**: tree correspondence currently works with all-in-only configs; pot-relative bet sizes need matching work

144 tests, clippy clean.
