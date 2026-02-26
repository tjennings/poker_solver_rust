---
# poker_solver_rust-yx1y
title: Restore strategy delta for MCCFR postflop early stopping
status: completed
type: feature
priority: normal
created_at: 2026-02-26T04:34:23Z
updated_at: 2026-02-26T04:43:29Z
---

Replace avg_positive_regret_flat with weighted_avg_strategy_delta in MCCFR postflop solver. Rename cfr_exploitability_threshold to cfr_convergence_threshold. The current proxy decays too quickly causing false convergence after ~100 iterations.

## Tasks
- [x] Restore weighted_avg_strategy_delta in postflop_abstraction.rs
- [x] Update mccfr_solve_one_flop to use strategy delta
- [x] Rename config field to cfr_convergence_threshold
- [x] Add tests for weighted_avg_strategy_delta
- [x] Update docs (training.md, architecture.md)
- [x] Update YAML configs
- [x] Run full verification


## Summary of Changes

Restored `weighted_avg_strategy_delta` as the convergence metric for MCCFR postflop early stopping, replacing `avg_positive_regret_flat` which decayed too quickly (dividing cumulative regret by buffer_size × iterations) causing false convergence after ~100 iterations.

**Code changes:**
- Added `weighted_avg_strategy_delta()`, `entry()`, `num_nodes()` back to `postflop_abstraction.rs`
- Updated `mccfr_solve_one_flop` to clone regrets before each iteration and compute strategy delta
- Renamed `cfr_exploitability_threshold` → `cfr_convergence_threshold` with 3 serde aliases for backward compat
- Made `avg_positive_regret_flat` test-only (`#[cfg(test)]`)
- 2 new tests for `weighted_avg_strategy_delta`

**Doc/config changes:**
- Updated `docs/architecture.md` and `docs/training.md`
- Updated `sample_configurations/AKQr_vs_234r.yaml` and `smoke.yaml`
