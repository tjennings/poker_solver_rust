---
# poker_solver_rust-8zw3
title: Replace cfr_delta_threshold with cfr_regret_threshold
status: completed
type: task
priority: normal
created_at: 2026-02-25T18:22:36Z
updated_at: 2026-02-25T18:33:24Z
---

Replace per-flop postflop CFR early stopping from strategy delta to avg positive regret. Rename config field, update both bucketed and mccfr solvers, remove prev_regrets clone, update FlopStage/FlopSolveResult, update docs.

## Summary of Changes\nReplaced cfr_delta_threshold with cfr_regret_threshold using avg positive regret. Removed weighted_avg_strategy_delta and the expensive per-iteration regret buffer clone. Updated both bucketed and MCCFR solvers, config, progress display, docs. Commit: 2a4719c
