---
# poker_solver_rust-seai
title: Add prune_regret_threshold and median regret TUI display
status: completed
type: feature
priority: normal
created_at: 2026-03-02T01:41:40Z
updated_at: 2026-03-02T02:06:02Z
---

Add configurable prune_regret_threshold to PostflopModelConfig (default 0.0 = current behavior), and display median positive/negative regret per flop in the TUI.\n\n- [ ] Add prune_regret_threshold config field\n- [ ] Use threshold in pruning logic\n- [ ] Add median regret fields to FlopStage and FlopTuiState\n- [ ] Compute median regrets in solve loop\n- [ ] Display median regrets in TUI\n- [ ] Update sample configs


## Summary of Changes

Added `prune_regret_threshold` config field (default 0.0) to control the negative regret level required to prune an action. Changed pruning condition from `< 0.0` to `< prune_regret_threshold`. Added median positive/negative regret computation (O(n) via `select_nth_unstable`) every 10 iterations, displayed per-flop in the TUI alongside pruning percentage.

Files changed: `postflop_model.rs`, `postflop_exhaustive.rs`, `postflop_abstraction.rs`, `postflop_mccfr.rs`, `tui_metrics.rs`, `tui.rs`, `main.rs`, `20spr.yaml`, `tiny.yaml`
