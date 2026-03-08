---
# poker_solver_rust-qi6p
title: Add avg positive regret sparkline to blueprint TUI
status: completed
type: feature
priority: normal
created_at: 2026-03-08T07:19:57Z
updated_at: 2026-03-08T07:28:37Z
---

Add average positive regret metric to the blueprint training TUI. This is the actual convergence signal (O(1/√T) bound) vs max regret which stays noisy.

## Tasks
- [x] Add avg_pos_regret() method to trainer.rs (sum of max(0,r) / count across all regret entries)
- [x] Add on_avg_pos_regret callback + push_avg_pos_regret() to BlueprintTuiMetrics
- [x] Add avg_pos_regret sparkline row to TUI dashboard
- [x] Wire callback in main.rs
- [x] Verify tests pass

## Summary of Changes
Added avg positive regret sparkline (Yellow) to TUI between min regret and prune bar. Computed as mean of max(0, r) across all regret entries. This is the true convergence signal per CFR theory.
