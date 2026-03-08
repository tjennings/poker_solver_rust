---
# poker_solver_rust-qi6p
title: Add avg positive regret sparkline to blueprint TUI
status: in-progress
type: feature
created_at: 2026-03-08T07:19:57Z
updated_at: 2026-03-08T07:19:57Z
---

Add average positive regret metric to the blueprint training TUI. This is the actual convergence signal (O(1/√T) bound) vs max regret which stays noisy.

## Tasks
- [ ] Add avg_pos_regret() method to trainer.rs (sum of max(0,r) / count across all regret entries)
- [ ] Add on_avg_pos_regret callback + push_avg_pos_regret() to BlueprintTuiMetrics
- [ ] Add avg_pos_regret sparkline row to TUI dashboard
- [ ] Wire callback in main.rs
- [ ] Verify tests pass
