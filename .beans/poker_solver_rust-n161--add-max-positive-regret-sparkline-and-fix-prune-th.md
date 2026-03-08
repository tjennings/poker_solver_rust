---
# poker_solver_rust-n161
title: Add max positive regret sparkline and fix prune_threshold default
status: in-progress
type: bug
created_at: 2026-03-08T01:58:55Z
updated_at: 2026-03-08T01:58:55Z
---

Two issues in blueprint TUI:
1. Feature: Add max_regret() method (symmetric to min_regret()) + TUI sparkline showing max positive regret (convergence indicator)
2. Bug: Default prune_threshold is -310_000_000 which effectively disables pruning. Should be 0 (standard RBP: prune any negative regret)

Changes needed:
- [ ] Add max_regret() to trainer.rs (symmetric to min_regret at line 600)
- [ ] Add on_max_regret callback to trainer (like on_min_regret)
- [ ] Add max_regret_history to BlueprintTuiMetrics + push_max_regret method
- [ ] Add max_regret_history to BlueprintTuiApp + render_max_regret sparkline
- [ ] Wire callback in main.rs (like line 586-588)
- [ ] Add sparkline row to TUI layout
- [ ] Fix default_prune_threshold from -310_000_000 to 0
- [ ] Update sample YAML configs that hardcode -310000000
