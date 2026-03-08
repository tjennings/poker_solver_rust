---
# poker_solver_rust-n161
title: Add max positive regret sparkline and fix prune_threshold default
status: completed
type: bug
priority: normal
created_at: 2026-03-08T01:58:55Z
updated_at: 2026-03-08T02:07:33Z
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

## Summary of Changes

1. **Fixed prune_threshold default**: Changed from -310_000_000 to 0 (standard RBP: prune any negative regret). Removed hardcoded -310M from sample configs.
2. **Added max positive regret sparkline**: New green sparkline in TUI showing max positive regret (convergence indicator — should trend toward 0). Mirrors the existing min_regret pattern through the full chain: trainer method → callback → metrics → TUI sparkline.
