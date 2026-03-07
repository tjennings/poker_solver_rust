---
# poker_solver_rust-zrbp
title: 'Blueprint V2 TUI: fix bugs + add delta sparkline and leaf movement metric'
status: completed
type: bug
priority: high
created_at: 2026-03-07T01:14:21Z
updated_at: 2026-03-07T01:28:06Z
---

Five issues:

1. **Throughput overlap bug**: print_metrics() eprintln! corrupts alternate screen. Suppress when TUI active.
2. **Strategy tables don't update**: callback stores flat probs in metrics but grid cells (what TUI renders) never refresh.
3. **Remove exploitability/[e]**: dead feature, remove from layout, hotkeys, metrics.
4. **Strategy delta sparkline**: push delta history from trainer to TUI, render sparkline.
5. **Leaf movement metric**: fraction of info sets with max action prob change >20%, display in TUI.

- [x] Fix throughput overlap (suppress eprintln when TUI active)
- [x] Fix strategy table refresh (extract full grid in callback, TUI tick applies it)
- [x] Remove exploitability/[e] from TUI
- [x] Add strategy delta sparkline
- [x] Add leaf movement metric
- [x] All tests pass

## Summary of Changes

- trainer.rs: added tui_active flag, on_strategy_delta/on_leaf_movement callbacks
- storage.rs: strategy_delta() returns (mean_delta, pct_moving) tuple
- blueprint_tui.rs: removed exploitability, added delta + leaf sparklines, strategy grid refresh in tick()
- blueprint_tui_metrics.rs: removed exploitability, added strategy_grids/delta_history/leaf_movement_history
- main.rs: wired all new callbacks, set tui_active=true
- 95 core + 62 trainer tests pass
