---
# poker_solver_rust-d2kj
title: TUI spot notation migration
status: completed
type: feature
priority: normal
created_at: 2026-03-27T18:17:29Z
updated_at: 2026-03-27T18:51:30Z
---

Replace TUI scenario config format with Tauri spot notation. Plan: docs/plans/2026-03-27-tui-spot-notation-impl.md

- [x] Task 1+2: Add format_tree_action_bb, match_action_by_label, resolve_spot
- [x] Task 3: Simplify ScenarioConfig to {name, spot}
- [x] Task 6: Update YAML config to spot notation
- [x] Task 4: Update main.rs scenario initialization
- [x] Task 5: Remove old resolve_action_path/match_action
- [x] Task 7: Final cleanup and verification

## Summary of Changes
- Added resolve_spot() that parses spot notation strings and walks the game tree
- Added format_tree_action_bb() for BB-based action labels matching Tauri format
- Simplified ScenarioConfig from 5 fields to {name, spot}
- Removed PlayerLabel enum, resolve_action_path, match_action
- Updated sample YAML config to use spot notation
- Net -107 lines (simplification)
