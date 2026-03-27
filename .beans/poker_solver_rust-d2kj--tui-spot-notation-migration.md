---
# poker_solver_rust-d2kj
title: TUI spot notation migration
status: in-progress
type: feature
created_at: 2026-03-27T18:17:29Z
updated_at: 2026-03-27T18:17:29Z
---

Replace TUI scenario config format with Tauri spot notation. Plan: docs/plans/2026-03-27-tui-spot-notation-impl.md

- [ ] Task 1+2: Add format_tree_action_bb, match_action_by_label, resolve_spot
- [ ] Task 3: Simplify ScenarioConfig to {name, spot}
- [ ] Task 6: Update YAML config to spot notation
- [ ] Task 4: Update main.rs scenario initialization
- [ ] Task 5: Remove old resolve_action_path/match_action
- [ ] Task 7: Final cleanup and verification
