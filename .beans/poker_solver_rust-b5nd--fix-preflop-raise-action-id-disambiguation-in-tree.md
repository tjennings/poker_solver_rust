---
# poker_solver_rust-b5nd
title: Fix preflop raise action ID disambiguation in tree walk
status: completed
type: bug
priority: normal
created_at: 2026-02-26T19:08:23Z
updated_at: 2026-02-26T19:13:49Z
---

When walking the preflop tree, all raise actions match the first raise at each node because `preflop_action_matches` ignores raise size. Action IDs like `r:3` and `r:4` (which encode the position within the node's action_labels array) all resolve to the first raise action. This means different raise sizes produce identical pot/stack states.

Root cause: `parse_preflop_action()` converts any `r:N` to `PreflopAction::Raise(0.0)`, and `preflop_action_matches()` matches `Raise(_)` against `Raise(_)` regardless of size. The `.position()` call returns the first match.

Fix: Extract the index from the action ID string and use it directly as the child position instead of searching by action type.

## Tasks
- [x] Fix `walk_preflop_tree_with_state` to use action index from ID string for raise actions
- [x] Verify different raise sizes produce different pot/stack states
- [x] Verify all-in (`r:A`) still works correctly

## Summary of Changes

Replaced `parse_preflop_action` + `preflop_action_matches` (which matched ANY raise to the first raise) with `find_action_position` that uses the index encoded in the action ID (`r:{idx}`) to select the correct raise action at each tree node. Updated the existing test to use the correct action index.
