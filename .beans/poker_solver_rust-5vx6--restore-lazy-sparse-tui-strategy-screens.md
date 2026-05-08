---
# poker_solver_rust-5vx6
title: Restore lazy sparse TUI strategy screens
status: completed
type: bug
priority: high
created_at: 2026-05-08T02:37:27Z
updated_at: 2026-05-08T02:42:03Z
---

Lazy sparse MP TUI launches without configured strategy scenarios, so the strategy screens disappear even when tui.scenarios are configured. Restore usable strategy screens for lazy sparse training without relying on eager public-node storage.

## Summary of Changes

- Added a lazy resolved-spot cursor so the TUI can resolve configured scenario spots without building the eager public tree.
- Restored lazy sparse MP strategy screens by resolving `tui.scenarios` against the lazy game and initializing hand grids from sparse average strategy.
- Added periodic lazy sparse grid refreshes to the TUI bridge so screens update while training runs.
- Updated training docs and added focused regression coverage for lazy TUI scenario resolution.
