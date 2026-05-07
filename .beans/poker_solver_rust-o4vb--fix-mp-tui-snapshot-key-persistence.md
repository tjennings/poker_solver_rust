---
# poker_solver_rust-o4vb
title: Fix MP TUI snapshot key persistence
status: completed
type: bug
priority: high
created_at: 2026-05-07T03:27:02Z
updated_at: 2026-05-07T03:42:26Z
---

Pressing [s] in the Blueprint MP TUI sets the shared snapshot trigger, but the MP training bridge does not consume it or write strategy/regret snapshot artifacts to the configured output_dir.

## Summary of Changes

- Wired Blueprint MP TUI snapshot requests into the training bridge so pressing [s] writes a numbered snapshot under the configured snapshots.output_dir.
- Added MP snapshot serialization for average strategy, raw regrets/strategy sums, and metadata.
- Added a regression test that saves and reloads an MP snapshot.
- Reduced the TUI scenario-resolution fixture stack so the previously timing-out test remains fast.

## Verification

- cargo test -q -p poker-solver-trainer mp_snapshot_save_creates_strategy_and_metadata
- cargo test -q -p poker-solver-trainer resolve_tui_scenarios_from_tree
- cargo test -q passed, but wall time was 159s, which remains above the repository guideline of under 1 minute.
