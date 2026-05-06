---
# poker_solver_rust-25o8
title: Allow unopened MP preflop calls
status: completed
type: task
priority: high
created_at: 2026-05-06T05:44:17Z
updated_at: 2026-05-06T05:53:23Z
---

Update blueprint_mp unopened preflop action generation in the current checkout so unopened positions can fold, call/limp, or use configured non-all-in open sizes. Keep raise and all-in unavailable until after a voluntary opening action.

## Summary of Changes

Implemented unopened MP preflop calls/limps in the current checkout. Unopened preflop nodes now expose Fold, Call, and configured non-all-in Lead open sizes; Raise and AllIn remain unavailable until after a voluntary open, and all-in-equivalent open sizes are still suppressed. Added 6-max core and TUI regressions, and raised the resolve_tui_scenarios_from_tree timed guard to account for the larger limp-inclusive tree.

## Verification

- cargo test -p poker-solver-core blueprint_mp::game_tree -- --nocapture
- cargo test -p poker-solver-trainer mp_tui_scenarios -- --nocapture
- cargo test -p poker-solver-trainer resolve_tui_scenarios_from_tree -- --nocapture
- cargo test

Full suite passed, but exceeded the repo's sub-1-minute target in this checkout.
