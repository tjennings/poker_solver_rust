---
# poker_solver_rust-3oxj
title: Show SB open through SB 4-bet in MP training TUI
status: completed
type: task
priority: normal
created_at: 2026-08-04T17:15:57Z
updated_at: 2026-08-04T17:45:37Z
---

Update the TUI section of sample_configurations/blueprint_mp_hu_50f_50t_50r.yaml so the training dashboard exposes the preflop sequence SB open, BB 3-bet, and SB 4-bet.

- [x] Research TUI spot syntax and action encoding
- [x] Brainstorm the minimal correct spot configuration
- [x] Implement in an isolated worktree
- [x] Review the configuration
- [x] Validate config and complete workspace suite
- [x] Update documentation if required
- [x] Integrate and summarize


## Summary of Changes

- Replaced ignored player/actions TUI fields with three ordered name/spot scenarios for SB open, BB 3-bet, and SB 4-bet versus the 6bb branch.
- Added a sample-backed LazyMpGame regression asserting exact paths, acting seats, empty boards, and both BB 3bb/6bb raise actions.
- Confirmed existing documentation already covers the spot syntax; no documentation edit was required.
- Passed release config inspection, focused regression, formatting/diff checks, and the complete workspace test suite.
