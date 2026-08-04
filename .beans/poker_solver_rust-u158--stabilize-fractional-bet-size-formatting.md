---
# poker_solver_rust-u158
title: Stabilize fractional bet-size formatting
status: in-progress
type: bug
priority: high
created_at: 2026-08-04T13:28:43Z
updated_at: 2026-08-04T13:33:41Z
---

The complete workspace suite reaches poker-solver-tauri but fails two precision-format tests: exploration::tests::blueprint_sizes_preserve_fractional_percentages and game_session::tests::format_bet_sizes_preserves_fractional_percentages. Actual output is 33.333333333333329% while the stable expected representation is 33.3333333333333%. Diagnose and restore deterministic shared formatting.

- [x] Research formatter implementation and regression origin
- [ ] Brainstorm the minimal stable formatting rule
- [ ] Plan and dispatch implementation in an isolated worktree
- [ ] Review the repair
- [ ] Confirm targeted and complete suites pass
- [ ] Summarize the outcome
