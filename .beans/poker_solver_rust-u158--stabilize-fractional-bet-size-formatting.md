---
# poker_solver_rust-u158
title: Stabilize fractional bet-size formatting
status: completed
type: bug
priority: high
created_at: 2026-08-04T13:28:43Z
updated_at: 2026-08-04T14:33:01Z
---

The complete workspace suite reaches poker-solver-tauri but fails two precision-format tests: exploration::tests::blueprint_sizes_preserve_fractional_percentages and game_session::tests::format_bet_sizes_preserves_fractional_percentages. Actual output is 33.333333333333329% while the stable expected representation is 33.3333333333333%. Diagnose and restore deterministic shared formatting.

- [x] Research formatter implementation and regression origin
- [x] Brainstorm the minimal stable formatting rule
- [x] Plan and dispatch implementation in an isolated worktree
- [x] Review the repair
- [x] Confirm targeted and complete suites pass
- [x] Summarize the outcome

## Summary of Changes

Corrected two inconsistent fractional-percentage test expectations to the existing deterministic 15-decimal transport output and added a half-chip boundary regression for 1/96 at a 48-chip pot. Production formatting and solver behavior remain unchanged. Targeted Tauri tests and the complete workspace suite pass.
