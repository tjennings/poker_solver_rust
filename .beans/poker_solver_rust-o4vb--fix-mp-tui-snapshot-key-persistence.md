---
# poker_solver_rust-o4vb
title: Fix MP TUI snapshot key persistence
status: in-progress
type: bug
priority: high
created_at: 2026-05-07T03:27:02Z
updated_at: 2026-05-07T03:27:02Z
---

Pressing [s] in the Blueprint MP TUI sets the shared snapshot trigger, but the MP training bridge does not consume it or write strategy/regret snapshot artifacts to the configured output_dir.
