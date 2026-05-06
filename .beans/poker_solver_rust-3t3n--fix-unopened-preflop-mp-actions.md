---
# poker_solver_rust-3t3n
title: Fix unopened preflop MP actions
status: in-progress
type: bug
priority: high
created_at: 2026-05-06T02:45:00Z
updated_at: 2026-05-06T02:45:00Z
---

In blueprint_mp, unopened preflop nodes should only offer fold and open. Call, raise, and all-in should become available only after an opening action has occurred.
