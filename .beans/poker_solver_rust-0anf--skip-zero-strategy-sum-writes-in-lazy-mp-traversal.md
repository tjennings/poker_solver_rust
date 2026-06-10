---
# poker_solver_rust-0anf
title: Skip zero strategy-sum writes in lazy MP traversal
status: completed
type: task
priority: normal
created_at: 2026-05-19T20:18:13Z
updated_at: 2026-05-19T20:19:45Z
---

Avoid sparse storage writes when a lazy MP strategy probability produces a zero strategy-sum delta.

## Summary of Changes

Skipped lazy MP sparse strategy-sum writes when the computed delta is zero and added a regression test proving zero-probability strategy updates do not allocate sparse rows.
