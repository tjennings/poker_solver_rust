---
# poker_solver_rust-u81t
title: Implement lazy MP betting-state traversal
status: todo
type: feature
priority: high
created_at: 2026-05-07T13:18:52Z
updated_at: 2026-05-07T13:18:52Z
parent: poker_solver_rust-5kvv
blocked_by:
    - poker_solver_rust-7n72
---

Avoid eager materialization of the full 100bb 6-max MP game tree. Implement traversal over compact public betting state with dynamic legal-action generation, optional memoization for hot states, and stable infoset keys for storage/export. Preserve current eager tree path for small/debug configs during migration.
