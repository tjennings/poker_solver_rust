---
# poker_solver_rust-7n72
title: Design lazy/sparse Blueprint MP trainer architecture
status: completed
type: task
priority: high
created_at: 2026-05-07T13:18:35Z
updated_at: 2026-05-07T13:44:45Z
parent: poker_solver_rust-5kvv
---

Design the 100bb-capable MP trainer architecture. Compare lazy state traversal, compact betting-state graph, sparse visited-infoset storage, compressed action-memory storage, and snapshot/export implications. Deliver a concrete migration plan with interfaces and test gates.

## Work Started

Design pass started. Dirty config changes are intentionally left untouched while documenting the 100bb-capable architecture path.

## Summary of Changes

- Added docs/plans/2026-05-07-blueprint-mp-100bb-design.md with the lazy public-state traversal, stable infoset key, sparse storage, snapshot, TUI, preflight, migration, and acceptance-criteria plan.
- Updated architecture docs to call out the current eager MP backend and the intended lazy/sparse 100bb path.
- Updated training docs to warn that 100bb is a target depth but still requires the planned lazy/sparse backend for large 6-max abstractions.

## Verification

- git diff --check -- docs/plans/2026-05-07-blueprint-mp-100bb-design.md docs/architecture.md docs/training.md
