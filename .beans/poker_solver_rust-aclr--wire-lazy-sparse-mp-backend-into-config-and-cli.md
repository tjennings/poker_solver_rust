---
# poker_solver_rust-aclr
title: Wire lazy_sparse MP backend into config and CLI
status: completed
type: task
priority: high
created_at: 2026-05-07T14:48:13Z
updated_at: 2026-05-07T15:04:51Z
parent: poker_solver_rust-5kvv
---

Add a user-selectable lazy_sparse backend for Blueprint MP training configs and train-blueprint-mp, route no-TUI runs through LazyTrainContext/SparseMpStorage, update inspect-mp-config to treat lazy_sparse as the 100bb-safe path, and add progress/sparse-storage heartbeat output.

## Work Started

Wiring the core lazy_sparse path into config/preflight/no-TUI execution and sparse progress telemetry.

## Summary of Changes

Added training.backend with eager default and lazy_sparse selection, routed train-blueprint-mp --no-tui through the lazy sparse core trainer, kept TUI on the eager path for now, updated inspect-mp-config to report selected backend while preserving eager-risk context, added sparse no-TUI heartbeat telemetry, documented lazy_sparse training usage, and added focused config/CLI regression tests.
