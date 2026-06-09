---
# poker_solver_rust-vh99
title: Wire MP lazy trainer and TUI through shared runtime
status: in-progress
type: task
priority: high
created_at: 2026-06-09T17:47:52Z
updated_at: 2026-06-09T17:47:52Z
parent: poker_solver_rust-tzv5
---

Route the existing train-blueprint-mp lazy_sparse CLI/TUI execution through the shared training runtime and LazySparseMpTrainingRuntimeAdapter while preserving current lazy sparse snapshot format, TUI controls, telemetry, and no-TUI heartbeat behavior. Scope includes designing trainer-side snapshot/resume hooks for sparse_entries.bin metadata if required by runtime integration, wiring pause/quit/snapshot/refresh requests through RuntimeControls, updating docs/training.md for any behavior changes, and adding focused tests. Non-goals: do not merge HU and MP traversal algorithms, do not change sparse storage identity, do not implement strategy pruning or disk eviction yet.
