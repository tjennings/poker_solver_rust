---
# poker_solver_rust-2ej9
title: Add MP lazy training runtime adapter
status: in-progress
type: task
priority: high
created_at: 2026-06-09T17:06:45Z
updated_at: 2026-06-09T17:06:45Z
parent: poker_solver_rust-tzv5
---

Add a multiplayer lazy sparse adapter for the shared training runtime. Scope: wrap the existing lazy MP training context with TrainingRuntimeBackend, preserve LazyMpGame traversal, sparse semantic storage identity, sampled/exact chance semantics, snapshot format, and existing training cadence. Seed RuntimeCounters from restored meta-iterations, cap batches with BatchBudget where applicable, bridge RuntimeControls into existing lazy MP quit/pause/snapshot/refresh surfaces or document missing pause support, and add focused adapter/parity tests. Non-goals: do not merge HU/MP traversal code, do not alter lazy_mccfr algorithms, do not change sparse storage keys, do not wire CLI/TUI yet.
