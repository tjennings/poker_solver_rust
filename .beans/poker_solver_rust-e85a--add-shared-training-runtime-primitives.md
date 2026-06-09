---
# poker_solver_rust-e85a
title: Add shared training runtime primitives
status: in-progress
type: task
priority: high
created_at: 2026-06-09T15:03:09Z
updated_at: 2026-06-09T15:03:09Z
parent: poker_solver_rust-tzv5
---

Add the neutral runtime primitives for trainer unification without changing HU or MP traversal behavior. Scope: new core training_runtime module exported from lib.rs; backend kind/unit labels, controls/counters/limits, batch outcome, telemetry sink, runtime backend trait, and fake-backend unit tests for stop/pause/snapshot/telemetry semantics. Non-goal: no HU or MP adapter, no CLI/TUI rewiring, no traversal changes.
