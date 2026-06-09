---
# poker_solver_rust-osss
title: Unify HU and multiplayer blueprint trainers
status: draft
type: epic
priority: high
created_at: 2026-06-09T13:03:45Z
updated_at: 2026-06-09T13:03:45Z
---

There are currently separate HU blueprint_v2 and multiplayer/6-max trainer paths. Plan and eventually migrate toward one trainer architecture across all player counts. This is intentionally large and requires a detailed architecture plan before implementation. The plan must cover shared game/tree abstractions, storage/key identity, traversal/sampling, action abstraction, snapshot/bundle format, TUI/metrics, validation baselines, migration compatibility, and incremental rollout slices.
