---
# poker_solver_rust-skcd
title: 'Phase 1 slice: blueprint_v2 sparse CFR storage backend'
status: in-progress
type: task
priority: high
created_at: 2026-06-03T19:36:06Z
updated_at: 2026-06-03T19:36:06Z
parent: poker_solver_rust-kqpn
---

First implementation slice for Phase 1 lazy/sparse blueprint trainer storage.

Scope:
- Add a blueprint_v2 CFR storage abstraction over MCCFR storage operations.
- Implement the abstraction for existing dense BlueprintStorage with no behavior change.
- Add HU sparse/lazy row storage with missing-row zero/uniform semantics, idempotent allocation, action-schema validation, dense projection helpers, and resident storage stats.
- Wire the existing differential harness so dense oracle can be compared against sparse candidate.
- Add focused tests for missing-row uniform fallback, idempotent allocation, schema mismatch rejection, deterministic dense projection, and dense-vs-sparse differential equivalence.

Non-goals: no pruning, no disk eviction, no lazy GameTree/child realization, no strategy.bin format change, no default sparse on-disk snapshot change.
