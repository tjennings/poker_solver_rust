---
# poker_solver_rust-skcd
title: 'Phase 1 slice: blueprint_v2 sparse CFR storage backend'
status: completed
type: task
priority: high
created_at: 2026-06-03T19:36:06Z
updated_at: 2026-06-03T19:48:23Z
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

## Summary of Changes

- Added a blueprint_v2 CFR storage trait, stable action-schema fingerprints, dense projection helpers, and storage instrumentation stats.
- Implemented the trait for existing dense BlueprintStorage without changing strategy/regret semantics.
- Added SparseBlueprintStorage with lazy row realization, missing-row zero/uniform fallback, schema mismatch rejection, idempotent realization, dense projection, and resident/dense-equivalent stats.
- Wired the deterministic MCCFR differential harness to compare dense oracle behavior against a sparse candidate while preserving the dense-vs-dense self-check.
- Added focused sparse storage tests for missing-row fallback, idempotent allocation, schema mismatch rejection, and deterministic dense projection.

## Verification

- `cargo test -p poker-solver-core blueprint_v2::mccfr --quiet` passed.
- `cargo test -p poker-solver-core blueprint_v2::sparse_storage --quiet` passed.
- `/usr/bin/time -p cargo test --quiet` passed warm in `real 42.48`, under the one-minute gate. A prior cold post-edit run passed but measured `real 120.12` due to recompilation.
