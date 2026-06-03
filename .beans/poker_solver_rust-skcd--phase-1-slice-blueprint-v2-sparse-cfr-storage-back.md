---
# poker_solver_rust-skcd
title: 'Phase 1 slice: blueprint_v2 sparse CFR storage backend'
status: in-progress
type: task
priority: high
created_at: 2026-06-03T19:36:06Z
updated_at: 2026-06-03T19:53:33Z
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

## Review Findings 2026-06-03

Independent review of commit `9f0e5bc2 Add sparse blueprint CFR storage` found the slice is not ready for production trainer integration.

Blocking fixes required:

- Sparse storage must preserve prediction-aware current-strategy semantics. Dense storage delegates current strategy to the configured optimizer and prediction buffer; sparse currently computes from regrets only and lacks a prediction read hook in the trait.
- Sparse storage must honor configured regret floors. Dense `BlueprintStorage` clamps `add_regret` using `training.regret_floor`; sparse has a private floor initialized to `i32::MIN` but no setter or trait/config hook.

Additional fixes required before closing the slice:

- Action-schema mismatch protection must not create false confidence. Normal sparse reads/writes and dense projection should validate against the active tree/layout where practical, or the limitations must be explicit and tested.
- The differential harness must exercise sparse candidate dense projection/save-load compatibility, not only the dense oracle round trip.

Review verification passed, but the recommendation is fix before proceeding:

- `cargo test -p poker-solver-core blueprint_v2::sparse_storage --quiet` passed.
- `cargo test -p poker-solver-core differential_harness_eager_dense_vs_sparse_candidate --quiet` passed.
- `cargo test -p poker-solver-core differential_harness_eager_dense_self_check --quiet` passed.
- `/usr/bin/time -p cargo test --quiet` passed in `real 41.31` in the review workspace.
