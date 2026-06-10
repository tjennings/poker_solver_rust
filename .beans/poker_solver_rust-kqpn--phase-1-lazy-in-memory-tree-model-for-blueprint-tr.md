---
# poker_solver_rust-kqpn
title: 'Phase 1: lazy in-memory tree model for blueprint trainer'
status: completed
type: feature
priority: high
created_at: 2026-06-03T18:08:59Z
updated_at: 2026-06-03T20:29:29Z
parent: poker_solver_rust-34kn
---

Phase 1 of the blueprint trainer tree roadmap.

Scope:
- Treat the current HU `blueprint_v2` eager `GameTree` arena as the correctness oracle; do not assume a hot-path map lookup exists in the HU trainer path.
- Design and implement an in-memory lazy/sparse trainer storage model for CFR rows and, where it is a clean fit, lazily realized decision state around stable node/public-row identity.
- Preserve deterministic traversal semantics, action ordering, bucket lookup behavior, terminal values, and existing resume/export surfaces.
- Preserve dense `strategy.bin`/bundle output for Explorer/Tauri compatibility, even if trainer internals become sparse/lazy.
- Add memory/accounting instrumentation: realized row/node count, row allocation rate, dense-projected footprint estimates, sparse resident storage estimates, and traversal/storage hot-path counters where appropriate.
- Keep strategy pruning and disk eviction completely out of this phase.
- Update `docs/architecture.md` and `docs/training.md` if trainer architecture/config behavior changes.

Acceptance criteria:
- Existing trainer behavior is preserved modulo storage representation.
- The Phase 1 prep differential harness can compare eager+dense oracle behavior against the new lazy/sparse path for a small deterministic fixture.
- Focused tests cover lazy realization idempotence, stable row/public identity, dense projection/export compatibility, and equivalence against the eager+dense oracle.
- Full test suite passes in under 1 minute, or any runtime violation is fixed/beaned before proceeding.
- Instrumentation makes resident tree/storage growth visible during a small trainer run.

Implementation must be delegated to rust-developer/worker agents; manager does not write Rust directly.

## Phase 1 Entry Gate

- Working tree was clean before Phase 1 implementation planning.
- `/usr/bin/time -p cargo test --quiet` passed on 2026-06-03 in `real 41.46`, under the one-minute project gate.
- Required research/architecture brainstorming has been dispatched before implementation.

## Phase 1 Implementation Plan

Research/architecture consensus: do not rewrite HU `GameTree` in the first production slice. HU `blueprint_v2` already traverses an eager arena by stable node index; the memory pressure Phase 1 should address is eager dense CFR row allocation across every `(decision node, bucket, action)` slot.

Planned implementation sequence:

- Add a `blueprint_v2` CFR storage abstraction over the operations used by MCCFR traversal: current strategy, average strategy, regret read/write, strategy-sum read/write, baseline/prediction hooks where needed, dense projection, load/save compatibility hooks, and storage stats.
- Keep existing `BlueprintStorage` as the dense implementation and preserve current behavior through the trait before adding sparse storage.
- Add HU sparse/lazy row storage keyed by stable decision identity plus bucket and action-schema fingerprint. Missing rows must be equivalent to dense all-zero rows: zero regrets/sums/predictions/baselines and uniform current/average strategy.
- Refactor the existing differential harness to compare dense oracle behavior against the sparse candidate using the completed trace/action/delta checks.
- Preserve dense `strategy.bin` and dense-compatible snapshot/export behavior for Explorer/Tauri. Sparse internals remain an in-memory trainer detail in this phase.
- Add instrumentation for realized rows/slots, inserts, read/write probes and hits, dense-equivalent slots/bytes, and approximate sparse resident bytes.
- Defer lazy child realization, strategy pruning, sparse on-disk defaults, and disk eviction to later phases.

First worker slice:

- Implement the storage trait and dense implementation with no behavior change.
- Add the HU sparse storage module with missing-row semantics, idempotent row realization, dense projection helpers, and stats.
- Wire just enough MCCFR/test-harness code to run dense oracle vs sparse candidate on the deterministic fixture.
- Add focused tests for missing-row uniform fallback, idempotent allocation, schema mismatch rejection, deterministic dense projection, and differential equivalence.

## Summary of Changes

Phase 1 is complete as an in-memory lazy/sparse CFR row model over the existing HU `blueprint_v2` eager arena `GameTree`. Phase 0 showed that rewriting HU traversal into a lazy `GameTree` would target the wrong pressure point; the implemented Phase 1 therefore keeps stable arena node identity for traversal and moves lazy realization into trainer storage rows.

Delivered:

- Added a `blueprint_v2` CFR storage abstraction used by MCCFR traversal.
- Preserved existing dense `BlueprintStorage` behavior as the default backend.
- Added `SparseBlueprintStorage` with lazy row realization, missing-row zero/uniform semantics, idempotent allocation, action-schema validation, prediction-aware optimizer support, regret-floor support, dense projection helpers, and resident/dense-equivalent storage stats.
- Extended the deterministic differential harness to compare eager dense oracle behavior against sparse candidate behavior, including traces, deltas, dense projection, and candidate save/load round trip.
- Added `training.storage_backend` with `dense` default and `sparse`/`lazy` opt-in for HU `blueprint_v2` training.
- Wired `BlueprintTrainer` through the storage abstraction so sparse training runs without dense projection in the hot MCCFR loop.
- Preserved dense `strategy.bin`/`regrets.bin` snapshot/export/resume compatibility for Explorer/Tauri consumers.
- Mirrored optimizer, SAPCFR+ prediction, baseline, and regret-floor setup for sparse storage; sparse+BRCFR+ is explicitly rejected in this phase with test coverage.
- Surfaced sparse storage instrumentation in trainer progress/logging.
- Updated `docs/architecture.md` and `docs/training.md` for the new backend and limitations.

Verification:

- `cargo test -p poker-solver-core blueprint_v2::sparse_storage --quiet` passed.
- `cargo test -p poker-solver-core differential_harness_eager_dense_vs_sparse_candidate --quiet` passed.
- `cargo test -p poker-solver-core differential_harness_eager_dense_self_check --quiet` passed.
- `cargo test -p poker-solver-core blueprint_v2::trainer::tests --quiet` passed.
- `cargo test -p poker-solver-core blueprint_v2 --quiet` passed.
- Manager full-suite gate: `/usr/bin/time -p cargo test --quiet` passed in `real 42.99`, under the one-minute requirement.

Independent reviews completed:

- Initial sparse storage slice review found blockers around prediction-aware strategy semantics, regret floors, schema validation confidence, and sparse candidate round-trip coverage; all were fixed in `8558fbfb`.
- Corrective sparse storage review found no blockers and recommended proceeding.
- Production trainer integration review found no blockers and recommended closing Phase 1.

Known follow-up:

- `poker_solver_rust-f14l` tracks a low-priority TUI startup display freshness gap in sparse resume mode. This is not a trainer traversal/export blocker and is deferred outside Phase 1.
