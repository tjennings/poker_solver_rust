---
# poker_solver_rust-kqpn
title: 'Phase 1: lazy in-memory tree model for blueprint trainer'
status: in-progress
type: feature
priority: high
created_at: 2026-06-03T18:08:59Z
updated_at: 2026-06-03T19:33:14Z
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
