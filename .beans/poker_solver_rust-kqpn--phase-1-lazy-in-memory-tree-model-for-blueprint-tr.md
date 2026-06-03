---
# poker_solver_rust-kqpn
title: 'Phase 1: lazy in-memory tree model for blueprint trainer'
status: todo
type: feature
priority: high
created_at: 2026-06-03T18:08:59Z
updated_at: 2026-06-03T18:31:46Z
parent: poker_solver_rust-34kn
blocked_by:
    - poker_solver_rust-zgkr
---

Phase 1 of the blueprint trainer tree roadmap.

Scope:
- Research current blueprint trainer map-based lookup and traversal ownership.
- Design an arena-backed NodeId tree model with lazy child realization.
- Replace hot-path map lookup with direct NodeId traversal where safe.
- Preserve any required canonicalization/interning outside the traversal hot path.
- Add memory/accounting instrumentation: realized node count, child realization count/rate, per-node/state storage estimates, and traversal hot-path counters where appropriate.
- Keep pruning and disk eviction completely out of this phase.
- Update docs/architecture.md and docs/training.md if trainer architecture/config behavior changes.

Acceptance criteria:
- Existing trainer behavior is preserved modulo storage/traversal representation.
- Full test suite passes in under 1 minute, or any runtime violation is fixed/beaned before proceeding.
- Focused tests cover lazy realization idempotence, stable NodeId traversal, and equivalence against the prior map-backed path for a small deterministic fixture.
- Instrumentation makes resident tree growth visible during a small trainer run.

Implementation must be delegated to rust-developer/worker agents; manager does not write Rust directly.
