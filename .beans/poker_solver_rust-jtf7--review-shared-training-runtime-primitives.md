---
# poker_solver_rust-jtf7
title: Review shared training runtime primitives
status: in-progress
type: task
priority: high
created_at: 2026-06-09T15:10:02Z
updated_at: 2026-06-09T15:21:58Z
parent: poker_solver_rust-tzv5
---

Review task for implementation bean poker_solver_rust-e85a. Focus on whether the generic API stays backend-neutral, preserves HU arena/lazy and MP semantic storage identities, avoids imposing shared traversal/chance/pruning semantics, and has adequate fake-backend tests.



Second review findings:
- P1: RuntimeCounters needs a public resume-seeding API so adapters can initialize restored snapshot progress before the runtime owns subsequent increments.
- P2: TrainingBackendKind labels should avoid inviting runtime coupling to traversal/storage internals.
- P2: adapters need a target-remaining budget affordance or runtime behavior that avoids batch overshoot.
