---
# poker_solver_rust-6y86
title: 'Phase 0: baseline and rewrite spec for blueprint trainer lazy tree'
status: todo
type: task
priority: high
created_at: 2026-06-03T18:10:49Z
updated_at: 2026-06-03T18:10:49Z
parent: poker_solver_rust-34kn
blocking:
    - poker_solver_rust-zgkr
    - poker_solver_rust-kqpn
    - poker_solver_rust-l6r9
    - poker_solver_rust-bgbz
    - poker_solver_rust-i4fy
---

Phase 0 planning/specification task for the blueprint trainer lazy tree roadmap.

Scope:
- Capture current blueprint trainer storage, traversal, snapshot, export, and resume behavior before implementation starts.
- Record baseline performance on tiny and representative configs: startup memory, visited decision count, allocated slots/rows, iterations/sec, snapshot size, and resume behavior.
- Define stable lazy node identity: public state key, action-history encoding, street, pot/stacks/bets, actor, chance boundary handling, legal action order, and terminal payoff identity.
- Define row identity separately from arena identity: node/bucket/action coordinates or an explicit sparse row key.
- Decide snapshot compatibility policy: legacy dense resume support, explicit versioned failure, and/or new sparse/lazy resume format.
- Document invariants that Phase 1 must preserve.

Acceptance criteria:
- Baseline report exists in the bean or docs and is specific enough for rust-developer agents to implement against.
- Invariants are documented for legal action equality, terminal payoff equality, bucket lookup consistency, action coordinate stability, deterministic seed behavior, snapshot/export compatibility, and DCFR discount behavior for lazy rows.
- The working tree is clean before implementation begins.
- The full test suite passes and completes under 1 minute before implementation begins, or a blocking bean is created/fixed first.

This blocks the differential harness and lazy tree implementation.
