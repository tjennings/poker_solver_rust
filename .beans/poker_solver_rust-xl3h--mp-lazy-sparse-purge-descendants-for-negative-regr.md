---
# poker_solver_rust-xl3h
title: 'MP lazy sparse: purge descendants for negative-regret actions'
status: in-progress
type: task
priority: high
created_at: 2026-05-14T14:13:07Z
updated_at: 2026-05-14T14:13:07Z
---

Implement an experimental lazy-sparse pruning policy where an action whose cumulative regret falls below zero causes its materialized child subtree to be purged. If DCFR discounting or future updates return the action to non-negative/positive regret, descendant rows are treated as first visits and re-materialized from default strategy/regret state.\n\nAcceptance criteria:\n- [ ] Define exact purge trigger semantics and reactivation semantics.\n- [ ] Add config flags so the behavior is opt-in and off by default.\n- [ ] Add sparse storage support for descendant purging or equivalent generation invalidation.\n- [ ] Ensure traversal does not allocate descendants under currently negative actions.\n- [ ] Add telemetry for purged rows, skipped allocations, and re-materializations.\n- [ ] Add focused tests for purge/reactivation behavior.\n- [ ] Update docs/training.md and docs/architecture.md.
