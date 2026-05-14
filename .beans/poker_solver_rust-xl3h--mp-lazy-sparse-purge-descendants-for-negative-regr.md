---
# poker_solver_rust-xl3h
title: 'Epic: negative-regret subtree purge for MP lazy sparse'
status: in-progress
type: epic
priority: high
created_at: 2026-05-14T14:13:07Z
updated_at: 2026-05-14T14:31:28Z
---

Implement an experimental lazy-sparse pruning policy where an action whose cumulative regret falls below zero causes its materialized child subtree to be purged. If DCFR discounting or future updates return the action to non-negative/positive regret, descendant rows are treated as first visits and re-materialized from default strategy/regret state.\n\nAcceptance criteria:\n- [ ] Define exact purge trigger semantics and reactivation semantics.\n- [ ] Add config flags so the behavior is opt-in and off by default.\n- [ ] Add sparse storage support for descendant purging or equivalent generation invalidation.\n- [ ] Ensure traversal does not allocate descendants under currently negative actions.\n- [ ] Add telemetry for purged rows, skipped allocations, and re-materializations.\n- [ ] Add focused tests for purge/reactivation behavior.\n- [ ] Update docs/training.md and docs/architecture.md.

## Design Decision\n\nDisable batch-level prune exploration for this experiment (`prune_explore_pct: 0.0`). The current 5% exploration batches disable pruning globally and can re-materialize cold subtrees, working against memory containment. Reactivation should instead come from DCFR negative-regret discounting plus explicit prune/reactivation thresholds: pruned actions remain allocation-blocked while below the prune threshold, and are treated as first-visit/default descendants only after regret crosses the reactivation threshold.\n\nOpen design detail: use hysteresis (`prune_below` < `reactivate_at`) to avoid allocate/purge thrash.
