---
# poker_solver_rust-xl3h
title: 'Epic: negative-regret subtree purge for MP lazy sparse'
status: completed
type: epic
priority: high
created_at: 2026-05-14T14:13:07Z
updated_at: 2026-05-14T16:08:17Z
---

Implement an experimental lazy-sparse pruning policy where an action whose cumulative regret falls below zero causes its materialized child subtree to be purged. If DCFR discounting or future updates return the action to non-negative/positive regret, descendant rows are treated as first visits and re-materialized from default strategy/regret state.

Acceptance criteria:
- [x] Define exact purge trigger semantics and reactivation semantics.
- [x] Add config flags so the behavior is opt-in and off by default.
- [x] Add sparse storage support for descendant purging or equivalent generation invalidation.
- [x] Ensure traversal does not allocate descendants under currently negative actions.
- [x] Add telemetry for purged rows, skipped allocations, and re-materializations.
- [x] Add focused tests for purge/reactivation behavior.
- [x] Update docs/training.md and docs/architecture.md.

## Design Decision

Disable batch-level prune exploration for this experiment (`prune_explore_pct: 0.0`). The current 5% exploration batches disable pruning globally and can re-materialize cold subtrees, working against memory containment. Reactivation should instead come from DCFR negative-regret discounting plus explicit prune/reactivation thresholds: pruned actions remain allocation-blocked while below the prune threshold, and are treated as first-visit/default descendants only after regret crosses the reactivation threshold.

Use hysteresis (`negative_action_prune_below` < `negative_action_reactivate_at`) to avoid allocate/purge thrash.

## Summary of Changes

Implemented the experimental negative-action subtree purge policy for MP lazy sparse training. The project now has opt-in config keys under `training`, edge-specific traversal gating, sparse descendant purge/repurge handling for stale concurrent rows, reactivation from first-visit state, no-TUI `neg_action[...]` telemetry, focused tests, and docs for the 6-max experiment workflow.
