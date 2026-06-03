---
# poker_solver_rust-nrz3
title: 'Fix regret pruning: skip regret/strategy updates for pruned actions'
status: completed
type: bug
priority: critical
created_at: 2026-03-02T02:25:49Z
updated_at: 2026-05-15T13:42:53Z
---

Regret-based pruning is broken: convergence reverses after warmup ends. Root cause: when an action is pruned (subtree not traversed), the code still updates regret with action_value=0, giving bogus regret deltas. In proper RBP (Brown & Sandholm), pruned actions should have regret frozen (no update). Fix: skip regret and strategy sum updates for pruned action indices.

## Summary of Changes

- Fixed MP traversal pruning so an action can only be skipped when its current regret-matched probability is zero.
- Applied the same guard to eager and lazy MP traversal pruning.
- Added eager and lazy regression tests for the all-nonpositive-regret case where regret matching falls back to uniform strategy; these actions must not be pruned despite being below prune_threshold.
- Updated architecture and training docs to describe strategy-probability-aware pruning.

## Validation

- cargo fmt --check
- cargo test -p poker-solver-core pruning_keeps_actions_with_current_strategy_mass -- --nocapture
- cargo test -p poker-solver-core lazy_pruning -- --nocapture
- cargo test -p poker-solver-core negative_action -- --nocapture
- git diff --check
