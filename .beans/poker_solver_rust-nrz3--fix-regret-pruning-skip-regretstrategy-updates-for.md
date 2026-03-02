---
# poker_solver_rust-nrz3
title: 'Fix regret pruning: skip regret/strategy updates for pruned actions'
status: in-progress
type: bug
priority: critical
created_at: 2026-03-02T02:25:49Z
updated_at: 2026-03-02T02:25:49Z
---

Regret-based pruning is broken: convergence reverses after warmup ends. Root cause: when an action is pruned (subtree not traversed), the code still updates regret with action_value=0, giving bogus regret deltas. In proper RBP (Brown & Sandholm), pruned actions should have regret frozen (no update). Fix: skip regret and strategy sum updates for pruned action indices.
