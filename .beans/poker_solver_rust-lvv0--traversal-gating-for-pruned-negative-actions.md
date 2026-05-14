---
# poker_solver_rust-lvv0
title: Traversal gating for pruned negative actions
status: todo
type: task
priority: high
created_at: 2026-05-14T14:31:57Z
updated_at: 2026-05-14T14:31:57Z
parent: poker_solver_rust-xl3h
blocked_by:
    - poker_solver_rust-1dav
---

Wire negative-action purge semantics into lazy MP traversal. When enabled, crossing below `training.negative_action_prune_below` purges the child subtree and marks/derives the edge as allocation-blocked. Traversal must not allocate descendants under blocked actions. DCFR discounting can move regret back toward zero; once regret reaches `training.negative_action_reactivate_at`, the edge is eligible again and descendants materialize from default first-visit state. Acceptance: batch-level exploration is not used for reactivation, and `prune_explore_pct: 0.0` remains compatible.
