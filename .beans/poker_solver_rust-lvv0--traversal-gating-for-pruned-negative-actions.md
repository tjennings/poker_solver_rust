---
# poker_solver_rust-lvv0
title: Traversal gating for pruned negative actions
status: in-progress
type: task
priority: high
created_at: 2026-05-14T14:31:57Z
updated_at: 2026-05-14T15:06:36Z
parent: poker_solver_rust-xl3h
blocked_by:
    - poker_solver_rust-1dav
---

Wire negative-action purge semantics into lazy MP traversal. When enabled, crossing below `training.negative_action_prune_below` purges the child subtree and marks/derives the edge as allocation-blocked. Traversal must not allocate descendants under blocked actions. DCFR discounting can move regret back toward zero; once regret reaches `training.negative_action_reactivate_at`, the edge is eligible again and descendants materialize from default first-visit state. Acceptance: batch-level exploration is not used for reactivation, and `prune_explore_pct: 0.0` remains compatible.

## Implementation Notes

Starting after sparse descendant purge primitive landed in commit 61677872. Initial implementation should preserve hysteresis semantics: prune/purge when an unblocked edge crosses below negative_action_prune_below, keep it allocation-blocked while below negative_action_reactivate_at, and reactivate from first-visit state once DCFR discounting brings the parent action regret back to the reactivation threshold.
