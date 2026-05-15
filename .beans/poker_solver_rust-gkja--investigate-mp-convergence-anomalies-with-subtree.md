---
# poker_solver_rust-gkja
title: Investigate MP convergence anomalies with subtree purge disabled
status: completed
type: bug
priority: high
created_at: 2026-05-15T12:56:04Z
updated_at: 2026-05-15T13:01:04Z
---

User reports unusual converged MP strategies even with negative_action_subtree_purge_enabled=false. Investigate remaining pruning/traversal/averaging paths that can bias lazy sparse MP training independently of persistent negative-action subtree purge, then patch and test any confirmed bug.

## Summary of Changes

- Confirmed negative_action_subtree_purge_enabled=false disables the persistent negative-action traversal gate and purge path.
- Found a separate MP averaging bug: opponent decision nodes sampled one action for recursion and only added that sampled action to average strategy sums, weighted by its probability. In expectation this records p^2 instead of the full strategy vector, distorting TUI/snapshot average strategies.
- Fixed eager and lazy MP traversal so every visited decision infoset records the full current strategy vector; opponent actions remain sampled only for recursive traversal.
- Added eager and lazy regression tests asserting opponent traversal updates every root action in the average strategy vector.
- Updated architecture docs to describe the correct external-sampling average-strategy accounting.

## Validation

- cargo fmt --check
- cargo test -p poker-solver-core opponent_traversal_updates_full_average_strategy_vector -- --nocapture
- cargo test -p poker-solver-core lazy_opponent_traversal_updates_full_average_strategy_vector -- --nocapture
- cargo test -p poker-solver-core lazy_pruning -- --nocapture
- cargo test -p poker-solver-core negative_action -- --nocapture
- git diff --check

## Note

The current sample config still has ordinary pruning enabled after 4,000,000 iterations with prune_threshold=-1 and prune_explore_pct=0.0. That is independent of negative-action subtree purge and can still aggressively lock out nonterminal branches after warmup if the run is intended to test no-pruning behavior.
