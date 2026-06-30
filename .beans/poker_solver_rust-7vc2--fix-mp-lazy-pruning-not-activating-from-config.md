---
# poker_solver_rust-7vc2
title: Fix MP lazy pruning not activating from config
status: in-progress
type: bug
priority: high
created_at: 2026-06-30T13:31:40Z
updated_at: 2026-06-30T13:33:58Z
parent: poker_solver_rust-osss
---

MP lazy sparse pruning appears configured but does not activate: TUI reports 0 pruned and training does not show the expected throughput improvement. Investigate config parsing, runtime adapter propagation, lazy MCCFR pruning/purge gates, TUI telemetry labels, and whether the intended pruning mode is traversal pruning or negative-action subtree purge. Acceptance: a config with pruning enabled produces nonzero pruning/blocked-edge telemetry under a small deterministic test or clear diagnostic, and the TUI reports the active pruning mode accurately.

## Diagnosis

Read-only inspection found the active HU MP lazy-sparse config does not currently enable either pruning path. `prune_explore_pct` only controls the fraction of post-warmup batches that disable ordinary traversal pruning once traversal pruning is already enabled. Ordinary MP traversal pruning requires `training.traversal_pruning_enabled: true` and `meta_iter >= prune_after_iterations`. Negative-action subtree purge requires `training.negative_action_subtree_purge_enabled: true` and the same warmup boundary. The TUI `Traversals pruned` metric is wired to ordinary traversal-pruning hits/total, not negative-action purge telemetry.
