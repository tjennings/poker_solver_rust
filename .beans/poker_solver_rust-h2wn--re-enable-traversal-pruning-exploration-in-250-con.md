---
# poker_solver_rust-h2wn
title: Re-enable traversal pruning exploration in 250 config
status: completed
type: task
priority: normal
created_at: 2026-05-19T14:55:38Z
updated_at: 2026-05-19T14:57:05Z
---

Set the active 250/100/20 MP config prune_explore_pct back to 0.05 so ordinary traversal pruning periodically explores full branches, while keeping negative-action subtree purge disabled.

## Summary of Changes

- Set sample_configurations/blueprint_mp_6max_250f_100t_20r.yaml prune_explore_pct to 0.05.
- Kept traversal_pruning_enabled true and negative_action_subtree_purge_enabled false.
- Validated the config with inspect-mp-config.
