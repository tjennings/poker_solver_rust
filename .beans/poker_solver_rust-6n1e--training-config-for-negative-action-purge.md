---
# poker_solver_rust-6n1e
title: Training config for negative-action purge
status: in-progress
type: task
priority: high
created_at: 2026-05-14T14:31:40Z
updated_at: 2026-05-14T14:35:20Z
parent: poker_solver_rust-xl3h
---

Add opt-in training-level config keys for the experiment. Config keys belong under `training`, not a separate `lazy_sparse` section. Proposed keys: `negative_action_subtree_purge_enabled`, `negative_action_prune_below`, `negative_action_reactivate_at`, and optionally `negative_action_purge_mode`. Ensure defaults keep current behavior unchanged, and update sample 6-max config to set `prune_explore_pct: 0.0` when running this experiment. Acceptance: config parses, defaults are tested, docs name the exact keys.
