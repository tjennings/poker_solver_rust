---
# poker_solver_rust-6n1e
title: Training config for negative-action purge
status: completed
type: task
priority: high
created_at: 2026-05-14T14:31:40Z
updated_at: 2026-05-14T14:52:17Z
parent: poker_solver_rust-xl3h
---

Add opt-in training-level config keys for the experiment. Config keys belong under `training`, not a separate `lazy_sparse` section. Proposed keys: `negative_action_subtree_purge_enabled`, `negative_action_prune_below`, `negative_action_reactivate_at`, and optionally `negative_action_purge_mode`. Ensure defaults keep current behavior unchanged, and update sample 6-max config to set `prune_explore_pct: 0.0` when running this experiment. Acceptance: config parses, defaults are tested, docs name the exact keys.

## Acceptance

- [x] Added training-level negative-action purge config keys with current-behavior defaults
- [x] Added serde snake_case purge mode enum with scan_history_prefix
- [x] Covered empty training defaults and explicit YAML parsing in core config tests
- [x] Updated 6-max 500/100/100 sample config with experiment keys and prune_explore_pct: 0.0
- [x] Documented exact training keys and batch-level prune exploration setting

## Summary of Changes

Added parsed/configured training options for the negative-action subtree purge experiment without implementing purge behavior. Updated test literals, sample configuration, and training docs. Verified formatting, focused core config tests, and a focused trainer test.
