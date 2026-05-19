---
# poker_solver_rust-mazy
title: Re-enable MP traversal pruning without subtree purge
status: completed
type: task
priority: high
created_at: 2026-05-19T14:44:55Z
updated_at: 2026-05-19T14:52:43Z
---

Add an explicit training config switch for ordinary MP traversal pruning, enable it in the active 250/100/20 config, and keep negative-action subtree purge disabled so pruning does not physically remove stored strategy rows/subtrees.

## Summary of Changes

Added `training.traversal_pruning_enabled` so ordinary MP regret-threshold traversal pruning can be explicitly enabled without enabling negative-action subtree purge. Updated the active 250/100/20 config to set `traversal_pruning_enabled: true`, use a conservative `prune_threshold: -100`, and keep `negative_action_subtree_purge_enabled: false` so stored strategy rows/subtrees are not physically removed. Updated docs and config tests.

## Validation

- `cargo fmt --check`
- `cargo run -p poker-solver-trainer --release -- inspect-mp-config -c sample_configurations/blueprint_mp_6max_250f_100t_20r.yaml`
- `cargo test -p poker-solver-core should_prune -- --nocapture`
- `cargo test -p poker-solver-core training_traversal_pruning_key_parses -- --nocapture`
- `cargo test -p poker-solver-trainer run_train_blueprint_mp_lazy_sparse_no_tui_zero_iters_completes -- --nocapture`
