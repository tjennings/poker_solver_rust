---
# poker_solver_rust-59he
title: Confirm MP prune exploration config
status: completed
type: task
priority: normal
created_at: 2026-05-20T18:19:51Z
updated_at: 2026-05-20T18:20:40Z
---

Trace and verify that the MP training prune exploration config variable is parsed and affects traversal pruning as intended.

## Summary of Changes

Confirmed that the MP pruning exploration variable is `prune_explore_pct`. It is parsed on `MpTrainingConfig`, defaults to 0.05, and the config parsing test `training_traversal_pruning_key_parses` passes. Runtime pruning uses `should_prune(meta_iter, config, rng)`: pruning is disabled before `prune_after_iterations`, disabled when `traversal_pruning_enabled` is false, and post-warmup pruning occurs when a sampled random value is greater than or equal to `prune_explore_pct`. Therefore `prune_explore_pct: 0.10` means roughly 10% of post-warmup batches explore all eligible actions instead of applying ordinary traversal pruning. The focused `should_prune` tests pass.
