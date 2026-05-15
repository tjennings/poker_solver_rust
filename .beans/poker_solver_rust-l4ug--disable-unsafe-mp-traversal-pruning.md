---
# poker_solver_rust-l4ug
title: Disable unsafe MP traversal pruning
status: completed
type: bug
priority: critical
created_at: 2026-05-15T13:53:11Z
updated_at: 2026-05-15T13:57:54Z
---

User still sees MP strategies collapse immediately after ordinary traversal pruning starts; telemetry shows pruning near 100%. The current regret-threshold pruning is unsafe for MP lazy training because pruned actions are not explicitly scheduled for re-entry and can starve traversal/averaging. Disable ordinary traversal pruning for MP while leaving negative-action subtree purge as the explicit memory experiment.

## Summary of Changes

Disabled ordinary MP regret-threshold traversal pruning because live telemetry showed it climbing toward 100% and starving strategy averaging after `prune_after_iterations`. Left `prune_after_iterations` as the warmup gate for the opt-in negative-action subtree purge experiment, updated unit tests, config comments, and docs to make the behavior explicit.

## Validation

- `cargo fmt --check`
- `cargo test -p poker-solver-core should_prune -- --nocapture`
- `cargo test -p poker-solver-core negative_action_traversal_config -- --nocapture`
