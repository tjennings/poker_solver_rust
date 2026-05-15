---
# poker_solver_rust-l4ug
title: Disable unsafe MP traversal pruning
status: in-progress
type: bug
priority: critical
created_at: 2026-05-15T13:53:11Z
updated_at: 2026-05-15T13:53:11Z
---

User still sees MP strategies collapse immediately after ordinary traversal pruning starts; telemetry shows pruning near 100%. The current regret-threshold pruning is unsafe for MP lazy training because pruned actions are not explicitly scheduled for re-entry and can starve traversal/averaging. Disable ordinary traversal pruning for MP while leaving negative-action subtree purge as the explicit memory experiment.
