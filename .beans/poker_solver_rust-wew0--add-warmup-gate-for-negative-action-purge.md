---
# poker_solver_rust-wew0
title: Add warmup gate for negative-action purge
status: in-progress
type: task
priority: high
created_at: 2026-05-14T16:39:07Z
updated_at: 2026-05-14T16:39:20Z
---

Add a warmup function for the experimental MP lazy sparse negative-action subtree purge. The gate should keep purge/block behavior disabled during an initial training warmup window, then enable prune/reactivation semantics after warmup. Prefer reusing training iteration semantics and keeping defaults current-behavior safe. Update config/docs/tests as needed.

## Semantics

Negative-action subtree purge must be inactive until pruning is active. Reuse the existing pruning warmup decision rather than adding an independent warmup clock if possible: before pruning starts, negative regrets should not purge descendants, mark blocked edges, or emit prune/reactivation telemetry. Once pruning starts, the existing negative-action prune/reactivate thresholds apply.
