---
# poker_solver_rust-7vc2
title: Fix MP lazy pruning not activating from config
status: in-progress
type: bug
priority: high
created_at: 2026-06-30T13:31:40Z
updated_at: 2026-06-30T13:31:40Z
parent: poker_solver_rust-osss
---

MP lazy sparse pruning appears configured but does not activate: TUI reports 0 pruned and training does not show the expected throughput improvement. Investigate config parsing, runtime adapter propagation, lazy MCCFR pruning/purge gates, TUI telemetry labels, and whether the intended pruning mode is traversal pruning or negative-action subtree purge. Acceptance: a config with pruning enabled produces nonzero pruning/blocked-edge telemetry under a small deterministic test or clear diagnostic, and the TUI reports the active pruning mode accurately.
