---
# poker_solver_rust-85ui
title: Telemetry for negative-action subtree purge
status: todo
type: task
priority: high
created_at: 2026-05-14T14:32:07Z
updated_at: 2026-05-14T14:32:07Z
parent: poker_solver_rust-xl3h
blocked_by:
    - poker_solver_rust-lvv0
---

Expose enough telemetry to evaluate the experiment during long 6-max runs. Track actions newly pruned, actions reactivated, subtree purge calls, rows purged, regret slots purged, strategy slots purged, allocations blocked under pruned edges, and purge scan time. Surface the values in existing no-TUI lazy sparse logs and TUI metrics where appropriate. Acceptance: late-run logs make it obvious whether memory is still growing from new inserts or being offset by purges.
