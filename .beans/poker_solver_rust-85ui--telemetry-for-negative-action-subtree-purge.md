---
# poker_solver_rust-85ui
title: Telemetry for negative-action subtree purge
status: completed
type: task
priority: high
created_at: 2026-05-14T14:32:07Z
updated_at: 2026-05-14T16:00:59Z
parent: poker_solver_rust-xl3h
blocked_by:
    - poker_solver_rust-lvv0
---

Expose enough telemetry to evaluate the experiment during long 6-max runs. Track actions newly pruned, actions reactivated, subtree purge calls, rows purged, regret slots purged, strategy slots purged, allocations blocked under pruned edges, and purge scan time. Surface the values in existing no-TUI lazy sparse logs and TUI metrics where appropriate. Acceptance: late-run logs make it obvious whether memory is still growing from new inserts or being offset by purges.

## Implementation Notes

Starting after traversal gating landed in commit c6ef944e. Focus this slice on operational telemetry: edge prune/reactivation counters, purge call/row/slot totals, and enough log/TUI visibility to know whether long runs are reducing resident sparse rows or continuing to grow.

## Summary of Changes

Added cumulative negative-action purge telemetry to sparse storage and exposed it in no-TUI lazy sparse heartbeat logs as `neg_action[...]`. Counters include blocked edges, newly pruned edges, reactivations, purge calls, rows and slots purged, blocked traversal skips, and purge scan duration. The log line keeps these beside resident sparse storage counters so long runs show both growth and purge offset. TUI widgets were not expanded in this slice because the existing MP TUI path does not have a small storage-telemetry surface.

Review follow-up corrected `blocked_skips` so it counts actual blocked traversal attempts rather than every blocked mask entry. Remaining known caveat: cumulative atomic snapshots can be momentarily inconsistent across fields during concurrent updates, which is acceptable for coarse heartbeat telemetry.
