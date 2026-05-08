---
# poker_solver_rust-6abz
title: Audit lazy MP bets per street against config
status: in-progress
type: bug
priority: high
created_at: 2026-05-08T01:42:37Z
updated_at: 2026-05-08T01:42:37Z
---

Lazy sparse training still inserts many low-SPR river infosets. Audit whether observed per-street raise counts exceed configured raise rows, including all-in aggression paths that may bypass sized raise limits.\n\nTasks:\n- [ ] Add lazy action-limit telemetry for per-street max raise count and over-config decision/action counts.\n- [ ] Report the audit fields in lazy_sparse no-TUI heartbeat.\n- [ ] Add focused tests showing all-in aggression can be observed past sized raise rows or is correctly counted.\n- [ ] Update training docs with the new action-limit audit fields.\n- [ ] Run focused lazy/trainer tests.
