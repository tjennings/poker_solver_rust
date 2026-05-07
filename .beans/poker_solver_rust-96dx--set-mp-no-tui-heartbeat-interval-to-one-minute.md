---
# poker_solver_rust-96dx
title: Set MP no-TUI heartbeat interval to one minute
status: completed
type: task
priority: normal
created_at: 2026-05-07T16:48:19Z
updated_at: 2026-05-07T16:49:28Z
---

Change Blueprint MP no-TUI console heartbeat from 10 seconds to 60 seconds so progress reporting does not create short CPU utilization oscillations during lazy_sparse training.\n\n- [x] Update heartbeat interval and message\n- [x] Add or update focused test coverage\n- [x] Run focused verification

## Summary of Changes

Changed Blueprint MP no-TUI heartbeat reporting from every 10 seconds to every 60 seconds for both eager and lazy_sparse training paths. Added a focused regression test asserting the one-minute heartbeat interval.

Verification: cargo test -q -p poker-solver-trainer mp_no_tui_heartbeat_interval_is_one_minute
