---
# poker_solver_rust-mniy
title: Add 100bb MP regression config and perf gate
status: in-progress
type: task
priority: high
created_at: 2026-05-07T13:18:56Z
updated_at: 2026-05-07T16:03:42Z
parent: poker_solver_rust-5kvv
blocked_by:
    - poker_solver_rust-xsga
    - poker_solver_rust-u81t
---

Add a 100bb 6-max Blueprint MP regression config with multiple preflop raise depths and a bounded smoke/perf test that verifies setup does not allocate dense 100bb-scale storage and can advance MCCFR iterations with stable heartbeat telemetry.

## Work Started

- [ ] Add dedicated 100bb lazy_sparse regression/sample config
- [ ] Add bounded inspect/train smoke test for two preflop raise rows
- [ ] Update training docs for the regression config
- [ ] Run focused verification
