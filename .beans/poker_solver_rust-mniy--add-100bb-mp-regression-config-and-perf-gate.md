---
# poker_solver_rust-mniy
title: Add 100bb MP regression config and perf gate
status: completed
type: task
priority: high
created_at: 2026-05-07T13:18:56Z
updated_at: 2026-05-07T16:06:52Z
parent: poker_solver_rust-5kvv
blocked_by:
    - poker_solver_rust-xsga
    - poker_solver_rust-u81t
---

Add a 100bb 6-max Blueprint MP regression config with multiple preflop raise depths and a bounded smoke/perf test that verifies setup does not allocate dense 100bb-scale storage and can advance MCCFR iterations with stable heartbeat telemetry.

## Work Started

- [x] Add dedicated 100bb lazy_sparse regression/sample config
- [x] Add bounded inspect/train smoke test for two preflop raise rows
- [x] Update training docs for the regression config
- [x] Run focused verification

## Summary of Changes

Added a committed 100bb 6-max lazy_sparse smoke config and a trainer regression test that inspects the config, verifies it remains an eager dense-risk shape, confirms lazy setup has no allocated sparse infosets, advances one meta-iteration, and asserts sparse storage stays bounded. Updated training docs to list the regression config.

Verification: cargo test -q -p poker-solver-trainer mp_100bb_lazy_sparse_smoke_config_advances_without_dense_setup; cargo test -q -p poker-solver-trainer lazy_sparse; cargo test -q -p poker-solver-trainer inspect_mp_config
