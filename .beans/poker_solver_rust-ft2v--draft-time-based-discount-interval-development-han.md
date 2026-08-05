---
# poker_solver_rust-ft2v
title: Draft time-based discount interval development handoff
status: completed
type: task
priority: normal
created_at: 2026-08-05T14:20:43Z
updated_at: 2026-08-05T14:21:51Z
---

Produce an implementation-ready summary for adding a time-based discount interval config key to blueprint_mp.

- [x] Define config semantics and precedence
- [x] Define runtime/checkpoint behavior and edge cases
- [x] Define acceptance tests and documentation changes
- [x] Deliver concise development handoff

## Summary of Changes

Prepared a development handoff proposing optional dcfr_discount_interval_seconds wall-clock scheduling, explicit schedule-mode selection, monotonic coordinator-owned deadlines, persisted pass epochs and remaining time across checkpoints, no catch-up discount bursts, fake-clock tests, telemetry, backward compatibility, and documentation updates. The handoff identifies the legacy exact-modulo defect and distinguishes a future stop-after-passes option from merely capping the DCFR factor epoch.
