---
# poker_solver_rust-e62l
title: Fix MP regret overflow in long training
status: in-progress
type: bug
priority: critical
created_at: 2026-05-06T13:23:38Z
updated_at: 2026-05-06T13:25:32Z
---

Latest 6-max MP trainer run has correct open actions but still saturates positive regret after roughly 1.5B iterations. Current MpStorage stores regrets as AtomicI16 with saturating adds, so long-running training can clamp the regret signal at i16::MAX. Investigate and implement a durable fix for billion-iteration MP runs.

## Implementation Plan

- [ ] Convert MP regret storage from AtomicI16 to AtomicI32
- [ ] Update regret delta, pruning threshold, DCFR discount, and tests for i32 regrets
- [ ] Update MP TUI regret telemetry scans for AtomicI32
- [ ] Run focused storage/trainer/TUI tests
- [ ] Close bean and commit implementation
