---
# poker_solver_rust-slxt
title: Add wall-clock DCFR discount scheduling to blueprint_mp
status: in-progress
type: feature
priority: high
created_at: 2026-08-05T14:24:48Z
updated_at: 2026-08-05T14:30:07Z
---

Implement the approved time-based discount interval design for blueprint_mp.

- [x] Verify algorithm and architecture decisions
- [ ] Establish clean baseline with full test suite under one minute
- [ ] Implement config, scheduler, checkpoint behavior, telemetry, and docs
- [ ] Add deterministic fake-clock and compatibility tests
- [ ] Complete independent code review and repairs
- [ ] Run the entire test suite under one minute
- [ ] Integrate an atomic implementation commit

Reference: poker_solver_rust-ft2v development handoff and poker_solver_rust-qh5t literature research.

## Baseline Blocker

Initial full `cargo test` was interrupted after 104.37 seconds while entering `gpu_range_solver`; the required under-one-minute baseline was not met. Feature implementation is paused while the test-runtime path is diagnosed and repaired.

## Approved Scope

Implement config, a deterministic elapsed-time scheduler, eager/lazy integration, explicit wall-clock pass epochs, legacy compatibility, startup/per-pass telemetry, tests, and training/architecture documentation. Preserve lazy purge after each actual pass. Do not add the 40-pass stopping rule or claim checkpoint restoration; MP resume does not currently exist and snapshots cannot atomically capture trainer-local scheduler state. Wall-clock mode uses monotonic process-up elapsed time, including pauses and sweep/checkpoint overhead.
