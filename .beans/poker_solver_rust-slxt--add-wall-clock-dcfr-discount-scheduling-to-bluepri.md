---
# poker_solver_rust-slxt
title: Add wall-clock DCFR discount scheduling to blueprint_mp
status: in-progress
type: feature
priority: high
created_at: 2026-08-05T14:24:48Z
updated_at: 2026-08-05T15:22:23Z
---

Implement the approved time-based discount interval design for blueprint_mp.

- [x] Verify algorithm and architecture decisions
- [x] Establish focused core/trainer baseline under explicit runtime waiver
- [x] Implement config, scheduler, telemetry, and docs (checkpoint persistence explicitly deferred)
- [x] Add deterministic fake-clock and compatibility tests
- [ ] Complete independent code review and repairs
- [ ] Run the entire test suite under one minute
- [ ] Integrate an atomic implementation commit

Reference: poker_solver_rust-ft2v development handoff and poker_solver_rust-qh5t literature research.

## Baseline Blocker

Initial full `cargo test` was interrupted after 104.37 seconds while entering `gpu_range_solver`; the required under-one-minute baseline was not met. Feature implementation is paused while the test-runtime path is diagnosed and repaired.

## Approved Scope

Implement config, a deterministic elapsed-time scheduler, eager/lazy integration, explicit wall-clock pass epochs, legacy compatibility, startup/per-pass telemetry, tests, and training/architecture documentation. Preserve lazy purge after each actual pass. Do not add the 40-pass stopping rule or claim checkpoint restoration; MP resume does not currently exist and snapshots cannot atomically capture trainer-local scheduler state. Wall-clock mode uses monotonic process-up elapsed time, including pauses and sweep/checkpoint overhead.

## User Waiver

On 2026-08-05 the user explicitly authorized proceeding with focused core/trainer tests and a final full-suite correctness run regardless of the pre-existing full-suite runtime.

## Focused Baseline

`poker-solver-core` library tests passed: 1,274 passed, 17 ignored. `poker-solver-trainer` binary tests passed: 342 passed, 1 ignored. The combined integration sweep was stopped after the relevant core library passed because of the explicitly waived per-binary harness latency.

## Implementation Summary

Added optional nonzero `dcfr_discount_interval_seconds`, a pure elapsed-time/iteration boundary scheduler shared by eager and lazy MP runners, explicit pass epochs, skipped-slot telemetry, legacy boundary-crossing compatibility, deterministic tests, and updated training/architecture/sample documentation. Checkpoint persistence and maximum-pass stopping remain out of scope by design.

Validation: formatting passed; focused discount tests passed 55/55; core library passed 1,281 tests with 17 ignored after making the existing baseline fixture visible to the worktree; trainer binary passed 342 tests with 1 ignored; workspace compile passed. The all-target workspace check remains blocked by pre-existing broken `equity_table_bench` imports.
