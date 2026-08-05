---
# poker_solver_rust-slxt
title: Add wall-clock DCFR discount scheduling to blueprint_mp
status: completed
type: feature
priority: high
created_at: 2026-08-05T14:24:48Z
updated_at: 2026-08-05T15:51:54Z
---

Implement the approved time-based discount interval design for blueprint_mp.

- [x] Verify algorithm and architecture decisions
- [x] Establish focused core/trainer baseline under explicit runtime waiver
- [x] Implement config, scheduler, telemetry, and docs (checkpoint persistence explicitly deferred)
- [x] Add deterministic fake-clock and compatibility tests
- [x] Complete initial independent code review and repairs
- [x] Complete independent re-review
- [x] Run the entire test suite for correctness under explicit runtime waiver
- [x] Integrate an atomic implementation commit

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

## Review Repair

Fixed construction-time wall-clock arming when a lazy stepper starts at or beyond warmup, corrected documented MP fallback defaults, separated lazy DCFR sweep timing from negative-action purge timing, and added scheduled-boundary plus execution-elapsed pass telemetry. Added deterministic equality/greater-than start tests while retaining the below-warmup crossing test. Nonzero-count, newly-zeroed-count, scan-volume, and aggregate overhead telemetry remain explicitly out of scope because they require performance-sensitive scan aggregation; no extra full-table scans were added.

## Summary of Changes

Implemented and integrated optional wall-clock DCFR scheduling for blueprint_mp. The new nonzero `dcfr_discount_interval_seconds` key overrides legacy iteration cadence, uses shared deterministic eager/lazy scheduling with executed-pass epochs and no catch-up sweeps, repairs legacy missed-boundary triggering without changing legacy epoch semantics, preserves lazy purge ordering, and adds accurate schedule/sweep/purge telemetry. Updated config fixtures, active 600-second HU sample, training documentation, and architecture documentation. Independent re-review approved with no remaining findings. Post-merge formatting and 57 discount tests passed. The complete workspace `cargo test` exited successfully in 1,088.47 seconds under the user-approved runtime waiver; focused branch validation also passed 1,283 core tests and 342 trainer tests.
