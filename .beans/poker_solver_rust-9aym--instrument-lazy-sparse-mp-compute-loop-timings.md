---
# poker_solver_rust-9aym
title: Instrument lazy sparse MP compute loop timings
status: completed
type: task
priority: high
created_at: 2026-05-07T17:05:30Z
updated_at: 2026-05-07T17:08:35Z
---

Add low-overhead timing telemetry for lazy_sparse MP training so CPU utilization pauses can be correlated with deal sampling, bucket assignment, traversal, discounting, and console stats collection.\n\n- [x] Add timing counters for lazy compute-loop components\n- [x] Print timing fields in lazy no-TUI heartbeat\n- [x] Include console stats timing so reporting pauses can be confirmed or ruled out\n- [x] Add focused regression coverage\n- [x] Run focused verification

## Summary of Changes

Added lazy sparse MP timing counters for batch wall time, deal sampling, bucket lookup, traversal, and DCFR discounting. The lazy no-TUI heartbeat now drains those counters and also times sparse storage stats collection, printing all component timings beside the existing throughput/storage fields so CPU utilization dips can be correlated with compute phases or console-side reporting overhead. Updated training docs to describe the timing buckets.

Verification: cargo test -q -p poker-solver-core lazy_timing_snapshot_tracks_compute_components; cargo test -q -p poker-solver-core lazy_; cargo test -q -p poker-solver-trainer lazy_sparse
