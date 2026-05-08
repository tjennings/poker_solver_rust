---
# poker_solver_rust-q0zk
title: Add lazy sparse storage activity telemetry
status: completed
type: task
priority: high
created_at: 2026-05-08T00:41:24Z
updated_at: 2026-05-08T00:44:44Z
---

Training throughput collapses after sparse storage reaches hundreds of millions of entries even with cheap stats. Add lightweight telemetry to separate allocation pressure from repeated lookup pressure and hit/miss behavior.

Tasks:
- [x] Add sparse storage activity counters for read probes/hits, write probes/hits, and inserts.
- [x] Report per-heartbeat activity deltas/rates in no-TUI lazy_sparse output.
- [x] Add focused tests for counter behavior and heartbeat formatting if appropriate.
- [x] Update training docs with the new telemetry fields.
- [x] Run focused sparse/lazy tests.

## Summary of Changes

Added cumulative sparse storage activity counters for read probes/hits, write probes/hits, and inserts. Lazy sparse no-TUI heartbeat now reports per-interval activity rates and hit percentages in an activity[...] block so throughput drops can be correlated with lookup pressure or allocation pressure. Added sparse storage counter coverage and documented the new fields.

Focused tests passed:
- cargo test -q -p poker-solver-core sparse_storage
- cargo test -q -p poker-solver-trainer lazy_sparse
