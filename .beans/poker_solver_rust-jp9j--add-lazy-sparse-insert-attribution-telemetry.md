---
# poker_solver_rust-jp9j
title: Add lazy sparse insert attribution telemetry
status: completed
type: task
priority: high
created_at: 2026-05-08T00:59:33Z
updated_at: 2026-05-08T01:03:19Z
---

Sparse activity telemetry shows high hit rates but sustained insert growth. Add insert attribution so we can tell whether new sparse infosets cluster by street, seat, SPR bucket, history length, or action count.

Tasks:
- [x] Add cumulative insert attribution counters to SparseMpStorage.
- [x] Report interval insert attribution in lazy_sparse no-TUI heartbeat.
- [x] Add focused tests for attribution counters.
- [x] Update training docs with attribution fields.
- [x] Run focused sparse/lazy tests.

## Summary of Changes

Added cumulative sparse insert attribution counters for street, seat, SPR bucket, history-length bins, and action-count shape. Lazy sparse no-TUI heartbeat now emits an insert_by[...] block with interval rates and top contributors, including cumulative max_seen values for history length and action count. Added focused storage tests and updated training docs.

Focused tests passed:
- cargo test -q -p poker-solver-core sparse_storage
- cargo test -q -p poker-solver-trainer lazy_sparse
