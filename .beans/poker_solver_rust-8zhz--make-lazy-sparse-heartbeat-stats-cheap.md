---
# poker_solver_rust-8zhz
title: Make lazy sparse heartbeat stats cheap
status: completed
type: bug
priority: high
created_at: 2026-05-07T18:58:27Z
updated_at: 2026-05-07T19:01:09Z
---

Lazy sparse MP no-TUI heartbeat currently calls SparseMpStorage::stats(), which walks every sparse node to count slots. At hundreds of millions of entries this telemetry blocks training for 60-130s and causes a major IPS collapse.

Tasks:
- [x] Add exact live sparse storage counters for entries, slots, and per-shard entry counts.
- [x] Change heartbeat-facing stats/entry_count to read counters instead of scanning all nodes.
- [x] Preserve snapshot/load behavior and stats correctness in tests.
- [x] Update training docs to clarify cheap live sparse telemetry.
- [x] Run focused sparse/lazy tests.

## Summary of Changes

Sparse MP storage now maintains exact live counters for total entries, action slots, and per-shard occupancy when new infosets are inserted. Heartbeat stats and entry_count read those counters instead of locking and scanning every sparse node. Added regression coverage for duplicate updates and snapshot restore counters, and documented that lazy sparse telemetry is O(shards).

Focused tests passed:
- cargo test -q -p poker-solver-core sparse_storage
- cargo test -q -p poker-solver-trainer lazy_sparse
