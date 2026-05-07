---
# poker_solver_rust-vq11
title: Smooth lazy sparse MP storage contention telemetry
status: completed
type: task
priority: high
created_at: 2026-05-07T16:53:30Z
updated_at: 2026-05-07T16:56:40Z
---

Raise the default lazy_sparse MP storage shard count and add heartbeat telemetry that can reveal shard imbalance and sparse allocation growth during 100bb training.\n\n- [x] Raise default sparse storage shard count\n- [x] Add shard imbalance/growth fields to sparse heartbeat\n- [x] Add focused sparse storage regression coverage\n- [x] Run focused verification

## Summary of Changes

Raised the default lazy sparse MP storage shard count from 256 to 4096, expanded sparse storage stats with shard count, nonempty shard count, and max entries per shard, and added lazy no-TUI heartbeat fields for entries/sec, storage bytes/sec, and shard distribution. Updated the 100bb training docs to describe the diagnostic fields.

Verification: cargo test -q -p poker-solver-core sparse_storage; cargo test -q -p poker-solver-trainer lazy_sparse; cargo test -q -p poker-solver-trainer mp_no_tui_heartbeat_interval_is_one_minute; cargo test -q -p poker-solver-core default_storage_uses_high_shard_count_for_large_lazy_runs; cargo test -q -p poker-solver-core stats_count_entries_and_slots
