---
# poker_solver_rust-7ri5
title: Parallelize lazy sparse MP DCFR discount
status: completed
type: task
priority: high
created_at: 2026-05-07T18:00:45Z
updated_at: 2026-05-07T18:02:31Z
---

Parallelize SparseMpStorage::discount across shards so lazy_sparse 100bb training does not collapse to one CPU core during DCFR discount passes.\n\n- [x] Parallelize sparse discount over shards\n- [x] Add focused regression coverage for discount correctness\n- [x] Run focused verification\n- [x] Keep discount timing telemetry available for before/after comparison

## Summary of Changes

Parallelized SparseMpStorage::discount across sparse storage shards with Rayon, preserving the existing discount timing telemetry for before/after comparison. Added a multi-shard discount regression test that validates positive regrets, negative regrets, and strategy sums are discounted correctly across many shard-owned entries. Updated training docs to call out the parallel discount behavior and the discount timing field.

Verification: cargo test -q -p poker-solver-core discount_updates_entries_across_many_shards; cargo test -q -p poker-solver-core sparse_storage; cargo test -q -p poker-solver-trainer lazy_sparse
