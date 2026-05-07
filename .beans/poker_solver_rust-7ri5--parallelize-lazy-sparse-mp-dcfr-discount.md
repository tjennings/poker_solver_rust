---
# poker_solver_rust-7ri5
title: Parallelize lazy sparse MP DCFR discount
status: in-progress
type: task
priority: high
created_at: 2026-05-07T18:00:45Z
updated_at: 2026-05-07T18:00:45Z
---

Parallelize SparseMpStorage::discount across shards so lazy_sparse 100bb training does not collapse to one CPU core during DCFR discount passes.\n\n- [ ] Parallelize sparse discount over shards\n- [ ] Add focused regression coverage for discount correctness\n- [ ] Run focused verification\n- [ ] Keep discount timing telemetry available for before/after comparison
