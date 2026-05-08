---
# poker_solver_rust-jp9j
title: Add lazy sparse insert attribution telemetry
status: in-progress
type: task
priority: high
created_at: 2026-05-08T00:59:33Z
updated_at: 2026-05-08T00:59:33Z
---

Sparse activity telemetry shows high hit rates but sustained insert growth. Add insert attribution so we can tell whether new sparse infosets cluster by street, seat, SPR bucket, history length, or action count.\n\nTasks:\n- [ ] Add cumulative insert attribution counters to SparseMpStorage.\n- [ ] Report interval insert attribution in lazy_sparse no-TUI heartbeat.\n- [ ] Add focused tests for attribution counters.\n- [ ] Update training docs with attribution fields.\n- [ ] Run focused sparse/lazy tests.
