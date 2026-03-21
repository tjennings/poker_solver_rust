---
# poker_solver_rust-a2cy
title: Fix CBV backward induction bucket clamping at Chance nodes
status: todo
type: bug
created_at: 2026-03-21T13:15:56Z
updated_at: 2026-03-21T13:15:56Z
---

cbv_compute.rs clamps bucket index at Chance nodes (child_bucket = bucket.min(next_buckets-1)) instead of properly mapping flop buckets to turn buckets. Flop bucket 500 has no relation to turn bucket 500. All precomputed CBVs are wrong. Fix: at each Chance node, enumerate concrete hands per bucket, deal next-street cards, look up correct next-street bucket, average the child CBVs. Requires retrain after fix.
