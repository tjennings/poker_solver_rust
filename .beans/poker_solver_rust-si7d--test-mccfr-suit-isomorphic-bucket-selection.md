---
# poker_solver_rust-si7d
title: Test MCCFR suit-isomorphic bucket selection
status: completed
type: task
priority: normal
created_at: 2026-05-20T18:39:35Z
updated_at: 2026-05-20T18:43:19Z
---

Add coverage that Blueprint MP MCCFR bucket selection maps suit-isomorphic A2 suited hands with equivalent flush-draw texture to the same postflop buckets, while distinguishing suited A2 with no flush draw when the bucket file encodes that distinction.



## Summary of Changes

Added a Blueprint MP trainer regression at the MCCFR bucket precompute boundary. The test builds a synthetic flop bucket file that assigns A2s with the two-tone board suit to a flush-draw bucket and A2s without that suit to a no-draw bucket, then verifies suit-rotated A2s/flop cases resolve to the same buckets through compute_deal_buckets -> AllBuckets::get_bucket. This confirms board canonicalization plus holding remap are active before MCCFR traversal consumes bucket ids. Focused test passes.
