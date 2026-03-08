---
# poker_solver_rust-ztc6
title: Wire BucketFile lookups into MCCFR get_bucket
status: todo
type: task
priority: high
created_at: 2026-03-08T18:50:46Z
updated_at: 2026-03-08T18:50:46Z
---

get_bucket() in mccfr.rs falls back to compute_equity() for all postflop streets even when BucketFile data is loaded. Need to implement the canonical board→index and hole-cards→combo_idx mappings so get_bucket() uses the precomputed cluster assignments from the clustering pipeline. This is the single biggest perf bottleneck in MCCFR training.
