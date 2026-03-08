---
# poker_solver_rust-9udu
title: Exhaustive canonical bucket file lookups for MCCFR
status: completed
type: feature
created_at: 2026-03-08T19:54:04Z
updated_at: 2026-03-08T19:54:04Z
---

Replace runtime compute_equity() fallback in MCCFR get_bucket() with O(1) precomputed lookups using exhaustive canonical board bucket files.

## Summary of Changes

- **PackedBoard** (bucket_file.rs): Compact u64 board key for hashing/serialization
- **combo_index** (cluster_pipeline.rs): Maps (Card, Card) → 0..1325 matching enumerate_combos ordering
- **Canonical board enumeration** (cluster_pipeline.rs): enumerate_canonical_flops (1,755), enumerate_canonical_turns (~20K), enumerate_canonical_rivers (~700K)
- **Weighted k-means** (clustering.rs): kmeans_1d_weighted and kmeans_emd_weighted for proper isomorphism frequency weighting
- **BucketFile v2** (bucket_file.rs): Board table stored between header and bucket data, backward compatible with v1
- **Canonical clustering pipeline** (cluster_pipeline.rs): cluster_river/turn/flop_canonical use exhaustive enumeration + weighted k-means
- **get_bucket wiring** (mccfr.rs): AllBuckets builds HashMap<PackedBoard, u32> at construction, get_bucket canonicalizes board + hole cards for O(1) lookup
- **Resume-safe caching** (trainer.rs): Skip clustering when bucket files already exist
