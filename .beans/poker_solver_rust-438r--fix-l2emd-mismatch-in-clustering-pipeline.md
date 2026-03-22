---
# poker_solver_rust-438r
title: Fix L2/EMD mismatch in clustering pipeline
status: completed
type: bug
priority: high
created_at: 2026-03-22T13:51:11Z
updated_at: 2026-03-22T14:58:03Z
parent: poker_solver_rust-60h7
---

cluster_histogram_exhaustive uses fast_kmeans_histogram (L2 k-means) for centroid finding but EMD for assignment. This metric mismatch causes postflop buckets to conflate strategically different hands, making the blueprint 3bet trash hands. Fix: replace L2 centroid phase with EMD k-means (kmeans_emd_weighted_u8 already exists in clustering.rs:571). Also port perf improvements from ../robopoker's clustering crate (elkan.rs, emd.rs). Consider reducing postflop buckets from 1000 to 200 (Pluribus used 200 with proper EMD).

## Summary of Changes

- Added `ElkanBounds` struct for triangle-inequality accelerated k-means
- Added `elkan_emd_weighted_u8` — Elkan (2003) algorithm with EMD distance
- Extracted shared `weighted_centroid_update_u8` helper (eliminates duplication)
- Extracted `compute_pairwise_and_midpoints` with symmetry optimization (K^2 → K*(K-1)/2)
- Extracted `init_elkan_bounds` initialization helper
- Replaced all 5 `fast_kmeans_histogram` (L2) call sites with Elkan EMD k-means
- Removed `fast_kmeans_histogram` and `nearest_centroid_l2` (dead code)
- Marked `kmeans_emd_weighted_u8` as `#[cfg(test)]` (test-only reference impl)
- Added 12 new tests including naive-equivalence anchor
- Verified: 629 tests pass, clippy clean
