---
# poker_solver_rust-438r
title: Fix L2/EMD mismatch in clustering pipeline
status: in-progress
type: bug
priority: high
created_at: 2026-03-22T13:51:11Z
updated_at: 2026-03-22T13:56:05Z
parent: poker_solver_rust-60h7
---

cluster_histogram_exhaustive uses fast_kmeans_histogram (L2 k-means) for centroid finding but EMD for assignment. This metric mismatch causes postflop buckets to conflate strategically different hands, making the blueprint 3bet trash hands. Fix: replace L2 centroid phase with EMD k-means (kmeans_emd_weighted_u8 already exists in clustering.rs:571). Also port perf improvements from ../robopoker's clustering crate (elkan.rs, emd.rs). Consider reducing postflop buckets from 1000 to 200 (Pluribus used 200 with proper EMD).
