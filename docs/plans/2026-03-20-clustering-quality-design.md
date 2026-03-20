# Clustering Quality Improvements

**Date:** 2026-03-20
**Status:** Approved
**Scope:** Global clustering pipeline only (not per-flop)

## Motivation

Analysis of robopoker's clustering implementation revealed several quality improvements we can borrow. Our current pipeline silently switched from EMD to L2 distance during the fastkmeans-rs migration, discards centroids after clustering, doesn't sort bucket IDs by strategic value, and treats all bucket gaps as uniform in EMD computation.

## Decisions

| Question | Answer |
|----------|--------|
| L2/EMD fix approach | Hybrid: L2 for centroid-finding, EMD for exhaustive assignment |
| Centroid storage | Separate file (`*.centroids`), not embedded in BucketFile |
| Ground distance approach | Weighted 1-D EMD using adjacent centroid equity gaps |
| Pipeline scope | Global pipeline only; per-flop untouched for now |

## Design

### 1. Centroid Persistence

A new `CentroidFile` stored alongside each `BucketFile`.

- **Format:** Magic `"CEN1"` (4 bytes) + street (1 byte) + K (2 bytes, u16) + dim (2 bytes, u16) + K×dim f64 values (little-endian)
- **Files produced:** `river.centroids`, `turn.centroids`, `flop.centroids`
- **Contents:** For river, K scalar equity centroids (dim=1). For turn/flop, K histogram centroids (dim = child-street bucket count).
- **Lifetime:** Written during `run_clustering_pipeline`, read back during the same pipeline run for downstream streets. Not needed at MCCFR runtime.
- **API:** `CentroidFile::save(path)`, `CentroidFile::load(path)`, `centroids() -> &[Vec<f64>]`

### 2. Sort Bucket IDs by Expected Equity

After k-means finds centroids, reorder cluster IDs so bucket 0 = lowest expected equity, bucket K-1 = highest.

**For river:** Already sorted — `fast_kmeans_1d` on equity values produces naturally ordered centroids. No change needed.

**For turn:** Each centroid is a histogram over river buckets. Compute expected equity as:
```
EV(turn_centroid_j) = Σ_i turn_centroid_j[i] × river_centroid_i_equity
```
Sort turn centroids by EV. Build a `remap[old_id] -> new_id` permutation. Apply remap to all bucket assignments in the turn BucketFile before saving.

**For flop:** Same logic using turn centroids' expected equities:
```
EV(flop_centroid_j) = Σ_i flop_centroid_j[i] × EV(turn_centroid_i)
```
Ordering propagates bottom-up: river → turn → flop.

**Where in pipeline:** Immediately after k-means returns labels+centroids, before the exhaustive assignment phase. Sorted centroids are what gets persisted and used for exhaustive assignment.

### 3. EMD for Exhaustive Assignment

In `cluster_histogram_exhaustive`, replace:
```rust
nearest_centroid_l2(&hist, &centroids)
```
with:
```rust
nearest_centroid_u8(&hist, &centroid_f64s)
```

The centroids from `fast_kmeans_histogram` (Vec<Vec<f32>>) are converted to Vec<Vec<f64>> (normalized probability distributions) for `nearest_centroid_u8` which calls `emd_u8_vs_f64`.

Centroid sorting (Section 2) happens between k-means and assignment, so exhaustive assignment uses correctly ordered centroids.

**Scope:** `cluster_turn_exhaustive` and `cluster_flop_exhaustive` (both via `cluster_histogram_exhaustive`). River already correct. Per-flop untouched.

### 4. Weighted 1-D EMD with Ground Distances

After sorting centroids by expected equity, adjacent centroids have non-uniform spacing. Weight EMD by actual equity gaps.

**Ground distance vector:** For K sorted centroids with expected equities `ev[0] < ev[1] < ... < ev[K-1]`:
```
gap[i] = ev[i+1] - ev[i]    for i in 0..K-1
```

**Weighted EMD formula:**
```
EMD(p, q) = Σ_{i=0}^{K-2} |CDF_p[i] - CDF_q[i]| × gap[i]
```

**New functions:** `emd_u8_vs_f64_weighted(counts, centroid, gaps)`. Gaps vector has length K-1.

**Where gaps come from:**
- Turn clustering uses river centroid equity gaps
- Flop clustering uses turn centroid EV gaps (computed recursively from river)

**Pipeline integration:** Load child-street centroids from `CentroidFile`, compute EVs, derive gaps. Pass gaps into `nearest_centroid_u8_weighted` (new variant) during exhaustive assignment.

## Dependency Order

```
1. CentroidFile (foundation — no dependencies)
2. Sort bucket IDs (requires centroids)
3. EMD for exhaustive assignment (requires sorted centroids)
4. Weighted EMD with ground distances (requires sorted centroids + child-street centroid EVs)
```

## Files Modified

- `crates/core/src/blueprint_v2/clustering.rs` — new weighted EMD functions, new `nearest_centroid_u8_weighted`
- `crates/core/src/blueprint_v2/cluster_pipeline.rs` — pipeline orchestration, sorting, centroid persistence
- `crates/core/src/blueprint_v2/centroid_file.rs` — new module for CentroidFile format
- `crates/core/src/blueprint_v2/mod.rs` — add centroid_file module
