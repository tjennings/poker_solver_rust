# U8 Histogram Quantization for Canonical Clustering

## Problem

Canonical turn/flop clustering stores histograms as `Vec<f64>` (8 bytes per bin).
With 600 buckets and ~15K canonical turns x 1,300 combos, peak memory reaches ~90 GB,
causing OOM on most machines.

## Solution

Store histograms as `Vec<u8>` raw integer counts instead of `Vec<f64>` normalized
probabilities. The underlying data is integer counts 0-47 (number of next-street
cards that land in each bucket), so u8 is lossless.

**Memory reduction: 8x** (4,800 bytes -> 600 bytes per histogram, ~90 GB -> ~11 GB).

## Approach

Dedicated u8 functions alongside existing f64 versions. Existing dead-code
sampling-based functions are untouched.

### New functions in `clustering.rs`

| Function | Purpose |
|----------|---------|
| `emd_u8(a: &[u8], b: &[u8]) -> f64` | EMD between two unnormalized count histograms |
| `emd_u8_vs_f64(counts: &[u8], centroid: &[f64]) -> f64` | EMD between u8 data point and f64 centroid |
| `nearest_centroid_u8(point: &[u8], centroids: &[Vec<f64>]) -> u16` | Assignment step |
| `kmeanspp_init_u8(data: &[Vec<u8>], k, rng) -> Vec<Vec<f64>>` | Initialization (normalizes chosen points to f64 centroids) |
| `farthest_point_u8(data: &[Vec<u8>], assignments, centroids) -> usize` | Empty cluster reseeding |
| `kmeans_emd_weighted_u8(data: &[Vec<u8>], weights, k, max_iter, seed, progress) -> Vec<u16>` | Main entry point |

Centroids remain `Vec<f64>` throughout (they are weighted averages of normalized distributions).

### Changes in `cluster_pipeline.rs`

| Function | Change |
|----------|--------|
| `build_next_street_histogram_u8` | New variant returning `Vec<u8>` raw counts (no normalization) |
| `cluster_turn_canonical` | `board_features` -> `Vec<Vec<Option<Vec<u8>>>>`, calls `kmeans_emd_weighted_u8` |
| `cluster_flop_canonical` | Same as above |

### Precision

Zero loss. The f64 histograms stored values like `count/46`. With u8 we store
the exact integer `count` and divide by `total` on-the-fly during EMD computation
using f64 arithmetic. The EMD distances are numerically identical.

### What stays the same

- Existing f64 k-means functions (used by sampling-based dead code)
- `into_iter` move optimization in flatten step
- Centroid representation (`Vec<f64>`)
- Output format (`Vec<u16>` cluster assignments)
