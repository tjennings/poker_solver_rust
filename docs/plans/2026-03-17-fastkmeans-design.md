# Replace Custom K-Means with fastkmeans-rs

**Date:** 2026-03-17
**Goal:** Replace custom k-means implementations with fastkmeans-rs (BLAS-accelerated via Apple Accelerate) for faster clustering.

## Changes

Replace `kmeans_1d_weighted` and `kmeans_emd_weighted_u8` with a `fast_kmeans` wrapper around `fastkmeans-rs`. L2 distance replaces EMD (acceptable tradeoff given dense per-flop histograms).

### Wrapper Function

```rust
fn fast_kmeans(
    features: &[Vec<f32>],
    k: usize,
    max_iters: u32,
    seed: u64,
) -> (Vec<u16>, Vec<Vec<f32>>)
```

Converts to `Array2<f32>`, calls `FastKMeans::fit_predict`, returns labels and centroids in our format.

### Callers

1. `cluster_river_exhaustive` — 1D equity
2. `cluster_single_flop` — 1D equity (river) + histogram (turn)
3. `cluster_histogram_exhaustive` — histogram (turn/flop, old pipeline)
4. `run_per_flop_pipeline` — histogram (global flop clustering)
5. `cluster_river_from_cfvnet` — 1D equity

### Progress

Per-iteration k-means progress bars removed (fastkmeans-rs has no callback). Callers report before/after instead. Phase-level progress (river per-turn, etc.) unaffected.

### Dependencies

```toml
fastkmeans-rs = { version = "0.1.8", features = ["accelerate"] }
ndarray = "0.16"
```

### What Does NOT Change

- `nearest_centroid_1d` / `nearest_centroid_u8` (exhaustive assignment phase)
- Per-flop pipeline structure, file formats, MCCFR integration
- Weighted samples: fastkmeans-rs doesn't support weights, but our per-flop pipeline uses uniform weights (1.0) everywhere. The old global pipeline used canonical board weights — for backwards compat, duplicate samples proportionally or accept uniform weighting.
