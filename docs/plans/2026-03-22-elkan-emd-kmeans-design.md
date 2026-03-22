# Elkan EMD K-Means for Clustering Pipeline

**Date:** 2026-03-22
**Bean:** poker_solver_rust-438r (Fix L2/EMD mismatch in clustering pipeline)
**Status:** Approved

## Problem

`cluster_histogram_exhaustive` (and 4 other call sites) use L2 k-means via
`fast_kmeans_histogram` (BLAS-accelerated) for centroid-finding, but the
assignment phase uses EMD. This metric mismatch degrades bucket quality —
centroids are optimized for L2 while assignments use EMD. The result: postflop
buckets conflate strategically different hands (e.g., blueprint 3bets trash
hands at 8%).

## Solution

Port robopoker's Elkan algorithm into `clustering.rs` as a runtime-sized EMD
k-means, then replace all 5 `fast_kmeans_histogram` call sites.

## Design

### New Types in `clustering.rs`

**`ElkanBounds`** — Per-point distance bounds (runtime-sized):
- `j: usize` — assigned centroid index
- `lower: Vec<f32>` — lower bounds on distance to each centroid
- `error: f32` — upper bound on distance to assigned centroid
- `stale: bool` — whether upper bound needs refreshing after centroid drift

Same logic as robopoker's `Bounds<K>` but with `Vec<f32>` instead of `[f32; K]`.

**`elkan_emd_weighted_u8`** — Top-level function:
```rust
fn elkan_emd_weighted_u8(
    data: &[Vec<u8>],
    weights: &[f64],
    k: usize,
    max_iterations: u32,
    seed: u64,
    progress: impl Fn(u32, u32),
) -> (Vec<u16>, Vec<Vec<f64>>)
```

### Elkan Iteration (ported from robopoker `step_elkan`)

1. Compute K×K pairwise centroid EMD distances
2. Compute midpoints: `s(c) = min_{c'≠c} d(c,c') / 2`
3. For each point where `upper > s(assigned)`:
   - Refresh stale upper bound
   - Check all centroids via triangle inequality pruning
4. Recompute centroids as weighted mean of assigned points
5. Compute centroid drift, update all bounds

### Key Adaptations from robopoker

- `const K/N` → runtime `k/n` with heap-allocated `Vec`s
- `Absorb` trait → inline weighted averaging (same as `kmeans_emd_weighted_u8`)
- `self.distance()` → `emd_u8_vs_f64()` for point-centroid, `emd()` for centroid-centroid
- Parallelism via Rayon `par_iter_mut` (same as robopoker)

### Call Site Changes

5 call sites in `cluster_pipeline.rs` replace `fast_kmeans_histogram` → `elkan_emd_weighted_u8`:

1. **Line 496** (`cluster_histogram_exhaustive`) — Main global pipeline for turn/flop.
   Already has `all_weights`. Centroid f32→f64 conversion block (lines 504-512)
   becomes unnecessary since `elkan_emd_weighted_u8` returns normalized f64 centroids.

2. **Line 172** (global turn-only path) — Currently ignores centroids. Pass
   uniform weights `vec![1.0; n]` and progress closure.

3. **Line 294** (global flop-only path) — Same pattern as #2.

4. **Line 1136** (per-flop turn clustering) — Same pattern.

5. **Line 1299** (per-flop flop clustering) — Same pattern.

### What Gets Deleted

- `fast_kmeans_histogram` function (`clustering.rs:942-959`)
- f32→f64 centroid conversion block in `cluster_histogram_exhaustive` (lines 504-512)
- `fastkmeans-rs` crate dependency (if no other callers)

### What Stays Unchanged

- `fast_kmeans_1d` for river clustering (1-D equity, L2 is correct)
- Exhaustive assignment phase (already uses proper EMD)
- `kmeanspp_init_u8` — reused by Elkan function
- All EMD distance functions (`emd`, `emd_u8`, `emd_u8_vs_f64`, etc.)

### Error Handling & Edge Cases

- **Empty clusters:** Same re-seeding strategy as `kmeans_emd_weighted_u8` —
  pick farthest point from all centroids. Reset bounds after reassignment.
- **k >= n:** Short-circuit to 1:1 assignment.
- **Early convergence:** Break if no assignments change.
- **Bounds precision:** f32 for bounds (matching robopoker). EMD values are
  small enough for pruning decisions.

### Testing

- **`elkan_naive_equivalence`** — Run both `elkan_emd_weighted_u8` and
  `kmeans_emd_weighted_u8` on same data with same seed, verify identical
  assignments.
- **Convergence** — Verify RMS decreases over iterations.
- **Integration** — Existing `cargo test -p poker-solver-core` covers the pipeline.

### Performance Expectations

- Elkan's pruning skips ~80-95% of distance computations after first few iterations
- Net: comparable or faster than BLAS L2 k-means for this data shape
  (N=tens of thousands, K=200-1000, D=200 bins)
- Parallelism: Rayon `par_iter_mut` on assignment/bound-update loop

## References

- Elkan (2003), "Using the Triangle Inequality to Accelerate k-Means"
- Ganzfried & Sandholm (AAAI 2014), "Potential-Aware Imperfect-Recall Abstraction with EMD"
- Brown & Sandholm (Science 2019), Pluribus supplementary: 200 buckets/street, EMD clustering
- robopoker `crates/clustering/src/elkan.rs` — source implementation
- robopoker `crates/clustering/src/bounds.rs` — Bounds struct
