# True Potential-Aware Clustering

**Date:** 2026-03-13
**Status:** Approved

## Problem

The clustering pipeline claims to implement Pluribus-style potential-aware abstraction, but the turn, flop, and preflop stages use **uniform equity binning** (`equity_to_bucket()`) instead of looking up actual bucket assignments from the previous street's clustering results.

In true potential-aware abstraction (Johanson et al., Brown & Sandholm 2019), the feature vector at street S is a **distribution over the bucket IDs at street S+1**. Our implementation builds distributions over uniform equity quantiles instead, which breaks the recursive linkage that makes potential-aware abstraction effective.

## Solution

Replace uniform equity binning with actual bucket ID lookups from the previous street's in-memory `BucketFile`. Run the full bottom-up pipeline in memory and write all 4 bucket files at the end.

## Pipeline Flow

```
1. cluster_river()     → river BucketFile (in memory)
2. cluster_turn()      → looks up river bucket IDs → turn BucketFile (in memory)
3. cluster_flop()      → looks up turn bucket IDs → flop BucketFile (in memory)
4. cluster_preflop()   → looks up flop bucket IDs → preflop BucketFile (in memory)
5. Write all 4 BucketFiles to disk
```

Currently stages 2-3 pass the previous `BucketFile` but only read `header.bucket_count`. Stage 4 ignores previous stages entirely (1-D equity k-means). After the fix, stages 2-4 all perform actual bucket lookups.

## New Histogram Builder

Replace `build_next_street_histogram_u8` with:

```rust
fn build_bucket_histogram_u8(
    combo: [Card; 2],
    board: &[Card],
    deck: &[Card],
    prev_bucket_file: &BucketFile,
    board_index_map: &HashMap<PackedBoard, u32>,
) -> Vec<u8>
```

For each possible next-street card:
1. Extend the board with that card
2. Pack as `PackedBoard`, look up `board_idx` via the hashmap
3. Call `prev_bucket_file.get_bucket(board_idx, combo_index(combo))` to get the bucket ID
4. Increment `histogram[bucket_id]`

## Board Index Map

Add a helper to `BucketFile`:

```rust
pub fn board_index_map(&self) -> HashMap<PackedBoard, u32>
```

Built once per stage from `self.boards`, O(num_boards) construction, O(1) lookups during histogram building.

## Preflop Restructure

Change from 1-D equity k-means to EMD k-means over flop bucket distributions:
- For each of 1326 combos, sample/enumerate flop boards
- For each (combo, flop): look up flop bucket ID from `flop_buckets` → build histogram over flop bucket IDs
- Cluster with `kmeans_emd_weighted_u8`

## Canonical vs Sampling Variants

Both the canonical (`cluster_*_canonical`) and sampling-based (`cluster_*`) variants receive the same fix. The canonical variants are currently only used in tests but should be consistent.

## What Gets Deleted

- `equity_to_bucket()` function
- `build_next_street_histogram()` (f64 uniform-bin version)
- `build_next_street_histogram_u8()` (u8 uniform-bin version)
- Old `cluster_preflop()` 1-D equity variant
- Intermediate `.save()` calls within `run_clustering_pipeline` (moved to end)

## What Stays the Same

- River clustering (equity-based, no previous street to reference)
- EMD distance function and all k-means algorithms
- `BucketFile` format and serialization
- Canonical board enumeration and isomorphism
- Weighted k-means with multiplicity weights
- u8 histogram memory optimization
- Output format: 4 `.buckets` files consumed by MCCFR trainer

## Testing

- Existing clustering tests continue to work (they test k-means mechanics)
- Add a test: cluster river → cluster turn with true lookups → verify hands with identical river bucket distributions get the same turn bucket
- `diag-clusters` cross-street transition matrix becomes the quality validation tool
