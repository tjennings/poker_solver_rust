# CFVnet-Based River Clustering

**Date:** 2026-03-14
**Status:** Approved

## Problem

River clustering currently computes showdown equity against a random hand for each (board, combo) pair. By the river, no opponent holds a random hand — they hold a strategically filtered range. This makes random-hand equity a poor proxy for strategic similarity.

## Solution

Use pre-solved cfvnet river training data (~3.1M records) as the equity source. Each record contains a board, realistic opponent ranges (generated via R(S,p)), and pot-relative CFVs for all 1326 combos from both OOP and IP perspectives. Both perspectives are used — IP vs OOP is a meaningful strategic distinction.

## Data Source

`local_data/cfvnet/river/v1/river_*.bin` — binary files containing `TrainingRecord` structs:
- board (5 cards), pot, stack, player (0=OOP, 1=IP)
- oop_range, ip_range (1326 x f32 each)
- cfvs (1326 x f32, pot-relative)
- valid_mask (1326 x u8)

CFV → equity conversion: `equity = clamp((cfv + 1.0) / 2.0, 0.0, 1.0)`

## Approach: Streaming Histogram K-means (Approach 3)

With ~3B data points, holding all in memory (~24GB as f64) is impractical. Instead use a 3-pass streaming approach:

**Pass 1 — Build equity histogram:**
- Stream all `river_*.bin` files
- Convert each valid (board, combo) CFV to equity in [0, 1]
- Accumulate into a 10,000-bin histogram
- Collect unique canonical boards (for BucketFile output)

**Pass 2 — K-means on histogram:**
- Run `kmeans_1d_weighted` on 10K bin midpoints weighted by counts
- Produces 500 centroid positions

**Pass 3 — Assign bucket labels:**
- Stream files again
- For each valid (board, combo): equity → nearest centroid → bucket ID
- Build BucketFile with canonical boards and 1326 combos per board

## Config Integration

Add optional `cfvnet_data_dir: Option<PathBuf>` to `ClusteringConfig`. When present, `run_clustering_pipeline` calls `cluster_river_from_cfvnet()` instead of `cluster_river()`.

## Bug Fix: kmeans_1d_weighted

Fix two bugs in the 1-D k-means (needed regardless of data source):
1. Add empty-cluster re-seeding (mirror `kmeans_emd_weighted` behavior)
2. Use weighted percentile initialization instead of unweighted index-based

## Pipeline Impact

Only the river stage changes. Turn, flop, preflop clustering remain the same — they cluster by EMD over next-street bucket distributions and automatically benefit from better river buckets:

river (CFV equity) → turn (EMD over river) → flop (EMD over turn) → preflop (canonical 169)

## BucketFile Output

Same format as today: canonical boards, 1326 combos per board, u16 bucket ID per entry. Downstream consumers (MCCFR solver, diagnostics) require no changes.
