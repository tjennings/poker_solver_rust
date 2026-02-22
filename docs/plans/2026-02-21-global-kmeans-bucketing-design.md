# Global K-Means Bucketing

**Date:** 2026-02-21
**Status:** Approved

## Problem

`cluster_per_texture` runs independent k-means per flop texture. Bucket IDs are texture-local: bucket 15 on texture A groups completely different EHS hands than bucket 15 on texture B. The solver treats bucket IDs as globally meaningful info sets, causing hands with EHS=1.0 to land in weak-centroid buckets and get trash-hand EVs.

Evidence from trace output: 401 cases of hands with EHS >= 0.8 in buckets >= 15 (weak). Within-bucket EHS ranges span 0.0–1.0 for most buckets. EVs are identical before/after the relabel fix because relabeling doesn't change which hands cluster together.

## Design

Replace per-texture k-means with a single global k-means over all pooled `(hand, texture)` 3D feature vectors.

### Approach: Pooled global k-means

1. Collect all non-NaN `(hand, texture)` feature vectors into one flat list (~8,450 points for 169 hands × 50 textures)
2. Run k-means once on the pooled points (same `kmeans()` function, just more data)
3. Map assignments back to `[hand][texture]` shape
4. Assign blocked/NaN hands to nearest global centroid via cross-texture average
5. Apply `relabel_by_centroid_ehs` for nice ordering (bucket 0 = strongest)

### Changes

**`crates/core/src/preflop/hand_buckets.rs`:**
- Add `cluster_global()` function
- `build_flop_buckets`, `build_turn_buckets`, `build_river_buckets` call `cluster_global` instead of `cluster_per_texture`
- Keep `cluster_per_texture` available but no longer default
- Keep `relabel_by_centroid_ehs` for bucket ordering

**No changes to:**
- `HandBucketMapping` / `BucketEquity` structs
- `postflop_abstraction.rs` (calls through `build_flop_buckets`)
- `hand_trace.rs` (reads `abstraction.buckets.flop_buckets`)
- Solver / info set / postflop tree code

### Expected outcomes
- Within-bucket EHS spread drops from ~0.98 to ~0.1–0.2
- Hands with high EHS always cluster with other high-EHS points regardless of texture
- Postflop EVs should become sensible (AA always positive EV)

### Validation
- Run `trace-hand` after clearing cache, verify within-bucket EHS spread
- Existing unit tests should pass (cluster shape unchanged)
- Add test: global clustering assigns high-EHS points to strong buckets across textures
