# Improved Transition Coherence Diagnostics

**Date:** 2026-03-20
**Status:** Approved

## Motivation

Current transition consistency diagnostics report raw EMD values that don't normalize across bucket counts and don't indicate whether clustering is capturing real structure. We need: (1) normalized scores, (2) a separation ratio that answers "how good is this clustering?", and (3) use of persisted centroids.

## Design

### 1. Normalize EMD to [0, 1]

Divide raw EMD by `(K - 1)` where K = number of child-street buckets (maximum possible 1-D EMD between point masses at opposite ends). All reported EMD values use this normalized scale.

### 2. Use persisted centroids

Add `centroid_file: Option<&CentroidFile>` parameter to `audit_transition_consistency`. When provided, use persisted centroid for each bucket instead of reconstructing from sampled histograms. Falls back to reconstructed centroids when not provided. CLI loads `{street}.centroids` if the file exists.

### 3. Centroid separation ratio

When centroids are available, compute mean pairwise EMD between all K centroids (K²/2 pairs, normalized). Report:
- `mean_between_emd`: average centroid-to-centroid EMD (normalized)
- `separation_ratio`: `mean_between_emd / mean_within_emd`

Higher separation ratio = better clustering.

## Output Format

```
Flop → Turn: 494 buckets, 30 sample boards
  Within-bucket EMD: mean=0.048, max=0.220 (normalized)
  Between-centroid EMD: mean=0.185
  Separation ratio: 3.85 (higher = better)
```

## Files Modified

- `crates/core/src/blueprint_v2/cluster_diagnostics.rs` — normalize EMD, accept optional centroids, compute separation ratio
- `crates/trainer/src/main.rs` — load centroid files and pass to audit function, update output formatting
