---
# poker_solver_rust-5uqj
title: Clustering quality improvements from robopoker analysis
status: completed
type: feature
created_at: 2026-03-20T15:09:06Z
updated_at: 2026-03-20T15:09:06Z
---

Implemented 4 clustering quality improvements borrowed from robopoker:

1. **CentroidFile** — new binary format (CEN1) for persisting k-means centroids alongside BucketFiles
2. **Sorted bucket IDs** — centroids sorted by expected equity so bucket 0 = weakest, bucket K-1 = strongest  
3. **EMD for exhaustive assignment** — replaced L2 distance with proper Earth Mover's Distance in the assignment phase
4. **Weighted EMD** — EMD now weighted by actual equity gaps between adjacent centroids

Branch: feat/clustering-quality-improvements (4 commits)
Scope: Global pipeline only (per-flop untouched)
