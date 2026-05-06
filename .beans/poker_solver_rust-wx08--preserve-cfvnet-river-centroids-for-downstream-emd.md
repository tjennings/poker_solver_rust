---
# poker_solver_rust-wx08
title: Preserve CFVNet river centroids for downstream EMD
status: todo
type: bug
priority: high
created_at: 2026-05-06T16:39:06Z
updated_at: 2026-05-06T16:39:06Z
parent: poker_solver_rust-hfnv
---

When cfvnet_river_data is used, cluster_river_from_cfvnet computes scalar river centroids but run_clustering_pipeline replaces them with an empty CentroidFile. That leaves turn clustering without river EV ordering or child-bucket gaps.

Acceptance:
- Return/persist a real river CentroidFile from CFVNet river clustering
- Thread river EVs and gaps into turn clustering exactly as with exhaustive river clustering
- Add regression coverage for cfvnet_river_data producing non-empty river.centroids
- Update diagnostics/docs if CFVNet river bucket semantics differ from showdown-equity river buckets
