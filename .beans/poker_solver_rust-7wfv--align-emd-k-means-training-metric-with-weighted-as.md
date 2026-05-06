---
# poker_solver_rust-7wfv
title: Align EMD k-means training metric with weighted assignment
status: todo
type: bug
priority: normal
created_at: 2026-05-06T16:39:11Z
updated_at: 2026-05-06T16:39:11Z
parent: poker_solver_rust-hfnv
---

cluster_histogram_exhaustive trains centroids with unweighted EMD, then exhaustive assignment can use weighted EMD gaps derived from child bucket EVs. The sampled centroid objective and final assignment metric should match.

Acceptance:
- Decide whether weighted child EV gaps should apply during both k-means and assignment or neither
- Implement weighted EMD centroid assignment/update path if needed
- Add tests showing training labels and exhaustive assignment use the same distance semantics
- Re-run/record cluster diagnostics comparing before/after separation and intra-bucket EMD
