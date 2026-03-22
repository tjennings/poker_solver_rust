---
# poker_solver_rust-9zzq
title: Add diff-clusters CLI command
status: completed
type: feature
priority: normal
created_at: 2026-03-22T15:12:39Z
updated_at: 2026-03-22T15:32:05Z
---

Compare two bucket sets: quality metrics (intra-bucket equity std, size distribution) + Adjusted Rand Index similarity. See docs/plans/2026-03-22-diff-clusters-design.md

## Summary of Changes

- Added sampling-based adjusted_rand_index() function
- Added ClusterDiffReport struct and diff_bucket_files() function
- Added diff-clusters CLI command to poker-solver-trainer
- Updated docs/training.md with usage
