---
# poker_solver_rust-tbc5
title: Validate bucket files against clustering config before reuse
status: todo
type: task
priority: normal
created_at: 2026-05-06T16:39:21Z
updated_at: 2026-05-06T16:39:21Z
parent: poker_solver_rust-hfnv
---

Cluster generation and training can reuse existing bucket files based mostly on file presence. Add stronger validation so stale or mismatched bucket directories do not silently feed the solver.

Acceptance:
- Validate street bucket counts, board counts, version, and expected per-flop/global mode before skipping cluster generation
- Validate training cluster_path against config.clustering before storage allocation/training starts
- Produce actionable errors for mismatched 200/1k bucket dirs, missing centroids when needed, or partial outputs
- Keep resume behavior for valid partial per-flop outputs
