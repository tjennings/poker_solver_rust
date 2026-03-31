---
# poker_solver_rust-b5q3
title: Implement per-flop blueprint training pipeline
status: todo
type: feature
priority: high
created_at: 2026-03-31T19:13:46Z
updated_at: 2026-03-31T19:13:46Z
---

Implementation plan at docs/plans/2026-03-31-per-flop-pipeline.md, design at docs/plans/2026-03-31-per-flop-pipeline-design.md.

7 tasks:
1. Add mode/flops/output_dir config fields to ClusteringConfig
2. Add lock_preflop to skip regret updates at preflop nodes
3. Add fixed_flop for constrained deal sampling
4. Load preflop strategy from global blueprint on startup
5. Per-combo flop bucketing in fixed_flop mode (no flop abstraction)
6. Tauri integration — load per-flop blueprint and buckets for subgame solving
7. End-to-end integration test

NOTE: Per-flop clustering pipeline already exists (cluster_single_flop, run_per_flop_pipeline). This work builds on that foundation — adding the training and integration pieces.

Depends on: a trained global blueprint (for the locked preflop strategy).
