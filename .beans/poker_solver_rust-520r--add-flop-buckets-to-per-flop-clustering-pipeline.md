---
# poker_solver_rust-520r
title: Add flop buckets to per-flop clustering pipeline
status: todo
type: task
created_at: 2026-03-18T06:23:12Z
updated_at: 2026-03-18T06:23:12Z
---

The per-flop bucket files (PerFlopBucketFile) store turn and river buckets but not flop-street buckets. Currently flop bucketing falls back to combo-index (ci % k) which is random and produces terrible strategy. The clustering pipeline (cluster_single_flop) should compute flop buckets from turn-bucket histograms and store them in the per-flop file format. This is the same histogram-based approach used for turn bucketing (histogram over river buckets) — just one level up.
