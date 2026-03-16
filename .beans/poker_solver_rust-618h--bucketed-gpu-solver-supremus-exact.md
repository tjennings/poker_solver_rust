---
# poker_solver_rust-618h
title: Bucketed GPU Solver (Supremus-exact)
status: in-progress
type: epic
priority: critical
tags:
    - gpu
    - bucket
    - supremus
created_at: 2026-03-16T17:47:25Z
updated_at: 2026-03-16T17:47:25Z
---

Supremus-exact bucketed GPU solver with configurable buckets (500/1000).
Builds on the concrete GPU solver epic (poker_solver_rust-twez) but redesigns
the core solver to operate in bucket space, matching the published architecture.

Key changes from concrete solver:
- Bucket-vs-bucket equity matrix instead of hand strength comparison
- No card blocking (baked into equity tables)
- Supremus DCFR+ (additive linear strategy weighting, delay d=100)
- CFV net: input 2×num_buckets+1, output 2×num_buckets (both players)
- Zero-sum enforcement in network architecture
- Preflop uses 169 hand classes (not buckets)

See docs/plans/2026-03-16-bucketed-gpu-solver-design.md
and docs/plans/2026-03-16-bucketed-gpu-solver-impl.md
