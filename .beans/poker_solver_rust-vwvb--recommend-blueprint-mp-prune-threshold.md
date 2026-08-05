---
# poker_solver_rust-vwvb
title: Recommend blueprint_mp prune threshold
status: completed
type: task
priority: high
created_at: 2026-08-05T18:50:41Z
updated_at: 2026-08-05T18:57:21Z
---

Determine an evidence-based prune_threshold recommendation for the active HU lazy_sparse blueprint_mp configuration.

- [x] Verify exact pruning eligibility, warmup, exploration, and regret units
- [x] Compare current -100 setting with literature and repository history
- [x] Assess risk of suppressing strategy evolution
- [x] Recommend a value and an empirical validation protocol

## Summary of Changes

Audited the active HU implementation and repository pruning history, reviewed primary Pluribus and regret-based-pruning evidence, and assessed DCFR interaction. `-100` can be crossed by one high-leverage sampled update and then limits recovery to 5% exploration batches. Recommend pruning disabled for the correctness reference and `prune_threshold: -20000` with `prune_explore_pct: 0.05` as the quality-first provisional production setting; test `-10000` to `-50000`, with `-6000` only as the most aggressive lower bound. Validate at equal work and equal wall time using exploitability/best-response EV, reach-weighted strategy divergence, prune-hit rate, throughput, and reactivation latency through post-discount-cap training.
