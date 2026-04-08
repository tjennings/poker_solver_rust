---
# poker_solver_rust-zry2
title: Lazy storage allocation for blueprint_mp
status: todo
type: task
priority: high
created_at: 2026-04-08T18:27:52Z
updated_at: 2026-04-08T18:27:52Z
---

Pre-allocated storage doesn't scale beyond ~18M nodes at 200 buckets (40GB). Pluribus used lazy allocation (only 62% of 664M sequences consumed memory). Implement a concurrent hash map approach for MpStorage where regret/strategy vectors are allocated on first visit during MCCFR traversal. This unlocks richer action abstractions (2+ lead sizes postflop) and higher bucket counts (500/street) for 6-player training.

Key changes:
- Replace flat Vec<AtomicI32>/Vec<AtomicI64> with DashMap<u32, NodeStorage> or similar
- Allocate NodeStorage (regrets + strategy_sums for bucket_count * num_actions) on first access
- Thread-safe: multiple MCCFR threads may hit the same node simultaneously
- Fallback: pre-allocate for small trees (2-3 player), lazy for large (4+ player)

Target: 350M node tree at 200 buckets fitting in ~300GB (vs current 781GB pre-allocated)
