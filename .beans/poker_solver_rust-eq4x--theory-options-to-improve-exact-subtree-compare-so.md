---
# poker_solver_rust-eq4x
title: Theory options to improve exact_subtree compare-solve
status: completed
type: task
priority: normal
created_at: 2026-05-04T00:04:54Z
updated_at: 2026-05-04T00:05:49Z
---

Develop theory-based options for making compare-solve --river-boundary exact_subtree recover the full-depth Exact solution more closely on the canonical JhTh9h7d spot.\n\n- [x] Identify likely algorithmic failure modes from the observed divergence\n- [x] Suggest ranked improvement options\n- [x] Recommend next validation experiments

## Summary of Changes\n\nReviewed compare-solve and exact_subtree evaluator flow. Identified theory-based improvement options around boundary CFV scaling/reach conditioning, subgame action-tree equivalence, safe resolving gadget usage, convergence diagnostics, and boundary cache/precompute stability.
