---
# poker_solver_rust-l8yt
title: Audit non-traverser action sampling probabilities
status: completed
type: task
priority: normal
created_at: 2026-05-20T18:35:43Z
updated_at: 2026-05-20T18:38:10Z
---

Confirm that Blueprint MP MCCFR samples non-traverser actions according to the regret-matched strategy probabilities, not uniformly or with an indexing bias.



## Summary of Changes

Confirmed dense Blueprint MP non-traverser traversal obtains current regret-matched strategy, samples a single child by cumulative probability, and records the full current strategy into average-strategy sums. Confirmed lazy_sparse does the same after masking permanently blocked negative-action edges and renormalizing the eligible action mass. Added seeded sampling regression tests for dense and lazy paths to verify probability frequencies and ensure zero-probability actions are not selected. Focused storage/traversal tests pass.
