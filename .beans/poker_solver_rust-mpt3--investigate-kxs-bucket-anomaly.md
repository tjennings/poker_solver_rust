---
# poker_solver_rust-mpt3
title: Investigate Kxs bucket anomaly
status: completed
type: task
priority: high
created_at: 2026-05-14T00:21:03Z
updated_at: 2026-05-14T00:26:25Z
---

Trace bucket assignments and preflop abstraction for Kx suited hands versus Q4s-Q6s to determine whether observed training folds can plausibly come from the bucket model.

- Investigated Kxs/Qxs training anomaly. Preflop bucketing is lossless 169; K8s=18, Q6s=31, Q4s=33 in both 500/50/50 and checked config paths.
- Sampled generated postflop bucket files for 500f_50t_50r_v1 and 500f_100t_100r_v1. K8s has higher bucket profile than Q6s/Q5s/Q4s on flop/turn/river; no gross bucket inversion found.
- Found a separate display caveat: lazy MP TUI hand grid maps 13x13 hands to canonical 169 indices modulo current street bucket count, so postflop grids are not actual board-aware bucket lookups. Preflop grids are unaffected when preflop bucket_count=169.
- Current suspicion: if anomaly is at preflop root, it is strategy/convergence/action/pruning/display-state rather than bucket generation. Need snapshot path to inspect exact root strategy sums/regrets.
