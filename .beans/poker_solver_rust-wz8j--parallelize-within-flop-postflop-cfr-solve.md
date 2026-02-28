---
# poker_solver_rust-wz8j
title: Parallelize within-flop postflop CFR solve
status: completed
type: feature
priority: high
created_at: 2026-02-28T21:18:02Z
updated_at: 2026-02-28T22:04:37Z
---

The exhaustive postflop solver is single-threaded within each flop solve. For high SPR values, this creates a major bottleneck: 169×169×2 = 57K traversals per iteration × 500 iterations, all sequential. The preflop solver already has a snapshot+delta-merge pattern that should be ported.

## Summary of Changes

Extracted shared ParallelCfr trait into cfr::parallel module. Both preflop and postflop solvers now use parallel_traverse() for snapshot + delta-merge parallelism over hand pairs. The postflop solver inner CFR loop went from 57K sequential traversals per iteration to fully parallelized across cores. compute_exploitability also parallelized.

Follow-up: poker_solver_rust-ccan (buffer allocation optimizations)
