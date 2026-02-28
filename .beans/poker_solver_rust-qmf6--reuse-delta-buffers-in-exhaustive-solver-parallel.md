---
# poker_solver_rust-qmf6
title: Reuse delta buffers in exhaustive solver parallel_traverse
status: in-progress
type: task
created_at: 2026-02-28T23:39:06Z
updated_at: 2026-02-28T23:39:06Z
---

Add parallel_traverse_into to cfr/parallel.rs that accepts pre-allocated output buffers. Update exhaustive_solve_one_flop to pre-allocate dr/ds before the iteration loop. Also fix minor string allocations in progress callback. See docs/plans/2026-02-28-exhaustive-solver-perf.md
