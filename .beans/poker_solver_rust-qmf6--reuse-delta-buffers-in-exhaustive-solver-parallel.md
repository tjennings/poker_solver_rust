---
# poker_solver_rust-qmf6
title: Reuse delta buffers in exhaustive solver parallel_traverse
status: completed
type: task
priority: normal
created_at: 2026-02-28T23:39:06Z
updated_at: 2026-02-28T23:46:16Z
---

Add parallel_traverse_into to cfr/parallel.rs that accepts pre-allocated output buffers. Update exhaustive_solve_one_flop to pre-allocate dr/ds before the iteration loop. Also fix minor string allocations in progress callback. See docs/plans/2026-02-28-exhaustive-solver-perf.md

## Summary of Changes

- Added parallel_traverse_into to crates/core/src/cfr/parallel.rs — reuses caller-provided buffers
- Updated exhaustive_solve_one_flop to pre-allocate dr/ds before iteration loop
- Pre-computed flop_name string, used .into() for static metric label
- Review: .fill(0.0) instead of iter_mut, debug_assert_eq for buffer size validation
