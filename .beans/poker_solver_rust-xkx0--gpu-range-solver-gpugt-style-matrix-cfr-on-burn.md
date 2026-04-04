---
# poker_solver_rust-xkx0
title: 'GPU Range Solver: GPUGT-style matrix CFR on burn'
status: completed
type: feature
priority: high
created_at: 2026-04-03T05:15:18Z
updated_at: 2026-04-03T07:55:42Z
---

Implement gpu-range-solve CLI command using level-synchronous DCFR on GPU via burn.

## Implementation Plan
See `docs/plans/2026-04-02-gpu-range-solver-impl.md` for full plan (12 tasks).

## Task Batches
- [x] Batch A (Tasks 1-2): Scaffold — public accessors on range-solver + new crate
- [x] Batch B (Tasks 3-4): Extraction — tree topology + terminal data
- [x] Batch C (Tasks 5-7): GPU core — tensor layout + regret matching + terminal eval
- [x] Batch D (Tasks 8-9): Solver core — forward/backward + full solve loop
- [x] Batch E (Tasks 10-12): Extension — cross-street + CLI + integration tests


## Summary of Changes

New `gpu-range-solver` crate (3,398 lines) implementing GPUGT-style level-synchronous DCFR on GPU via burn:

- **extract.rs** (713 lines): BFS tree topology extraction + terminal data (fold payoffs, showdown outcome matrices)
- **tensors.rs** (327 lines): `StreetSolver<B>` GPU tensor layout with edge-based storage
- **solver.rs** (1,726 lines): Forward/backward pass, regret matching, DCFR discount, full solve loop, exploitability computation, cross-street chance node handling
- **terminal.rs** (426 lines): Fold evaluation (card-blocking via gather/scatter) + showdown evaluation (outcome matrix matmul)
- **lib.rs** (206 lines): Public API (`gpu_solve_game`) + integration tests

Also: 14 lines of public accessor additions to `range-solver`, 269 lines of CLI command in `trainer`.

**79 tests passing.** GPU solver converges to match CPU solver within 0.01 tolerance on river games, and produces reasonable convergence on turn games.
