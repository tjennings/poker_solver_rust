---
# poker_solver_rust-xkx0
title: 'GPU Range Solver: GPUGT-style matrix CFR on burn'
status: in-progress
type: feature
priority: high
created_at: 2026-04-03T05:15:18Z
updated_at: 2026-04-03T05:15:18Z
---

Implement gpu-range-solve CLI command using level-synchronous DCFR on GPU via burn.

## Implementation Plan
See `docs/plans/2026-04-02-gpu-range-solver-impl.md` for full plan (12 tasks).

## Task Batches
- [ ] Batch A (Tasks 1-2): Scaffold — public accessors on range-solver + new crate
- [ ] Batch B (Tasks 3-4): Extraction — tree topology + terminal data
- [ ] Batch C (Tasks 5-7): GPU core — tensor layout + regret matching + terminal eval
- [ ] Batch D (Tasks 8-9): Solver core — forward/backward + full solve loop
- [ ] Batch E (Tasks 10-12): Extension — cross-street + CLI + integration tests
