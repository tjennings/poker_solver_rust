---
# poker_solver_rust-mblq
title: Add Criterion benchmark for exhaustive postflop solver
status: completed
type: feature
priority: normal
created_at: 2026-03-04T02:10:41Z
updated_at: 2026-03-04T02:59:17Z
---

Implement the postflop solve benchmark plan from docs/plans/2026-03-03-postflop-solve-bench.md

## Tasks
- [x] Task 1: Promote internal visibility for benchmarking (FlopBuffers, PostflopLayout, annotate_streets, exhaustive_solve_one_flop)
- [x] Task 2: Create benchmark file and Cargo.toml entry
- [x] Task 3: Verify profiling workflow with samply

## Summary of Changes
- Promoted FlopBuffers, PostflopLayout, annotate_streets, exhaustive_solve_one_flop to pub
- Created crates/core/benches/postflop_solve_bench.rs with Criterion iter_batched benchmark
- Cache-first equity loading: equity_tables.bin > rank_arrays.bin > compute from scratch
- Uses canonical_flops()[0] for guaranteed cache key match
- Baseline: ~58.6s per iteration (10 samples)
- samply profiling verified working
