---
# poker_solver_rust-7d5j
title: Optimize compute_equity_table performance
status: in-progress
type: feature
priority: normal
created_at: 2026-03-01T03:48:19Z
updated_at: 2026-03-01T04:26:32Z
---

Flamegraph-confirmed: compute_equity_table is ~98% of postflop solve runtime. Implement bitmask optimization, card index pre-computation, SPR equity caching, and criterion benchmarks. Plan: docs/plans/2026-02-28-equity-table-perf.md

## Tasks

- [ ] Task 1: Add criterion benchmark for compute_equity_table
- [x] Task 2: Replace used.contains() with 64-bit bitmask
- [x] Task 3: Pre-compute card index array to eliminate From<u8> conversions
- [x] Task 4: Cache equity tables across SPR solves
- [x] Task 5: End-to-end timing validation
