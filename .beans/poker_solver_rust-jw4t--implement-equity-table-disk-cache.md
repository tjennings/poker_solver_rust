---
# poker_solver_rust-jw4t
title: Implement equity table disk cache
status: completed
type: feature
priority: high
created_at: 2026-03-02T03:51:45Z
updated_at: 2026-03-02T04:04:23Z
---

Precompute all 1755 canonical flop equity tables to disk via `precompute-equity` CLI, then auto-load in solve-postflop. Design: docs/plans/2026-03-01-equity-table-cache-design.md, Plan: docs/plans/2026-03-01-equity-table-cache.md

## Tasks

- [x] Task 1: Create EquityTableCache module in core
- [x] Task 2: Add precompute-equity CLI subcommand
- [x] Task 3: Auto-detect equity cache in solve-postflop handler
- [x] Task 4: Verify single-SPR cache path works (covered by Task 3)
- [x] Task 5: Add cache/ to .gitignore (already present)
- [x] Task 6: Integration test — build/save/load/extract round-trip
- [x] Task 7: Update docs/training.md

## Summary of Changes

Implemented all 7 tasks in a single commit on feature/equity-table-cache branch. Core module with build/save/load/extract, CLI subcommand with progress bar, auto-detection in solve-postflop, and docs. Expensive tests marked #[ignore].
