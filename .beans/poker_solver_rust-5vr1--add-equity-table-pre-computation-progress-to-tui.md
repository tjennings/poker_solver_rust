---
# poker_solver_rust-5vr1
title: Add equity table pre-computation progress to TUI
status: completed
type: task
priority: normal
created_at: 2026-03-02T03:42:58Z
updated_at: 2026-03-02T03:47:06Z
---

## Problem
When running with 2+ SPRs, equity tables are pre-computed in a batch before solving starts. The TUI shows 'Idle' during this phase with no progress indication.

## Solution
Add progress reporting for the equity table pre-computation block in main.rs so the TUI shows a progress bar (e.g. 'Computing equity tables: 423/1755').

## Tasks
- [x] Add equity table progress phase to TuiMetrics
- [x] Update TUI rendering to show equity table progress
- [x] Wire atomic counter into the pre-computation par_iter in main.rs
- [ ] Test with 2+ SPRs config (user to verify)

## Summary of Changes
- Added phase 4 (ComputingEquityTables) to TUI with progress bar showing done/total and elapsed time
- Added equity_tables_completed/total atomic counters to TuiMetrics
- Wired AtomicU32 counter into the par_iter pre-computation block in main.rs
