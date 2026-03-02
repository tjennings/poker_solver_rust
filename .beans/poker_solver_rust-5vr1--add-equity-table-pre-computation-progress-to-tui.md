---
# poker_solver_rust-5vr1
title: Add equity table pre-computation progress to TUI
status: in-progress
type: task
created_at: 2026-03-02T03:42:58Z
updated_at: 2026-03-02T03:42:58Z
---

## Problem
When running with 2+ SPRs, equity tables are pre-computed in a batch before solving starts. The TUI shows 'Idle' during this phase with no progress indication.

## Solution
Add progress reporting for the equity table pre-computation block in main.rs so the TUI shows a progress bar (e.g. 'Computing equity tables: 423/1755').

## Tasks
- [ ] Add equity table progress phase to TuiMetrics
- [ ] Update TUI rendering to show equity table progress
- [ ] Wire atomic counter into the pre-computation par_iter in main.rs
- [ ] Test with 2+ SPRs config
