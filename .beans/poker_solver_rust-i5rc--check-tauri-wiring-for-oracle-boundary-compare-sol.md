---
# poker_solver_rust-i5rc
title: Check Tauri wiring for oracle boundary compare-solve changes
status: in-progress
type: task
priority: normal
created_at: 2026-05-04T04:50:36Z
updated_at: 2026-05-04T04:50:36Z
parent: poker_solver_rust-e90m
---

Verify whether the recent compare-solve oracle-boundary diagnostic and finalization changes need any Tauri/devserver/frontend wiring, and record any missing hooks.

Checklist:

[ ] Inspect Tauri command surface and devserver API for compare-solve or related solver diagnostics.
[ ] Inspect frontend invoke/API usage for compare-solve, oracle-boundary, exact_subtree, or root trace controls.
[ ] Determine whether the recent changes are CLI-only/internal or require Explorer UI updates.
[ ] Run the narrowest relevant build/type checks if frontend or Tauri wiring is touched.
