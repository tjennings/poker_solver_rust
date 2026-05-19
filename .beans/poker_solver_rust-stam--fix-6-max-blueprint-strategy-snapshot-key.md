---
# poker_solver_rust-stam
title: Fix 6-max blueprint strategy snapshot key
status: in-progress
type: bug
priority: high
created_at: 2026-05-19T18:05:35Z
updated_at: 2026-05-19T18:18:04Z
---

Pressing [s] during train-blueprint-mp appears not to write usable strategy snapshots for 6-max games, even though heads-up snapshots have worked previously.\n\n- [x] Reproduce or isolate the snapshot key path for train-blueprint-mp\n- [x] Identify why 6-max snapshot generation is failing\n- [x] Implement the fix without disturbing unrelated config changes\n- [x] Add focused regression coverage\n- [x] Run the relevant trainer test suite under the project time budget
- [ ] Fix review finding: publish queued snapshot status before trigger

## Research Notes\n\n`train-blueprint-mp` with the 6-max config uses the `lazy_sparse` backend. The `[s]` key sets `BlueprintTuiMetrics::snapshot_trigger`; the lazy MP bridge consumes it and writes `snapshot_NNNN/sparse_entries.bin` plus `metadata.json`, not a heads-up-style `strategy.bin`. The likely user-visible failure is lack of TUI feedback while a large sparse snapshot scans/writes, plus format mismatch with previous HU snapshots.

## Summary of Changes

Added MP TUI snapshot status for queued, writing, saved, and failed manual snapshot requests. Eager MP and lazy sparse MP bridges now update the shared status around snapshot writes, while lazy sparse checkpoint format remains sparse_entries.bin plus metadata.json. Updated training docs and verified the full poker-solver-trainer test suite.
