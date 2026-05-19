---
# poker_solver_rust-stam
title: Fix 6-max blueprint strategy snapshot key
status: in-progress
type: bug
priority: high
created_at: 2026-05-19T18:05:35Z
updated_at: 2026-05-19T18:05:35Z
---

Pressing [s] during train-blueprint-mp appears not to write usable strategy snapshots for 6-max games, even though heads-up snapshots have worked previously.\n\n- [ ] Reproduce or isolate the snapshot key path for train-blueprint-mp\n- [ ] Identify why 6-max snapshot generation is failing\n- [ ] Implement the fix without disturbing unrelated config changes\n- [ ] Add focused regression coverage\n- [ ] Run the relevant and full test suites under the project time budget
