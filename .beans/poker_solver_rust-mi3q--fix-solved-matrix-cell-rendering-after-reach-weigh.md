---
# poker_solver_rust-mi3q
title: Fix solved matrix cell rendering after reach weighting
status: in-progress
type: bug
priority: high
created_at: 2026-05-05T16:01:19Z
updated_at: 2026-05-05T16:01:19Z
---

After the solved-node matrix fix, matrix cells render as large solid green blocks/columns. The backend now sends normalized reach weights through the existing cell weight field, which the frontend uses for visual reach/availability masking.

## Report

User screenshot 2026-05-05 11:00 shows Subgame solved turn matrix with broken cell rendering after commit ea090047.

## TODOs

- [ ] Separate action aggregation reach weighting from the UI display weight field.
- [ ] Preserve solved child matrices while restoring sane matrix cell rendering.
- [ ] Add regression coverage for solved matrix display weight scale.
- [ ] Run targeted and full verification.
- [ ] Merge to local main.
