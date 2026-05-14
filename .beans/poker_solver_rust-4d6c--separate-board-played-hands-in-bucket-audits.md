---
# poker_solver_rust-4d6c
title: Separate board-played hands in bucket audits
status: completed
type: task
priority: high
created_at: 2026-05-14T00:11:37Z
updated_at: 2026-05-14T00:16:10Z
---

Update hand-class bucket diagnostics to distinguish board-only, one-hole, and two-hole made-hand contribution before interpreting class mixing and strength inversions; rerun the 500f_50t_50r_v1 audit.

## Summary of Changes\n\n- Added hole-contribution labels to hand-class bucket diagnostics: board, 1h, and 2h.\n- Split class/strength spread grouping and bucket class-mix labels by contribution so board-played made hands no longer masquerade as private-card-made hands.\n- Added tests for board-only, one-hole, and two-hole contribution classification.\n- Reran the 500f_50t_50r_v1 hand-class audit with 200 boards.
