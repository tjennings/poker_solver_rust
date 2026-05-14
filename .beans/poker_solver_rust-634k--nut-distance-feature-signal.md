---
# poker_solver_rust-634k
title: Nut-distance feature signal
status: in-progress
type: task
priority: high
created_at: 2026-05-14T01:14:02Z
updated_at: 2026-05-14T01:56:49Z
parent: poker_solver_rust-03j0
---

Define and expose nut-distance features for each hand/board assignment.\n\nScope:\n- made hand rank\n- current nut gap\n- draw nut gap\n- blocker rank\n- reverse implied risk\n- board-lock/chop likelihood\n\nAcceptance: diagnostics can report nut-distance spans and class/nut separation independent of clustering changes.

## Implementation Notes

Exposed the first nut-distance signal in diagnostics via a sampled river bucket audit. The scorecard now records per-bucket spans for same-family class gap, dominance margin, global rank percentile, blocker-to-nuts share, and top made classes. This covers current river nut-distance observability; flop/turn draw nut gaps, reverse implied risk, and board-lock/chop features remain future work.
