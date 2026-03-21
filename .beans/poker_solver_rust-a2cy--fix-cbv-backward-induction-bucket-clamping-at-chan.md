---
# poker_solver_rust-a2cy
title: Fix CBV backward induction bucket clamping at Chance nodes
status: todo
type: bug
priority: normal
created_at: 2026-03-21T13:15:56Z
updated_at: 2026-03-21T18:41:15Z
---

Phase 1 of the two-phase CBV backward induction STILL clamps at nested Chance nodes (turn->river). The transition matrix fix only applies in phase 2 to the outermost Chance node. Deeper Chance nodes use the old buggy clamping. Needs full recursive transition support at every Chance level, or a multi-pass approach that processes each street boundary independently.
