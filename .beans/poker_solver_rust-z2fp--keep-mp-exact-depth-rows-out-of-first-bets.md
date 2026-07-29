---
# poker_solver_rust-z2fp
title: Keep MP exact depth rows out of first bets
status: in-progress
type: bug
priority: high
created_at: 2026-07-29T14:48:50Z
updated_at: 2026-07-29T14:48:50Z
parent: poker_solver_rust-g7yj
---

The exact action parser currently flattens lead and raise-depth rows into the first-bet vector. Keep depth 0 in BetSizeOptions.bet and preserve depth-specific rows only in per_num_bets so Turn/River exact trees do not expose raise-only sizes as first bets.
