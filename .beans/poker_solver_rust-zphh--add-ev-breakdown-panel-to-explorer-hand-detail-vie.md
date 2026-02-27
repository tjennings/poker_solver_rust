---
# poker_solver_rust-zphh
title: Add EV breakdown panel to Explorer hand detail view
status: completed
type: feature
priority: normal
created_at: 2026-02-27T16:05:40Z
updated_at: 2026-02-27T17:26:44Z
---

When clicking a hand in the grid, show an EV breakdown below the raise breakdown: EV average vs all hands, and EV vs a specific hand (defaulting to AA).


## Summary of Changes

9 commits on `feat/ev-breakdown-panel`:

1. `feat: add optional ev_table.bin to PreflopBundle` — new `hand_avg_values` field + `with_ev_table()` constructor
2. `feat: save postflop EV table into preflop bundle during training` — clone hand_avg_values before attach_postflop consumes it
3. `feat: add PostflopBundle::load_hand_avg_values for co-located solve.bin fallback` — deserialize just hand_avg_values from standalone solve.bin
4. `refactor: read hand_avg_values from PreflopBundle, fallback to postflop/ dir` — 3-tier fallback chain in explorer loader
5. `feat: extend get_hand_equity with villain_hand matchup lookup` — MatchupEquity struct, updated API + devserver
6. `feat: add MatchupEquity type for villain-specific EV lookup` — frontend TypeScript types
7. `feat: add EV vs specific hand input to Explorer equity panel` — villainHand state, two-section UI
8. `feat: style villain hand input and EV subheaders` — CSS for input and subheaders
9. `fix: accept plain float raise sizes as pot fractions for backward compat` — serde Visitor for YAML numeric values

End-to-end verified with `full_postflop` bundle (has co-located solve.bin): AKs vs range = +1.18, AKs vs AA = -2.98 (pot fraction units).
