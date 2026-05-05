---
# poker_solver_rust-7ajt
title: Investigate subgame nut-fold all-in spot
status: in-progress
type: bug
priority: critical
created_at: 2026-05-05T18:23:04Z
updated_at: 2026-05-05T18:23:04Z
---

At spot sb:2bb,bb:10bb,sb:22bb,bb:call|Ks8d3c|bb:check,sb:15bb,bb:call|Js|bb:check,sb:24bb,bb:all-in, the subgame appears to make BB jam 100% because SB folds every hand, including KK.\n\n## TODOs\n\n- [ ] Reproduce the exact spot in a backend test or diagnostic.\n- [ ] Inspect the range-solver root configuration, player/seat mapping, and terminal payoff orientation at the BB all-in node.\n- [ ] Determine whether the bad strategy comes from the solver, boundary/payoff setup, or matrix extraction.\n- [ ] Fix the underlying bug or document the narrowed root cause if it is outside this patch.\n- [ ] Add regression coverage for SB not folding nut hands to a jam when call is clearly profitable.\n- [ ] Run targeted and full verification.\n- [ ] Merge to local main.
