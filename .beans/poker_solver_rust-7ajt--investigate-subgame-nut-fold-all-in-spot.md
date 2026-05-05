---
# poker_solver_rust-7ajt
title: Investigate subgame nut-fold all-in spot
status: in-progress
type: bug
priority: critical
created_at: 2026-05-05T18:23:04Z
updated_at: 2026-05-05T18:23:04Z
---

At spot sb:2bb,bb:10bb,sb:22bb,bb:call|Ks8d3c|bb:check,sb:15bb,bb:call|Js|bb:check,sb:24bb,bb:all-in, the subgame appears to make BB jam 100% because SB folds every hand, including KK.

## TODOs

- [x] Reproduce the exact spot in a backend test or diagnostic.
- [x] Inspect the range-solver root configuration, player/seat mapping, and terminal payoff orientation at the BB all-in node.
- [x] Determine whether the bad strategy comes from the solver, boundary/payoff setup, or matrix extraction.
- [ ] Fix the underlying bug or document the narrowed root cause if it is outside this patch.
- [ ] Add regression coverage for SB not folding nut hands to a jam when call is clearly profitable.
- [ ] Run targeted and full verification.
- [ ] Merge to local main.

## Findings

- The comparison tool was rebuilding a fresh street root instead of the navigated solve root, which made prior diagnostics misleading.
- With root-aware comparison, the all-exact post-jam node has strong SB hands calling 100%, including AA/KK/JJ.
- The pathological overjam reproduces one decision earlier with `--river-boundary exact_subtree`: exact calls AA/KK/JJ, while the depth-limited exact-subtree solve jams them.
- `--river-boundary exact_oracle` remained close to exact in the same spot, so the next target is the exact-subtree boundary CFV contract/scale.
