---
# poker_solver_rust-bw9a
title: Trace suspect Blueprint MP infoset regret updates
status: in-progress
type: bug
priority: high
created_at: 2026-05-20T13:27:50Z
updated_at: 2026-08-05T18:05:05Z
parent: poker_solver_rust-kiqt
---

Instrument or otherwise inspect one suspicious preflop infoset to determine whether action values and regret deltas favor implausible folds.

## Subtasks

- [ ] Choose one primary suspect infoset, e.g. BTN unopened A5s or UTG AJs
- [ ] Run a tiny deterministic training or unit-level traversal with fixed seed/deal stream
- [ ] Log action values, node value, regret deltas, current strategy, average strategy, and terminal payoffs
- [ ] Verify fold EV, open EV, call EV, raise EV signs and magnitudes
- [ ] If fold is incorrectly preferred, audit payoff/blind accounting and apply_action sequence
- [ ] If open is preferred but average stays folded, audit strategy-sum accumulation and discounting

## 2026-08-05 Regret-extrema diagnostic

Investigate observed HU blueprint_mp telemetry at roughly 400 seconds: max positive regret 14,106 versus max negative regret -1,364,769. Determine whether asymmetry is mathematically expected, a telemetry-population/scaling artifact, or evidence of faulty regret updates/discounting.

## Diagnostic conclusion

No regret-update bug was found. The approximately 96.75x magnitude skew is mathematically plausible because only strategy-weighted instantaneous regret deltas sum to zero; cumulative extrema need not mirror. The TUI values are correctly scaled raw i32 chip regrets, but they are extrema from a bounded sample of at most 8,192 sparse rows and likely originate from different infosets. At 400 seconds the first post-warmup wall-clock discount had not occurred. Deferred follow-ups: relabel TUI extrema as sampled and correct stale x1000 MP regret documentation; provenance-based exact snapshot diagnostics would provide stronger validation.
