---
# poker_solver_rust-nv3f
title: Boundary CFV magnitude calibration — rollout values too large
status: in-progress
type: bug
priority: normal
created_at: 2026-03-26T13:35:42Z
updated_at: 2026-03-26T14:23:53Z
---

The range-solver subgame solve converges to 0.0 exploitability on river (no boundaries). On flop with depth_limit=0, exploitability plateaus at ~12 due to boundary CFV approximation from rollout. The solver itself is correct — the floor is from rollout inaccuracy.

## Validated
- [x] River solve: expl=0.0000 (no boundaries, solver works perfectly)
- [x] Flop solve: expl~12 (boundary CFVs are approximate, not exact)
- [x] Boundary values in correct range (±1-2 pot fractions after stack cap fix)

## Remaining
- [ ] Improve boundary CFV accuracy (more rollout samples, better SPR mapping, or equity-based fallback)
- [ ] The abstract tree walk to find next-street node follows first child only — may not find the right chance node in all tree shapes
- [ ] Consider caching boundary evaluations across solve iterations
