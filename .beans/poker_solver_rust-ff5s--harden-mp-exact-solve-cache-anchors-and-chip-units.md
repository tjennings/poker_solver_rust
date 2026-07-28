---
# poker_solver_rust-ff5s
title: Harden MP exact solve cache anchors and chip units
status: completed
type: bug
priority: high
created_at: 2026-07-28T18:21:54Z
updated_at: 2026-07-28T18:32:35Z
parent: poker_solver_rust-mk2k
---

Review follow-up for UniversalMpLazy exact flop solve.

- Clear exact status/overlay before returning from stale anchor checks in MP state/action/back paths.
- Replace hardcoded 2-chip big blind conversions in MP exact cached action matching with configured chip units.
- Add regression coverage for stale solve navigation and nonstandard big-blind amounts.
- Preserve HU/eager/legacy exact behavior and the existing two-player flop capability boundary.


## Summary of Changes

- MP exact overlays now require a matching solve anchor before publishing exact status or cached matrices.
- MP exact matrices, cached actions, and semantic matching use the configured big-blind chip units.
- Added stale-navigation and nonstandard-big-blind regression coverage while retaining the asymmetric flop exact test.
