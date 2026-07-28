---
# poker_solver_rust-6fui
title: Implement root-reach-weighted postflop lazy MP display
status: completed
type: bug
priority: normal
created_at: 2026-07-28T01:26:08Z
updated_at: 2026-07-28T01:34:45Z
---

Extend lazy MP explorer display from preflop marginal reach weighting into flop matrices. Preserve per-seat semantics, add regression coverage and explorer documentation. Todo: [x] research and architecture design; [x] implement root-reach-weighted flop display; [x] add integration regression; [x] update docs; [x] run formatting, diff check, and focused test.

## Summary of Changes

Added concrete 1326-combo flop reach replay with per-seat marginal weighting, blocker handling, regression coverage for the SB raise / BB reraise / SB call path, and updated Explorer boundary documentation.
