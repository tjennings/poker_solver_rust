---
# poker_solver_rust-2jc1
title: Fix turn-root subgame check default matrix
status: in-progress
type: bug
priority: high
created_at: 2026-05-05T15:18:40Z
updated_at: 2026-05-05T15:18:40Z
---

After solving a subgame at the turn root, clicking BB Check advances to SB but the Subgame tab shows a default/blueprint-looking matrix instead of the solved child matrix.

## Report

User still reproduces on local main after edb8c66c: solve at turn root, BB check, SB shows default matrix.

## TODOs

- [ ] Reproduce why the live BB Check path misses solved child matrix overlay.
- [ ] Fix source-specific solved matrix selection for turn-root action navigation.
- [ ] Add regression coverage for the exact turn root BB Check -> SB child case.
- [ ] Run targeted and full verification.
- [ ] Merge to local main for testing.
