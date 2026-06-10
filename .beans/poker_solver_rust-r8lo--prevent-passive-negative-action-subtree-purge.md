---
# poker_solver_rust-r8lo
title: Prevent passive negative-action subtree purge
status: completed
type: bug
priority: critical
created_at: 2026-05-14T17:51:17Z
updated_at: 2026-05-14T17:54:23Z
---

After first negative-action purge, open strategies for positions after UTG appear wiped. Root cause is likely purging/blocking passive MP action edges such as UTG fold, whose child history contains later players' unopened spots. Restrict persistent negative-action subtree purge to aggressive actions and update regression tests/docs.

## Summary of Changes

- Restricted lazy MP negative-action persistent blocking/purge to aggressive actions only.
- Passive edges (`Fold`, `Check`, `Call`, and all-in calls that do not increase the current max bet) no longer enter the blocked-edge set or get skipped by negative-action masking.
- Added a focused regression test proving passive edges are ignored by the negative-action gate, while existing aggressive-edge tests still pass.
- Updated training and architecture docs to describe aggressive-only purge semantics.

## Validation

- cargo fmt --check
- cargo test -p poker-solver-core negative_action -- --nocapture
- git diff --check
