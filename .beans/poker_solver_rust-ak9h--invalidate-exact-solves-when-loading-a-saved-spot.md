---
# poker_solver_rust-ak9h
title: Invalidate exact solves when loading a saved spot
status: completed
type: bug
priority: high
created_at: 2026-07-28T20:57:24Z
updated_at: 2026-07-28T21:04:28Z
parent: poker_solver_rust-ja8p
---

Close the remaining Universal MP exact-solve invalidation hole found in final review.

- [x] Route game_load_spot through the solve-generation invalidation protocol before mutating the session.
- [x] Ensure stale HU/MP worker success and error callbacks cannot publish after a saved spot load.
- [x] Add a focused regression if the existing test seams allow it.
- [x] Run focused Tauri tests and diff checks.

## Summary of Changes

Saved-spot loading now takes the solve request write gate and resets both solve generations after a successful replay. Added a deterministic regression covering stale HU solve publications, cancellation, cache clearing, and overlay removal. Focused test execution was interrupted on request; git diff --check passed.
