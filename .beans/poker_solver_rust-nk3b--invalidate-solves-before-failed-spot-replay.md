---
# poker_solver_rust-nk3b
title: Invalidate solves before failed spot replay
status: completed
type: bug
priority: high
created_at: 2026-07-28T21:09:53Z
updated_at: 2026-07-28T21:25:38Z
parent: poker_solver_rust-ja8p
---

Close the failed saved-spot load invalidation hole from final review.

- [x] Invalidate both solve generations before replay begins, so malformed/partial spot loads cannot leave old worker results valid.
- [x] Add a regression for failed or partial saved-spot replay.
- [x] Run focused Tauri tests and diff checks.

## Summary of Changes

Invalidated both exact-solve generations before saved-spot replay so malformed or partial loads cannot leave stale worker overlays authoritative. Added a regression for failed replay, and the final explorer target passed 27 tests; rustfmt and git diff --check also passed.
