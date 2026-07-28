---
# poker_solver_rust-nk3b
title: Invalidate solves before failed spot replay
status: in-progress
type: bug
priority: high
created_at: 2026-07-28T21:09:53Z
updated_at: 2026-07-28T21:09:53Z
parent: poker_solver_rust-ja8p
---

Close the failed saved-spot load invalidation hole from final review.

- [ ] Invalidate both solve generations before replay begins, so malformed/partial spot loads cannot leave old worker results valid.
- [ ] Add a regression for failed or partial saved-spot replay.
- [ ] Run focused Tauri tests and diff checks.
