---
# poker_solver_rust-ak9h
title: Invalidate exact solves when loading a saved spot
status: in-progress
type: bug
priority: high
created_at: 2026-07-28T20:57:24Z
updated_at: 2026-07-28T20:57:24Z
parent: poker_solver_rust-ja8p
---

Close the remaining Universal MP exact-solve invalidation hole found in final review.

- [ ] Route game_load_spot through the solve-generation invalidation protocol before mutating the session.
- [ ] Ensure stale HU/MP worker success and error callbacks cannot publish after a saved spot load.
- [ ] Add a focused regression if the existing test seams allow it.
- [ ] Run focused Tauri tests and diff checks.
