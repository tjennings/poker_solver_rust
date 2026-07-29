---
# poker_solver_rust-i61b
title: Prevent exact action matching panic on empty match sets
status: in-progress
type: bug
priority: high
created_at: 2026-07-29T19:42:49Z
updated_at: 2026-07-29T19:51:24Z
---

The UniversalMpLazy exact post-solve action matcher panics at crates/tauri-app/src/game_session.rs:2918 when no cached action matches the scaled descriptor. The expression `(matches.len() == 1).then_some(matches[0])` indexes matches eagerly even when it is empty.

- [x] Replace eager indexing with guarded extraction
- [x] Add regression coverage for zero-match and ambiguous-match cases
- [ ] Run focused Tauri exact explorer tests
- [ ] Document verification and close the bean

## Notes\n\nImplemented the guarded slice-pattern match and focused helper-level regression tests. Per the implementation request, no tests or builds were run in this turn.
