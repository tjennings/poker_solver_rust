---
# poker_solver_rust-i61b
title: Prevent exact action matching panic on empty match sets
status: completed
type: bug
priority: high
created_at: 2026-07-29T19:42:49Z
updated_at: 2026-07-29T19:59:03Z
---

The UniversalMpLazy exact post-solve action matcher panics at crates/tauri-app/src/game_session.rs:2918 when no cached action matches the scaled descriptor. The expression `(matches.len() == 1).then_some(matches[0])` indexes matches eagerly even when it is empty.

- [x] Replace eager indexing with guarded extraction
- [x] Add regression coverage for zero-match and ambiguous-match cases
- [x] Run focused Tauri exact explorer tests
- [x] Document verification and close the bean

## Notes\n\nThe implementation worker skipped builds; focused helper and exact explorer verification was run after integration.

## Summary of Changes

Replaced eager `then_some(matches[0])` indexing with guarded slice matching. Added helper-level regression tests for zero matches and ambiguous scaled descriptors. Verified with `cargo test -p poker-solver-tauri --lib unique_cached_action_index -- --nocapture` (2 passed) and `cargo test -p poker-solver-tauri --test universal_explorer_integration two_player_lazy_exact -- --test-threads=1` (6 passed).
