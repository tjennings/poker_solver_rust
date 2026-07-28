---
# poker_solver_rust-pzuf
title: Repair Tauri game session borrow-checker regressions
status: completed
type: bug
priority: normal
created_at: 2026-07-28T02:19:56Z
updated_at: 2026-07-28T02:22:19Z
---

Repair only the E0502 borrow-checker regressions introduced by commit 8d944b74 in crates/tauri-app/src/game_session.rs. Preserve lazy MP bucket behavior and cache/path semantics.

- [x] Inspect the affected methods and baseline compiler errors
- [x] Apply a conservative borrow-structure repair
- [x] Run rustfmt and focused verification
- [x] Commit the repair and report files/tests

## Summary of Changes

Fixed the three E0502 borrow-checker regressions in LazyMpSession by cloning board/history inputs before mutable state_at calls. Preserved lazy MP bucket loading and existing cache/path behavior. Focused tauri lib tests passed: 376 passed, 6 ignored, 0 failed.
