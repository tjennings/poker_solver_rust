---
# poker_solver_rust-pzuf
title: Repair Tauri game session borrow-checker regressions
status: in-progress
type: bug
created_at: 2026-07-28T02:19:56Z
updated_at: 2026-07-28T02:19:56Z
---

Repair only the E0502 borrow-checker regressions introduced by commit 8d944b74 in crates/tauri-app/src/game_session.rs. Preserve lazy MP bucket behavior and cache/path semantics.

- [ ] Inspect the affected methods and baseline compiler errors
- [ ] Apply a conservative borrow-structure repair
- [ ] Run rustfmt and focused verification
- [ ] Commit the repair and report files/tests
