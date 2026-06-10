---
# poker_solver_rust-f14l
title: Fix sparse resume TUI startup storage freshness
status: todo
type: task
priority: low
created_at: 2026-06-03T20:29:13Z
updated_at: 2026-06-03T20:29:13Z
parent: poker_solver_rust-kqpn
---

Follow-up found during Phase 1 sparse trainer integration review.

In sparse HU `blueprint_v2` mode after resume, TUI startup scenario/audit resolution still receives `&trainer.storage` directly. That storage may be the dense projection/stub until the first projected refresh callback, so the initial TUI display may be stale even though trainer traversal/export correctness is unaffected.

Scope:
- Audit `crates/trainer/src/main.rs` startup scenario/audit resolution around the blueprint_v2 trainer storage reference.
- Route startup display through the same active/projected storage path used after sparse refresh, or force an initial sparse-to-dense projection before TUI display.
- Add a focused regression test if practical.

Non-goal: do not change training semantics, sparse storage format, or Explorer/Tauri bundle formats.
