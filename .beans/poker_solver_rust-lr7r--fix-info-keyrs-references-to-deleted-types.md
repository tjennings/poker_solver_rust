---
# poker_solver_rust-lr7r
title: Fix info_key.rs references to deleted types
status: completed
type: task
priority: normal
created_at: 2026-02-27T23:18:08Z
updated_at: 2026-02-27T23:29:25Z
---

Remove any imports from crate::game that reference HunlPostflop, PostflopState, or Game. PostflopConfig and AbstractionMode still exist via game/config.rs.
