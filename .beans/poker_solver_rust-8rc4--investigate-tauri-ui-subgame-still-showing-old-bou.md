---
# poker_solver_rust-8rc4
title: Investigate Tauri UI subgame still showing old boundary behavior
status: in-progress
type: bug
priority: critical
created_at: 2026-05-05T20:01:00Z
updated_at: 2026-05-05T20:01:00Z
---

User still sees original Tauri UI behavior after main merge: solving subgame at turn root, BB checks, SB shows default/incorrect subgame matrix and odd all-in/fold response persists. Verify whether Tauri frontend/backend uses fixed solve root and boundary stack/exact-subtree path, and repair if needed.
