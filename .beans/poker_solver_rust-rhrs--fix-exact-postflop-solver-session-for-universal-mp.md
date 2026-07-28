---
# poker_solver_rust-rhrs
title: Fix exact postflop solver session for Universal MP lazy
status: in-progress
type: bug
priority: high
created_at: 2026-07-28T17:48:03Z
updated_at: 2026-07-28T17:48:03Z
parent: poker_solver_rust-mk2k
---

The Explorer exact postflop solver fails on a loaded universal_mp_lazy game with `No game session active`. Trace the Exact solve command from the frontend through Tauri/core and make it work for the active Universal MP lazy two-player postflop session without regressing HU/eager/legacy exact solves. Add focused regression coverage and update explorer docs if behavior or capability messaging changes. Loading performance is explicitly out of scope.
