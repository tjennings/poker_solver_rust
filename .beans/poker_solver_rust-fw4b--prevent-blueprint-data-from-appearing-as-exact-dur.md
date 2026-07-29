---
# poker_solver_rust-fw4b
title: Prevent Blueprint data from appearing as Exact during solve startup
status: in-progress
type: bug
priority: high
created_at: 2026-07-29T14:20:48Z
updated_at: 2026-07-29T14:20:48Z
parent: poker_solver_rust-g7yj
---

When an Exact solve is starting and no exact snapshot exists yet, UniversalMpLazy currently leaves the Blueprint matrix in place while the UI switches to the Exact tab. Clear or explicitly mark the matrix until the exact snapshot is published, and add a regression for the startup state.
