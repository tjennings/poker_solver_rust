---
# poker_solver_rust-r664
title: Build stratified validation split generation
status: completed
type: task
priority: normal
created_at: 2026-05-05T02:57:09Z
updated_at: 2026-05-05T04:19:03Z
parent: poker_solver_rust-q93y
---

Create frozen validation splits by action depth, SPR, pot, stack, and board texture so offline metrics expose weak strata instead of only aggregate error.

Completed frozen validation split generation. Turn-boundary datagen now emits validation_split.yaml with deterministic global record indices and per-stratum counts keyed by raise depth, boundary label, SPR, pot, stack, and board texture. Python training prefers the frozen split over random_split when present.
