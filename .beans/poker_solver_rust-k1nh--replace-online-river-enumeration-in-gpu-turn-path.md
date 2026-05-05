---
# poker_solver_rust-k1nh
title: Replace online river enumeration in GPU turn path
status: todo
type: task
priority: high
created_at: 2026-05-05T02:57:52Z
updated_at: 2026-05-05T02:57:52Z
parent: poker_solver_rust-n5l6
---

Use the turn-boundary evaluator for boundary leaves in the GPU turn solver so boundary evaluation scales with batch * boundaries * players rather than rivers.
