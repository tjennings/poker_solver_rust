---
# poker_solver_rust-nycm
title: Add capped exact MP tree-size estimator
status: todo
type: task
priority: normal
created_at: 2026-05-07T13:50:55Z
updated_at: 2026-05-07T13:50:55Z
parent: poker_solver_rust-5kvv
---

Extend inspect-mp-config with an exact tree-size counter that reuses MP action generation and state transitions but aborts at a configurable node limit without storing arena nodes. Report exact counts when under the cap and truncated lower bounds when over it.
