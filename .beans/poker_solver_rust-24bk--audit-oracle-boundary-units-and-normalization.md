---
# poker_solver_rust-24bk
title: Audit oracle-boundary units and normalization
status: todo
type: task
priority: high
created_at: 2026-05-04T01:09:27Z
updated_at: 2026-05-04T01:09:27Z
parent: poker_solver_rust-e90m
---

Confirm whether each boundary evaluator and depth-boundary injection path expects raw chip CFVs, pot-normalized values, or half-pot-normalized BCFVs. Compare raw exact continuation values to values consumed by regret updates.
