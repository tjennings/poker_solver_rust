---
# poker_solver_rust-4ig1
title: Document turn-boundary CFVNet input/output contract
status: completed
type: task
priority: high
created_at: 2026-05-05T02:54:42Z
updated_at: 2026-05-05T03:03:13Z
parent: poker_solver_rust-ewjj
---

Write the model contract for the turn-boundary CFVNet: 2720-float input with 4-card board, two 1326 reach vectors, pot/stack normalized features, player bit, and 1326 normalized bcfv output. Include target semantics and unit conversion to raw GPU leaf CFVs.



Completed with docs/plans/2026-05-05-turn-boundary-cfvnet-contract.md. The contract defines the direct 4-card turn-boundary input/output shape, range semantics, target oracle, dataset schema, manifest metadata, and validation gates.
