---
# poker_solver_rust-acpe
title: Define flop-boundary model/data contract
status: completed
type: task
priority: high
created_at: 2026-05-08T19:23:38Z
updated_at: 2026-05-08T19:43:48Z
parent: poker_solver_rust-2ek4
---

Specify the direct flop-boundary BoundaryNet contract: 3-card board input, range normalization/blocker rules, pot/stack/game_value normalization, output shape, and chip/unit semantics. Confirm whether existing BoundaryRecord can carry 3-card boards or needs schema evolution.

## Summary of Changes

Defined the flop-boundary data/model contract in docs/plans/2026-05-08-flop-boundary-cfvnet-contract.md. Decision: reuse the variable-board-size TrainingRecord binary layout with board_size=3; generalize manifests and validation to DatasetStreet::FlopBoundary; keep BoundaryNet input/output shape unchanged; direct inference must accept 3-card boards.
