---
# poker_solver_rust-0zvh
title: Canonicalize turn-boundary datagen ranges before writing records
status: completed
type: bug
priority: high
created_at: 2026-05-07T04:39:55Z
updated_at: 2026-05-07T04:46:46Z
---

Turn-boundary datagen currently stores raw blueprint reach weights in TrainingRecord inputs while the oracle solves with board-blocked ranges. Canonicalize stored ranges so model inputs are board-blocked, normalized, and aligned with the oracle inputs.\n\n- [x] Confirm test baseline before code changes\n- [x] Store canonical board-blocked normalized ranges in turn-boundary records\n- [x] Recompute record game values against canonical ranges\n- [x] Add/elevate datagen evaluator checks for range mass and blockers\n- [x] Validate with focused tests/evaluator run

## Summary of Changes\n\nCanonicalized turn-boundary range inputs before record writing, recomputed game values from the canonical ranges, and added datagen-eval range contract checks for finite non-negative weights, board blockers, and unit total mass. Verified with cfvnet tests plus clean and intentionally bad evaluator runs.
