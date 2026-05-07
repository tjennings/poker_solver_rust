---
# poker_solver_rust-0zvh
title: Canonicalize turn-boundary datagen ranges before writing records
status: in-progress
type: bug
priority: high
created_at: 2026-05-07T04:39:55Z
updated_at: 2026-05-07T04:39:55Z
---

Turn-boundary datagen currently stores raw blueprint reach weights in TrainingRecord inputs while the oracle solves with board-blocked ranges. Canonicalize stored ranges so model inputs are board-blocked, normalized, and aligned with the oracle inputs.\n\n- [ ] Confirm test baseline before code changes\n- [ ] Store canonical board-blocked normalized ranges in turn-boundary records\n- [ ] Recompute record game values against canonical ranges\n- [ ] Add/elevate datagen evaluator checks for range mass and blockers\n- [ ] Validate with focused tests/evaluator run
