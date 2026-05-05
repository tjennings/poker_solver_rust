---
# poker_solver_rust-i2nb
title: Add oversampling policy for deep raise and high SPR states
status: completed
type: task
priority: high
created_at: 2026-05-05T02:57:06Z
updated_at: 2026-05-05T03:57:13Z
parent: poker_solver_rust-q93y
---

Deliberately oversample scarce but important 3-bet, 4-bet, 5-bet-plus, tiny-pot, and high-SPR boundary states so the turn-boundary net avoids the current river-net coverage hole.



Completed a weighted turn-boundary sampling policy. datagen.turn_boundary_sampling.strata can now oversample named regions by overriding pot_intervals and spr_intervals per sample, while carrying raise_depth and boundary_ordinal labels into manifest coverage. The sample config now includes normal, tiny-pot/high-SPR, and low-SPR/all-in-pressure strata.
