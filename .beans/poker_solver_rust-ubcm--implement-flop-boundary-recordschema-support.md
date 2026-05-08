---
# poker_solver_rust-ubcm
title: Implement flop-boundary record/schema support
status: completed
type: task
priority: high
created_at: 2026-05-08T19:23:49Z
updated_at: 2026-05-08T20:03:35Z
parent: poker_solver_rust-8rxb
---

Extend cfvnet storage/manifest/config paths as needed so direct boundary records with 3-card flop boards round-trip cleanly without breaking river/turn datasets.

## Summary of Changes

- Added first-class flop-boundary manifest/schema support with board_size=3, turn_net/exact_turn target sources, and flop shard helpers.
- Confirmed TrainingRecord binary storage round-trips and counts 3-card board records.
- Allowed cfvnet game config/street helpers to represent flop_boundary datasets while preserving existing turn/river behavior.
- Kept turn-boundary datagen from accidentally accepting flop-boundary target sources.

## Verification

- cargo test -p cfvnet
- cargo test
