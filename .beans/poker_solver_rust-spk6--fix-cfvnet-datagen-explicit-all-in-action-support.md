---
# poker_solver_rust-spk6
title: Fix CFVNet datagen explicit all-in action support
status: in-progress
type: bug
priority: critical
created_at: 2026-05-04T15:16:21Z
updated_at: 2026-05-04T15:16:21Z
---

River CFVNet datagen currently parses "a" out of bet_sizes and disables all-in thresholds, so all-in is not guaranteed as an explicit action despite configs and Supremus/DeepStack abstractions including all-in. Fix the datagen action-tree construction, validate with tests/eval, update config/docs as needed, and prepare a new data generation command.
