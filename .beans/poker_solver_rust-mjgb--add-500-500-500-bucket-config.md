---
# poker_solver_rust-mjgb
title: Add 500-500-500 bucket config
status: completed
type: task
priority: normal
created_at: 2026-05-07T02:16:57Z
updated_at: 2026-05-07T02:22:49Z
---

Create a Blueprint V2-shaped sample clustering config for shared 500 flop / 500 turn / 500 river buckets and provide the command to run it.

## Summary of Changes

Added sample_configurations/blueprint_v2_500f_500t_500r.yaml as a Blueprint V2-shaped shared clustering config for 169 preflop / 500 flop / 500 turn / 500 river buckets. The config writes to ./local_data/buckets/500f_500t_500r_v1 and includes exhaustive turn sample_boards=16432 plus river sample_boards=500000, matching the existing 500-bucket style.

Verification:
- Pre-change full cargo test passed, but runtime was 1:06.52; rerun with cargo test -q passed in 1:01.03, still just over the one-minute repo budget.
- CLI parse sanity check passed and reported preflop=169, flop=500, turn=500, river=500.
- Post-change cargo test -q passed, but runtime was 1:11.01 under current machine load.

Note: The required full-suite runtime gate is currently not met on this machine despite passing tests; no Rust code was changed for this config-only task.
