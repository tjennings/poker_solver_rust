---
# poker_solver_rust-075w
title: Add 500-100-100 bucket config
status: completed
type: task
priority: normal
created_at: 2026-05-07T02:32:05Z
updated_at: 2026-05-07T02:34:05Z
---

Create a Blueprint V2-shaped sample clustering config for shared 500 flop / 100 turn / 100 river buckets and provide the command to run it.



## Summary of Changes

Added sample_configurations/blueprint_v2_500f_100t_100r.yaml as a Blueprint V2-shaped clustering config for shared 500 flop / 100 turn / 100 river buckets.

Verification:
- CLI parse sanity check passed and reported preflop=169, flop=500, turn=100, river=100.
- cargo test -q passed.

Note: The full-suite runtime was roughly 62 seconds on this machine, so the existing sub-minute workflow budget remains slightly unmet even though no Rust code changed for this config-only task.
