---
# poker_solver_rust-ekir
title: Update 6max sample bucket counts
status: completed
type: task
priority: normal
created_at: 2026-05-06T20:54:41Z
updated_at: 2026-05-06T20:57:43Z
---

Set the 6max blueprint sample configuration to use 500 flop buckets, 50 turn buckets, and 50 river buckets, then provide the clustering command.

## Summary of Changes

Updated sample_configurations/blueprint_mp_6max_simplified_actions.yaml to use 169 preflop buckets, 500 flop buckets, 50 turn buckets, and 50 river buckets. Updated the matching training.cluster_path to ./local_data/buckets/500f_50t_50r_v1 so the clustering output path and training config stay aligned.

Verification:
- Pre-change time cargo test passed in 55.958s.
- Post-change time cargo test passed in 56.746s.
