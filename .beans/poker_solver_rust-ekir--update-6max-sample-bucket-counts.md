---
# poker_solver_rust-ekir
title: Update 6max sample bucket counts
status: completed
type: task
priority: normal
created_at: 2026-05-06T20:54:41Z
updated_at: 2026-05-06T21:02:28Z
---

Set the 6max blueprint sample configuration to use 500 flop buckets, 50 turn buckets, and 50 river buckets, then provide the clustering command.

## Summary of Changes

Updated sample_configurations/blueprint_mp_6max_simplified_actions.yaml to use 169 preflop buckets, 500 flop buckets, 50 turn buckets, and 50 river buckets. Updated the matching training.cluster_path to ./local_data/buckets/500f_50t_50r_v1 so the clustering output path and training config stay aligned.

Verification:
- Pre-change time cargo test passed in 55.958s.
- Post-change time cargo test passed in 56.746s.

## Follow-up Adjustment

Added sample_configurations/blueprint_v2_500f_50t_50r.yaml as a Blueprint V2-shaped companion clustering config because the cluster subcommand parses BlueprintV2Config, while the 6-max training config is Blueprint MP-shaped. Verified the cluster CLI parses the companion config and reports preflop=169, flop=500, turn=50, river=50 using a skip-path sanity check.

Additional verification:
- Post-companion time cargo test passed with compile work in 1:06.95.
- Warm-cache time cargo test passed in 58.095s.
