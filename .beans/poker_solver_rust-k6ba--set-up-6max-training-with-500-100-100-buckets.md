---
# poker_solver_rust-k6ba
title: Set up 6max training with 500-100-100 buckets
status: completed
type: task
priority: normal
created_at: 2026-05-07T02:51:39Z
updated_at: 2026-05-07T02:56:00Z
---

Create or update a 6max Blueprint MP training configuration that uses the generated shared 500 flop / 100 turn / 100 river bucket files, and provide the training command.



## Summary of Changes

Added sample_configurations/blueprint_mp_6max_500f_100t_100r.yaml as a 20bb 6-max Blueprint MP training config using the generated shared 500/100/100 bucket directory at ./local_data/buckets/500f_100t_100r_v1.

Updated docs/training.md to list the new sample config and include the training command.

Verification:
- Ruby YAML sanity check passed for num_players=6, bucket counts 169/500/100/100, and cluster_path ./local_data/buckets/500f_100t_100r_v1.
- Confirmed all four bucket files exist in local_data/buckets/500f_100t_100r_v1.
- cargo test -q passed after the config/doc change, in roughly 55 seconds wall-clock.

Note: The training run itself was not launched; the config keeps tui.enabled=true and a 7200 minute time limit.
