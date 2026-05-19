---
# poker_solver_rust-w9cv
title: Set up 6max 250-100-20 bucket config
status: completed
type: task
priority: high
created_at: 2026-05-19T14:10:08Z
updated_at: 2026-05-19T14:18:30Z
---

Update the active 6-max Blueprint MP training config to use 250 flop / 100 turn / 20 river buckets, generate the matching bucket files, and document the training command before returning to pruning work.

## Summary of Changes

Added a 250/100/20 bucket-generation config and a matching 6-max lazy-sparse training config that preserves the current no-limp, DCFR schedule, and purge-disabled settings. Generated bucket files into `local_data/buckets/250f_100t_20r_nut_high_cap_0p5_v1` and updated training docs with the new run command.

## Validation

- `cargo run -p poker-solver-trainer --release -- inspect-mp-config -c sample_configurations/blueprint_mp_6max_250f_100t_20r.yaml`
- `cargo run -p poker-solver-trainer --release -- cluster -c sample_configurations/blueprint_v2_250f_100t_20r_nut_high_cap_0p5.yaml -o local_data/buckets/250f_100t_20r_nut_high_cap_0p5_v1`
- `cargo run -p poker-solver-trainer --release -- diag-clusters -d local_data/buckets/250f_100t_20r_nut_high_cap_0p5_v1`
