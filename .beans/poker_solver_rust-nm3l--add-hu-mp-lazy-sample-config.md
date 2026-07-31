---
# poker_solver_rust-nm3l
title: Add HU MP lazy sample config
status: completed
type: task
priority: normal
created_at: 2026-06-24T13:34:03Z
updated_at: 2026-06-24T13:43:24Z
---

Create a heads-up Blueprint MP lazy-sparse YAML config derived from sample_configurations/blueprint_mp_6max_500f_100t_100r.yaml so the MP backend can be run in a 2-player setup. Keep the 500/100/100 bucket base from the source config, change player/blind/scenario/output identity for HU, validate parsing/inspection, and commit the config plus tracker.

## Summary of Changes

- Added `sample_configurations/blueprint_mp_hu_500f_100t_100r.yaml`, derived from `sample_configurations/blueprint_mp_6max_500f_100t_100r.yaml`.
- Converted the MP lazy-sparse base config to a 2-player heads-up game with seat 0 small blind and seat 1 big blind.
- Preserved the source config's 500/100/100 bucket family, lazy sparse backend, training cadence, DCFR settings, pruning settings, and bucket path.
- Set a distinct output directory under `local_data/blueprints/mp_hu_500f_100t_100r_nut_high_cap_0p5_v1` and replaced 6-max TUI scenarios with HU-oriented spots.

Verification:

- `cargo run -p poker-solver-trainer --release -- inspect-mp-config --config sample_configurations/blueprint_mp_hu_500f_100t_100r.yaml` passed and reported `Players: 2`, `Stack: 200.0 chips (100.0bb, BB=2.0 chips)`, buckets `169/500/100/100`, and backend `lazy_sparse`.
- `git diff --check` passed.
- Full redirected workspace suite passed once with `real 416.37`, then immediate hot retry passed under the gate with `real 44.70`, `user 101.82`, `sys 15.46`.
