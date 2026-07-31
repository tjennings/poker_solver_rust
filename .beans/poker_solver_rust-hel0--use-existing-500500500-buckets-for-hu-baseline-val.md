---
# poker_solver_rust-hel0
title: Use existing 500/500/500 buckets for HU baseline validation sample
status: completed
type: task
priority: high
created_at: 2026-06-04T07:41:02Z
updated_at: 2026-06-04T07:41:59Z
---

Fix the Phase 2 HU 20bb baseline validation sample after runtime reported missing 200-bucket flop files. Use the existing 500f_500t_500r_v2 bucket set and update docs/run guidance.

## Summary of Changes

- Updated `sample_configurations/blueprint_v2_hu_20bb_baseline_validation.yaml` to use the existing `./local_data/buckets/500f_500t_500r_v2` bucket set.
- Changed postflop bucket counts from 200/200/200 to 500/500/500 and added `training.cluster_path`.
- Updated `docs/training.md` to document that the reproducible Phase 2 sample uses the existing 500/500/500 postflop bucket files while baseline validation remains preflop-only.

Verification:
- Parsed the YAML and confirmed `cluster_path`, 169 preflop buckets, and 500/500/500 postflop counts.
- Confirmed target bucket files exist in `local_data/buckets/500f_500t_500r_v2`.
- `cargo test -p poker-solver-core blueprint_v2::config::tests::test_baseline_config --quiet` passed.
