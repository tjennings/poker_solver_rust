---
# poker_solver_rust-1w5r
title: Street-specific nut-distance guardrails
status: completed
type: task
priority: normal
created_at: 2026-05-14T01:14:02Z
updated_at: 2026-05-14T07:04:55Z
parent: poker_solver_rust-03j0
---

Implement and validate street-specific guardrails: modest nut awareness on flop, stronger draw/nut separation on turn, strong nut hierarchy preservation on river.

Acceptance: generated candidates reduce nut-distance span without destroying potential-consistency metrics.

## Metric-shape implementation plan

Build shape controls for the nut-distance channel so it acts as a guardrail rather than a second potential metric.

Acceptance checklist:
- [x] Add config-compatible nut-distance cap and nonlinear transform controls with defaults preserving existing behavior.
- [x] Apply cap/transform only to the normalized nut-distance contribution in weighted child-bucket EMD gaps.
- [x] Add focused config and metric-combination tests.
- [x] Update training/architecture docs for the new controls.
- [x] Add stage-three candidate configs and runner for capped/nonlinear evaluation.
- [x] Run focused tests and script/config smoke checks.

## Summary of Changes

Implemented shaped nut-distance metric controls and completed the stage-three capped/nonlinear sweep on 2026-05-14.

Code changes:
- Added `nut_distance_transform` (`linear`, `sqrt`, `log1p`) and optional `nut_distance_cap` to per-street bucket metric config.
- Applied cap/transform only after nut-distance channel normalization, preserving existing behavior for old configs.
- Wrote focused config and metric-combination tests.
- Updated training/architecture docs and sweep reporting.
- Added four stage-three candidate configs plus `scripts/run_bucket_metric_sweep_stage3.sh`.

Sweep artifacts:
- Report: `local_data/bucket_sweeps/500f_100t_100r_nut_stage3_v1/analysis.md`
- Logs: `local_data/bucket_sweeps/500f_100t_100r_nut_stage3_v1/logs`

Result:
- Best candidate by current audit signal: `high_cap_0p5`.
- `high_cap_0p5` improved sampled flop max intra-bucket std by 25.0% vs baseline and 31.9% vs `first_high`, while retaining a 10.5% turn max-std improvement vs baseline.
- `high_cap_1p0` preserved turn improvement but worsened flop tail; `sqrt` and `log1p` also worsened flop tail.

Interpretation: bounded linear nut-distance is the first promising complementary shape. Next promotion step is short-run strategy sanity against baseline/first_high before replacing training buckets.
