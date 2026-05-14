---
# poker_solver_rust-3lma
title: Evaluate bucket candidates in three layers
status: in-progress
type: task
priority: normal
created_at: 2026-05-14T01:14:02Z
updated_at: 2026-05-14T05:29:17Z
parent: poker_solver_rust-03j0
---

Evaluate bucket candidates through cluster diagnostics, short-run strategy sanity, and sampled quality proxies.\n\nAcceptance: comparison report ranks candidates against baseline and highlights regressions.



## Comprehensive nut-distance sweep

2026-05-14: Started a second-stage comprehensive evaluation after the first low/med/high/river-heavy sweep. This pass will focus on asymmetric flop/turn weights to test whether flop needs much lower nut-distance pressure while turn keeps the hierarchy improvement seen in the high-weight candidate.

Planned candidates:
- flop_tiny_turn_high: flop 0.025, turn 0.50
- flop_low_turn_high: flop 0.05, turn 0.50
- flop_low_turn_med: flop 0.05, turn 0.25
- flop_none_turn_high: flop 0.00, turn 0.50
- flop_tiny_turn_med: flop 0.025, turn 0.25
- flop_low_turn_075: flop 0.05, turn 0.75

Outputs will be compared against baseline 500f_100t_100r_v1 and the first-pass nut_high candidate.



## Stage-2 sweep results

Completed the asymmetric nut-distance sweep on 2026-05-14.

Artifacts:
- Runner: scripts/run_bucket_metric_sweep_stage2.sh
- Report: local_data/bucket_sweeps/500f_100t_100r_nut_stage2_v1/analysis.md
- Logs: local_data/bucket_sweeps/500f_100t_100r_nut_stage2_v1/logs

Result:
- first_high remains the best current candidate by combined audit signal.
- turn nut_distance_weight=0.50 is the only tested turn setting that preserved the strong sampled turn max-spread improvement.
- turn nut_distance_weight=0.25 and 0.75 both regressed turn max spread versus baseline.
- reducing flop nut_distance_weight to 0.00, 0.025, or 0.05 did not repair the flop tail and generally worsened sampled flop max intra-bucket spread versus first_high.

Interpretation: the next useful step is not another scalar-weight sweep. The algorithm needs a shape change: capped, gated, or nonlinear nut-distance contribution, most likely applied selectively to strategic made-hand/nut-contested regions.
