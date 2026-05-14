---
# poker_solver_rust-6jse
title: Generate weighted bucket candidates
status: completed
type: task
priority: normal
created_at: 2026-05-14T01:14:02Z
updated_at: 2026-05-14T04:22:50Z
parent: poker_solver_rust-03j0
---

Generate competing 500/100/100 bucket candidates with low, medium, high, and river-heavy nut weights.\n\nAcceptance: candidate directories are reproducible and include config metadata for comparison.



Completed sweep setup and run on 2026-05-13/14.

Artifacts:
- Sweep runner: scripts/run_bucket_metric_sweep.sh
- Analyzer: scripts/analyze_bucket_sweep.py
- Candidate configs: low/med/high/river_heavy nut-distance variants for 500f/100t/100r
- Output: local_data/bucket_sweeps/500f_100t_100r_nut_v1/analysis.md

Result summary:
- low improved max equity span most in scorecard but worsened sampled flop max intra-bucket equity std.
- high produced the best inversion count and strongest turn max intra-bucket equity std improvement in diff-clusters.
- river_heavy regressed both sampled turn and flop equity spread proxies and should not be promoted.
