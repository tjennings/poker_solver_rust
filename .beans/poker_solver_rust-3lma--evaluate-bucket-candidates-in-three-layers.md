---
# poker_solver_rust-3lma
title: Evaluate bucket candidates in three layers
status: in-progress
type: task
priority: normal
created_at: 2026-05-14T01:14:02Z
updated_at: 2026-05-14T04:27:11Z
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
