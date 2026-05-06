---
# poker_solver_rust-he94
title: Design bucket model evaluation heuristics
status: in-progress
type: task
priority: high
created_at: 2026-05-06T19:07:21Z
updated_at: 2026-05-06T19:07:21Z
parent: poker_solver_rust-hfnv
---

Define diagnostics and scoring heuristics for postflop bucket models so we can tune bucket count, potential-awareness, and nut-distance weighting against strategy quality and resource cost. Acceptance:\n- Define offline abstraction-quality metrics that do not require full retraining\n- Define strategy-quality gates using exact subgame/oracle comparisons on validation spots\n- Define potential-awareness and nut-distance-specific diagnostics\n- Propose a Pareto scoring method for bucket count versus quality/cost\n- Identify an initial CLI/report implementation path
