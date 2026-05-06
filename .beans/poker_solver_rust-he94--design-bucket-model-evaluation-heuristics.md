---
# poker_solver_rust-he94
title: Design bucket model evaluation heuristics
status: in-progress
type: task
priority: high
created_at: 2026-05-06T19:07:21Z
updated_at: 2026-05-06T19:07:59Z
parent: poker_solver_rust-hfnv
---

Define diagnostics and scoring heuristics for postflop bucket models so we can tune bucket count, potential-awareness, and nut-distance weighting against strategy quality and resource cost. Acceptance:\n- Define offline abstraction-quality metrics that do not require full retraining\n- Define strategy-quality gates using exact subgame/oracle comparisons on validation spots\n- Define potential-awareness and nut-distance-specific diagnostics\n- Propose a Pareto scoring method for bucket count versus quality/cost\n- Identify an initial CLI/report implementation path

## Initial Heuristic Framework

Evaluation should be two-stage:

1. Cheap abstraction diagnostics for every bucket candidate: occupancy entropy, empty/dead buckets, within-bucket child-histogram EMD, held-out assignment distortion, potential transition preservation, nut-distance collision rate, and canonical lookup consistency.
2. Expensive strategy diagnostics for finalists: solve a fixed validation spot suite with exact/no-abstraction or stronger reference subgames and compare action distributions, CFVs, exploitability proxy, and high-value hand class decisions.

Bucket-count selection should use a Pareto frontier over quality versus cost, not a fixed bucket target. Candidate score: quality_gain / log(memory_or_train_cost), with a default knee rule that rejects a larger model when it improves strategy EV by less than roughly 1 mbb/100 or reduces action/CFV error by less than 5% relative while increasing cost by 50%+.

Potential-awareness diagnostics should measure whether same-bucket hands preserve future child-bucket distributions: normalized EMD variance inside buckets, nearest-centroid held-out EMD, and correlation between bucket distance and exact future EV distance.

Nut-distance diagnostics should measure whether dominated made hands collide with true nut holdings: same-family nut collision rate, within-bucket class_gap variance, dominance_margin variance, blocker-to-nuts mixing, and top-set/nut-flush/nut-straight separation on targeted board textures.

Tuning should sweep bucket counts and metric weights, including nut weight values 0.0, 0.1, 0.25, 0.5, and 1.0. Each metric component must be normalized before combination so potential EMD stays primary and nut distance acts as a board-aware regularizer.
