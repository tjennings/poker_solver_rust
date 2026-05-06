---
# poker_solver_rust-heuc
title: Design nut-distance-aware potential bucketing features
status: in-progress
type: feature
priority: high
created_at: 2026-05-06T16:39:30Z
updated_at: 2026-05-06T16:43:58Z
parent: poker_solver_rust-hfnv
---

Potential-aware bucket histograms currently capture future bucket distributions, but they may underrepresent dominance within made-hand families: nut flush vs K-high flush, nut straight vs lower straight, top set vs bottom set, etc. Design an extension that keeps potential-aware linkage while modeling nut distance / reverse-implied-odds risk.

Candidate directions:
- Add ordered side-channel features per situation: nut rank class, nut distance within class, blocker/nut-blocker flags, redraw class, and made-hand dominance margin
- Use a product or lexicographic metric: EMD over future buckets plus weighted L1/Hamming over nut-distance features
- Split child buckets by strength-family/nut-distance before EMD so future distributions carry dominance information
- Evaluate per-flop rather than global texture-agnostic features, since nut topology is board-dependent

Acceptance:
- Produce a design comparing feature-side-channel vs bucket-splitting vs heuristic_v3-style axes
- Define exact feature formulas for flop/turn/river with board-aware nut ordering
- Prototype diagnostics that show whether buckets separate nut flushes/straights/sets from dominated versions
- Recommend a default weighting and validation experiment before full retraining

## Build Start

Initial implementation slice:
- First produce a small design/spec for normalized potential EMD + nut-distance scaling.
- Then implement river-side nut dominance features and diagnostics before changing flop/turn clustering.
- Validate with weight sweeps: wn = 0.0, 0.1, 0.25, 0.5, 1.0.
- Keep potential EMD primary; nut distance starts as a board-aware regularizer.
