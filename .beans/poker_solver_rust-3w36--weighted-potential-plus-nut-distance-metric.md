---
# poker_solver_rust-3w36
title: Weighted potential plus nut distance metric
status: completed
type: task
priority: high
created_at: 2026-05-14T01:14:02Z
updated_at: 2026-05-14T02:06:19Z
parent: poker_solver_rust-03j0
---

Add an experimental clustering distance metric combining potential-aware EMD, nut-distance vector distance, and equity distance.\n\nAcceptance: config can select weighted metric per street without changing default behavior.

## Implementation Notes\n\nAdded opt-in per-street cluster metric weights under <street>.metric. Defaults keep the existing potential-aware EMD behavior unchanged. When enabled for turn/flop, clustering blends uniform potential movement, child centroid equity gaps, and sampled river nut-distance gaps into the child-bucket ground distance used by both sampled k-means and exhaustive assignment. Added config parsing tests, gap-blending tests, and docs for the new knobs.

## Summary of Changes\n\nImplemented the first experimental weighted metric path for bottom-up potential-aware clustering. The implementation keeps river equity clustering unchanged, estimates sampled river bucket nut-distance scores, propagates those scores through turn centroids, and uses configured weights to blend potential movement, equity gaps, and nut-distance gaps for turn/flop EMD training and assignment.
