---
# poker_solver_rust-h0o9
title: Normalize bucket distance channels
status: completed
type: task
priority: normal
created_at: 2026-05-14T01:14:02Z
updated_at: 2026-05-14T03:21:11Z
parent: poker_solver_rust-03j0
---

Normalize potential, nut-distance, and equity channels before weighting so metric weights have comparable scale.\n\nAcceptance: diagnostics report scale estimates and clustering uses normalized channel distances.

## Summary of Changes\n\nNormalized experimental clustering distance channels before applying metric weights. Potential, equity-gap, and nut-distance-gap channels now divide by their mean positive adjacent gap, and clustering writes the observed channel scales to metric_scales.json. The bucket scorecard loads that file when present so candidate comparisons include the actual scale factors. Verified with cargo check -p poker-solver-trainer and focused cluster metric tests.
