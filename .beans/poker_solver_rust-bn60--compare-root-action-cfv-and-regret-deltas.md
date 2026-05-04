---
# poker_solver_rust-bn60
title: Compare root action CFV and regret deltas
status: in-progress
type: task
priority: high
created_at: 2026-05-04T04:29:26Z
updated_at: 2026-05-04T04:29:26Z
parent: poker_solver_rust-e90m
---

Add or run a focused diagnostic that compares exact vs depth-limited root action counterfactual values and regret deltas for the canonical oracle-boundary spot.

Checklist:

[ ] Inspect solver storage/query APIs for root action CFVs and regrets.
[ ] Add the narrowest compare-solve diagnostic needed to print per-action root values/deltas.
[ ] Run canonical exact_oracle comparison at one early iteration and a higher-divergence setting.
[ ] Record whether divergence appears before or after boundary value injection.
[ ] Update research notes with the result.
[ ] Run focused tests and full warm test suite under 1 minute.
