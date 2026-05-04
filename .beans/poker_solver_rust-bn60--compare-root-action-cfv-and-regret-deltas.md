---
# poker_solver_rust-bn60
title: Compare root action CFV and regret deltas
status: completed
type: task
priority: high
created_at: 2026-05-04T04:29:26Z
updated_at: 2026-05-04T04:48:13Z
parent: poker_solver_rust-e90m
---

Add or run a focused diagnostic that compares exact vs depth-limited root action counterfactual values and regret deltas for the canonical oracle-boundary spot.

Checklist:

[x] Inspect solver storage/query APIs for root action CFVs and regrets.
[x] Add the narrowest compare-solve diagnostic needed to print per-action root values/deltas.
[x] Run canonical exact_oracle comparison at one early iteration and a higher-divergence setting.
[x] Record whether divergence appears before or after boundary value injection.
[x] Update research notes with the result.
[x] Run focused tests and full warm test suite under 1 minute.

## Summary of Changes

Added a hidden compare-solve root update trace that captures exact/subgame root action CFVs and regret-update deltas around selected iterations. Re-exported range-solver root CFV/regret helpers for the diagnostic.

The diagnostic exposed a stale evaluator-backed boundary CFV cache before finalization. Clearing those cached boundary values before final average-strategy finalization corrected the oracle aligned 1000/1000 subgame exploitability from 94.81 mbb/hand to 3.92 mbb/hand; exact remained 2.29 mbb/hand.

Root traces showed iteration 0 exact/subgame pre-update values and regret updates were identical. By iteration 999, the largest root action CFV/regret gap was about 0.171 chips on QhAs at the 24bb action, while reach-weighted action means stayed tiny. Focused tests passed, and the warm full cargo test pass completed in 52.77s.
