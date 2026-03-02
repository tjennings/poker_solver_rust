---
# poker_solver_rust-7ih3
title: Fix LCFR convergence regression on deep SPR trees
status: completed
type: bug
priority: normal
created_at: 2026-03-02T07:05:48Z
updated_at: 2026-03-02T07:08:32Z
---

## Problem
AhKsQd at SPR 6 regresses from ~300 to 676 mBB/h exploitability at iteration 12/30.

Root cause: LCFR (alpha=beta=1.0) decays positive and negative regrets equally, letting negative regrets swamp the signal on deep trees (+5875 / -84495 ratio).

Secondary: exhaustive solver passes 0-indexed iteration to iteration_weights(), wasting the first iteration.

## Tasks
- [x] Root cause investigation
- [x] Change default cfr_variant from Linear to Dcfr in postflop_model.rs
- [x] Fix 0-indexed iteration in postflop_exhaustive.rs to use iter+1
- [x] Make AKQr_vs_234r_postflop.yaml explicit about cfr_variant: dcfr
- [x] Run tests to verify no regressions


## Summary of Changes

1. Changed default `cfr_variant` from Linear to Dcfr — asymmetric decay (α=1.5, β=0.5) decays negative regrets ~8× faster
2. Fixed 0-indexed iteration in exhaustive solver — first iteration no longer wasted (weight=0→1)
3. Made AKQr_vs_234r_postflop.yaml explicit about cfr_variant: dcfr

51 lib tests pass, 0 failures.
