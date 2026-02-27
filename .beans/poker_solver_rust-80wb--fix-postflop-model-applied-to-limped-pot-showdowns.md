---
# poker_solver_rust-80wb
title: Fix postflop model applied to limped pot showdowns causing SB never-raise
status: completed
type: bug
priority: critical
created_at: 2026-02-27T00:43:16Z
updated_at: 2026-02-27T00:54:04Z
---

## Problem
SB never raises preflop — always limps (100% call for AA). The postflop model (trained for raised pots at SPR=3.5) gives avg_ev ~2.7 pot fractions for AA, inflating showdown values to ~10.87 chips vs fold terminal value of ~2 chips. This 5x ratio makes fold terminals negligibly valuable, creating degenerate limp-trap equilibrium.

## Root Cause
`postflop_showdown_value` applies the raised-pot postflop model to ALL showdown terminals, including limped pots. The `raise_counts` vector was precomputed for pot-type differentiation but is never read.

## Fix
- [x] In `postflop_showdown_value`, use `raise_counts[node_idx]` to detect limped pots (raise_count=0)
- [x] For limped pot showdowns, fall back to raw equity instead of postflop model
- [x] Add test verifying AA prefers raising over limping when postflop model is attached
- [x] Run full solve with --claude-debug to verify SB raises with strong hands


## Summary of Changes

**Root cause**: `postflop_showdown_value` applied the raised-pot postflop model (SPR=3.5) to ALL showdown terminals including limped pots. This inflated limped-pot showdown values ~5x vs fold terminals, making the solver learn to never raise (limp-trap degeneracy).

**Fix** (`crates/core/src/preflop/solver.rs`):
- Check `raise_counts[preflop_node_idx]` in `postflop_showdown_value`
- For limped pots (raise_count=0), fall back to raw equity instead of postflop model
- Added test `postflop_showdown_value_limped_pot_uses_equity`

**Result**: SB now raises 90-100% for most hands (was 0%). AA still shows ~63% call due to BB raise-back creating raised pots that use the postflop model — a known remaining limitation requiring pot-type-specific postflop models.
