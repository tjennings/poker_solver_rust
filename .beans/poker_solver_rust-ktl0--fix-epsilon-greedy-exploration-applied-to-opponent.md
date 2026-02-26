---
# poker_solver_rust-ktl0
title: Fix epsilon-greedy exploration applied to opponent nodes in preflop CFR
status: completed
type: bug
priority: normal
created_at: 2026-02-26T18:22:12Z
updated_at: 2026-02-26T18:24:38Z
---

## Problem

Preflop DCFR with exploration (epsilon-greedy) plateaus at ~81 mBB/hand exploitability and never converges further. Without exploration, DCFR converges to <14 mBB in 200 iterations.

## Root Cause

In `crates/core/src/preflop/solver.rs`, the epsilon-greedy exploration is applied at ALL decision nodes (both hero and opponent). It should only be applied at hero nodes. When applied at opponent nodes, it biases the hero's regret estimates because the opponent is forced to explore, inflating their exploitability. The average strategy converges to a fixed point adapted to the opponent's exploration noise rather than the true strategy.

## Fix

In `cfr_traverse`, only apply exploration when `is_hero` is true. Pass `intended` (not `traversal`) to `traverse_opponent`.

## Validation

- [x] Fix the exploration bug
- [x] Run DCFR with exploration and verify convergence below 20 mBB (8.33 mBB in 200 iters)
- [x] Run existing tests to verify no regressions (all pass)


## Summary of Changes

Fixed epsilon-greedy exploration in `crates/core/src/preflop/solver.rs:cfr_traverse`. The exploration was incorrectly applied to both hero AND opponent decision nodes. Moved exploration logic inside the `is_hero` branch so opponent nodes use the pure regret-matched strategy.

**Before:** DCFR with 5% exploration plateaued at ~81 mBB/hand exploitability
**After:** Converges to 8.33 mBB/hand in 200 iterations (early stop threshold reached)
