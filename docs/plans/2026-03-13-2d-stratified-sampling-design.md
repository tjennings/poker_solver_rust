# 2D Stratified Sampling for River NN Datagen

**Date:** 2026-03-13
**Status:** Approved

## Problem

The current `sample_pot_stack_by_spr` function picks an SPR bucket uniformly first, then samples pot from the feasible range. For high SPR values, the feasibility constraint `pot <= initial_stack / (spr + 0.5)` forces pot to be small. This creates a ~10:1 skew in pot distribution (62k samples in the 4-14 pot bin vs ~3k in upper bins), wasting compute on strategically less interesting small-pot spots.

## Solution

**Joint 2D rejection sampling**: pick both an SPR bucket AND a pot bucket uniformly, then sample within their intersection. Infeasible (SPR, pot) cells are rejected and retried.

## Algorithm

```
fn sample_pot_stack_by_spr(pot_intervals, spr_intervals, initial_stack, rng):
    loop:
        // 1. Pick BOTH buckets uniformly
        spr_idx = uniform(0..spr_intervals.len())
        pot_idx = uniform(0..pot_intervals.len())

        target_spr = uniform(spr_intervals[spr_idx])
        [pot_lo, pot_hi) = pot_intervals[pot_idx]

        // 2. Feasibility (same math as today)
        feasible_lo = ceil(5 / target_spr)
        feasible_hi = floor(initial_stack / (target_spr + 0.5))

        // 3. Intersect chosen pot bucket with feasible range
        lo = max(pot_lo, feasible_lo)
        hi = min(pot_hi, feasible_hi + 1)
        if lo >= hi → continue  // infeasible cell, retry

        // 4. Sample pot, derive stack (same as today)
        pot = uniform(lo..hi)
        max_stack = initial_stack - pot / 2
        target_stack = round(target_spr * pot), clamp(5, max_stack)

        // 5. Verify actual SPR in bucket (same as today)
        actual_spr = stack / pot
        if actual_spr in spr_bucket → return (pot, stack)
```

## What Changes

- **sampler.rs**: `sample_pot_stack_by_spr` — replace the segment-collection loop with single-bucket intersection. Adds ~2 lines, removes ~15 lines (segment logic). Net simpler.
- **Tests**: Update `spr_stratified_covers_all_buckets` to also verify pot bucket coverage. Add a new test for 2D cell coverage.

## What Doesn't Change

- `sample_situation` interface
- `sample_pot` (non-SPR path)
- `DatagenConfig` / YAML config
- Board sampling, range generation

## Rejection Rate

With `initial_stack=200`, 5 SPR × 4 pot = 20 cells. ~4-6 cells are infeasible (high SPR + large pot). Acceptance rate ~70-80%, negligible cost vs solver iterations.
