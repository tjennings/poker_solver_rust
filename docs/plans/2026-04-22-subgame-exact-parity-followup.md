# Subgame Exact Parity Followup

## Status: FAIL (worst_delta=0.99)

Best delta achieved: 0.99 (no meaningful improvement from baseline 0.99)
Exploitability delta: 2575-2652 mbb/hand (should be ~0)

## Iterations Taken: 6 of 8 hard cap

## Fixes Applied (committed)

1. **`798e7c16` - Remove spurious turn actions**: The subtree was built with the parent's `turn_bet_sizes`, creating a full turn decision layer before the river. Since the boundary is AFTER turn decisions, this extra layer let players re-act at the turn. Fixed by setting flop/turn bet sizes to empty defaults.

2. **`798e7c16` - Remap CFVs to parent ordering**: The subtree's `PostFlopGame` may have fewer private_cards than the parent (zero-reach combos excluded from Range). Added `remap_cfvs_to_parent()` for card-pair lookup mapping.

3. **`a611f028` - Correct CFV scaling with parent weights**: Build subtree with parent's initial weights (not boundary reaches) so `num_combinations` matches. Added `root_cfvalues_with_reach()` to range-solver. Convert cfvalues to bcfv by dividing out `half_pot / N * cfreach_adj`.

4. **`96ce05a6` - Use default subtree solve iters (500 vs 100)**: The explicit `.with_solve_iters(100)` was insufficient for convergence.

5. **`55353b01` - Use actual boundary reach for cfvalue computation**: Use `root_cfvalues_with_reach` with actual boundary reach and `cfv_to_bcfv` conversion. Set subtree iters to 200 for time budget.

## Root Cause Analysis

The fundamental issue is the **CFV-to-bcfv conversion formula**. The boundary evaluator framework expects bcfv values that, when multiplied by `half_pot / num_combinations * cfreach_adj` in `evaluate_boundary_single`, produce the correct counterfactual values at the boundary.

### What works (SPR=0 equity evaluator)

For showdown-only boundaries (SPR=0), the `SolveBoundaryEvaluator` computes:
```
bcfv = (weighted_equity - 0.5) * 2.0
```

This works because equity is a "per-unit-of-opponent-reach" quantity:
```
result = bcfv * half_pot/N * cfreach_adj
       = (cfreach_win - cfreach_lose)/cfreach_adj * half_pot/N * cfreach_adj
       = half_pot/N * (cfreach_win - cfreach_lose)
```
This exactly matches the full-tree showdown computation.

### What fails (SPR>0 exact subtree evaluator)

For game subtrees with betting, cfvalues are NOT simply proportional to cfreach_adj. The betting tree creates complex interactions between hands - a hand's value depends on the individual opponent reach values, not just their sum. The formula `bcfv = cfv / (half_pot/N * cfreach_adj)` assumes linearity that doesn't hold with betting.

Specifically:
- `cfv_subtree[h]` = sum over river cards of complex terminal/betting evaluations
- `cfreach_adj[h]` = sum of non-blocked opponent reach
- The ratio `cfv / cfreach_adj` is NOT constant across different reach distributions

## Approaches That Failed

1. **Pot-normalised cfvalue formula** (`cfv * N * (w_raw/w_norm) / half_pot`): This is the `expected_values_detail` conversion. With actual boundary reach vs initial weights, the w_raw/w_norm factor is wrong (it compensates for normalization that doesn't apply to custom reach).

2. **Reach-independent bcfv** (solve with parent weights, extract pot-normalised EVs): Returns the same bcfv regardless of reach, which is wrong - the SPR=0 evaluator correctly produces different values for different reaches.

3. **cfreach_adj division** (`cfv * N / (half_pot * cfreach_adj)`): Mathematically correct for showdown-only terminals but fails for betting trees because cfvalues are non-linear functions of reach.

## Top 3 Suspected Remaining Root Causes

### 1. The boundary evaluator interface is fundamentally incompatible with exact subtree evaluation at SPR > 0

`evaluate_boundary_single` assumes `bcfv[h]` is a per-hand quantity that gets linearly scaled by `cfreach_adj`. This works for showdown equity but not for games with betting. The fix may require modifying `evaluate_boundary_single` to accept raw cfvalues directly (bypassing the `bcfv * payoff_scale * cfreach_adj` formula), or using a different boundary evaluation path for exact subtree evaluators.

### 2. The subtree solver should use the actual boundary reach as initial weights (not parent weights or uniform)

The equilibrium strategy depends on which hands are present and their relative weights in some game-tree configurations (e.g., when isomorphism is used). Building the subtree with incorrect weights may produce subtly wrong strategies, even if the Nash equilibrium is theoretically weight-independent.

### 3. The check-check turn layer introduces a chance_factor mismatch

The subtree starts at `BoardState::Turn` with a chance node that deals river cards. The `compute_cfvalue_recursive` divides cfreach by `chance_factor` at this node. But `evaluate_boundary_single` in the parent doesn't apply any chance factor - it uses the raw boundary cfreach. This means the subtree's cfvalues include a 1/C factor that the parent doesn't expect. The bcfv conversion should account for this but may not do so correctly.

## Recommended Next Steps

1. **Build a minimal test**: Create a 2-hand-per-player river-only game where the exact answer is known analytically. Verify that the boundary evaluator produces correct cfvalues for this case. This would isolate the scaling issue.

2. **Bypass evaluate_boundary_single**: Instead of returning bcfv, directly inject cfvalues into `boundary_cfvs` using `set_boundary_cfvs()` before each iteration. This avoids the `bcfv * payoff_scale * cfreach_adj` formula entirely. The challenge is computing the cfvalues in exactly the format that the solver expects.

3. **Per-river-card subtree approach**: Instead of a Turn game with river enumeration, solve each river card individually as a River game (5-card board). This eliminates the chance_factor issue entirely. The challenge is averaging correctly over river cards and handling blocker effects.
