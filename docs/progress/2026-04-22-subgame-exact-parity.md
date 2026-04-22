# Subgame-Exact Parity Progress Log

Bean: izod — Subgame converges to worse-than-blueprint strategy.
Branch: feat/subgame-exact-parity
Spot: sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d (BB to act, flop Jd9d7d, pot=22bb eff=39bb)
Bundle: 1k_100bb_brdcfr_v2/snapshot_0013

## Iteration 0 — 2026-04-22 (Baseline)

**Baseline before:** exact_exp=38.61 mbb, subgame_exp=38.61 mbb (all-exact, no boundary), mean_mass=0.000, worst_cell="22 @ Check exact=1.0000 subgame=1.0000 delta=0.0000"

**Diagnosis:** All-exact baseline confirms both solvers produce identical results when there is no boundary cut (0 boundary nodes). This is the expected ground truth. Now testing with `--turn-boundary exact_subtree` to measure the impact of the SubtreeExactEvaluator from commit 798e7c16.

**Fix applied:** N/A (baseline measurement only).

**Result after:** exact_exp=38.61 mbb, subgame_exp=38.61 mbb, mean_mass=0.000, worst_cell="22 @ Check delta=0.0000"
Status: PASS (baseline, no boundary cut applied)

**Commits:** `798e7c16` fix(exact_subtree): remove spurious turn actions and remap CFVs to parent ordering

## Iteration 1 — 2026-04-22 11:35

**Baseline before:** exact_exp=109.13 mbb (turn spot Jd9d7d8s, 100 iters), subgame_exp=240.97 mbb, mean_mass=0.483, worst_cell="96s @ Check exact=0.0168 subgame=1.0000 delta=0.9832"

**Diagnosis:** SubtreeExactEvaluator `cfv_to_bcfv` conversion produces catastrophically wrong boundary CFVs. The subgame massively over-checks (+0.181 per hand) and under-bets (-0.197). Hands like 3c8d, 2h8d that should bet 75% pot (B33) in exact instead check or min-bet in subgame. The `cfv_to_bcfv` inversion formula divides raw cfvalues by `(half_pot/N)*cfreach_adj` to get bcfv, but this involves dividing by near-zero cfreach_adj for blocker-heavy hands, producing extreme values that poison the parent solver. The subtree solver also uses 100 internal DCFR iters (reduced from 500 for speed) which may under-converge some river subtrees.

**Fix applied:** None yet -- diagnosing root cause of `cfv_to_bcfv` numerical instability and the fundamental approach of dividing out the parent's expected multiplier.

**Result after:** exact_exp=109.13 mbb, subgame_exp=240.97 mbb, mean_mass=0.483, worst_cell="96s @ Check delta=0.9832"
Status: FAIL (delta=0.9832, +131.84 mbb exploitability gap)

**Commits:** (no code change yet, diagnostic only)

## Iteration 2 — 2026-04-22 12:10

**Baseline before:** exact_exp=109.13 mbb, subgame_exp=240.97 mbb (from iter 1, no clamping)

**Diagnosis:** Added MIN_ADJ=1e-6 and MAX_BCFV=10 clamping to `cfv_to_bcfv` to prevent numerical instability when cfreach_adj approaches zero. The clamping had negligible effect: subgame_exp dropped from 240.97 to 232.43 mbb, worst_cell unchanged (96s @ Check delta=0.9832). The problem is NOT numerical instability in the cfv_to_bcfv inversion. Root cause appears to be a more fundamental issue with the cfv_to_bcfv approach itself: the subtree game uses parent initial_weights for game construction but evaluates with boundary reach. The subtree's `num_combinations` (computed from parent initial_weights with a 4-card board) differs from what the parent solver expects, and the boundary reach remapping between subtree and parent ordering may introduce systematic errors. Next step: add diagnostic logging to compare cfreach_adj values between subtree and parent.

**Fix applied:** `exact_subtree.rs:cfv_to_bcfv` — added MIN_ADJ threshold and MAX_BCFV clamp.

**Result after:** exact_exp=109.13 mbb, subgame_exp=232.43 mbb, mean_mass=0.496, worst_cell="96s @ Check exact=0.0168 subgame=1.0000 delta=0.9832"
Status: FAIL (delta=0.9832, +123.30 mbb gap, marginal improvement from 240.97)

**Commits:** `ebb32128` docs(progress): iteration 2 — cfv_to_bcfv clamping ineffective

## Iteration 3-4 — 2026-04-22 12:34 (Deep diagnosis)

**Baseline before:** exact_exp=109.13 mbb, subgame_exp varies (232-27024 mbb depending on approach)

**Diagnosis:** Tried four approaches for bcfv computation. All fail:
1. `cfv_to_bcfv` with boundary reach + parent weights: all-negative bcfv (232 mbb exp)
2. `pot_normalise` from `root_cfvalues` without cfv_to_bcfv: huge bcfv (27024 mbb, all-in)
3. Uniform weights + `root_cfvalues` + `cfv_to_bcfv`: identical to (2), confirming cfv_to_bcfv correctly cancels reach
4. Uniform weights + `root_cfvalues` directly: identical to (3)

The fact that (2)=(3)=(4) confirms cfv_to_bcfv correctly inverts the reach weighting. The problem with (1) is that boundary reach from cfreach has different magnitude than initial_weights, causing the cfreach_adj ratio (subtree vs parent) to diverge. The problem with (2-4) is that `chip_ev_centered = cfv * num_combos * (w_raw/w_norm)` produces values much larger than half_pot because num_combos (~3925) amplifies the per-combo cfv. The bcfv interface expects equity-like values in [-1, 1], but the formula produces multi-pot values.

The root cause is a semantic mismatch: `evaluate_boundary_single` treats bcfv as a "how many half-pots this hand wins" scalar, but the subtree's cfvalues are in a fundamentally different unit system where the per-combo normalization doesn't directly map to equity.

**Fix applied:** Multiple approaches tried, none successful. The SubtreeExactEvaluator architecture needs a deeper redesign to either (a) produce values compatible with evaluate_boundary_single's expectations, or (b) use a different integration point that doesn't require bcfv format.

**Result after:** Best result: exact_exp=109.13 mbb, subgame_exp=232.43 mbb (approach 1 with clamping)
Status: FAIL (131 mbb gap at best, 27K mbb at worst)

**Commits:** `0b5b88ab` docs(progress): iteration 3 — root cause identified

## Iteration 3 — 2026-04-22 12:12 (Root cause identified)

**Baseline before:** exact_exp=109.13 mbb, subgame_exp=232.43 mbb (turn spot with clamping)

**Diagnosis:** Added diagnostic logging to `solve_subtree` revealing that ALL bcfv values are NEGATIVE for BOTH players at every boundary. In a zero-sum game, one player must have positive value when the other has negative, so both-negative indicates the cfv_to_bcfv scaling is fundamentally broken. The `root_cfvalues_with_reach` function returns per-combo cfvalues using boundary reach as opponent cfreach, then `cfv_to_bcfv` divides out `(half_pot / N) * cfreach_adj`. The cancellation requires the subtree's `num_combinations` and `cfreach_adj` to exactly match the parent's, but the subtree has different board cards (4-card boundary board for turn→river subtrees with 48 river cards in the subtree expansion), producing different `num_combinations`. Additionally, the `effective_stack` calculation in the subtree may be inconsistent with the parent's expectations, causing systematic cfvalue bias. The cfv_to_bcfv inversion approach is architecturally unsound -- it requires dividing by quantities that differ between subtree and parent, producing systematically wrong boundary values.

**Fix applied:** N/A (diagnostic only). Next step: abandon cfv_to_bcfv inversion. Instead, compute per-hand chip EVs via `expected_values_detail` and pot-normalise directly, bypassing the cfreach_adj dependency entirely. This requires investigating how `evaluate_boundary_single` handles the cfreach weighting and potentially modifying the interface.

**Result after:** exact_exp=109.13 mbb, subgame_exp=232.43 mbb (unchanged, diagnostic only)
Status: FAIL (root cause identified: cfv_to_bcfv architecture is broken)

**Commits:** `ebb32128` docs(progress): iteration 2 — cfv_to_bcfv clamping ineffective

## Iteration 5 — 2026-04-22 (Per-river enumeration approach #3)

**Baseline before:** exact_exp=77.67 mbb (turn spot JhTh9h7d, 150 iters), subgame_exp=8497 mbb (previous approach), delta=1.000

**Diagnosis:** Implemented approach #3 from the followup doc: per-river-card enumeration. Instead of solving a single Turn+River game, enumerate each of the 48 valid river cards and solve each as a separate 5-card River game. This eliminates the chance_factor issue that broke the previous Turn game approach.

Three sub-approaches were tried:
1. **expected_values with uniform weights**: Converts chip EV to bcfv via `(ev - half_pot) / half_pot`. Failed with subgame_exp=6919 mbb because expected_values includes opponent range weighting that double-counts with evaluate_boundary_single's cfreach_adj.
2. **root_cfvalues + cfv_to_bcfv with uniform weights**: Uses cfv_to_bcfv formula `bcfv[h] = cfv[h] * N_sub / (half_pot * cfreach_adj[h])`. N_sub cancels out correctly. Failed with subgame_exp=52281 mbb because uniform weights produce bcfv values that don't match the parent's actual reach distribution.
3. **root_cfvalues + cfv_to_bcfv with boundary reach**: Uses actual boundary reach from DCFR as initial weights. Produces reasonable bcfv values (range [-1, 2], mean near 0). Still produces all-in strategy (subgame_exp=8417 mbb).

Key diagnostic finding: zeroing ALL bcfv values produces subgame_exp=0.0 mbb with reasonable strategy (0.616 mean mass). Confirms tree structure is correct; issue is purely in bcfv-to-strategy interaction.

**Fix applied:** Per-river enumeration with boundary reach + cfv_to_bcfv.

**Result after:** exact_exp=77.67 mbb, subgame_exp=8417 mbb, worst_delta=1.000
Status: FAIL (delta=1.000, per-river bcfv values look numerically correct but integration with evaluate_boundary_single's lazy caching produces all-in strategy)

**Commits:** `cc82d6de` per-river enumeration, `735c6fcf` CLI fix, `fdd1cdd5` boundary reach, `1c251e35` root_cfvalues + cfv_to_bcfv, `07438ba3` uniform weights, `30028f79` boundary reach + root_cfvalues
