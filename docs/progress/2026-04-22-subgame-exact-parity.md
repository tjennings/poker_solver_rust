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

## Iteration 6 — 2026-04-22 (Round 3: Raw CFV path via compute_raw_cfvs_both)

**Baseline before:** exact_exp=77.67 mbb (turn spot JhTh9h7d, 150 iters), subgame_exp=8417 mbb (bcfv path), delta=1.000

**Diagnosis:** Implemented approach #3 from the followup plan: new `compute_raw_cfvs_both` trait method that returns per-hand chip CFVs already integrated over opponent reach, bypassing the `bcfv × payoff_scale × cfreach_adj` formula in `evaluate_boundary_single`.

Three-part implementation:
1. **BoundaryEvaluator trait** (`game/mod.rs`): Added `compute_raw_cfvs_both` with default `None` return. Added `boundary_is_raw` per-ordinal flag to switch between `evaluate_boundary_single` (legacy) and new `evaluate_boundary_raw` (direct write).
2. **evaluate_internal** (`evaluation.rs`): K<=1 path tries `compute_raw_cfvs_both` first; if `Some(...)`, stores raw values and sets `boundary_is_raw` flag; new `evaluate_boundary_raw` writes values directly without cfreach_adj multiplication.
3. **SubtreeExactEvaluator** (`exact_subtree.rs`): Builds Turn+River game once (check-check turn → chance → river decisions), caches the solved game, then calls `root_cfvalues_with_reach(game, player, boundary_opp_reach)` each iteration.

**Key findings:**
- **Zero-value sanity check PASSES:** Setting all raw CFVs to 0.0 produces subgame_exp=0.0 mbb, confirming the raw path plumbing works correctly.
- **Integration test PASSES with zero error:** Small ranges (AA,KK,QQ vs TT,99,88) with turn bet sizes — bounded game root cfvalues match full game exactly.
- **0.01x scaling test:** Reducing raw values by 100x drops subgame_exp from 2346 to 22.7 mbb, suggesting the raw values are correct in direction but the magnitude interacts badly with the seeded Turn strategy.
- **Root cause:** The subtree evaluator solves a fresh game from scratch (un-seeded), while the parent solver's Turn strategy is seeded from the blueprint. The subtree's Nash equilibrium river strategy differs from the exact game's blueprint-seeded strategy, producing different boundary CFVs that push the Turn strategy in the wrong direction.

**Fix applied:** `compute_raw_cfvs_both` trait method + `evaluate_boundary_raw` + Turn+River game approach.

**Result after:** exact_exp=77.67 mbb, subgame_exp=2346 mbb, worst_delta=0.975
Status: FAIL (delta=0.975, +2269 mbb gap — similar to previous bcfv-path result, confirming the issue is subtree strategy divergence, not the cfv formula)

**Commits:** `6ccb55ca` trait method + wiring, `d626e19d` SubtreeExactEvaluator impl, `c8e56ee3` Turn+River game + integration test, `ed0d4a07` cleanup

## Iteration 7 — 2026-04-22 (cfv_to_bcfv + clamping restore, JhTh9h spot)

**Baseline before:** exact_exp=77.67 mbb (river spot JhTh9h7d, 150 iters), subgame_exp=2346 mbb (raw CFV path from iter 6), worst_delta=0.975

**Diagnosis:** Reverted to the `cfv_to_bcfv` approach with `MIN_ADJ=1e-6` + `MAX_BCFV=10.0` clamping (commit `484939a4`). Testing whether the original bcfv path on the JhTh9h spot performs differently than the raw CFV path. The bcfv path re-introduces the `cfv_to_bcfv` inversion formula that divides raw cfvalues by `(half_pot/N)*cfreach_adj` and clamps extreme outputs. This was previously measured on the Jd9d7d turn spot (iter 2) where it produced 232 mbb exploitability. Now measuring on the JhTh9h7d river spot to compare against the raw CFV path (iter 6: 2346 mbb).

**Fix applied:** `484939a4` revert to cfv_to_bcfv + MIN_ADJ/MAX_BCFV clamping (revert of raw CFV path).

**Result after:** exact_exp=77.67 mbb, subgame_exp=3140.12 mbb, mean_mass=0.556, worst_cell="TT @ Check exact=0.9921 subgame=0.0041 delta=0.9880"
Status: FAIL (delta=0.9880, +3062.45 mbb exploitability gap — worse than raw CFV path's 2346 mbb)

Wall times: exact=0.3s, subgame=458.8s
Top 3 hands by mass moved:
1. AcAd mass=1.000 — exact=[X:1.00] subgame=[B55:0.60 A:0.40] (should check, goes aggro)
2. 7hKs mass=1.000 — exact=[B55:0.57 A:0.42] subgame=[X:1.00] (should bet/shove, checks instead)
3. 7hQd mass=1.000 — exact=[B55:0.78 A:0.21] subgame=[X:1.00] (should bet, checks instead)

**Commits:** `484939a4` revert(exact_subtree): cfv_to_bcfv approach + MIN_ADJ/MAX_BCFV clamping

## Iteration 8 — 2026-04-23 (DeepStack range gadget with ConstantOptOut(0.0))

**Baseline before:** exact_exp=77.67 mbb (river spot JhTh9h7d, 150 iters), subgame_exp=3140.12 mbb (iter 7 cfv_to_bcfv), worst_delta=0.988

**Diagnosis:** Implemented DeepStack-style range gadget (GadgetEvaluator) as a BoundaryEvaluator wrapper. The gadget clamps each opponent hand's bcfv upward to an opt-out value (ConstantOptOut(0.0) for this iteration). Architecture:
- OptOutProvider trait returns per-hand opt-out CFVs
- GadgetEvaluator wraps inner SubtreeExactEvaluator + OptOutProvider
- compute_cfvs_both delegates to inner, then clamps opponent values upward
- 9 unit tests + 2 integration tests all pass

The gadget correctly applies per-hand clamping (verified by unit tests with StubEvaluator). However, the underlying cfv_to_bcfv formula in SubtreeExactEvaluator still produces values with incorrect magnitude/scale (the root cause from iterations 1-7). Clamping already-wrong values to 0.0 adds error rather than correcting it.

**Fix applied:** `6057ff3f` feat(gadget): GadgetEvaluator + OptOutProvider + ConstantOptOut + --gadget CLI flag.

**Result after:** exact_exp=77.67 mbb, subgame_exp=34079.20 mbb, mean_mass=0.460, worst_cell="77 @ All-in exact=0.0000 subgame=1.0000 delta=1.0000"
Status: FAIL (delta=1.000, +34002 mbb — gadget with ConstantOptOut(0.0) WORSENS the result because clamping already-incorrect bcfv values to 0 amplifies the error)

Wall times: exact=1.6s, subgame=337.6s
Top 3 hands by mass moved:
1. 7hKs mass=1.000 — exact=[B55:0.57 A:0.42] subgame=[X:1.00] (should bet, checks)
2. 3hAh mass=1.000 — exact=[X:0.98] subgame=[A:1.00] (should check, shoves)
3. QhAd mass=1.000 — exact=[B55:0.62 A:0.38] subgame=[X:1.00] (should bet, checks)

**Root cause confirmed:** The gadget architecture is correct (unit tests verify clamping behavior). The blocker is the underlying SubtreeExactEvaluator's cfv_to_bcfv formula producing incorrectly-scaled boundary values. The gadget cannot fix wrong inputs — it needs correct bcfv values to clamp meaningfully.

**Commits:** `6057ff3f` feat(gadget): DeepStack range gadget for safe subgame boundary evaluation

## Iteration 9 — 2026-04-23 (500-iter convergence probe)

**Baseline before:** exact_exp=77.67 mbb (150 iters), subgame_exp=3140.12 mbb (iter 7, cfv_to_bcfv + clamping), worst_delta=0.988

**Hypothesis:** More iterations (500 vs 150) might close the gap if the subgame strategy just needs longer to converge.

**Result after:** exact_exp=5.02 mbb, subgame_exp=82.62 mbb, mean_mass=0.593, worst_cell="J8s @ Check exact=1.0000 subgame=0.0000 delta=1.0000"
Status: FAIL (delta=1.000, +77.61 mbb exploitability gap)

Wall times: exact=0.8s, subgame=1406.3s (2.81s/iter avg)
Cache-hit ratio at end: 11% (641 hits / 5500 calls, avg solve 2751ms)

**Comparison to iter 7 (150 iters):**
- exact_exp: 77.67 -> 5.02 mbb (exact solver converged better at 500 iters, as expected)
- subgame_exp: 3140.12 -> 82.62 mbb (massive improvement with more iters)
- exploitability gap: +3062 -> +77.61 mbb (gap narrowed ~40x)
- worst_delta: 0.988 -> 1.000 (unchanged — individual worst cell is still max-delta)
- Hands still fully inverted: 7 of top-10 flip from bet/shove to pure check

**Convergence verdict: PARTIAL.** The exploitability *numbers* improved dramatically (82.62 vs 3140 mbb), but this is largely because the exact solver also converged much tighter (5.02 vs 77.67 mbb). The *relative* gap (subgame/exact ratio) went from 40x to 16x. The worst-cell delta remains 1.000, and the same structural pathology persists: hands that should bet/shove instead check (and vice versa). More iters narrow the exploitability gap in absolute terms but do NOT fix the strategy divergence. The bcfv formula is still the root cause.

Per-action-class bias: Bet/Raise -0.233, Check +0.054, AllIn +0.179
Top 3 hands by mass moved:
1. 2d7h mass=1.000 — exact=[B55:0.50 A:0.50] subgame=[X:1.00] (should bet/shove, checks)
2. 2h7s mass=1.000 — exact=[B55:0.33 A:0.67] subgame=[X:1.00] (should shove, checks)
3. 2hJc mass=1.000 — exact=[B55:0.50 A:0.50] subgame=[X:1.00] (should bet/shove, checks)

**Commits:** measurement only, no code changes

## Iteration 10 — 2026-04-23 (cfvnet baseline, no gadget)

**Approach:** river-boundary=cfvnet (ONNX model `checkpoint_epoch675.onnx`), no gadget. This is the baseline for the Libratus gadget A/B comparison (iterations 10-12).

**Result:** exact_exp=77.67 mbb, subgame_exp=20932.49 mbb, worst_delta=1.0000, worst_cell="77 @ All-in exact=0.0000 subgame=1.0000"
mean_mass=0.543, max_mass=1.000 at hand 2s7c.
Status: FAIL (established cfvnet baseline for gadget comparison)

**Wall time:** exact=0.3s, subgame=1.5s.

Per-action-class bias (subgame - exact):
- AllIn: +0.217
- Bet/Raise: -0.015
- Check: -0.202

Top 3 hands by mass moved:
1. 2s7c mass=1.000 -- exact=[X:0.89 B24:0.11] subgame=[A:1.00] (should check, goes all-in)
2. 7cAd mass=1.000 -- exact=[X:0.99] subgame=[A:1.00] (should check, goes all-in)
3. 4h5h mass=1.000 -- exact=[X:0.98] subgame=[A:1.00] (should check, goes all-in)

**Key observation:** cfvnet boundary produces much larger exploitability (20932 mbb) than exact_subtree at same iters (3140 mbb iter 7). The cfvnet model's boundary values are far worse than even the broken cfv_to_bcfv formula. The subgame universally shoves with hands that should check.

**Commits:** measurement only, no code changes

## Iteration 11 — 2026-04-23 (cfvnet + BlueprintCbvOptOut gadget -- PANIC)

**Approach:** river-boundary=cfvnet + --gadget --gadget-provider=blueprint-cbv. Intended to test whether the Libratus-style gadget closes the cfvnet exploitability gap.

**Result:** PANIC at `crates/core/src/blueprint_v2/cbv.rs:48` -- index out of bounds: the len is 243 but the index is 7916.

**Root cause:** `BlueprintCbvOptOut::from_cbv_context` receives `current_node` (the abstract tree node index of the current spot, e.g. 7916) as the `boundary_node_idx` parameter. But the CbvTable is indexed by boundary ordinal (0..N where N = number of chance nodes in the abstract tree, e.g. 27 for a 3-street tree), not by the full tree node ID. The unit tests only exercise a 1-node CbvTable where ordinal 0 always works, so this never crashed in testing.

**Design flaw:** `BlueprintCbvOptOut` pre-computes CBV values for a single boundary node at construction, but there are 11 boundary nodes (one per river card) and each needs its own CBV lookup. The `opt_out_cfvs` method ignores the `boundary_ordinal` parameter.

**Fix needed:**
1. Map each range-solver boundary ordinal to the corresponding CBV table boundary node index (requires traversing the abstract tree from `current_node` to find its chance children and their CBV ordinals).
2. Either (a) make the provider compute lazily per-boundary using the ordinal, or (b) pre-compute all 11 boundaries at construction with the correct CBV indices.

Status: BLOCKED (code bug in Phase 1-3 -- `BlueprintCbvOptOut::from_cbv_context` index mapping is wrong)

**Commits:** no code changes (measurement-only run revealed the bug)

## Iteration 12 — 2026-04-23 (sanity check, dominated opt-out -999.0)

**Approach:** river-boundary=cfvnet + --gadget --gadget-provider=constant --gadget-constant=-999.0. Sanity check: a dominated opt-out should be a no-op, producing results identical to the baseline (iter 10).

**Result:** exact_exp=77.67 mbb, subgame_exp=20932.49 mbb, worst_delta=1.0000, worst_cell="77 @ All-in exact=0.0000 subgame=1.0000"
mean_mass=0.543, max_mass=1.000 at hand 2s7c.
Status: PASS (sanity)

**Comparison to iter 10 (no gadget):**

| Metric | Iter 10 (no gadget) | Iter 12 (constant -999) | Delta |
|--------|-------------------|------------------------|-------|
| exact_exp | 77.67 mbb | 77.67 mbb | 0.00 |
| subgame_exp | 20932.49 mbb | 20932.49 mbb | 0.00 |
| worst_delta | 1.0000 | 1.0000 | 0.00 |
| mean_mass | 0.543 | 0.543 | 0.00 |

All values match exactly. The dominated opt-out (-999.0) is a perfect no-op. The GadgetEvaluator wrapper correctly passes through inner evaluator values when the opt-out is dominated. This confirms the gadget wrapping infrastructure works correctly; the blocker is only in BlueprintCbvOptOut's CBV index mapping.

**Wall time:** exact=0.3s, subgame=1.6s.

**Commits:** measurement only, no code changes

## Iteration 11 (fixed) -- 2026-04-22 (cfvnet + BlueprintCbvOptOut gadget, bug fix applied)

**Bug fix:** `BlueprintCbvOptOut::from_cbv_context` had two bugs:
1. It was passed the abstract tree's decision node ID (e.g. 7916) but `CbvTable::lookup` expected a dense boundary ordinal (0..N). Added `CbvTable::build_node_to_ordinal_map()` and `CbvTable::require_ordinal()` to convert sparse arena indices to dense ordinals.
2. The provider only stored CBV values for a single boundary but the subgame has multiple boundaries (one per action path reaching a street transition). Redesigned `BlueprintCbvOptOut` to store per-boundary values (`Vec<[Vec<f32>; 2]>`) and accept `abstract_root` (decision node) instead of a single chance node. The constructor now finds all chance node descendants via DFS and computes CBVs for each.
3. `GadgetEvaluator` now carries its `boundary_ordinal` and passes it to `opt_out_cfvs`, so each boundary gets the correct opt-out values.

**Approach:** river-boundary=cfvnet + --gadget --gadget-provider=blueprint-cbv (same as iter 11 but with the CBV index bug fixed).

**Result:** exact_exp=77.67 mbb, subgame_exp=318119.14 mbb, worst_delta=1.0000, worst_cell="32o @ All-in exact=0.0000 subgame=1.0000"
mean_mass=0.818, max_mass=1.000 at hand 4h5c.
Status: FAIL (gadget made things WORSE)

**Comparison to iter 10 (cfvnet baseline, no gadget):**

| Metric | Iter 10 (no gadget) | Iter 11-fixed (CBV gadget) | Delta |
|--------|-------------------|---------------------------|-------|
| exact_exp | 77.67 mbb | 77.67 mbb | 0.00 |
| subgame_exp | 20932.49 mbb | 318119.14 mbb | +297186.65 |
| worst_delta | 1.0000 | 1.0000 | 0.00 |
| mean_mass | 0.543 | 0.818 | +0.275 |

Per-action-class bias (subgame - exact):
- AllIn: +0.818 (vs +0.217 in iter 10)
- Bet/Raise: -0.262 (vs -0.015 in iter 10)
- Check: -0.556 (vs -0.202 in iter 10)

**Wall time:** exact=0.3s, subgame=1.7s.

**Verdict:** The Libratus-style BlueprintCbvOptOut gadget with abstract-tree CBVs does NOT close the gap -- it makes it ~15x worse. The abstract tree's coarse bucketed CBVs produce inaccurate opt-out floors that distort the strategy, causing near-universal all-in. The CBV values from 2-bucket equity-fallback abstraction are too noisy to serve as meaningful lower bounds.

This confirms that the Libratus static-CBV approach is insufficient with the current abstraction quality. The DeepStack-proper approach (bean poker_solver_rust-akg3: retrain cfvnet to produce per-hand CBV values directly) is the correct next step.
