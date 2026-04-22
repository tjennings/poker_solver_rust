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

**Commits:** (clamping change, uncommitted)
