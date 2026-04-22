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
