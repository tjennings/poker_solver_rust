---
# poker_solver_rust-izod
title: Subgame converges to worse-than-blueprint strategy — likely sampling bias
status: todo
type: bug
priority: critical
created_at: 2026-04-19T02:04:29Z
updated_at: 2026-04-19T02:04:29Z
---

**Severity: critical — subgame solver is anti-refining the blueprint.**

User observation: on SB:2bb, BB:10bb, SB:22bb, BB:call | Jd9d7d, a full 4000-iteration subgame solve produces a strategy WORSE than the blueprint seed, while Exact produces the expected best result. Expected ordering is Blueprint < Subgame < Exact; observed is Subgame < Blueprint < Exact.

**Hypothesis:** the depth-gated MCCFR sampling introduced in commit 963e0822 (now-merged on main) has systematic bias (not just variance) at boundary CFVs. The validate-rollout harness reports mean_abs_diff of 0.71-1.0 mbb/hand between sampled and exhaustive CFVs — this was accepted under the assumption it was zero-mean noise that DCFR averaging would absorb. If the bias is sign-consistent, DCFR over 4000 iters converges to a strategy optimal against biased CFVs rather than true CFVs — exactly producing this symptom.

**Why this hypothesis is suspicious-by-construction:** the rollout uses the BLUEPRINT strategy to compute boundary values, not the current DCFR iterate. So the sampled CFVs are stable across iterations. Any per-call bias accumulates identically every DCFR step instead of being averaged away.

**Diagnostic tests (in priority order):**

1. **Enum. Depth = 255 experiment (free, zero-code):** set `rollout_enumerate_depth = 255` in the Tauri Settings panel, re-run subgame solve on the same spot. If result matches Exact, our sampling has bias. If result still worse than blueprint, bug is elsewhere (seeding, DCFR math, boundary CFV application).

2. **Re-examine validate-rollout for sign-consistency:** current harness reports `mean_abs_diff`. Should also report `mean_signed_diff` — if that's far from zero, that's the smoking gun for systematic bias.

3. **Compare strategy at root** between Enum.Depth=2 and Enum.Depth=255 for a small test case to measure the bias magnitude.

**Candidate root causes if sampling is confirmed biased:**
- `bias_strategy_into` normalization incorrect (unlikely; it's identical to the original Vec-returning version).
- `sample_action_index` inverse-CDF edge case biasing toward last index due to fp-drift fallthrough.
- The sampling changes per-branch expected value semantics in a way that no longer estimates the same quantity as enumeration. Specifically: when we sample one action at depth ≥2, we're computing an unbiased estimator of the expectimax VALUE of the biased blueprint strategy — but we're using this value as the 'biased continuation value' in Modicum's K=4 framework, where the intent was to compute the value *under* each biased continuation independently. If K=Unbiased vs K=Fold/Call/Raise sample from the *same* deep-tree structure in meaningfully similar ways, they collapse to producing similar CFV estimates, and Modicum's robustness guarantee is violated.

**Blast radius:** every subgame solve on main (since commit 963e0822). Exact mode unaffected.

**Workaround:** users can set Enum. Depth = 255 until fixed. Slower but correct.

**Validation after fix:**
- re-run user's scenario; expect Blueprint < Subgame < Exact
- validate-rollout `mean_signed_diff` should be small (< 0.5 mbb/hand) under fixed settings

Context: Found by user during 2026-04-19 post-merge testing. Related beans: jpwu (parallelize precompute), eqkm (completed sampling work).
