---
# poker_solver_rust-eqkm
title: Subgame rollout perf + hands/sec telemetry
status: completed
type: feature
priority: high
created_at: 2026-04-17T16:20:42Z
updated_at: 2026-04-18T20:29:32Z
---

Close the rollout-throughput gap to MCCFR's ~100k hands/sec.

**Design:** docs/plans/2026-04-17-subgame-rollout-perf-design.md
**Plan:** docs/plans/2026-04-17-subgame-rollout-perf-impl.md
**Branch:** perf/subgame-rollout (worktree at ../poker_solver_rust.rollout-perf)

## Commits (todo)
- [x] 1. feat(postflop): add rollout hands/sec telemetry (baseline measurement) — HEAD ed239b67, all 3 review stages green
- [x] 1b. chore(trainer): add rollout-bench diagnostic — HEAD 6d3eec31, baseline 8243 ms/call
- [x] 2. perf(continuation): depth-gated MCCFR sampling — HEAD 75eb8468, 50.2 ms/call (164× speedup), all review stages green
- [x] 3. test(trainer): validate-rollout harness — HEAD cf4fed95
- [x] 4. perf(postflop): per-call RNG seed rotation — HEAD c1aee56b
- [x] 5. perf(rollout): configurable knobs + UI + gate deletion — HEAD 15baf12a, defaults D=2/S=8 (51.6 ms/call), exhaustive gate removed
- [x] Consolidated review — spec GREEN, quality found 2 must-fix boundary bugs (propagation of enumerate_decision_depth and call_counter to sub-evaluators in game_session.rs), simplicity found function-length violations
- [x] Review fixes amended into commit 5 (HEAD 5054e90f post-rebase)
- [x] Docs pass: architecture.md, training.md, 2026-04-17 plan marked superseded
- [x] Final verification: 961 pass (1 known pre-existing failure v55b); clippy 1 pre-existing warning (bxip); TS 2 pre-existing errors in GameExplorer.tsx
- [x] Merged to main (fast-forward, linear history, 8 commits on top of e15f732b)
- [ ] F. Final verification + PR open

## Test gate
`cargo test -p poker-solver-core -p poker-solver-tauri --quiet` (runs in ~22s)
Known pre-existing failure: blueprint_mp::mccfr::tests::traverse_updates_strategy_sums (tracked in poker_solver_rust-v55b) — unrelated to this PR's code paths.

## Measurement target
~100k hands/sec, matching MCCFR. Each commit records before/after in PR description.

## Plan Refinement 2026-04-18

Bench baseline (commit 6c1de413, Ks7h2c flop, 1176 combos, TT+/AQs+/AKo vs JJ+/AKs/AQs): **46M terminal visits/sec, 8s per rollout_chip_values_with_state call**.

Raw hands/sec is NOT the bottleneck. Per-evaluator-call latency is. Reframing attribution metric from `hands/sec` to `ms/call` (and `calls/sec` derived).

The 5 planned optimizations remain valid — they reduce per-call work (fewer allocs per combo × opp_sample, cached buckets, deck mask, scratch buffers, better parallelism). Target: drive the 8s/call down meaningfully for the 1176-combo reference scenario.

## Plan Pivot 2026-04-18

ml-researcher findings pivoted the plan. Modicum/Pluribus use sampled rollouts at depth boundaries, not exhaustive. Dropped commits 2-6 (per-hand alloc fixes; only ~2× net after accounting for variance).

New plan: 2026-04-18-subgame-rollout-sampling-impl.md
- Commit 2: hybrid depth-gated sampling in rollout_inner (enumerate depth 0-1, sample deeper, bump chance samples)
- Commit 3: validation (sampled vs exhaustive CFVs within 2 mbb/hand), delete exhaustive gate on pass

Expected: 8.2s → 80-200ms per call (50-100×).

## Summary of Changes

**Achievement:** 164x speedup on subgame rollout evaluator-call latency — 8,243 ms/call → 50.2 ms/call at default settings on the 200_100bb_sapcfr bundle (1,176-combo scenario).

**Approach:** Pivoted mid-stream from the original 5-commit per-hand allocation plan (~5-10x expected) to depth-gated MCCFR sampling after ml-researcher findings pointed to Modicum (Brown/Sandholm/Amos NeurIPS 2018).

**Commits (8):**
1. 27a6b4d5 feat(postflop): rollout hands/sec telemetry
2. c1205462 chore(trainer): bench-rollout diagnostic
3. 0a88a901 docs(plans): pivot to depth-gated sampling
4. 963e0822 perf(continuation): depth-gated MCCFR sampling at decision nodes
5. ed483a1b test(trainer): validate-rollout harness
6. 738ca30d perf(postflop): per-call RNG seed rotation
7. 5054e90f perf(rollout): configurable accuracy/perf tradeoff + UI settings
8. 4cf35da1 docs: architecture/training updates

**Validation findings:** No setting passes the 2 mbb/hand max_abs_diff criterion (irreducible per-call opponent-sampling variance floor ~2.9 mbb even at D=4/S=64). Mean bias stays < 1 mbb across all configs — estimator unbiased, per-call max averages out across DCFR iterations. Defaults kept at D=2/S=8 (cheapest setting).

**User-facing:** 'Enum. Depth' input added to Tauri Settings panel alongside existing opponent-samples control. Both knobs tunable per-solve.

**Tech debt filed:** v55b (test-suite runtime + GPU test fixes), bxip (build_subgame_solver 14+ param refactor).
