---
# poker_solver_rust-mx1j
title: Design hybrid MCCFR-at-boundary solver mode
status: in-progress
type: feature
priority: high
created_at: 2026-04-19T17:57:41Z
updated_at: 2026-04-20T04:53:18Z
---

Extend full-resolution solver with configurable depth limit + MCCFR sampling at boundary leaves. Replaces Modicum K-continuation rollout precompute with live/on-demand MC sampling during DCFR iterations. Tracks: brainstorm, design doc, implementation plan.

## Design complete 2026-04-19

Design doc: `docs/plans/2026-04-19-hybrid-mccfr-solver-design.md`
Research context: `docs/research/2026-04-19-subgame-solving-literature.md`

### Decisions
- (C) Replace existing Subgame mode in place.
- (A) Live per-iteration MCCFR sampling with bulk refresh every R iterations.
- (D1) Drop K=4 biased continuations; sample against unbiased blueprint only.
- Three config knobs: `depth_limit`, `boundary_refresh_interval`, `samples_per_refresh`.
- Boundary evaluator API takes OOP/IP ranges, returns both players' CFVs in one sampling pass.
- Tauri UI gets new 'Hybrid Solver' panel + Solver Telemetry panel for MCCFR metrics.
- Success criteria: depth=1 Hybrid within 500 mbb/hand of Exact at ≤2× wall-time.

### Next
- [ ] Invoke hex:writing-plans to generate implementation plan
- [ ] Create follow-up beans for each plan step
- [ ] Dispatch rust-developer agents per CLAUDE.md manager-mode workflow

## Implementation progress

### Phase 1 + 3 complete (2026-04-19)
- Phase 1: `BoundaryCfvs` + `sample_boundary_cfvs` + convergence test — merged (commits 8ee06afb / d5344c84 / 90a74fc8)
- Phase 3: `compute_cfvs_both` default method on `BoundaryEvaluator` — merged (commits 67ce8be5 / 43c4624b)
- Spec-compliance review on Phase 1: PASS_WITH_NOTES (Vec<f64> chosen over Vec<f32> — plan amended)
- All 228 range-solver tests + tauri tests pass

### Phase 2 complete (2026-04-19)
- HybridBoundaryEvaluator with cache + refresh logic — merged (commits 986cac14 / d5dac025 / 7e2e14e3)
- Agent chose approach (c): BoundarySampler trait with Box<dyn BoundarySampler> field (enables test mock injection without leaking Option into prod code)
- Phase 4 must impl BoundarySampler on RolloutLeafEvaluator to integrate
- BoundaryCfvs duplicate from stale worktree base resolved (commit: fix(hybrid): dedupe BoundaryCfvs)
- 216 tauri tests pass

### Phase 4A complete (2026-04-19, evening)
- BoundarySampler impl on RolloutLeafEvaluator — merged (fad15683)
- HybridBoundaryAdapter bridges HybridBoundaryEvaluator to BoundaryEvaluator trait — merged (acaaa7ce)
- 5 tests added, 221 total tauri tests pass
- This phase intentionally left the live solve path untouched (all scaffolding additive)

### Phase 4B complete + spec-review fix (2026-04-19/20)
- Phase 4B landed the live-path integration (commits 28d77cc3 / b388a67f / dc4c54a4)
- Spec-compliance review caught FAIL: per_boundary_evaluators was dead code — evaluation.rs never read it. Fixed by agent in 98af2953 / ac71345b.
- Current state: 221 tauri + 230 range-solver tests pass. Hybrid path is structurally end-to-end — can be driven via Tauri. compare-solve CLI is not yet wired for hybrid (Phase 7.1 task).
- Pre-existing flaky test: poker-solver-core blueprint_mp::mccfr::traverse_updates_strategy_sums fails on de40cdd4 base too — not our regression.

### Remaining phases
- Phase 5 (deletions — K=4 cleanup, ~3 tasks)
- Phase 6 (metrics — HybridRefreshMetrics/HybridSolveMetrics + stream, ~3 tasks)
- Phase 7 (CLI — wire compare-solve hybrid flags for izod validation, ~2 tasks)
- Phase 8 (Tauri mode string canonicalization — mostly trivial rename, ~2 tasks)
- Phase 9 (Frontend — Hybrid Solver panel + Telemetry panel, ~2 tasks)
- Phase 10 (Validation — run izod repro, verify success criteria, close beans, ~4 tasks)
