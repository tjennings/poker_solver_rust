---
# poker_solver_rust-mx1j
title: Design hybrid MCCFR-at-boundary solver mode
status: in-progress
type: feature
priority: high
created_at: 2026-04-19T17:57:41Z
updated_at: 2026-04-19T20:42:49Z
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
