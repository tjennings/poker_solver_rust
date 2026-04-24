---
# poker_solver_rust-40vv
title: 'Implement Option A2: per-boundary single-sided gadget with traverser-disable'
status: in-progress
type: feature
priority: high
created_at: 2026-04-24T17:02:45Z
updated_at: 2026-04-24T20:16:07Z
blocked_by:
    - poker_solver_rust-02wj
---

Architectural pivot from Option A (root-only two-sided nested gadget) to Option A2 (per-boundary single-sided gadget pairs with traverser-dependent activation).

Motivation: ml-researcher comparison to published work (Burch 2014, Brown-Sandholm 2017, DeepStack-Leduc) showed Option A's three novelties:
1. Root-only vs per-infoset gadget placement — Option A loses per-boundary guarantees that Libratus provides
2. Bucketed CBV opt-out source — looser than Libratus Estimate (akg3's scope)
3. Neutralized non-gadget-player CFV at Terminate — breaks zero-sum, outside published theoretical cover

Option A2 design:
- Per-boundary gadget pair at each cfvnet boundary (not at root)
- Single-sided per pass: gadget owned by non-traverser is active; traverser's own gadget is disabled (σ forced to (0,1), no regret update, behaves as if gadget doesn't exist)
- Zero-sum complement at Terminate: non-gadget player gets -⟨reach, opt_out⟩/⟨reach⟩ (scalar, per-iter)
- Opt-out source: existing BlueprintCbvOptOut::from_cbv_context (per-boundary per-player, already implemented)
- Root-level Option A gadget retired entirely

Phases:
- A: gadget_owner node flag + per-boundary tree injection + disable root injection
- B: solver activation logic (traverser-disable)
- C: zero-sum complement terminal CFVs
- D: retire root gadget code + wiring updates (CLI, Tauri, explorer)
- E: tests + bench
- F: docs + bean management

Blocked-by poker_solver_rust-02wj (Option A merged; A2 replaces it on main).
Supersedes bgkr (GadgetEvaluator retirement) — A2 removes the post-clamp path implicitly.
Branch: feat/option-a2-per-boundary-gadget

Design basis: ml-researcher comparison preserved in session transcript; design doc addendum pending.



## Phase progress 2026-04-24

### Completed phases (all committed to feat/option-a2-per-boundary-gadget, merged to main at 8af0d107)

- **Phase A** — per-boundary gadget tree injection. Commit `bbda235b`. PLAYER_GADGET_FLAG added; inject_per_boundary_gadgets + 17 new tests.
- **Phase B** (initial, wrong condition) — Commit `ac19a5b1`. 5 new tests.
- **Phase B fix** — flipped to Option Y semantics (owner's pass active, non-owner's pass disabled). Commit `e2252a3d`. Per-boundary safety invariant test PASSES at 0.01 tolerance after 1000 iters.
- **Phase C** — SKIPPED. Option Y's disable-non-owner-pass semantics means the non-gadget player never traverses through the gadget Terminate, so zero-sum complement at that terminal is unreachable. No code needed.
- **Phase D** — retire Option A + wire A2 through CLI / Tauri / explorer. Commit `d55d427b`. Net -571 lines. 970/971 tests pass on main post-merge.
- **Merge to main** — `8af0d107`. All Option A code removed.

### Still outstanding

- **Phase F** — docs (dispatched; see commit TBD).
- **E2E harness on JhTh9h|…|7d** — user to run directly from main post-docs merge.
- **Close akg3** — triage decision based on iter-15/16 E2E results.
