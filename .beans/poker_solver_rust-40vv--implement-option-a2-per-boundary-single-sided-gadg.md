---
# poker_solver_rust-40vv
title: 'Implement Option A2: per-boundary single-sided gadget with traverser-disable'
status: in-progress
type: feature
priority: high
created_at: 2026-04-24T17:02:45Z
updated_at: 2026-04-24T17:03:10Z
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
