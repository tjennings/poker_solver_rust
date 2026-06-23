---
# poker_solver_rust-wa53
title: 'lazy-2p-equivalence: prove lazy sparse MP matches HU blueprint_v2 at 2 players (GO/NO-GO gate)'
status: in-progress
type: task
priority: high
created_at: 2026-06-18T18:43:16Z
updated_at: 2026-06-23T05:41:54Z
parent: poker_solver_rust-osss
blocked_by:
    - poker_solver_rust-8jan
---

GATING EVIDENCE for retiring HU training. Build an HU<->MP-lazy differential harness on a small shared 2-player game (same stacks/blinds/action abstraction); run both to convergence; compare per-infoset strategies within a stated tolerance; verify HU blind/button conventions (HU pins SB=0/BB=1; MP uses generic Vec<ForcedBet>); record iteration count, runtime, peak memory deltas (suspected 10-30% sparse overhead). Output is a GO/NO-GO verdict + tolerance, NOT a removal. If NO-GO, retire-hu is blocked and HU training stays. Size: medium. From Phase 7 plan 2026-06-18.

## 2026-06-23 Start Notes

Activated after lazy sparse action identity landed. Preflight:

- `/usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_wa53_preflight.log 2>&1'` passed but missed the gate with `real 75.62`, likely cache/build noise.\n- Hot rerun `/usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_wa53_preflight_hot.log 2>&1'` passed under the gate with `real 43.27`, `user 97.30`, `sys 14.82`.\n\nScope checklist:\n\n- [ ] Research HU blueprint_v2 and MP lazy 2-player training semantics, blind/button conventions, action abstraction mapping, storage identity, and comparison surfaces.\n- [x] Brainstorm the smallest credible GO/NO-GO harness that avoids false equivalence from incompatible abstractions or stochastic noise.\n- [ ] Implement or add a reusable validation harness/report path for a tiny shared 2-player game.\n- [ ] Compare HU and MP-lazy strategies with explicit tolerances and report runtime/memory evidence.\n- [ ] Document the GO/NO-GO result and update architecture/training docs if the validation workflow becomes user-visible.\n- [ ] Verify focused tests and hot redirected full workspace suite under the one-minute gate.\n

## Research / Brainstorming Notes (2026-06-23)

Required research and architecture passes completed. `hex:brainstorming` was not exposed in this session, so the brainstorming pass was delegated to architecture/research agents and synthesized here.

Strong conclusion: this slice should NOT remove HU training or add a production TUI feature. The smallest safe slice is an evidence-producing HU-vs-MP-lazy validation harness, preferably in `crates/convergence-harness`, with a reusable GO/NO-GO report.

Current evidence is NO-GO until proven otherwise. Major equivalence risks:

- HU `blueprint_v2` average-strategy accounting updates traverser nodes only, while MP eager/lazy update sampled opponent nodes as well. This can make average strategies differ even if regret behavior is close.
- Action graphs are not automatically identical: HU uses `Bet`/`Raise`, MP uses `Lead`/`Raise`; postflop min-open and lazy river SPR-zero suppression can diverge.
- Chance mode must be pinned to sampled full deals for MP lazy. Same-seed trajectory equality is not valid because HU and MP lazy seed/deal schedules differ.
- Raw integer regret deltas are not comparable because HU and MP storage use different regret scales.

Credible gate criteria:

- Structural checks: 2 players, HU seat 0 maps to MP SB/dealer seat 0, seat 1 maps to BB seat 1, same stacks/blinds/rake/limp flag/action sizes/bucket counts, MP chance mode `SampledFullDeal`, pruning/baselines/purge disabled.
- Action schema checks compare normalized public action descriptors (`Bet` normalized to `Lead`, chip amount tolerance 0.01) rather than raw node ids.
- The initial implementation should report GO/NO-GO and diagnostics; it should not claim HU can be retired. Treat mismatched action schema, missing matched rows, accounting mismatch, or uncalibrated tolerance as NO-GO.

Recommended implementation slice:

- Add `crates/convergence-harness/src/hu_mp_lazy.rs` plus module/export wiring.
- Define `HuMpLazyHarnessConfig`, `HuMpLazyReport`, verdict enum, structural/action mismatch records, and `run_hu_mp_lazy_diff`.
- Use tiny equivalent in-memory configs; first tests should assert report generation and structural/schema checks, not force a GO verdict.
- Save artifacts similar to the existing reporter: `summary.json`, strategy/schema CSVs, and `report.txt`.
- Focused tests: report serialization, blind mismatch rejection, action schema mismatch rejection, root schema match for the 2p fixture, and a tiny end-to-end smoke that produces an explicit verdict.
