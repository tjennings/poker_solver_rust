---
# poker_solver_rust-wa53
title: 'lazy-2p-equivalence: prove lazy sparse MP matches HU blueprint_v2 at 2 players (GO/NO-GO gate)'
status: in-progress
type: task
priority: high
created_at: 2026-06-18T18:43:16Z
updated_at: 2026-06-23T05:37:05Z
parent: poker_solver_rust-osss
blocked_by:
    - poker_solver_rust-8jan
---

GATING EVIDENCE for retiring HU training. Build an HU<->MP-lazy differential harness on a small shared 2-player game (same stacks/blinds/action abstraction); run both to convergence; compare per-infoset strategies within a stated tolerance; verify HU blind/button conventions (HU pins SB=0/BB=1; MP uses generic Vec<ForcedBet>); record iteration count, runtime, peak memory deltas (suspected 10-30% sparse overhead). Output is a GO/NO-GO verdict + tolerance, NOT a removal. If NO-GO, retire-hu is blocked and HU training stays. Size: medium. From Phase 7 plan 2026-06-18.

## 2026-06-23 Start Notes

Activated after lazy sparse action identity landed. Preflight:

- `/usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_wa53_preflight.log 2>&1'` passed but missed the gate with `real 75.62`, likely cache/build noise.\n- Hot rerun `/usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_wa53_preflight_hot.log 2>&1'` passed under the gate with `real 43.27`, `user 97.30`, `sys 14.82`.\n\nScope checklist:\n\n- [ ] Research HU blueprint_v2 and MP lazy 2-player training semantics, blind/button conventions, action abstraction mapping, storage identity, and comparison surfaces.\n- [ ] Brainstorm the smallest credible GO/NO-GO harness that avoids false equivalence from incompatible abstractions or stochastic noise.\n- [ ] Implement or add a reusable validation harness/report path for a tiny shared 2-player game.\n- [ ] Compare HU and MP-lazy strategies with explicit tolerances and report runtime/memory evidence.\n- [ ] Document the GO/NO-GO result and update architecture/training docs if the validation workflow becomes user-visible.\n- [ ] Verify focused tests and hot redirected full workspace suite under the one-minute gate.\n
