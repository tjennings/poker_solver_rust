---
# poker_solver_rust-wa53
title: 'lazy-2p-equivalence: prove lazy sparse MP matches HU blueprint_v2 at 2 players (GO/NO-GO gate)'
status: todo
type: task
priority: high
created_at: 2026-06-18T18:43:16Z
updated_at: 2026-06-18T18:43:36Z
parent: poker_solver_rust-osss
blocked_by:
    - poker_solver_rust-8jan
---

GATING EVIDENCE for retiring HU training. Build an HU<->MP-lazy differential harness on a small shared 2-player game (same stacks/blinds/action abstraction); run both to convergence; compare per-infoset strategies within a stated tolerance; verify HU blind/button conventions (HU pins SB=0/BB=1; MP uses generic Vec<ForcedBet>); record iteration count, runtime, peak memory deltas (suspected 10-30% sparse overhead). Output is a GO/NO-GO verdict + tolerance, NOT a removal. If NO-GO, retire-hu is blocked and HU training stays. Size: medium. From Phase 7 plan 2026-06-18.
