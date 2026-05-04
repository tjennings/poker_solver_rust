---
# poker_solver_rust-qcgs
title: Audit oracle-boundary reach semantics
status: completed
type: task
priority: high
created_at: 2026-05-04T01:09:31Z
updated_at: 2026-05-04T04:04:05Z
parent: poker_solver_rust-e90m
---

Compare subgame boundary reaches against exact-game reaches for identical public histories. Check blocker zeroing, opponent-vs-own reach usage, conditional vs unnormalized reach, and chance-card probability treatment.



## Work Notes

Starting reach-semantics audit after orientation and scalar unit audits did not recover the exact root policy.

Audit checklist:

- [ ] Map when range-solver records boundary_reach for each player at a depth boundary.
- [ ] Add diagnostics for exact_oracle cache timing: which player visits first, whether own/opponent reaches are initial, empty, or current.
- [ ] Compare cached raw CFVs computed on first visit against recomputed raw CFVs after both boundary reaches are available.
- [ ] Run the canonical JhTh9h7d exact_oracle spot and document whether stale or asymmetric reach explains the divergence.
- [ ] Identify the next fix or next audit.



## Result

Completed reach-semantics audit.

Checklist:

[x] Mapped when range-solver records boundary_reach for each player at depth boundaries.
[x] Added diagnostic coverage for exact_oracle raw-cache timing with reach-dependent values.
[x] Compared first-visit cached raw CFVs against current-reach recomputation by enforcing one raw-cache fill per traversing player.
[x] Ran canonical JhTh9h7d exact_oracle spot after the fix.
[x] Identified next audit: boundary value injection/iteration semantics.

Summary: raw CFV boundaries were caching both players on the first boundary visit. The first visit has current opponent reach only for the visiting player, so the opposite players raw value could be cached against initial/stale reach. Fixed raw boundary caching to populate only the current players slot and documented the result in docs/research/oracle_boundary_reach_audit_2026-05-04.md.
