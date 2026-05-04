---
# poker_solver_rust-qcgs
title: Audit oracle-boundary reach semantics
status: in-progress
type: task
priority: high
created_at: 2026-05-04T01:09:31Z
updated_at: 2026-05-04T03:52:31Z
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
