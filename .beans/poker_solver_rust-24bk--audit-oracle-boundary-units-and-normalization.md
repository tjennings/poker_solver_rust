---
# poker_solver_rust-24bk
title: Audit oracle-boundary units and normalization
status: in-progress
type: task
priority: high
created_at: 2026-05-04T01:09:27Z
updated_at: 2026-05-04T03:38:09Z
parent: poker_solver_rust-e90m
---

Confirm whether each boundary evaluator and depth-boundary injection path expects raw chip CFVs, pot-normalized values, or half-pot-normalized BCFVs. Compare raw exact continuation values to values consumed by regret updates.



## Work Notes

Starting units/normalization audit after orientation audit ruled out simple player/sign transforms.

Audit checklist:

- [ ] Map unit conventions for compare-solve oracle, exact_subtree, cfvnet/gadget helpers, and range-solver boundary injection.
- [ ] Add or run diagnostics comparing raw exact continuation CFVs against the values consumed at boundary nodes.
- [ ] Test plausible scale transforms on the canonical JhTh9h7d exact_oracle spot.
- [ ] Document whether a unit/normalization mismatch explains the divergence and identify the next fix/audit.
