---
# poker_solver_rust-24bk
title: Audit oracle-boundary units and normalization
status: completed
type: task
priority: high
created_at: 2026-05-04T01:09:27Z
updated_at: 2026-05-04T03:48:14Z
parent: poker_solver_rust-e90m
---

Confirm whether each boundary evaluator and depth-boundary injection path expects raw chip CFVs, pot-normalized values, or half-pot-normalized BCFVs. Compare raw exact continuation values to values consumed by regret updates.



## Work Notes

Starting units/normalization audit after orientation audit ruled out simple player/sign transforms.

Audit checklist:

- [x] Map unit conventions for compare-solve oracle, exact_subtree, cfvnet/gadget helpers, and range-solver boundary injection.
- [x] Add or run diagnostics comparing raw exact continuation CFVs against the values consumed at boundary nodes.
- [x] Test plausible scale transforms on the canonical JhTh9h7d exact_oracle spot.
- [x] Document whether a unit/normalization mismatch explains the divergence and identify the next fix/audit.



## Summary of Changes

Completed the unit/normalization audit. Added hidden compare-solve --oracle-scale diagnostic for exact_oracle raw CFV scaling experiments and documented the unit map plus canonical JhTh9h7d scale sweep in docs/research/oracle_boundary_units_audit_2026-05-04.md.

Result: exact_oracle already uses the raw per-combination CFV path; converting or scaling toward bcfv/chip-style units does not recover the exact root strategy. Downscales can reduce subgame exploitability, but root strategy mass moves to roughly 0.5-0.8 with max mass near 1.0, so scalar normalization is not a valid fix.

Next: audit reach semantics, especially raw-CFV caching before both players boundary reach vectors are available in the current iteration.

Verification: cargo test -p poker-solver-trainer compare_solve; cargo build --release -p poker-solver-trainer; warm time cargo test in 52.401s.
