---
# poker_solver_rust-jmed
title: Build tiny oracle-boundary contract test
status: completed
type: task
priority: high
created_at: 2026-05-04T01:09:18Z
updated_at: 2026-05-04T01:27:28Z
parent: poker_solver_rust-e90m
---

Create a minimal one-boundary game where full exact and depth-limited exact_oracle should match within numerical noise. Use it to prove whether the boundary handoff fails before poker complexity enters.

## Work Notes

Starting step 2. Goal: add a tiny one-boundary contract test that compares a depth-limited exact-oracle handoff against the full exact solve, before investigating orientation/unit/reach variants.

## Summary of Changes

Added a normal compare-solve regression test that constructs a pruned turn-root game with exactly one depth boundary, solves the same game full-depth and through exact_oracle, and asserts exploitability and root strategy mass remain within tight tolerance.

Observed result: the one-boundary oracle contract passes exactly in this tiny case: exact_exp=0.000000, subgame_exp=0.000000, exp_delta=0.000000, mean_mass=0.000000, max_mass=0.000000. This suggests the current large-spot divergence is not caused by a universal raw boundary handoff failure.

Verification:
- cargo test -p poker-solver-trainer compare_solve::tests::oracle_boundary_one_boundary_contract_matches_exact -- --nocapture
- cargo test -p poker-solver-trainer compare_solve
- time cargo test (warm): passed in 51.807s
