---
# poker_solver_rust-366o
title: 'CFVNet IO: add boundary contract regression tests'
status: completed
type: task
priority: high
created_at: 2026-05-14T01:10:59Z
updated_at: 2026-05-14T01:35:52Z
parent: poker_solver_rust-8e9f
blocked_by:
    - poker_solver_rust-kai4
---

Pin the normalized IO contract with targeted tests.\n\n- [x] Test 1326 range normalization after blockers\n- [x] Test river blocker adjustment renormalizes per river\n- [x] Test chip EV and legacy bcfv conversion formulas\n- [x] Test raw-CFV path bypasses legacy boundary formula\n- [x] Add targeted cfvnet and range-solver test commands to docs or bean summary\n\nPrimary files: crates/cfvnet/src/eval/boundary_evaluator.rs, crates/range-solver/src/game/evaluation.rs

## Summary of Changes

Regression coverage was added across the CFVNet IO implementation commits: canonical 1326 range validation/normalization after blockers, river-enumerated per-river renormalization, normalized-output conversion to chip EV and legacy half-pot units, constructor mode safety, and neural raw-CFV support. Existing range-solver raw-CFV tests verify the raw boundary evaluator bypasses the legacy BCFV formula. Targeted verification commands: cargo test -p cfvnet boundary_evaluator; cargo test -p range-solver raw_cfv; cargo test -p poker-solver-trainer compare_solve_street_boundary_cli_flags_parse.
