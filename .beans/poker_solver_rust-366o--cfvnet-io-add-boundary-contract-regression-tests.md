---
# poker_solver_rust-366o
title: 'CFVNet IO: add boundary contract regression tests'
status: todo
type: task
priority: high
created_at: 2026-05-14T01:10:59Z
updated_at: 2026-05-14T01:10:59Z
parent: poker_solver_rust-8e9f
blocked_by:
    - poker_solver_rust-kai4
---

Pin the normalized IO contract with targeted tests.\n\n- [ ] Test 1326 range normalization after blockers\n- [ ] Test river blocker adjustment renormalizes per river\n- [ ] Test chip EV and legacy bcfv conversion formulas\n- [ ] Test raw-CFV path bypasses legacy boundary formula\n- [ ] Add targeted cfvnet and range-solver test commands to docs or bean summary\n\nPrimary files: crates/cfvnet/src/eval/boundary_evaluator.rs, crates/range-solver/src/game/evaluation.rs
