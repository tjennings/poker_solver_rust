---
# poker_solver_rust-5tzu
title: 'CFVNet IO: standardize output units and raw-CFV adapter'
status: todo
type: task
priority: high
created_at: 2026-05-14T01:10:45Z
updated_at: 2026-05-14T01:10:45Z
parent: poker_solver_rust-8e9f
blocked_by:
    - poker_solver_rust-kai4
---

Make model output units explicit and prefer raw chip CFVs for range-solver integration.\n\n- [ ] Treat model output as chip_cfv / (pot + effective_stack) everywhere\n- [ ] Implement neural evaluator compute_raw_cfvs_both returning chip EVs\n- [ ] Fix or isolate legacy compute_cfvs_both half-pot conversion\n- [ ] Ensure Burn and ONNX paths decode identically\n\nPrimary files: crates/cfvnet/src/eval/boundary_evaluator.rs, crates/range-solver/src/game/mod.rs, crates/range-solver/src/game/evaluation.rs
