---
# poker_solver_rust-etzw
title: 'CFVNet IO: normalize evaluator inference ranges'
status: todo
type: task
priority: high
created_at: 2026-05-14T01:10:39Z
updated_at: 2026-05-14T01:10:39Z
parent: poker_solver_rust-8e9f
blocked_by:
    - poker_solver_rust-kai4
---

Normalize runtime evaluator inputs to match training.\n\n- [ ] Use canonical helper when mapping game-order reaches to 1326 ranges\n- [ ] Zero board-blocked combos before model inference\n- [ ] Renormalize ranges after river blocker adjustment in river-enumerated mode\n- [ ] Apply to Burn and ONNX evaluator paths\n\nPrimary file: crates/cfvnet/src/eval/boundary_evaluator.rs
