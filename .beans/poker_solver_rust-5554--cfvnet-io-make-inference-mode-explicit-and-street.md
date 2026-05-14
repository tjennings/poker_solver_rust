---
# poker_solver_rust-5554
title: 'CFVNet IO: make inference mode explicit and street-safe'
status: todo
type: task
priority: high
created_at: 2026-05-14T01:10:52Z
updated_at: 2026-05-14T01:10:52Z
parent: poker_solver_rust-8e9f
---

Remove unsafe reliance on river_enumerated_turn as an implicit default for direct boundary models.\n\n- [ ] Audit constructors and loader helpers that use BoundaryInferenceMode::default\n- [ ] Make direct turn/flop models pass Direct explicitly\n- [ ] Keep RiverEnumeratedTurn only for river-model adapter use\n- [ ] Update trainer CLI defaults or validation so model kind is street-safe\n\nPrimary files: crates/cfvnet/src/eval/boundary_evaluator.rs, crates/trainer/src/main.rs, tauri/devserver call sites if any
