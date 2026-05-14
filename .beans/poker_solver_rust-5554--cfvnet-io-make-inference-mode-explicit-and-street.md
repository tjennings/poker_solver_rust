---
# poker_solver_rust-5554
title: 'CFVNet IO: make inference mode explicit and street-safe'
status: completed
type: task
priority: high
created_at: 2026-05-14T01:10:52Z
updated_at: 2026-05-14T01:35:17Z
parent: poker_solver_rust-8e9f
---

Remove unsafe reliance on river_enumerated_turn as an implicit default for direct boundary models.\n\n- [x] Audit constructors and loader helpers that use BoundaryInferenceMode::default\n- [x] Make direct turn/flop models pass Direct explicitly\n- [x] Keep RiverEnumeratedTurn only for river-model adapter use\n- [x] Update trainer CLI defaults or validation so model kind is street-safe\n\nPrimary files: crates/cfvnet/src/eval/boundary_evaluator.rs, crates/trainer/src/main.rs, tauri/devserver call sites if any

## Summary of Changes

Added explicit load_neural_boundary_evaluator_with_mode APIs for Burn and ONNX boundary evaluators. Legacy no-mode constructors/loaders now explicitly choose RiverEnumeratedTurn and are documented as legacy adapter behavior. compare-solve model-kind defaults now use direct for flop/turn/river, while explicit river_enumerated_turn remains supported for legacy river-model adapter use. Verified with cargo test -p cfvnet boundary_evaluator and cargo test -p poker-solver-trainer compare_solve_street_boundary_cli_flags_parse.
