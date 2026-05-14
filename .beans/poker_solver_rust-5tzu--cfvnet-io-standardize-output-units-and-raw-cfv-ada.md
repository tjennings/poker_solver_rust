---
# poker_solver_rust-5tzu
title: 'CFVNet IO: standardize output units and raw-CFV adapter'
status: completed
type: task
priority: high
created_at: 2026-05-14T01:10:45Z
updated_at: 2026-05-14T01:30:17Z
parent: poker_solver_rust-8e9f
blocked_by:
    - poker_solver_rust-kai4
---

Make model output units explicit and prefer raw chip CFVs for range-solver integration.\n\n- [x] Treat model output as chip_cfv / (pot + effective_stack) everywhere\n- [x] Implement neural evaluator compute_raw_cfvs_both returning chip EVs\n- [x] Fix or isolate legacy compute_cfvs_both half-pot conversion\n- [x] Ensure Burn and ONNX paths decode identically\n\nPrimary files: crates/cfvnet/src/eval/boundary_evaluator.rs, crates/range-solver/src/game/mod.rs, crates/range-solver/src/game/evaluation.rs

## Implementation Notes\n\n- Added shared normalized-output conversion helpers for chip EV and legacy half-pot BCFV units.\n- Implemented NeuralBoundaryEvaluator::compute_raw_cfvs_both for Burn and ONNX paths, returning chip EVs in game private-card order.\n- Kept compute_cfvs_both on the legacy half-pot BCFV contract and routed both backends through identical decode/mapping helpers.\n- Added focused cfvnet tests for normalized-to-game-hand unit mapping and raw CFV support.

## Verification

- cargo test -p cfvnet boundary_evaluator: passed, 15 tests.
- cargo test -p range-solver raw_cfv: passed, 4 tests.
