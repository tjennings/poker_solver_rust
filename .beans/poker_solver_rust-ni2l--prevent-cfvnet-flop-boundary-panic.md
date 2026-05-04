---
# poker_solver_rust-ni2l
title: Prevent CFVNet flop-boundary panic
status: completed
type: bug
priority: high
created_at: 2026-05-04T05:00:40Z
updated_at: 2026-05-04T05:10:16Z
---

Using cfvnet as a flop boundary can attach NeuralBoundaryEvaluator to a 3-card boundary board, which panics because the ONNX evaluator supports only 4-card and 5-card boards.

Checklist:

[x] Confirm the failing Tauri/frontend configuration path.
[x] Add validation so cfvnet cannot be configured for unsupported flop boundaries.
[x] Keep exact_subtree behavior intact for flop/turn/river boundaries.
[x] Add or update focused tests for frontend params and/or Tauri boundary config validation.
[x] Run focused checks and full warm cargo test under 1 minute.

## Summary of Changes

Confirmed that a flop-root CFVNet cut can resolve to depth 0 and hand 3-card flop boards to the ONNX boundary evaluator. Added shared backend validation before Tauri solve-thread startup and in compare-solve, kept Exact Subtree valid for earlier cuts, and updated the frontend solve config path so only the river CFVNet boundary mode is accepted. Verified with focused Rust/frontend checks and a full warm cargo test.
