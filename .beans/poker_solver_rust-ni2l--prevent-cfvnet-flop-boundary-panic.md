---
# poker_solver_rust-ni2l
title: Prevent CFVNet flop-boundary panic
status: in-progress
type: bug
priority: high
created_at: 2026-05-04T05:00:40Z
updated_at: 2026-05-04T05:00:40Z
---

Using cfvnet as a flop boundary can attach NeuralBoundaryEvaluator to a 3-card boundary board, which panics because the ONNX evaluator supports only 4-card and 5-card boards.

Checklist:

[ ] Confirm the failing Tauri/frontend configuration path.
[ ] Add validation so cfvnet cannot be configured for unsupported flop boundaries.
[ ] Keep exact_subtree behavior intact for flop/turn/river boundaries.
[ ] Add or update focused tests for frontend params and/or Tauri boundary config validation.
[ ] Run focused checks and full warm cargo test under 1 minute.
