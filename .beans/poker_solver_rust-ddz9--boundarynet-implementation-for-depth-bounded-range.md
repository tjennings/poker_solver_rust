---
# poker_solver_rust-ddz9
title: BoundaryNet implementation for depth-bounded range solving
status: in-progress
type: task
priority: high
created_at: 2026-04-06T00:52:01Z
updated_at: 2026-04-06T01:30:27Z
---

Add BoundaryNet model to cfvnet crate for depth-bounded range solving.

Design doc: docs/plans/2026-04-05-boundary-net-design.md
Impl plan: docs/plans/2026-04-05-boundary-net-impl.md

## Tasks
- [x] Task 1: BoundaryNet model struct
- [x] Task 2: BoundaryNet dataset encoding
- [x] Task 3: BoundaryNet training loop
- [x] Task 4: Validation metrics (normalized MAE)
- [x] Task 5: CLI commands (train-boundary, eval-boundary)
- [x] Task 6: Boundary evaluator for range-solver
- [ ] Task 7: BoundaryEvaluator struct with batch inference
- [ ] Task 8: Tauri integration for turn solving
- [ ] Task 9: Exploitability comparison command
- [ ] Task 10: Full workspace build/test verification + docs
