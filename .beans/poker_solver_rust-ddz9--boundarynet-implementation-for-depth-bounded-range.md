---
# poker_solver_rust-ddz9
title: BoundaryNet implementation for depth-bounded range solving
status: completed
type: task
priority: high
created_at: 2026-04-06T00:52:01Z
updated_at: 2026-04-06T02:14:15Z
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
- [x] Task 7: BoundaryEvaluator struct with batch inference
- [x] Task 8: Tauri integration for turn solving
- [x] Task 9: Exploitability comparison command
- [x] Task 10: Full workspace build/test verification + docs


## Summary of Changes

Implemented the full BoundaryNet pipeline:
- **Model**: `BoundaryNet` struct sharing `HiddenBlock` with `CfvNet`, outputting normalized EVs
- **Encoding**: `encode_boundary_record()` with pot/stack as fractions of total stake
- **Training**: `train_boundary()` with streaming dataloader, cosine LR, checkpointing
- **Validation**: `compute_normalized_mae()` metric, per-SPR bucket breakdown
- **CLI**: `train-boundary`, `eval-boundary`, `compare-boundary` commands
- **Integration**: `NeuralBoundaryEvaluator` implementing `BoundaryEvaluator` trait for range-solver
- **Explorer**: Opt-in boundary model loading in Tauri for turn solving
- **Docs**: Updated architecture.md and training.md
