---
# poker_solver_rust-qg6w
title: GPU turn datagen pipeline
status: in-progress
type: feature
created_at: 2026-04-09T03:33:46Z
updated_at: 2026-04-09T03:33:46Z
---

GPU turn datagen using incremental DCFR with BoundaryNet re-evaluation.

## Status
- [x] Design doc: docs/plans/2026-04-08-gpu-turn-datagen-design.md
- [x] Implementation plan: docs/plans/2026-04-08-gpu-turn-datagen.md
- [x] Task 1: Incremental solving API (GpuBatchSolver)
- [x] Task 2: BoundaryNet GPU evaluator (ORT + TensorRT EP)
- [x] Task 3: Turn datagen orchestrator (pipeline.rs)
- [ ] Task 4: Integration test + validation
- [ ] Code review for Task 3

## Known Issues
- Turn games with RSP ranges have ~1128 hands, exceeding CUDA 1024-thread block limit
- CPU fallback handles this; GPU optimization deferred to future work

## Branch
feat/gpu-turn-datagen
