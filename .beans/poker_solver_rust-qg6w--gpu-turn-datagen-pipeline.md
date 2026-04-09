---
# poker_solver_rust-qg6w
title: GPU turn datagen pipeline
status: in-progress
type: feature
priority: normal
created_at: 2026-04-09T03:33:46Z
updated_at: 2026-04-09T03:38:55Z
---

GPU turn datagen using incremental DCFR with BoundaryNet re-evaluation.

## Status
- [x] Design doc: docs/plans/2026-04-08-gpu-turn-datagen-design.md
- [x] Implementation plan: docs/plans/2026-04-08-gpu-turn-datagen.md
- [x] Task 1: Incremental solving API (GpuBatchSolver) — reviewed, fixes applied
- [x] Task 2: BoundaryNet GPU evaluator (ORT + TensorRT EP) — reviewed, fixes applied
- [x] Task 3: Turn datagen orchestrator (pipeline.rs) — reviewed, critical bug fixed
- [ ] Task 4: Integration test + validation

## Known Issues
- Turn games with RSP ranges have ~1128 hands, exceeding CUDA 1024-thread block limit
- CPU fallback handles this; GPU optimization deferred to future work
- Boundary re-evaluation currently uses root ranges (same as CPU path) — TODO: use reach at boundary nodes

## Branch
feat/gpu-turn-datagen
