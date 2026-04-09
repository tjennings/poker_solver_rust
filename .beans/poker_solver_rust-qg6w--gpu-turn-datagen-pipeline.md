---
# poker_solver_rust-qg6w
title: GPU turn datagen pipeline
status: completed
type: feature
priority: normal
created_at: 2026-04-09T03:33:46Z
updated_at: 2026-04-09T03:46:59Z
---

GPU turn datagen using incremental DCFR with BoundaryNet re-evaluation.

## Summary of Changes
- Incremental solving API on GpuBatchSolver (prepare_batch/run_iterations/extract_results)
- CUDA kernel accepts start_iteration/end_iteration for partial runs
- BoundaryNet GPU evaluator via ORT with TensorRT EP (48-river batched inference)
- Turn datagen orchestrator in DomainPipeline::run_gpu_turn()
- Sample config at sample_configurations/turn_gpu_datagen.yaml
- Validated: 5 samples → 10 records, correct binary format

## Known Limitations
- Turn games with RSP ranges have ~1128 hands (>1024 CUDA thread limit) → CPU fallback
- Boundary re-evaluation uses root ranges (matches CPU path) — reach-based re-eval is TODO
- ONNX model required for boundary evaluation; zero-CFV fallback when no model

## Branch
feat/gpu-turn-datagen
