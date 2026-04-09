---
# poker_solver_rust-qg6w
title: GPU turn datagen pipeline
status: completed
type: feature
priority: normal
created_at: 2026-04-09T03:33:46Z
updated_at: 2026-04-09T04:28:32Z
---

GPU turn datagen using incremental DCFR with BoundaryNet re-evaluation.

## Summary of Changes
- Incremental solving API on GpuBatchSolver (prepare_batch/run_iterations/extract_results)
- CUDA kernel stride loop for >1024 hands (C(48,2)=1128 turn hands now work on GPU)
- BoundaryNet GPU evaluator via ORT with TensorRT EP (48-river batched inference)
- Reach-based boundary re-evaluation using average strategy from strategy_sum
- compute_reach_at_nodes() factors out forward-walk from compute_evs_from_strategy_sum
- Turn datagen orchestrator in DomainPipeline::run_gpu_turn()
- river_model_path required — no zero-CFV fallback

## Branch
feat/gpu-turn-datagen
