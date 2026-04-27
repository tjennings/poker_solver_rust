---
# poker_solver_rust-7civ
title: Review crates/cfvnet/python for GPU-only solver readiness
status: completed
type: task
priority: normal
created_at: 2026-04-27T15:43:18Z
updated_at: 2026-04-27T15:46:35Z
---

Evaluate crates/cfvnet/python against the goal of a Supremus-style GPU-only GTO solver: GPU DCFR subgame solving feeding CFVNet boundary-model training.\n\n- [x] Inspect package structure and entry points\n- [x] Review GPU DCFR/datagen implementation\n- [x] Review CFVNet model/training implementation\n- [x] Summarize risks, gaps, and recommendations

## Summary of Changes\n\nReviewed crates/cfvnet/python via delegated read-only review passes plus local test execution. Found that Python is a BoundaryNet training/export package, not a GPU DCFR implementation. Key risks: CUDA validation samples training buffer, GPU buffer refill can silently copy corrupt zero rows if worker processes fail, multiprocessing forks after CUDA allocation on Linux, GPU sampling is biased when buffer capacity is smaller than the corpus, eval is CPU/default ONNXRuntime oriented, and the Python e2e test currently fails due to strict ONNX tolerance.
