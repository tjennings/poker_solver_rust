---
# poker_solver_rust-ofou
title: 'Phase 3: Turn CFVNet Training on GPU'
status: completed
type: feature
priority: normal
tags:
    - gpu
    - cfvnet
created_at: 2026-03-15T04:14:34Z
updated_at: 2026-03-15T19:31:44Z
parent: poker_solver_rust-twez
---

GPU datagen for turn subgames using Phase 2 river model as leaf evaluator.

Tasks:
- [ ] leaf_eval kernel: batch-invoke river CFVNet at river-boundary nodes
- [ ] Shared CUDA context between cudarc solver and burn-cuda inference
- [ ] Turn datagen: solve random turn subgames with river model at leaves
- [ ] Turn model training on GPU
- [ ] CLI: gpu-datagen turn --river-model <path>, gpu-train turn

Requires chance node support in the flat tree builder for 
turn→river transitions.
