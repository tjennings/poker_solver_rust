---
# poker_solver_rust-9v41
title: 'Phase 2: River CFVNet Training on GPU'
status: in-progress
type: feature
priority: high
tags:
    - gpu
    - cfvnet
created_at: 2026-03-15T04:14:26Z
updated_at: 2026-03-15T04:33:09Z
parent: poker_solver_rust-twez
---

Generate river training data and train river CFVNet model entirely on GPU.

Tasks:
- [ ] GPU batch solver: solve N independent river subgames in parallel
- [ ] Random situation sampling (board, ranges, pot/stack)
- [ ] CFV extraction at root as training targets
- [ ] GPU-resident training data pipeline (no host round-trip)
- [ ] River CFVNet training loop via burn-cuda
- [ ] CLI: gpu-datagen river, gpu-train river
- [ ] Compare validation loss vs CPU-trained model
- [ ] Benchmark: training examples/second vs current pipeline

This is where GPU parallelism pays off — batch-solving 1000+ 
independent subgames simultaneously saturates GPU cores.
