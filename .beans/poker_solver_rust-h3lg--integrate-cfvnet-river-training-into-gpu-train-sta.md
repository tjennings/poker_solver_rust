---
# poker_solver_rust-h3lg
title: Integrate cfvnet river training into gpu-train-stack pipeline
status: todo
type: feature
priority: high
tags:
    - gpu
    - cfvnet
created_at: 2026-03-16T01:36:52Z
updated_at: 2026-03-16T01:36:52Z
parent: poker_solver_rust-twez
---

Wire the existing cfvnet river training (CPU datagen + burn training) as the
river phase of gpu-train-stack. The cfvnet pipeline has proven R(S,p) range
sampling, stratified pot/SPR distribution, and quality training.

Tasks:
- [ ] Call cfvnet datagen + train from stack_config::train_full_stack()
- [ ] Add river-specific config to YAML (pot_intervals, spr_intervals, epochs, target_mae)
- [ ] MAE-based early stopping on validation split
- [ ] gpu-eval-model command for post-training MAE evaluation
- [ ] Ensure river model output is compatible with CudaNetInference for turn leaf eval

Existing trained river model: local_data/models/river_v7/checkpoint_epoch340.mpk.gz
