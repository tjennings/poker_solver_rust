---
# poker_solver_rust-bvcw
title: 'Milestone: train turn-boundary CFVNet'
status: in-progress
type: feature
priority: high
created_at: 2026-05-05T02:55:44Z
updated_at: 2026-05-05T04:24:53Z
parent: poker_solver_rust-fp06
---

Train the turn-boundary CFVNet from oracle data, initially reusing the BoundaryNet/CfvNet architecture and training infrastructure where possible.\n\n## Acceptance\n\n- Training config exists for turn-boundary data.\n- Model can train, checkpoint, validate, and export ONNX.\n- Validation is stratified and reports weighted CFV error by important poker strata.\n- Training can consume large sharded datasets without loading everything into memory.

Started the training milestone with dataset-contract enforcement and a turn-boundary training config. Remaining work: first train/export run and stratified validation metrics.
