---
# poker_solver_rust-gnh2
title: Audit CFVNet runtime versus training target encoding
status: in-progress
type: bug
priority: high
created_at: 2026-05-04T14:46:28Z
updated_at: 2026-05-04T14:46:28Z
parent: poker_solver_rust-e90m
---

CFVNet root attribution sweep shows broad positive bet/all-in pressure on both paired-K and non-paired Js turn spots. Audit whether the ONNX evaluator runtime input/target conventions match the datagen/training target consumed by range-solver boundaries.\n\n## Tasks\n\n- [ ] Map runtime ONNX boundary inputs and output normalization in NeuralBoundaryEvaluator.\n- [ ] Map datagen/training target construction for the river boundary model checkpoint.\n- [ ] Compare board/action-history, pot, remaining-stack, player/reach, and CFV unit conventions.\n- [ ] Identify the first concrete fix or validation experiment.
