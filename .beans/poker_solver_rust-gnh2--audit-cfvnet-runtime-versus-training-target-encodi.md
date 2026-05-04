---
# poker_solver_rust-gnh2
title: Audit CFVNet runtime versus training target encoding
status: in-progress
type: bug
priority: high
created_at: 2026-05-04T14:46:28Z
updated_at: 2026-05-04T14:48:20Z
parent: poker_solver_rust-e90m
---

CFVNet root attribution sweep shows broad positive bet/all-in pressure on both paired-K and non-paired Js turn spots. Audit whether the ONNX evaluator runtime input/target conventions match the datagen/training target consumed by range-solver boundaries.\n\n## Tasks\n\n- [x] Map runtime ONNX boundary inputs and output normalization in NeuralBoundaryEvaluator.\n- [ ] Map datagen/training target construction for the river boundary model checkpoint.\n- [ ] Compare board/action-history, pot, remaining-stack, player/reach, and CFV unit conventions.\n- [ ] Identify the first concrete fix or validation experiment.


## 2026-05-04 Runtime Map

Runtime ONNX evaluator path is cfvnet::eval::boundary_evaluator::NeuralBoundaryEvaluator. Inputs are 1326 OOP reach, 1326 IP reach, board one-hot, rank presence, pot/(pot+remaining_stack), remaining_stack/(pot+remaining_stack), and player id. For turn boundaries, runtime enumerates all valid rivers, zeroes river-blocked combos in both ranges, runs the river model for each river, and averages per combo.

Potential convention conflict found: ONNX runtime treats model output as BoundaryNet normalized target and rescales by (pot + remaining_stack) / pot, which produces the stored pot-relative CFV target. The Burn runtime path multiplies output by (pot + remaining_stack). The GPU boundary-eval path used by datagen continuation code also multiplies net output by (pot + effective_stack). This means there are at least two incompatible output-denormalization conventions in the tree; the active compare-solve/Tauri ONNX path is the smaller pot-relative one.
