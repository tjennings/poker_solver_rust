---
# poker_solver_rust-gnh2
title: Audit CFVNet runtime versus training target encoding
status: in-progress
type: bug
priority: high
created_at: 2026-05-04T14:46:28Z
updated_at: 2026-05-04T14:49:11Z
parent: poker_solver_rust-e90m
---

CFVNet root attribution sweep shows broad positive bet/all-in pressure on both paired-K and non-paired Js turn spots. Audit whether the ONNX evaluator runtime input/target conventions match the datagen/training target consumed by range-solver boundaries.\n\n## Tasks\n\n- [x] Map runtime ONNX boundary inputs and output normalization in NeuralBoundaryEvaluator.\n- [ ] Map datagen/training target construction for the river boundary model checkpoint.\n- [x] Compare board/action-history, pot, remaining-stack, player/reach, and CFV unit conventions.\n- [ ] Identify the first concrete fix or validation experiment.


## 2026-05-04 Runtime Map

Runtime ONNX evaluator path is cfvnet::eval::boundary_evaluator::NeuralBoundaryEvaluator. Inputs are 1326 OOP reach, 1326 IP reach, board one-hot, rank presence, pot/(pot+remaining_stack), remaining_stack/(pot+remaining_stack), and player id. For turn boundaries, runtime enumerates all valid rivers, zeroes river-blocked combos in both ranges, runs the river model for each river, and averages per combo.

Potential convention conflict found: ONNX runtime treats model output as BoundaryNet normalized target and rescales by (pot + remaining_stack) / pot, which produces the stored pot-relative CFV target. The Burn runtime path multiplies output by (pot + remaining_stack). The GPU boundary-eval path used by datagen continuation code also multiplies net output by (pot + effective_stack). This means there are at least two incompatible output-denormalization conventions in the tree; the active compare-solve/Tauri ONNX path is the smaller pot-relative one.


## 2026-05-04 Training Target Map

River training records store cfvs as pot-relative break-even-centered values: (ev_chips - half_pot) / half_pot. The Python BoundaryNet encoder then trains the model on target = stored_cfv * pot / (pot + effective_stack), with the same pot/(pot+stack), stack/(pot+stack), board one-hot, rank-presence, and player features used by the active ONNX runtime.

Comparison so far:

- Board and player features match between Python training and ONNX runtime.
- Pot/stack feature normalization matches: pot/(pot+stack), stack/(pot+stack).
- Runtime uses both OOP/IP reach vectors and zeroes river-blocked combos during turn-to-river enumeration, matching the intended river-model input shape.
- Active ONNX output inversion returns the stored pot-relative CFV, which is consistent with Python training tests and comments.
- However, Burn NeuralBoundaryEvaluator and gpu_boundary_eval use different output denormalization comments/logic, including multiplying by pot+stack. That is likely stale or for a different consumer, but it is a real convention fork that should be cleaned up or locked by tests.

Current read: the active Tauri/compare-solve ONNX path is probably not suffering a simple scaling inversion bug. The broad root bet-pressure failure is more likely model accuracy/domain coverage, especially because exact_subtree controls are much closer while using the same range-solver boundary injection path.
