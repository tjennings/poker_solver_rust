---
# poker_solver_rust-obst
title: 'Epic: flop-boundary CFVNet for preflop/flop resolving'
status: in-progress
type: epic
priority: high
created_at: 2026-05-08T19:21:58Z
updated_at: 2026-05-08T19:24:54Z
---

Build the next-street neural boundary model one street earlier than the turn-boundary CFVNet. Generate flop-to-turn oracle data using the trained direct turn-boundary model as the leaf evaluator, train a direct flop-boundary model on 3-card boards, and wire it so preflop/flop boundary solvers can use it.\n\nTarget outcome: a validated direct flop-boundary ONNX model suitable for use as a preflop/flop boundary evaluator.

## Milestones\n\n- poker_solver_rust-2ek4: flop-boundary design and data contract\n- poker_solver_rust-8rxb: flop-to-turn oracle datagen\n- poker_solver_rust-ue4b: flop-boundary dataset validation\n- poker_solver_rust-tz8h: train and export flop-boundary CFVNet\n- poker_solver_rust-mk2k: preflop/flop runtime integration
