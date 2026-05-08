---
# poker_solver_rust-dk9t
title: Map flop oracle design to turn-boundary evaluator
status: completed
type: task
priority: high
created_at: 2026-05-08T19:23:38Z
updated_at: 2026-05-08T19:43:48Z
parent: poker_solver_rust-2ek4
---

Design how flop-root targets are produced from flop subgame solves whose turn leaves use the direct turn-boundary ONNX model. Document sampling, chance averaging, depth limit, exploitability target, and expected target noise from using a learned turn boundary.

## Summary of Changes

Mapped the flop-to-turn oracle design in docs/plans/2026-05-08-flop-boundary-cfvnet-contract.md. Flop data generation should solve flop subgames to turn boundary leaves, evaluate those leaves with the direct turn-boundary ONNX model, extract root CFVs, and write one canonical flop-boundary record per player.
