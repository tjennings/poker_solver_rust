---
# poker_solver_rust-5r6c
title: Fix flop boundary gpu-turn-datagen feature gate
status: completed
type: bug
priority: normal
created_at: 2026-05-08T20:24:21Z
updated_at: 2026-05-08T20:24:53Z
---

Review found gpu-turn-datagen enables onnx-gpu but not onnx, so flop_boundary_generate.rs compiles the disabled stub despite ORT-capable feature selection.

- [x] Wire gpu-turn-datagen to enable the onnx-gated implementation
- [x] Run targeted compile/test verification

## Summary of Changes

Changed cfvnet's onnx-gpu feature to include the crate-level onnx feature, so gpu-turn-datagen activates the onnx-gated flop-boundary implementation. Verified with cargo check -p cfvnet --features gpu-turn-datagen.
