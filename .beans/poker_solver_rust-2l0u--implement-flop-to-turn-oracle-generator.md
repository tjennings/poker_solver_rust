---
# poker_solver_rust-2l0u
title: Implement flop-to-turn oracle generator
status: completed
type: task
priority: high
created_at: 2026-05-08T19:23:49Z
updated_at: 2026-05-08T20:25:38Z
parent: poker_solver_rust-8rxb
---

Add a datagen path that samples flop situations, builds flop games, solves to turn boundaries, evaluates turn leaves with the direct turn-boundary model, and writes direct flop-boundary records with manifest metadata.

## Summary of Changes

Implemented the first flop-to-turn oracle datagen path. It samples flop situations, builds depth-limited flop games with turn boundary leaves, evaluates those turn leaves through the direct turn-boundary ONNX model, writes flop-boundary records/manifests, and documents the pilot command/config.

Verification:
- cargo test -p cfvnet --features onnx flop_boundary --no-default-features
- cargo test -p cfvnet
- cargo test
- cargo check -p cfvnet --features gpu-turn-datagen
- git diff --check
