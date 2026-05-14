---
# poker_solver_rust-6zw6
title: Audit CFVNet target-unit alignment end to end
status: in-progress
type: bug
priority: high
created_at: 2026-05-14T04:46:09Z
updated_at: 2026-05-14T04:46:09Z
parent: poker_solver_rust-lnpl
---

Trace a small set of boundary records through Rust datagen, Python loading, model target scaling, ONNX output interpretation, Rust evaluator conversion, and compare-solve oracle comparison. Prove the same value unit is used at every boundary before more large training runs.
