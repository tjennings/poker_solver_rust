---
# poker_solver_rust-h2yt
title: 'Milestone: runtime evaluator integration'
status: in-progress
type: feature
priority: high
created_at: 2026-05-05T02:57:37Z
updated_at: 2026-05-06T01:22:37Z
parent: poker_solver_rust-fp06
---

Add the runtime inference path for a direct turn-boundary CFVNet while keeping the existing river-enumeration evaluator available as oracle and fallback.

Progress: started direct turn-boundary runtime evaluator integration. Goal is to avoid old river-enumeration ONNX path for 4-card turn-boundary models, then gate with compare-solve boundary CFV diagnostics.
