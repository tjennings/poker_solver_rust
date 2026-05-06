---
# poker_solver_rust-h2yt
title: 'Milestone: runtime evaluator integration'
status: completed
type: feature
priority: high
created_at: 2026-05-05T02:57:37Z
updated_at: 2026-05-06T01:43:32Z
parent: poker_solver_rust-fp06
---

Add the runtime inference path for a direct turn-boundary CFVNet while keeping the existing river-enumeration evaluator available as oracle and fallback.

Progress: started direct turn-boundary runtime evaluator integration. Goal is to avoid old river-enumeration ONNX path for 4-card turn-boundary models, then gate with compare-solve boundary CFV diagnostics.


## Summary of Changes 2026-05-06

Implemented explicit cfvnet boundary inference contracts. Existing behavior remains the default river_enumerated_turn adapter for legacy river models; new direct mode evaluates the supplied 4-card turn boundary board directly. The ONNX evaluator now batches OOP/IP rows together, so direct compare-solve performs one ONNX session run per boundary cache fill. Wired the mode through StreetBoundaryConfig, compare-solve CLI --*-model-kind flags, Tauri solver setup, frontend types, and docs. Verified with targeted evaluator/config tests, a full cargo test --workspace --quiet pass, and a compare-solve smoke using --river-model-kind direct against checkpoint_epoch200.onnx.
