---
# poker_solver_rust-t0b3
title: 'Phase 1 slice: integrate sparse storage into blueprint trainer'
status: in-progress
type: task
priority: high
created_at: 2026-06-03T20:05:02Z
updated_at: 2026-06-03T20:05:02Z
parent: poker_solver_rust-kqpn
---

Second implementation slice for Phase 1 lazy/sparse blueprint trainer storage.

Scope:
- Add a production trainer configuration path for selecting dense vs sparse/lazy CFR storage in HU `blueprint_v2`, defaulting to dense unless the existing config conventions clearly support safe opt-in.
- Wire `BlueprintTrainer` and MCCFR execution to use the storage abstraction for the selected backend while preserving existing dense behavior by default.
- Preserve dense `strategy.bin` bundle/export and dense-compatible resume behavior for Explorer/Tauri.
- Ensure sparse storage receives the same optimizer, prediction, baseline, and regret-floor configuration as dense storage or rejects unsupported combinations with explicit errors and tests.
- Surface sparse/dense-equivalent storage instrumentation during training progress or diagnostics for a small trainer run.
- Update `docs/architecture.md` and `docs/training.md` for the new storage backend/config behavior.
- Add tests that run a tiny trainer path with dense and sparse backends and compare dense export/projection where practical.

Non-goals: no strategy pruning, no disk eviction, no lazy GameTree/child realization, no `strategy.bin` format change, no sparse on-disk snapshot default.
