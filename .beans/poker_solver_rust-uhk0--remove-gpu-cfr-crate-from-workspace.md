---
# poker_solver_rust-uhk0
title: Remove gpu-cfr crate from workspace
status: completed
type: task
priority: normal
created_at: 2026-02-27T19:52:45Z
updated_at: 2026-02-27T20:17:41Z
---

Remove the gpu-cfr crate (GPU-accelerated sequence-form CFR using wgpu) from the workspace. Delete crate directory, update workspace Cargo.toml, trainer dependencies, trainer main.rs GPU code, and documentation.

- [x] Delete crates/gpu-cfr/ directory
- [x] Remove from workspace Cargo.toml members
- [x] Remove from trainer Cargo.toml (dep + feature)
- [x] Remove GPU code from trainer main.rs
- [x] Update docs/training.md
- [x] Update docs/architecture.md
- [x] Update CLAUDE.md crate map
- [x] Verify: cargo build, cargo test, cargo clippy

## Summary of Changes

Removed the gpu-cfr crate entirely from the workspace:
- Deleted crates/gpu-cfr/ directory
- Removed from workspace members in root Cargo.toml
- Removed optional dependency and gpu feature from trainer Cargo.toml
- Removed SolverMode::Gpu variant, tile_size config field, GPU impl blocks, and run_gpu_training() from trainer main.rs
- Updated docs/training.md (6 GPU references removed)
- Updated docs/architecture.md (crate tree entry and description removed)
- Updated CLAUDE.md crate map

Verification: cargo build passes, tests same as baseline (pre-existing stochastic failures only), clippy clean on changed crates (pre-existing tauri-app error unchanged).
