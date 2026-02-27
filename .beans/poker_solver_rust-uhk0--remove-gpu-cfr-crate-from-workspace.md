---
# poker_solver_rust-uhk0
title: Remove gpu-cfr crate from workspace
status: in-progress
type: task
created_at: 2026-02-27T19:52:45Z
updated_at: 2026-02-27T19:52:45Z
---

Remove the gpu-cfr crate (GPU-accelerated sequence-form CFR using wgpu) from the workspace. Delete crate directory, update workspace Cargo.toml, trainer dependencies, trainer main.rs GPU code, and documentation.

- [ ] Delete crates/gpu-cfr/ directory
- [ ] Remove from workspace Cargo.toml members
- [ ] Remove from trainer Cargo.toml (dep + feature)
- [ ] Remove GPU code from trainer main.rs
- [ ] Update docs/training.md
- [ ] Update docs/architecture.md
- [ ] Update CLAUDE.md crate map
- [ ] Verify: cargo build, cargo test, cargo clippy
