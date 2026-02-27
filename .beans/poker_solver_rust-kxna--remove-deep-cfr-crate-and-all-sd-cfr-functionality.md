---
# poker_solver_rust-kxna
title: Remove deep-cfr crate and all SD-CFR functionality
status: completed
type: task
priority: normal
created_at: 2026-02-27T18:54:23Z
updated_at: 2026-02-27T19:10:33Z
---

Remove the deep-cfr crate (neural network SD-CFR solver using candle) and all related code from the workspace.

## Tasks
- [x] Delete crates/deep-cfr/ directory
- [x] Update workspace Cargo.toml (remove deep-cfr from members)
- [x] Update trainer Cargo.toml (remove deep-cfr dependency)
- [x] Delete lhe_diagnose.rs
- [x] Clean up lhe_viz.rs (remove deep-cfr imports and SD-CFR functions)
- [x] Clean up main.rs (remove SD-CFR structs, functions, CLI commands)
- [x] Update documentation (training.md, architecture.md, CLAUDE.md)
- [x] Remove plan docs related to deep-cfr
- [x] Verify: cargo test, clippy, help output


## Summary of Changes

Removed the deep-cfr crate and all SD-CFR functionality:

- Deleted `crates/deep-cfr/` (13 source files, Cargo.toml)
- Deleted `crates/trainer/src/lhe_diagnose.rs`
- Removed `EvalLhe`, `TraceLhe` CLI commands and `SolverMode::SdCfr`
- Removed all `SdCfr*` config structs and `run_sdcfr_*` functions from main.rs
- Cleaned up `lhe_viz.rs` (removed SD-CFR matrix functions, kept preflop solver functions)
- Updated Cargo.toml manifests (workspace + trainer)
- Updated docs: training.md, architecture.md, CLAUDE.md, README.md
- Removed 4 SD-CFR plan docs from docs/plans/
- All 24 trainer tests pass, no new clippy warnings
