---
# poker_solver_rust-ad96
title: Add MP tree/storage sizing preflight
status: completed
type: task
priority: high
created_at: 2026-05-07T13:18:39Z
updated_at: 2026-05-07T13:51:10Z
parent: poker_solver_rust-5kvv
---

Add a lightweight sizing/preflight mode for Blueprint MP configs that reports stack depth, action-depth rows, estimated/eager tree nodes, storage slots, and virtual bytes before committing huge mmap/storage. Use it to fail fast with a useful diagnostic when eager mode is selected for a 100bb-scale tree.

## Work Started

Starting implementation of MP config sizing/preflight so 100bb eager-mode configs report tree/storage scale and can fail fast before dense allocation.

## Summary of Changes

- Added inspect-mp-config CLI command for Blueprint MP configs.
- Reports effective stack in BB, bucket counts, action row counts, and eager-backend risk warnings without building the dense tree.
- Added train-blueprint-mp fail-fast guard for the known unsafe 100bb 6-max multi-preflop-raise-row eager backend pattern.
- Documented inspect-mp-config in docs/training.md.
- Created follow-up poker_solver_rust-nycm for capped exact tree-size counting.

## Verification

- cargo test -q -p poker-solver-trainer inspect_mp_config_cli_parses
- cargo test -q -p poker-solver-trainer inspect_mp_config_flags_100bb_multi_preflop_raise_rows
- cargo run -q -p poker-solver-trainer --release -- inspect-mp-config --config sample_configurations/blueprint_mp_6max_500f_100t_100r.yaml
- git diff --check -- crates/trainer/src/main.rs docs/training.md
