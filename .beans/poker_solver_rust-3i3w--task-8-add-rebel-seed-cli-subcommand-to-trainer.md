---
# poker_solver_rust-3i3w
title: 'Task 8: Add rebel-seed CLI subcommand to trainer'
status: completed
type: task
priority: normal
created_at: 2026-03-21T03:09:56Z
updated_at: 2026-03-21T03:16:23Z
---

## Summary of Changes

Added the `rebel-seed` CLI subcommand to the trainer crate for generating PBS training data from blueprint play for ReBeL offline seeding.

### Changes:
- `crates/trainer/Cargo.toml` — Added `rebel` crate dependency
- `crates/trainer/src/main.rs` — Added `RebelSeed` CLI variant and `run_rebel_seed()` handler
- `crates/rebel/src/generate.rs` — Created PBS generation module with `generate_pbs()` function
- `crates/rebel/src/lib.rs` — Registered the `generate` module
- `sample_configurations/rebel_river_seed.yaml` — Sample ReBeL seed config

### Handler workflow:
1. Parse ReBeL YAML config
2. Load blueprint strategy (`strategy.bin`) from blueprint path
3. Load bucket files from cluster directory (with per-flop auto-detection)
4. Build game tree from blueprint's `config.yaml`
5. Create disk buffer for PBS records
6. Run parallel PBS generation via rayon thread pool
7. Report PBS count and buffer records
