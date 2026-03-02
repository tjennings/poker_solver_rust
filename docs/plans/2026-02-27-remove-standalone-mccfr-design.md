# Design: Remove Standalone MCCFR Solver

**Date:** 2026-02-27
**Goal:** Remove the standalone MCCFR/sequence postflop solver while preserving solve-preflop, solve-postflop, and the Tauri explorer app.

## Context

The project has two postflop solving pipelines:
1. **Preflop LCFR + 169-hand postflop pipeline** — the primary solver (solve-preflop, solve-postflop commands)
2. **Standalone MCCFR** — a generic Game-trait-based solver using HunlPostflop with EHS2/HandClassV2 abstractions (train command)

Pipeline #2 is no longer needed. Its removal simplifies the codebase significantly.

## What Gets Deleted

### Trainer CLI Subcommands

| Command | Reason |
|-|-|
| `train` | Standalone MCCFR/sequence training |
| `generate-deals` | Deal generation for sequence solver |
| `merge-deals` | Batch merging for sequence solver |
| `inspect-deals` | Deal inspection for sequence solver |
| `tree-stats` | Game tree size estimation (uses HunlPostflop) |
| `tree` | Bundle tree explorer (reads MCCFR bundles) |

### Core Files — Delete Entirely

| File | Reason |
|-|-|
| `cfr/mccfr.rs` | Standalone MCCFR solver |
| `cfr/sequence_cfr.rs` | Sequence-form solver |
| `cfr/game_tree.rs` | Game tree materialization |
| `cfr/exploitability.rs` | Only used by standalone CFR tests |
| `cfr/vanilla.rs` | Only used by standalone CFR tests |
| `cfr/convergence.rs` | Only used by standalone train loop |
| `game/hunl_postflop.rs` | HunlPostflop struct + Game impl (types extracted first) |
| `game/kuhn.rs` | Kuhn poker — test-only game |
| `game/limit_holdem.rs` | Limit hold'em — unused |
| `abstract_game.rs` | Exhaustive deal enumeration for sequence solver |
| `tree.rs` | Tree display for `tree` command |

### Core Files — Reduce

- `cfr/mod.rs` → keep only `regret.rs` export (used by `blueprint/subgame_cfr.rs`)
- `game/mod.rs` → remove Game trait, keep `Action`, `Player`, `Actions`, `ALL_IN`, `MAX_ACTIONS`, re-export from new `config.rs`

### Sample Configs — Delete

Standalone MCCFR configs: `smoke.yaml`, `ultra_fast.yaml`, `fast_buckets.yaml`, `full.yaml`, `AKQr_vs_234r.yaml`, `mccfr_smoke.yaml`

### Examples & Tests — Delete

- `examples/bench_mccfr.rs` and `examples/hand_class_histogram.rs` (if dependent on deleted code)
- Integration tests in `crates/core/tests/` that test standalone MCCFR

## What Gets Created

- `game/config.rs` — `PostflopConfig` and `AbstractionMode` extracted from `hunl_postflop.rs`

## What Stays Untouched

- `preflop/` entire module (has its own internal MCCFR in `postflop_mccfr.rs`)
- `blueprint/` (strategy, bundle, subgame, subgame_cfr, subgame_tree)
- `abstraction/`
- `info_key.rs`, `hand_class.rs`, `showdown_equity.rs`
- `agent.rs`, `simulation.rs`, `flops.rs`, `hands.rs`
- Tauri app and devserver
- Trainer: `solve-preflop`, `solve-postflop`, `flops`, `trace-hand`, `diag-buckets`
- Sample configs: `minimal_postflop.yaml`, `minimal_preflop.yaml`, `preflop_medium.yaml`

## Entangled Dependency

`PostflopConfig` and `AbstractionMode` are defined in `hunl_postflop.rs` but used by `blueprint/bundle.rs`, `simulation.rs`, and the Tauri app. Solution: extract these types into `game/config.rs` before deleting `hunl_postflop.rs`.

## Verification

After removal, all of these must pass:
- `cargo test`
- `cargo clippy`
- `cargo build -p poker-solver-tauri-app`
