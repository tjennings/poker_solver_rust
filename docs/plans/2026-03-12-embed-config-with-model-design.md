# Design: Embed Config With Model, Eliminate Hardcoded Values

**Date:** 2026-03-12
**Status:** Approved

## Problem

The `compare` commands use `DatagenConfig::default()` when no `--config` is provided, which includes `pot_intervals: [[4,20],[20,80],[80,200],[200,400]]`. If the model was trained with different intervals (e.g. `[[4,20],[20,80],[80,200]]`), compare generates out-of-distribution spots, producing misleading results.

Additionally, several values are hardcoded throughout the pipeline:
- Model architecture `7, 500` in `cmd_evaluate` and `cmd_compare`
- Compare seed `42` instead of using `datagen.seed`
- Solver params `500` iterations / `0.005` exploitability in `default_solve_config()`

## Solution

Save the full `CfvnetConfig` alongside the model at training time. Compare commands load it from the model directory — no `--config` flag needed, no defaults to get wrong.

## Changes

### 1. Save config at training time
- After `cmd_train` saves model weights, write `config.yaml` to the output directory
- Contains the full `CfvnetConfig` used for training

### 2. Add `load_model_config` helper
- `load_model_config(model_dir: &Path) -> CfvnetConfig`
- Reads `{model_dir}/config.yaml`, errors with clear message if missing

### 3. `cmd_compare` (river)
- Remove `--config` CLI arg
- Load config from model dir via `load_model_config`
- Use `training.hidden_layers` / `training.hidden_size` for model construction (replaces hardcoded `7, 500`)
- Use `datagen.seed` instead of hardcoded `42`
- Derive board_size from `datagen.street` via `board_cards_for_street()`

### 4. `cmd_compare_net` / `cmd_compare_exact` (turn)
- Remove `--config` CLI arg
- Load config from model dir via `load_model_config`
- Use `datagen.seed` instead of hardcoded `42`

### 5. `cmd_evaluate`
- Load config from model dir (same pattern as compare)
- Use `training.hidden_layers` / `training.hidden_size` (replaces hardcoded `7, 500`)

### 6. `compare.rs` — remove `default_solve_config()`
- `run_comparison` already receives `&DatagenConfig` — pass `solver_iterations` and `target_exploitability` through to the solve config instead of hardcoding `500` / `0.005`

### 7. `compare.rs` — board size from street
- `generate_comparison_spot` derives board_size from `datagen.street` via `board_cards_for_street()` instead of hardcoding `5`

### 8. Delete `load_config_or_default`
- No longer needed — all consumers load from model dir

## What stays unchanged
- Generate and Train CLI (already require `--config`)
- Config structs themselves (no field changes)
- `Default` impls remain for tests
- compare_turn.rs board size `4` (inherently turn-specific)
