---
# poker_solver_rust-qy46
title: Rake support + blueprint name
status: completed
type: feature
priority: normal
created_at: 2026-03-08T18:02:38Z
updated_at: 2026-03-08T18:23:20Z
---

Add rake_rate, rake_cap to blueprint MCCFR and range solver. Add required name field to blueprint config for Tauri UX. Plan: docs/plans/2026-03-08-rake-support-impl.md


## Summary of Changes

### Core (`crates/core/src/blueprint_v2/`)
- **config.rs**: Added `name: String` (required), `rake_rate: f64` (default 0.0), `rake_cap: f64` (default 0.0) to `GameConfig`
- **mccfr.rs**: `terminal_value()` now applies `rake = min(pot * rate, cap)` — winner pays rake, loser pays full investment, ties split rake. Threaded through `traverse_external`/`traverse_traverser`/`traverse_opponent`.
- **trainer.rs**: `run_batch` passes `config.game.rake_rate/rake_cap` to traversal

### Tauri Backend (`crates/tauri-app/src/`)
- **exploration.rs**: `BundleInfo` has `rake_rate`/`rake_cap` fields; `BlueprintListEntry` uses config name; BlueprintV2 arms populate name/rake from config
- **postflop.rs**: `PostflopConfig` has `rake_rate`/`rake_cap`; `TreeConfig` uses config values instead of hardcoded 0.0

### Frontend (`frontend/src/`)
- **types.ts**: `BundleInfo`, `BlueprintConfig`, `PostflopConfig`, `PreflopRanges` all carry `rake_rate`/`rake_cap`
- **PostflopExplorer.tsx**: Populates rake from blueprint, shows in config card, editable in config modal
- **Explorer.tsx**: Shows bundle name and rake in action strip; passes rake through BlueprintConfig

### Sample Configs
- All 3 blueprint_v2 YAML configs have required `name` field
