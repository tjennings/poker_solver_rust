# Remove Legacy Preflop/Postflop Solver Modules

**Date:** 2026-03-11
**Status:** Approved

## Goal

Remove the old LCFR preflop solver (`preflop/`) and the old subgame-based postflop solver (`blueprint/`) along with all CLI commands, tests, benchmarks, configs, and documentation specific to them. The modern `blueprint_v2/` MCCFR system, Tauri explorer, cfvnet GPU module, and simulation (agents + v2 blueprints) must continue to function.

## What Gets Deleted

### Core Crate Modules
- `crates/core/src/preflop/` — entire directory (LCFR solver, postflop model, exhaustive equity, equity caches, bundles, tree, exploitability)
- `crates/core/src/blueprint/` — entire directory (subgame CFR, CBV, range narrower, solver dispatch, cache, full depth solver, old strategy/bundle)

### Trainer CLI Commands (and their handler functions)
- `solve-preflop` / `solve-postflop`
- `diag-buckets`, `trace-hand`, `precompute-equity`, `precompute-equity-delta`
- `dump-tree`, `inspect-preflop`, `trace-regrets`, `decompose-regrets`, `trace-terminals`
- `flops`

### Trainer Support Files
- `tui.rs`, `tui_metrics.rs` — old solve-postflop TUI dashboard
- `bucket_diagnostics.rs` — diag-buckets implementation
- `hand_trace.rs` — trace-hand implementation
- `tree_dump.rs` — dump-tree implementation
- `lhe_viz.rs` — preflop visualization

### Tests
- `crates/core/tests/preflop_convergence.rs`
- `crates/core/tests/preflop_avg_regret_test.rs`
- `crates/core/tests/postflop_imperfect_recall.rs`
- `crates/core/tests/postflop_diagnostics.rs`
- `crates/core/tests/abstraction_integration.rs`
- `crates/trainer/tests/blueprint_tui_integration.rs`

### Benchmarks
- `crates/core/benches/preflop_solver_bench.rs`
- `crates/core/benches/postflop_solve_bench.rs`
- Corresponding `[[bench]]` entries in `crates/core/Cargo.toml`

### Sample Configurations
- All `preflop_*.yaml` files
- All `*postflop*.yaml`, `perf_*.yaml`, `*spr*.yaml`, `*bkt*.yaml` files
- Any configs that reference the old `postflop_model` section exclusively

## What Gets Kept

- `crates/core/src/abstraction/` — used by `blueprint_v2::cluster_pipeline`
- `crates/core/src/blueprint_v2/` — modern full-game MCCFR solver
- `crates/cfvnet/` — GPU training module
- `crates/range-solver/` — exact DCFR for single-spot analysis
- Trainer commands: `train-blueprint`, `cluster`, `diag-clusters`, `range-solve`
- Blueprint v2 TUI: `blueprint_tui.rs`, `blueprint_tui_config.rs`, `blueprint_tui_metrics.rs`, `blueprint_tui_scenarios.rs`, `blueprint_tui_widgets.rs`
- Agent system: `crates/core/src/agent.rs`, `agents/*.toml`
- Game engine, hand eval, hand classes, info keys (refactored)

## What Gets Refactored

### 1. `crates/core/src/simulation.rs`
- **Remove:** `RealTimeSolvingAgent`, `RealTimeSolvingAgentGenerator` and all old `blueprint::*` imports
- **Keep:** `RuleBasedAgent`, `RuleBasedAgentGenerator`, `BlueprintV2Agent`
- **Add:** `BlueprintV2AgentGenerator` that wraps `BlueprintV2Strategy` for simulation lookups (replacing the old real-time solving agent)

### 2. `crates/tauri-app/src/exploration.rs`
- **Remove variants:** `StrategySource::Bundle`, `StrategySource::PreflopSolve`, `StrategySource::SubgameSolve`
- **Remove functions:** `load_preflop_solve_core`, `load_bundle_core`, old bundle loading
- **Remove:** preflop imports in test modules (lines 954, 1045, 1152, 1187)
- **Keep:** `StrategySource::Agent`, `StrategySource::BlueprintV2`
- **Update:** all match arms that handled removed variants

### 3. `crates/tauri-app/src/simulation.rs`
- **Remove:** `blueprint::cbv::CbvTable`, `blueprint::solver_dispatch::SolverConfig`, `blueprint::StrategyBundle` imports
- **Update:** simulation setup to use v2 blueprint + agents only

### 4. `crates/tauri-app/src/lib.rs`
- **Remove:** `load_bundle`, `load_preflop_solve_core`, `load_bundle_core` exports
- **Keep:** `load_blueprint_v2`, agent loading

### 5. `crates/core/src/info_key.rs`
- **Remove:** `use crate::blueprint::{AbstractionModeConfig, BlueprintStrategy}`
- **Refactor:** any functions that depend on old blueprint types (move or remove)

### 6. `crates/core/src/lib.rs`
- **Remove:** `pub mod preflop`, `pub mod blueprint`

### 7. `crates/trainer/src/main.rs`
- **Remove:** all old command enum variants and their handler functions
- **Remove:** all `use poker_solver_core::preflop::*` imports
- **Keep:** `TrainBlueprint`, `Cluster`, `DiagClusters`, `RangeSolve` and v2-specific commands

### 8. `crates/core/Cargo.toml`
- **Remove:** `[[bench]]` entries for `preflop_solver_bench` and `postflop_solve_bench`
- **Review:** dependencies that were only needed by old modules

## Documentation Updates

### `docs/training.md`
- Remove solve-preflop, solve-postflop workflow sections
- Remove diag-buckets, trace-hand, precompute-equity, dump-tree, inspect-preflop, trace-regrets, decompose-regrets, trace-terminals documentation
- Keep train-blueprint, cluster, diag-clusters, range-solve docs

### `docs/architecture.md`
- Remove old LCFR preflop solver pipeline description
- Remove old subgame-based postflop solver description
- Keep blueprint_v2 MCCFR architecture description

### `docs/explorer.md`
- Remove references to old bundle format loading
- Keep blueprint_v2 and agent loading documentation

### `CLAUDE.md`
- Update crate map (remove preflop solver references)
- Update key files list
- Remove preflop solver notes from architecture section
- Update MEMORY.md to reflect new state

## Risk Mitigation

- Run full test suite after each major deletion phase to catch cascading breakage early
- The `abstraction/` module's only external consumer is `blueprint_v2::cluster_pipeline` (via `CanonicalBoard`) — verify this link survives
- `cfvnet` depends on `blueprint_v2::trainer` — verify no transitive dependency on old modules
- `range-solver` is independent — should be unaffected
