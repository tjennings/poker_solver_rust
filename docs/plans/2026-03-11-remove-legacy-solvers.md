# Remove Legacy Solvers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Remove the legacy preflop LCFR solver (`preflop/`) and old subgame-based postflop solver (`blueprint/`) while keeping `blueprint_v2/`, `abstraction/`, Tauri UX, cfvnet, and simulation (agents + v2 blueprints) fully functional.

**Architecture:** Bottom-up deletion — start with leaf consumers (CLI commands, tests, benchmarks, configs), then remove the core modules, then refactor the remaining consumers (simulation, exploration, info_key) to stop importing deleted types.

**Tech Stack:** Rust, Tauri, Cargo

---

### Task 1: Delete Old Tests and Benchmarks

**Files:**
- Delete: `crates/core/tests/preflop_convergence.rs`
- Delete: `crates/core/tests/preflop_avg_regret_test.rs`
- Delete: `crates/core/tests/postflop_imperfect_recall.rs`
- Delete: `crates/core/tests/postflop_diagnostics.rs`
- Delete: `crates/core/tests/abstraction_integration.rs`
- Delete: `crates/trainer/tests/blueprint_tui_integration.rs`
- Delete: `crates/core/benches/preflop_solver_bench.rs`
- Delete: `crates/core/benches/postflop_solve_bench.rs`
- Modify: `crates/core/Cargo.toml` (remove `[[bench]]` entries at lines 36-38 and 44-46)

**Step 1: Delete test files**

```bash
rm crates/core/tests/preflop_convergence.rs
rm crates/core/tests/preflop_avg_regret_test.rs
rm crates/core/tests/postflop_imperfect_recall.rs
rm crates/core/tests/postflop_diagnostics.rs
rm crates/core/tests/abstraction_integration.rs
rm crates/trainer/tests/blueprint_tui_integration.rs
```

**Step 2: Delete bench files**

```bash
rm crates/core/benches/preflop_solver_bench.rs
rm crates/core/benches/postflop_solve_bench.rs
```

**Step 3: Remove `[[bench]]` entries from Cargo.toml**

Remove the `[[bench]]` sections for `preflop_solver_bench` and `postflop_solve_bench` from `crates/core/Cargo.toml` (lines ~36-38 and ~44-46). Keep the `equity_table_bench` entry.

**Step 4: Commit**

```bash
git add -A && git commit -m "chore: delete legacy preflop/postflop tests and benchmarks"
```

---

### Task 2: Delete Old Sample Configurations

**Files:**
- Delete: `sample_configurations/preflop_on_4spr.yaml`
- Delete: `sample_configurations/preflop_medium.yaml`
- Delete: `sample_configurations/preflop_full_model.yaml`
- Delete: `sample_configurations/minimal_preflop.yaml`
- Delete: `sample_configurations/AKQr_vs_234r_preflop.yaml`
- Delete: `sample_configurations/AKQr_vs_234r_postflop.yaml`
- Delete: `sample_configurations/minimal_postflop.yaml`
- Delete: `sample_configurations/perf_postflop.yaml`
- Delete: `sample_configurations/perf_test_1flop.yaml`
- Delete: `sample_configurations/perf_test_2flop_2spr.yaml`
- Delete: `sample_configurations/20spr.yaml`
- Delete: `sample_configurations/100bkt.yaml`
- Delete: `sample_configurations/600bkt.yaml`
- Delete: `sample_configurations/equity_delta.yaml`

Keep: `blueprint_v2_*.yaml`, `river_cfvnet.yaml`, `debug.yaml`, `full.yaml`, `test.yaml`, `tiny.yaml`

**Step 1: Delete old configs**

```bash
cd sample_configurations
rm preflop_on_4spr.yaml preflop_medium.yaml preflop_full_model.yaml minimal_preflop.yaml
rm AKQr_vs_234r_preflop.yaml AKQr_vs_234r_postflop.yaml minimal_postflop.yaml
rm perf_postflop.yaml perf_test_1flop.yaml perf_test_2flop_2spr.yaml
rm 20spr.yaml 100bkt.yaml 600bkt.yaml equity_delta.yaml
```

**Step 2: Check remaining configs don't reference old `postflop_model` section**

Grep remaining YAML files for `postflop_model` or `preflop`. If any reference old fields, edit them.

**Step 3: Commit**

```bash
git add -A && git commit -m "chore: delete legacy preflop/postflop sample configurations"
```

---

### Task 3: Delete Trainer Support Files for Old Commands

**Files:**
- Delete: `crates/trainer/src/tui.rs` (old solve-postflop TUI — NOT the blueprint_tui_*.rs files)
- Delete: `crates/trainer/src/tui_metrics.rs`
- Delete: `crates/trainer/src/bucket_diagnostics.rs`
- Delete: `crates/trainer/src/hand_trace.rs`
- Delete: `crates/trainer/src/tree_dump.rs`
- Delete: `crates/trainer/src/lhe_viz.rs`

**Step 1: Delete files**

```bash
rm crates/trainer/src/tui.rs
rm crates/trainer/src/tui_metrics.rs
rm crates/trainer/src/bucket_diagnostics.rs
rm crates/trainer/src/hand_trace.rs
rm crates/trainer/src/tree_dump.rs
rm crates/trainer/src/lhe_viz.rs
```

**Step 2: Remove `mod` declarations from trainer main.rs**

In `crates/trainer/src/main.rs`, remove `mod` declarations for `tui`, `tui_metrics`, `bucket_diagnostics`, `hand_trace`, `tree_dump`, `lhe_viz`. These are near the top of the file.

**Step 3: Commit**

```bash
git add -A && git commit -m "chore: delete legacy trainer support modules (tui, diagnostics, lhe_viz)"
```

---

### Task 4: Remove Old CLI Commands from Trainer

**Files:**
- Modify: `crates/trainer/src/main.rs`

This is the largest refactoring task. Remove these command variants and their dispatch + handler functions:

**Step 1: Remove old command enum variants**

From the `Commands` enum, remove these variants:
- `SolvePreflop` (lines ~53-72)
- `SolvePostflop` (lines ~74-84)
- `Flops` (lines ~86-93)
- `DiagBuckets` (lines ~95-105)
- `TraceHand` (lines ~107-111)
- `PrecomputeEquity` (lines ~115-119)
- `DumpTree` (lines ~121-137)
- `InspectPreflop` (lines ~139-149)
- `TraceRegrets` (lines ~151-164)
- `DecomposeRegrets` (lines ~166-179)
- `TraceTerminals` (lines ~193-206)
- `PrecomputeEquityDelta` (lines ~228-232)

**Step 2: Remove dispatch arms**

In the `match` block (around line 291+), remove the `Commands::SolvePreflop`, `Commands::SolvePostflop`, `Commands::Flops`, `Commands::DiagBuckets`, `Commands::TraceHand`, `Commands::PrecomputeEquity`, `Commands::DumpTree`, `Commands::InspectPreflop`, `Commands::TraceRegrets`, `Commands::DecomposeRegrets`, `Commands::TraceTerminals`, `Commands::PrecomputeEquityDelta` match arms.

**Step 3: Remove handler functions**

Delete the following functions:
- `run_solve_preflop` (starts ~line 1273)
- `run_solve_postflop` (starts ~line 1249)
- Any helper functions that only served the removed commands (inspect_preflop, trace_regrets, decompose_regrets, trace_terminals, etc.)

**Step 4: Remove old imports**

Remove all `use poker_solver_core::preflop::*` import lines (lines ~25-35). Remove any `use poker_solver_core::blueprint::*` imports (not `blueprint_v2`).

**Step 5: Remove dead helper structs/functions**

Remove `PreflopSolveConfig` (or similar wrapper struct, ~line 879) and any other types that only existed for old commands.

**Step 6: Verify compilation**

```bash
cargo check -p poker-solver-trainer
```

**Step 7: Commit**

```bash
git add -A && git commit -m "refactor: remove legacy CLI commands from trainer (solve-preflop, solve-postflop, diagnostics)"
```

---

### Task 5: Delete Core Preflop Module

**Files:**
- Delete: `crates/core/src/preflop/` (entire directory)
- Modify: `crates/core/src/lib.rs` (remove `pub mod preflop` at line 28)

**Step 1: Delete the preflop directory**

```bash
rm -rf crates/core/src/preflop
```

**Step 2: Remove module declaration**

In `crates/core/src/lib.rs`, remove the line `pub mod preflop;`.

**Step 3: Verify compilation**

```bash
cargo check -p poker-solver-core 2>&1 | head -40
```

This will likely show errors in `simulation.rs`, `info_key.rs`, and `exploration.rs` — those are fixed in later tasks.

**Step 4: Commit**

```bash
git add -A && git commit -m "refactor: delete legacy preflop solver module"
```

---

### Task 6: Delete Core Blueprint Module

**Files:**
- Delete: `crates/core/src/blueprint/` (entire directory)
- Modify: `crates/core/src/lib.rs` (remove `pub mod blueprint` at line 17)

**Step 1: Delete the blueprint directory**

```bash
rm -rf crates/core/src/blueprint
```

**Step 2: Remove module declaration**

In `crates/core/src/lib.rs`, remove the line `pub mod blueprint;`.

**Step 3: Commit (will not compile yet)**

```bash
git add -A && git commit -m "refactor: delete legacy blueprint solver module"
```

---

### Task 7: Refactor simulation.rs — Remove Old Agents

**Files:**
- Modify: `crates/core/src/simulation.rs`

**Step 1: Remove old blueprint imports**

Remove these lines (30-34):
```rust
use crate::blueprint::cbv::CbvTable;
use crate::blueprint::full_depth_solver::{self, FullDepthConfig, rs_poker_card_to_id};
use crate::blueprint::range_narrower::RangeNarrower;
use crate::blueprint::solver_dispatch::{self, SolverChoice, SolverConfig, Street};
use crate::blueprint::{BlueprintStrategy, BundleConfig};
```

**Step 2: Remove BlueprintAgent and BlueprintAgentGenerator**

Delete the entire `BlueprintAgent` struct, its `impl` blocks, `BlueprintAgentGenerator`, and all associated code (lines ~108-254). These use old `BlueprintStrategy` and `BundleConfig`.

**Step 3: Remove RealTimeSolvingAgent and RealTimeSolvingAgentGenerator**

Delete the entire `RealTimeSolvingAgent` struct (lines ~273-302), all its `impl` blocks (lines ~304-680), `RealTimeSolvingAgentGenerator` struct and its impls (lines ~684-735).

**Step 4: Remove old agent tests**

Delete all tests that reference `BlueprintAgent`, `BlueprintAgentGenerator`, `RealTimeSolvingAgent`, or `RealTimeSolvingAgentGenerator` (lines ~1400-1754 approximately — inspect the test module to find exact range).

**Step 5: Clean up unused imports**

After removing the old agents, some imports may become unused. Remove:
- `use crate::info_key::{InfoKey, canonical_hand_index, compute_hand_bits_v2, spr_bucket}` — if only used by old agents
- Any other now-dead imports

Keep imports used by `RuleBasedAgent` and the generic `run_simulation` function.

**Step 6: Verify exports**

Ensure `pub` exports still include: `RuleBasedAgentGenerator`, `SimProgress`, `SimResult`, `run_simulation`. Remove from public API: `BlueprintAgentGenerator`, `RealTimeSolvingAgentGenerator`, `RealTimeSolvingAgent`.

**Step 7: Verify compilation**

```bash
cargo check -p poker-solver-core 2>&1 | head -40
```

**Step 8: Commit**

```bash
git add -A && git commit -m "refactor: remove legacy blueprint and real-time solving agents from simulation"
```

---

### Task 8: Refactor info_key.rs — Remove Old Blueprint Dependencies

**Files:**
- Modify: `crates/core/src/info_key.rs`

**Step 1: Remove old import**

Remove line 16: `use crate::blueprint::{AbstractionModeConfig, BlueprintStrategy};`

**Step 2: Handle AbstractionModeConfig usage**

The `describe_key`, `parse_hand_bits`, and `hand_label_from_bits` functions use `AbstractionModeConfig`. Two options:

**Option A (recommended):** Inline a simple enum directly in `info_key.rs`:
```rust
/// Which abstraction mode to use for key interpretation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AbstractionMode {
    #[default]
    Ehs2,
    HandClassV2,
}

impl AbstractionMode {
    pub fn is_hand_class(self) -> bool {
        matches!(self, Self::HandClassV2)
    }
}
```

Then rename all `AbstractionModeConfig` references in the file to `AbstractionMode`.

**Step 3: Remove BlueprintStrategy parameter**

The `describe_key` function takes `blueprint: Option<&BlueprintStrategy>`. Change this to `action_count: Option<usize>` (the only thing it uses from the blueprint is the number of actions). Update callers.

**Step 4: Verify compilation**

```bash
cargo check -p poker-solver-core 2>&1 | head -40
```

**Step 5: Commit**

```bash
git add -A && git commit -m "refactor: decouple info_key from legacy blueprint types"
```

---

### Task 9: Refactor Tauri exploration.rs — Remove Old Strategy Sources

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs`
- Modify: `crates/tauri-app/src/lib.rs`

**Step 1: Remove old imports**

From `exploration.rs`, remove imports of:
- `poker_solver_core::blueprint::*` (anything from the old blueprint module)
- `poker_solver_core::preflop::*`
- `lru::LruCache` (if only used by SubgameSolve)

**Step 2: Remove old StrategySource variants**

From the `StrategySource` enum, delete:
- `Bundle { config, blueprint }` variant
- `PreflopSolve { config, strategy, hand_avg_values }` variant
- `SubgameSolve { blueprint, blueprint_config, subgame_config, solve_cache }` variant

Keep:
- `Agent(AgentConfig)`
- `BlueprintV2 { config, strategy, tree, decision_map }`

**Step 3: Remove old loading functions**

Delete these functions:
- `load_bundle_core` (starts ~line 249)
- `load_preflop_solve_core` (starts ~line 375)
- Any subgame-solve loading functions

**Step 4: Update all match arms**

Search the file for every `match` on `StrategySource` and remove arms for `Bundle`, `PreflopSolve`, `SubgameSolve`. Many of these will be in `get_strategy_matrix`, `get_bundle_info`, `get_action_labels`, `get_position_info`, etc.

**Step 5: Remove preflop test imports**

Remove `use poker_solver_core::preflop::*` from the `#[cfg(test)]` module (lines ~954, 1045, 1152, 1187) and delete or update any tests that used old strategy sources.

**Step 6: Update lib.rs exports**

In `crates/tauri-app/src/lib.rs`, remove exports for:
- `load_bundle`, `load_bundle_core`
- `load_preflop_solve_core`
Keep: `load_blueprint_v2`, `load_blueprint_v2_core`, agent loading.

**Step 7: Verify compilation**

```bash
cargo check -p poker-solver-tauri 2>&1 | head -40
```

**Step 8: Commit**

```bash
git add -A && git commit -m "refactor: remove legacy strategy sources from Tauri explorer"
```

---

### Task 10: Refactor Tauri simulation.rs — Remove Old Blueprint Dependencies

**Files:**
- Modify: `crates/tauri-app/src/simulation.rs`

**Step 1: Remove old imports**

Remove:
```rust
use poker_solver_core::blueprint::cbv::CbvTable;
use poker_solver_core::blueprint::solver_dispatch::SolverConfig;
use poker_solver_core::blueprint::StrategyBundle;
```

Remove import of `RealTimeSolvingAgentGenerator` from `poker_solver_core::simulation`.

**Step 2: Remove `build_solver_agent_generator` function**

Delete the entire `build_solver_agent_generator` function (lines ~286-325) and the `load_cbv` helper (lines ~328-331).

**Step 3: Update `build_agent_generator`**

The `build_agent_generator` function (line ~337) currently falls through to `build_solver_agent_generator` for bundle paths. Change this to return an error for non-agent, non-builtin paths (or load a V2 blueprint agent if we add one in simulation.rs).

For now, the simplest approach: if the path is not a `.toml` agent and not a `builtin:` prefix, return an error: `"Bundle-based simulation requires blueprint_v2 format (not yet wired)"`.

**Step 4: Verify compilation**

```bash
cargo check -p poker-solver-tauri 2>&1 | head -40
```

**Step 5: Commit**

```bash
git add -A && git commit -m "refactor: remove legacy blueprint agent from Tauri simulation"
```

---

### Task 11: Clean Up Cargo Dependencies and Examples

**Files:**
- Modify: `crates/core/Cargo.toml` — remove dependencies only used by deleted modules
- Delete: `crates/core/examples/debug_load.rs` (uses `BundleConfig`)
- Check: `crates/trainer/Cargo.toml` — remove dependencies only used by deleted commands

**Step 1: Identify dead dependencies**

Check what dependencies were only used by `preflop/` and `blueprint/` (e.g., `sled` if only used by blueprint cache, `indicatif` if only used by old TUI, etc.). Grep the remaining source for each suspect dependency.

**Step 2: Remove dead dependencies from Cargo.toml**

**Step 3: Delete dead examples**

```bash
rm crates/core/examples/debug_load.rs
```

Remove `[[example]]` entry from `crates/core/Cargo.toml` if present.

**Step 4: Full build check**

```bash
cargo check --workspace
```

**Step 5: Commit**

```bash
git add -A && git commit -m "chore: remove dead dependencies and examples after legacy module deletion"
```

---

### Task 12: Run Full Test Suite

**Step 1: Run all tests**

```bash
cargo test --workspace 2>&1 | tail -30
```

**Step 2: Fix any failures**

Address any test failures caused by removed types or missing imports.

**Step 3: Run clippy**

```bash
cargo clippy --workspace 2>&1 | tail -30
```

**Step 4: Fix any clippy warnings**

**Step 5: Commit any fixes**

```bash
git add -A && git commit -m "fix: resolve test failures and clippy warnings after legacy removal"
```

---

### Task 13: Update Documentation

**Files:**
- Modify: `docs/training.md`
- Modify: `docs/architecture.md`
- Modify: `docs/explorer.md`
- Modify: `CLAUDE.md`

**Step 1: Update training.md**

Remove all sections about `solve-preflop`, `solve-postflop`, `diag-buckets`, `trace-hand`, `precompute-equity`, `dump-tree`, `inspect-preflop`, `trace-regrets`, `decompose-regrets`, `trace-terminals`, `precompute-equity-delta`, `flops`. Keep `train-blueprint`, `cluster`, `diag-clusters`, `range-solve`.

**Step 2: Update architecture.md**

Remove sections describing the LCFR preflop solver pipeline and the old subgame-based postflop solver. Keep `blueprint_v2` MCCFR architecture, abstraction module, and cfvnet descriptions.

**Step 3: Update explorer.md**

Remove references to old bundle format loading (`load_bundle`, preflop solve loading). Keep `load_blueprint_v2` and agent loading.

**Step 4: Update CLAUDE.md**

Update the crate map, build commands, and key files sections to remove preflop/blueprint references. Remove the "Preflop Solver (LCFR)" section from key config files. Update any references to old CLI commands.

**Step 5: Commit**

```bash
git add -A && git commit -m "docs: update training, architecture, and explorer docs after legacy removal"
```

---

### Task 14: Update Memory Files

**Files:**
- Modify: `/Users/ltj/.claude/projects/-Users-ltj-Documents-code-poker-solver-rust/memory/MEMORY.md`

**Step 1: Update MEMORY.md**

Remove or update sections about:
- Preflop Solver (LCFR) section
- References to old `blueprint/` module
- Old CLI commands
- `RealTimeSolvingAgent` references
- Old test counts

Add note about legacy removal date and that `blueprint_v2` is now the only solver.

**Step 2: Commit beans and memory**

```bash
git add -A && git commit -m "chore: update project memory after legacy solver removal"
```

---

### Task 15: Final Verification

**Step 1: Full workspace build**

```bash
cargo build --workspace
```

**Step 2: Full test suite**

```bash
cargo test --workspace
```

Verify tests complete in under 1 minute.

**Step 3: Clippy clean**

```bash
cargo clippy --workspace
```

**Step 4: Verify Tauri app compiles**

```bash
cargo check -p poker-solver-tauri
```

**Step 5: Spot-check that cfvnet still compiles**

```bash
cargo check -p cfvnet
```

**Step 6: Final commit if any fixes needed, then mark bean complete**
