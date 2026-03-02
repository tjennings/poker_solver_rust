# Remove Standalone MCCFR Solver — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the standalone MCCFR/sequence postflop solver while preserving solve-preflop, solve-postflop, and the Tauri explorer app.

**Architecture:** The standalone solver (train command, Game trait, HunlPostflop, generic CFR modules) is fully separable from the preflop pipeline which has its own internal MCCFR in `preflop/postflop_mccfr.rs`. The one entanglement is `PostflopConfig` and `AbstractionMode` types defined in `hunl_postflop.rs` but imported by `blueprint/bundle.rs` and transitively by tauri/simulation — these get extracted to a new `game/config.rs` first.

**Tech Stack:** Rust, Cargo workspace

---

### Task 1: Extract PostflopConfig and AbstractionMode to game/config.rs

**Why first:** Every subsequent deletion depends on this. `blueprint/bundle.rs`, `simulation.rs`, and tauri all import `PostflopConfig` and `AbstractionMode` from `crate::game`. We extract them before deleting `hunl_postflop.rs`.

**Files:**
- Create: `crates/core/src/game/config.rs`
- Modify: `crates/core/src/game/mod.rs`

**Step 1: Create `game/config.rs`**

Copy `PostflopConfig`, its `Default` impl, `default_max_raises`, and `AbstractionMode` (with its imports) from `hunl_postflop.rs` lines 23-67 into a new file:

```rust
// crates/core/src/game/config.rs
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::abstraction::CardAbstraction;

/// Card abstraction mode for information set construction.
pub enum AbstractionMode {
    /// EHS2-based bucketing (expensive Monte Carlo, fine-grained).
    Ehs2(Arc<CardAbstraction>),
    /// Hand-class V2: class ID + intra-class strength + equity bin + draw flags.
    HandClassV2 {
        strength_bits: u8,
        equity_bits: u8,
    },
}

/// Configuration for the postflop game.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostflopConfig {
    pub stack_depth: u32,
    pub bet_sizes: Vec<f32>,
    #[serde(default = "default_max_raises")]
    pub max_raises_per_street: u8,
}

impl Default for PostflopConfig {
    fn default() -> Self {
        Self {
            stack_depth: 100,
            bet_sizes: vec![0.33, 0.5, 0.75, 1.0],
            max_raises_per_street: 3,
        }
    }
}

fn default_max_raises() -> u8 {
    3
}
```

**Step 2: Update `game/mod.rs` to export from `config.rs` instead of `hunl_postflop`**

Add `mod config;` and add `PostflopConfig`, `AbstractionMode` to the `pub use` from it. Keep the existing `hunl_postflop` module for now (it will be deleted in Task 3).

**Step 3: Verify it compiles**

Run: `cargo check -p poker-solver-core 2>&1 | head -30`
Expected: compiles (may have warnings about duplicate types — that's fine, we'll remove the originals in Task 3)

**Step 4: Commit**

```
git add crates/core/src/game/config.rs crates/core/src/game/mod.rs
git commit -m "refactor: extract PostflopConfig and AbstractionMode to game/config.rs"
```

---

### Task 2: Delete standalone CFR modules

**Files to delete:**
- `crates/core/src/cfr/mccfr.rs`
- `crates/core/src/cfr/sequence_cfr.rs`
- `crates/core/src/cfr/game_tree.rs`
- `crates/core/src/cfr/exploitability.rs`
- `crates/core/src/cfr/vanilla.rs`
- `crates/core/src/cfr/convergence.rs`

**Files to modify:**
- `crates/core/src/cfr/mod.rs` — reduce to only `pub mod regret;`

**Step 1: Replace `cfr/mod.rs` contents**

```rust
pub mod regret;
```

**Step 2: Delete the six files**

```bash
rm crates/core/src/cfr/mccfr.rs
rm crates/core/src/cfr/sequence_cfr.rs
rm crates/core/src/cfr/game_tree.rs
rm crates/core/src/cfr/exploitability.rs
rm crates/core/src/cfr/vanilla.rs
rm crates/core/src/cfr/convergence.rs
```

**Step 3: Verify `blueprint/subgame_cfr.rs` still compiles**

It depends on `cfr::regret` which we kept.

Run: `cargo check -p poker-solver-core 2>&1 | head -40`
Expected: errors from other files that import deleted modules (lib.rs, hunl_postflop.rs, etc.) — these are cleaned up in subsequent tasks.

**Step 4: Commit**

```
git add -A crates/core/src/cfr/
git commit -m "refactor: remove standalone CFR modules (keep regret only)"
```

---

### Task 3: Delete game implementations and Game trait

**Files to delete:**
- `crates/core/src/game/hunl_postflop.rs`
- `crates/core/src/game/kuhn.rs`
- `crates/core/src/game/limit_holdem.rs`

**Files to modify:**
- `crates/core/src/game/mod.rs` — remove Game trait, remove deleted module declarations, keep Action/Player/Actions/ALL_IN/MAX_ACTIONS and re-exports from config.rs

**Step 1: Rewrite `game/mod.rs`**

```rust
mod config;

use arrayvec::ArrayVec;

pub use config::{AbstractionMode, PostflopConfig};

/// Maximum number of actions at any decision point.
pub const MAX_ACTIONS: usize = 8;

/// Sentinel value for all-in bets/raises.
pub const ALL_IN: u32 = u32::MAX;

/// Stack-allocated action list.
pub type Actions = ArrayVec<Action, MAX_ACTIONS>;

/// Player in a two-player game
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    Player1,
    Player2,
}

impl Player {
    #[must_use]
    pub const fn opponent(self) -> Self {
        match self {
            Self::Player1 => Self::Player2,
            Self::Player2 => Self::Player1,
        }
    }
}

/// Actions available in poker games.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Fold,
    Check,
    Call,
    Bet(u32),
    Raise(u32),
}
```

**Step 2: Delete the three game files**

```bash
rm crates/core/src/game/hunl_postflop.rs
rm crates/core/src/game/kuhn.rs
rm crates/core/src/game/limit_holdem.rs
```

**Step 3: Verify**

Run: `cargo check -p poker-solver-core 2>&1 | head -40`
Expected: errors from `lib.rs` (still exports `Game`), `info_key.rs` (may reference `hunl_postflop` types), and trainer. Addressed in Tasks 4-6.

**Step 4: Commit**

```
git add -A crates/core/src/game/
git commit -m "refactor: remove Game trait, HunlPostflop, KuhnPoker, LimitHoldem"
```

---

### Task 4: Delete abstract_game.rs and tree.rs from core

**Files to delete:**
- `crates/core/src/abstract_game.rs`
- `crates/core/src/tree.rs`

**Files to modify:**
- `crates/core/src/lib.rs` — remove `pub mod abstract_game;`, `pub mod tree;`, `pub use game::Game;` and the standalone CFR re-exports

**Step 1: Update `lib.rs`**

Remove these lines:
- `pub mod abstract_game;`
- `pub mod tree;`
- `pub use game::{Action, Game, Player};` → change to `pub use game::{Action, Player};`

Also remove any re-exports that reference deleted types (e.g., `Game`, `KuhnPoker`, etc.).

**Step 2: Delete the files**

```bash
rm crates/core/src/abstract_game.rs
rm crates/core/src/tree.rs
```

**Step 3: Verify core compiles**

Run: `cargo check -p poker-solver-core 2>&1 | head -40`
Expected: core should now compile cleanly (or close to it). Any remaining issues will be in `info_key.rs` if it references deleted types.

**Step 4: Commit**

```
git add crates/core/src/lib.rs
git add -A crates/core/src/abstract_game.rs crates/core/src/tree.rs
git commit -m "refactor: remove abstract_game and tree modules from core"
```

---

### Task 5: Fix info_key.rs references to deleted types

**File:** `crates/core/src/info_key.rs`

`info_key.rs` imports `AbstractionModeConfig` from `blueprint` and `PostflopConfig`-related types. It also has the `describe_key` function that may reference `HunlPostflop`-specific types.

**Step 1: Audit imports and fix**

Remove any imports from `crate::game` that no longer exist (e.g., `HunlPostflop`, `PostflopState`). The types `PostflopConfig` and `AbstractionMode` still exist via `crate::game::config`.

The `describe_key` / `describe_from_hex` functions use `bet_sizes` from `PostflopConfig` — these should still work.

Check for any functions that instantiate `HunlPostflop` or call `Game` trait methods and remove them.

**Step 2: Verify**

Run: `cargo check -p poker-solver-core 2>&1 | head -40`

**Step 3: Commit**

```
git add crates/core/src/info_key.rs
git commit -m "fix: update info_key.rs imports for removed game types"
```

---

### Task 6: Delete standalone MCCFR tests, examples, and benchmarks

**Files to delete:**
- `crates/core/tests/mccfr_hunl_integration.rs`
- `crates/core/tests/postflop_cfr_mechanics.rs`
- `crates/core/tests/cfr_mechanics.rs`
- `crates/core/tests/kuhn_convergence.rs`
- `crates/core/tests/lcfr_convergence.rs`
- `crates/core/tests/convergence_metrics_test.rs`
- `crates/core/examples/bench_mccfr.rs`
- `crates/core/examples/hand_class_histogram.rs`
- `crates/core/benches/sequence_cfr_bench.rs`

**Files to keep:**
- `tests/abstraction_integration.rs` — tests card abstraction (shared)
- `tests/full_pipeline.rs` — tests preflop pipeline (shared)
- `tests/postflop_diagnostics.rs` — tests postflop pipeline (shared)
- `tests/postflop_imperfect_recall.rs` — tests postflop pipeline (shared)
- `tests/preflop_avg_regret_test.rs` — tests preflop solver (shared)
- `tests/preflop_convergence.rs` — tests preflop solver (shared)
- `examples/debug_load.rs` — loads bundles (shared)
- `examples/validate_flops.rs` — tests flop enumeration (shared)
- `benches/preflop_solver_bench.rs` — benchmarks preflop (shared)

**Files to modify:**
- `crates/core/Cargo.toml` — remove `[[bench]] name = "sequence_cfr_bench"` section

**Step 1: Delete the files**

```bash
rm crates/core/tests/mccfr_hunl_integration.rs
rm crates/core/tests/postflop_cfr_mechanics.rs
rm crates/core/tests/cfr_mechanics.rs
rm crates/core/tests/kuhn_convergence.rs
rm crates/core/tests/lcfr_convergence.rs
rm crates/core/tests/convergence_metrics_test.rs
rm crates/core/examples/bench_mccfr.rs
rm crates/core/examples/hand_class_histogram.rs
rm crates/core/benches/sequence_cfr_bench.rs
```

**Step 2: Remove bench entry from Cargo.toml**

Remove the `[[bench]] name = "sequence_cfr_bench" harness = false` block from `crates/core/Cargo.toml`.

**Step 3: Verify**

Run: `cargo test -p poker-solver-core 2>&1 | tail -20`
Expected: remaining tests pass, no compilation errors from deleted test files.

**Step 4: Commit**

```
git add -A crates/core/tests/ crates/core/examples/ crates/core/benches/ crates/core/Cargo.toml
git commit -m "refactor: remove standalone MCCFR tests, examples, and benchmarks"
```

---

### Task 7: Clean up trainer CLI

**Files to modify:**
- `crates/trainer/src/main.rs` — remove `train`, `generate-deals`, `merge-deals`, `inspect-deals`, `tree-stats`, `tree` subcommands and all their handler functions + supporting types (`SolverMode`, `DealSortOrder`, `TrainingConfig`, `TrainingParams`, etc.)

**Files to delete:**
- `crates/trainer/src/tree.rs` — entire module (the `tree` command handler uses `HunlPostflop` and `core::tree`)

**Files to keep:**
- `crates/trainer/src/main.rs` — with only `solve-preflop`, `solve-postflop`, `flops`, `diag-buckets`, `trace-hand`
- `crates/trainer/src/bucket_diagnostics.rs` — used by `diag-buckets`
- `crates/trainer/src/hand_trace.rs` — used by `trace-hand`
- `crates/trainer/src/lhe_viz.rs` — used by `solve-preflop` output

**Step 1: Delete `crates/trainer/src/tree.rs`**

```bash
rm crates/trainer/src/tree.rs
```

**Step 2: Clean up `main.rs`**

- Remove `mod tree;` declaration
- Remove imports: `poker_solver_core::Game`, `poker_solver_core::abstract_game`, `poker_solver_core::cfr::*`, `poker_solver_core::game::{HunlPostflop, PostflopState}`, `poker_solver_core::HandClass`, `poker_solver_core::hand_class::HandClassification`
- Keep imports: `poker_solver_core::blueprint::*`, `poker_solver_core::flops::*`, `poker_solver_core::game::{AbstractionMode, Action, PostflopConfig}`, `poker_solver_core::info_key::*`, `poker_solver_core::preflop::*`
- Remove `Commands` variants: `Train`, `TreeStats`, `GenerateDeals`, `InspectDeals`, `MergeDeals`, `Tree`
- Remove types: `SolverMode`, `DealSortOrder`, `TrainingConfig`, `TrainingParams`, all `default_*` helper fns for training params
- Remove handler functions: `run_training`, `run_tree_stats`, `run_generate_deals`, `run_inspect_deals`, `run_merge_deals`, `run_tree` (and the match arms in `main()`)
- Keep: `run_solve_preflop`, `run_solve_postflop`, `run_flops`, `run_diag_buckets`, `run_trace_hand`

**Step 3: Fix `lhe_viz.rs`**

The `#[cfg(test)] use poker_solver_core::game::Action;` import should still work since `Action` is still exported from `game/mod.rs`.

**Step 4: Verify trainer compiles and runs**

Run: `cargo check -p poker-solver-trainer 2>&1 | head -30`
Run: `cargo run -p poker-solver-trainer --release -- --help`
Expected: only shows solve-preflop, solve-postflop, flops, diag-buckets, trace-hand

**Step 5: Commit**

```
git add -A crates/trainer/src/
git commit -m "refactor: remove standalone MCCFR commands from trainer CLI"
```

---

### Task 8: Delete standalone MCCFR sample configs

**Files to delete:**
- `sample_configurations/smoke.yaml`
- `sample_configurations/ultra_fast.yaml`
- `sample_configurations/fast_buckets.yaml`
- `sample_configurations/full.yaml`
- `sample_configurations/AKQr_vs_234r.yaml`
- `sample_configurations/mccfr_smoke.yaml`

**Files to keep:**
- `sample_configurations/minimal_postflop.yaml`
- `sample_configurations/minimal_preflop.yaml`
- `sample_configurations/preflop_medium.yaml`
- `sample_configurations/preflop_full_model.yaml`
- `sample_configurations/AKQr_vs_234r_postflop.yaml` — this is a *postflop-only* config (uses `postflop_model:` section), keep it

**Step 1: Verify which configs are standalone MCCFR**

Check each file for `game:` + `training:` top-level keys (standalone MCCFR format) vs. `postflop_model:` key (postflop pipeline format) or preflop format.

**Step 2: Delete standalone configs**

```bash
rm sample_configurations/smoke.yaml
rm sample_configurations/ultra_fast.yaml
rm sample_configurations/fast_buckets.yaml
rm sample_configurations/full.yaml
rm sample_configurations/AKQr_vs_234r.yaml
rm sample_configurations/mccfr_smoke.yaml
```

**Step 3: Commit**

```
git add -A sample_configurations/
git commit -m "refactor: remove standalone MCCFR sample configs"
```

---

### Task 9: Update documentation

**Files to modify:**
- `docs/training.md` — remove `train`, `generate-deals`, `merge-deals`, `inspect-deals`, `tree-stats`, `tree` command sections; remove HUNL Training Config section; remove Abstraction Modes section; remove Solver Backends section; remove Advanced Options section; update Sample Configs table
- `docs/architecture.md` — remove references to standalone MCCFR; update module descriptions
- `docs/strategic_coverage.md` — remove "Standalone MCCFR" references from the pipeline table and all "EHS2 bucket" entries
- `CLAUDE.md` — update Crate Map if needed; remove `train` command references

**Step 1: Update each doc file**

Focus on removing all references to the `train` command, `HunlPostflop`, `Game` trait, EHS2 standalone mode, sequence solver, deal generation pipeline.

**Step 2: Verify no broken internal links**

Grep docs for references to deleted commands/types.

**Step 3: Commit**

```
git add docs/ CLAUDE.md
git commit -m "docs: update documentation for standalone MCCFR removal"
```

---

### Task 10: Full verification

**Step 1: Run full test suite**

```bash
cargo test 2>&1 | tail -30
```

Expected: all remaining tests pass.

**Step 2: Run clippy**

```bash
cargo clippy --workspace 2>&1 | tail -30
```

Expected: clean (or only pre-existing warnings).

**Step 3: Build tauri app**

```bash
cargo build -p poker-solver-tauri-app 2>&1 | tail -10
```

Expected: builds successfully.

**Step 4: Smoke test trainer commands**

```bash
cargo run -p poker-solver-trainer --release -- --help
cargo run -p poker-solver-trainer --release -- flops --format csv 2>&1 | head -5
```

Expected: help shows only kept commands; flops command works.

**Step 5: Commit any final fixes**

If anything failed in steps 1-4, fix and commit.
