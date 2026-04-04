# Remove Old Datagen Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Remove the 4000+ line old datagen pipeline (`turn_generate.rs`) by moving shared helpers into the domain module and routing all datagen through `DomainPipeline`.

**Architecture:** Three tasks: (1) Move 5 helper functions + their dependencies into a new `domain/game_tree.rs` module, (2) Delete `turn_generate.rs` and its old pipeline code, (3) Update CLI dispatch in `main.rs` to route directly to `DomainPipeline`.

**Tech Stack:** Rust, range-solver

---

## Context for Implementer

The old `turn_generate.rs` (4400 lines) has 4 pipeline implementations (CUDA, wgpu, exact, iterative) plus ~20 helper functions. The domain pipeline (`DomainPipeline`) replaces all of them.

**5 functions the domain module still calls from turn_generate.rs:**
1. `build_turn_game(board, pot, stack, ranges, bet_sizes) -> Option<PostFlopGame>` — line 535
2. `build_turn_game_exact(board, pot, stack, ranges, bet_sizes) -> Option<PostFlopGame>` — line 546
3. `build_turn_game_inner(board, pot, stack, ranges, bet_sizes, exact) -> Option<PostFlopGame>` — line 468 (internal, called by the above two)
4. `parse_bet_sizes_all(config) -> Vec<Vec<f64>>` — line 352 (depends on `parse_bet_sizes_depth` at line 337)
5. `fuzz_bet_sizes(bet_sizes, fuzz, rng) -> Vec<Vec<f64>>` — line 560
6. `u8_to_rs_card(id) -> Card` — line 306

**Transitive dependencies of `build_turn_game_inner`:**
- `range_solver::range::Range as RsRange`
- `range_solver::bet_size::{BetSize, BetSizeOptions}`
- `range_solver::card::{CardConfig, NOT_DEALT}`
- `range_solver::action_tree::{ActionTree, BoardState, TreeConfig}`
- `range_solver::game::PostFlopGame`
- `crate::datagen::range_gen::NUM_COMBOS`

**External reference to `turn_generate` from main.rs:**
```rust
"turn" | "river" => cfvnet::datagen::turn_generate::generate_turn_training_data(&cfg, &file_output, backend),
```

---

### Task 1: Move helpers into domain/game_tree.rs

**Files:**
- Create: `crates/cfvnet/src/datagen/domain/game_tree.rs`
- Modify: `crates/cfvnet/src/datagen/domain/mod.rs`
- Modify: `crates/cfvnet/src/datagen/domain/game.rs` — update imports
- Modify: `crates/cfvnet/src/datagen/domain/neural_net_evaluator.rs` — update imports
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs` — update imports

**Step 1: Create `game_tree.rs` with all 6 helper functions**

Move these functions (and their imports) from `turn_generate.rs` into `game_tree.rs`:

```rust
// game_tree.rs — Game tree construction helpers.

use poker_solver_core::poker::{Card, Suit, Value};
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::{BetSize, BetSizeOptions};
use range_solver::card::{CardConfig, NOT_DEALT};
use range_solver::game::PostFlopGame;
use range_solver::range::Range as RsRange;
use rand::Rng;

use crate::config::BetSizeConfig;
use crate::datagen::range_gen::NUM_COMBOS;

/// Convert a range-solver u8 card to an rs_poker Card.
pub fn u8_to_rs_card(id: u8) -> Card { ... }

/// Parse bet size strings into pot fractions.
fn parse_bet_sizes_depth(sizes: &[String]) -> Vec<f64> { ... }

/// Parse all depths from a BetSizeConfig.
pub fn parse_bet_sizes_all(config: &BetSizeConfig) -> Vec<Vec<f64>> { ... }

/// Perturb bet sizes by a random factor.
pub fn fuzz_bet_sizes(bet_sizes: &[Vec<f64>], fuzz: f64, rng: &mut impl Rng) -> Vec<Vec<f64>> { ... }

/// Internal: build a PostFlopGame with configurable depth limit.
fn build_game_inner(board_u8: &[u8], pot: f64, effective_stack: f64, ranges: &[[f32; NUM_COMBOS]; 2], bet_sizes: &[Vec<f64>], exact: bool) -> Option<PostFlopGame> { ... }

/// Build a depth-limited turn game (model mode).
pub fn build_turn_game(board_u8: &[u8], pot: f64, effective_stack: f64, ranges: &[[f32; NUM_COMBOS]; 2], bet_sizes: &[Vec<f64>]) -> Option<PostFlopGame> { ... }

/// Build a full turn+river game (exact mode).
pub fn build_turn_game_exact(board_u8: &[u8], pot: f64, effective_stack: f64, ranges: &[[f32; NUM_COMBOS]; 2], bet_sizes: &[Vec<f64>]) -> Option<PostFlopGame> { ... }
```

Copy the function bodies verbatim from `turn_generate.rs`.

**Step 2: Add to mod.rs**

```rust
pub mod game_tree;
```

**Step 3: Update domain callers to import from game_tree**

In `game.rs`:
```rust
// Old: crate::datagen::turn_generate::build_turn_game(...)
// New: super::game_tree::build_turn_game(...)
```

In `neural_net_evaluator.rs`:
```rust
// Old: use crate::datagen::turn_generate::u8_to_rs_card;
// New: use super::game_tree::u8_to_rs_card;
```

In `pipeline.rs`:
```rust
// Old: crate::datagen::turn_generate::parse_bet_sizes_all(...)
// New: super::game_tree::parse_bet_sizes_all(...)
```

**Step 4: Verify**

Run: `cargo build -p cfvnet --release 2>&1 | tail -5`
Run: `cargo test -p cfvnet domain 2>&1 | tail -5`

Both must pass. The old turn_generate.rs still exists (it's not deleted yet).

**Step 5: Commit**

```
git commit -m "refactor: move game tree helpers from turn_generate into domain/game_tree"
```

---

### Task 2: Delete turn_generate.rs

**Files:**
- Delete: `crates/cfvnet/src/datagen/turn_generate.rs`
- Modify: `crates/cfvnet/src/datagen/mod.rs` — remove `pub mod turn_generate;`

**Step 1: Remove the module declaration**

In `crates/cfvnet/src/datagen/mod.rs`, remove `pub mod turn_generate;`.

**Step 2: Delete the file**

```bash
rm crates/cfvnet/src/datagen/turn_generate.rs
```

**Step 3: Fix compilation errors**

The main.rs reference `cfvnet::datagen::turn_generate::generate_turn_training_data` will break. Replace with a direct call to `DomainPipeline::run`. Also, the old `generate.rs` (river-specific) may still be referenced.

In `main.rs`, change line 422:
```rust
// Old:
"turn" | "river" => cfvnet::datagen::turn_generate::generate_turn_training_data(&cfg, &file_output, backend),
// New:
"turn" | "river" => cfvnet::datagen::domain::pipeline::DomainPipeline::run(&cfg, &file_output),
```

Note: the `backend` parameter is no longer needed (DomainPipeline handles GPU internally).

**Step 4: Check for other references**

```bash
grep -rn "turn_generate" crates/cfvnet/src/ | grep -v test | grep -v target
```

Fix any remaining references.

**Step 5: Also check eval/compare modules**

```bash
grep -rn "turn_generate" crates/cfvnet/src/eval/ | head -10
```

The eval/compare modules may reference helpers from turn_generate.rs. If so, update them to use `domain::game_tree::` instead.

**Step 6: Verify**

Run: `cargo build -p cfvnet --release 2>&1 | tail -5`
Run: `cargo test -p cfvnet 2>&1 | grep "test result:" | head -5`

ALL tests must pass (not just domain tests — the full crate).

**Step 7: Commit**

```
git commit -m "refactor: delete turn_generate.rs — all datagen routes through DomainPipeline"
```

---

### Task 3: Clean up remaining references

**Files:**
- Possibly modify: `crates/cfvnet/src/eval/compare_turn.rs`
- Possibly modify: `crates/cfvnet/src/datagen/generate.rs`
- Possibly modify: `crates/cfvnet/src/main.rs`

**Step 1: Fix any remaining compilation errors**

After deleting turn_generate.rs, some eval/compare modules may reference functions that were in it. Check:

```bash
cargo build -p cfvnet --release 2>&1 | grep "error" | head -20
```

For each error:
- If it references a helper function (u8_to_rs_card, build_turn_game, etc.), update the import to `crate::datagen::domain::game_tree::*`
- If it references an old pipeline function (generate_turn_training_data_cuda, etc.), it can be deleted or updated

**Step 2: Consider deleting `generate.rs`**

The old `generate.rs` was the original river-only datagen pipeline. Since `DomainPipeline` handles river mode (board_size >= 5 → SolveStrategy::Exact), this file may be unnecessary.

Check if anything references it:
```bash
grep -rn "datagen::generate::" crates/cfvnet/src/main.rs | head -5
```

If referenced, update the dispatch. If not referenced, delete it.

**Step 3: Verify full workspace builds**

```bash
cargo build 2>&1 | tail -5
cargo test -p cfvnet 2>&1 | grep "test result:" | head -5
```

**Step 4: Commit**

```
git commit -m "chore: clean up remaining references to old pipeline code"
```

---

## Important Notes

1. **The `generate.rs` (old river datagen)** may still be referenced from `main.rs` for the default street. Check and update.

2. **eval/compare_turn.rs** likely uses `build_turn_game` and `u8_to_rs_card` from turn_generate.rs. These need to be updated to import from `domain::game_tree`.

3. **The `backend` CLI parameter** becomes unused after this change since `DomainPipeline` handles GPU internally. Consider removing it from the Generate CLI command or ignoring it.

4. **Old tests in turn_generate.rs** will be deleted with the file. The domain tests (58+) replace them. Verify no critical test coverage is lost.

5. **The CUDA feature flag** becomes dead code. The `NeuralNetEvaluator` uses wgpu only. CUDA support would need to be re-added as a different `BoundaryEvaluator` implementation later.
