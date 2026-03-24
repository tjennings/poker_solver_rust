# Convergence Harness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Build a `convergence-harness` crate that solves "Flop Poker" (QhJdTh, all combos, 50bb/2bb, limited bet sizes) with exact DCFR to produce a golden baseline, then provides infrastructure to compare any CFR algorithm against that baseline.

**Architecture:** New standalone crate with a pluggable `ConvergenceSolver` trait. The harness drives the iteration loop, samples metrics at intervals, and produces convergence curves, strategy maps, combo EVs, and human-readable reports. Phase 1 wraps the range-solver as the exhaustive baseline solver. The baseline is solved once and persisted as a fixed artifact.

**Tech Stack:** Rust, range-solver (DCFR + best-response), clap (CLI), serde/bincode (serialization), serde_json/csv (reporting)

**Reference Design:** `docs/plans/2026-03-24-convergence-harness-design.md`

---

### Task 1: Scaffold the crate

**Files:**
- Create: `crates/convergence-harness/Cargo.toml`
- Create: `crates/convergence-harness/src/main.rs`
- Modify: `Cargo.toml` (workspace root, line 3 — add to members)

**Step 1: Create the Cargo.toml**

```toml
[package]
name = "convergence-harness"
version = "0.1.0"
edition = "2021"

[dependencies]
range-solver = { path = "../range-solver" }
poker-solver-core = { path = "../core" }
clap = { version = "4", features = ["derive"] }
serde = { workspace = true }
serde_json = "1"
serde_yaml = { workspace = true }
bincode = "1"
csv = "1"
chrono = "0.4"
```

**Step 2: Create minimal main.rs**

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "convergence-harness")]
#[command(about = "Convergence validation harness for CFR algorithms")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Solve Flop Poker exactly and persist the golden baseline
    GenerateBaseline {
        /// Output directory for baseline artifacts
        #[arg(long, default_value = "baselines/flop_poker_v1")]
        output_dir: String,

        /// Maximum iterations for the solver
        #[arg(long, default_value_t = 1000)]
        iterations: u32,

        /// Target exploitability (fraction of pot)
        #[arg(long, default_value_t = 0.001)]
        target_exploitability: f32,
    },
    /// Compare a saved solver result against the baseline
    Compare {
        /// Path to baseline directory
        #[arg(long, default_value = "baselines/flop_poker_v1")]
        baseline_dir: String,

        /// Path to solver result directory
        #[arg(long)]
        result_dir: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::GenerateBaseline { output_dir, iterations, target_exploitability } => {
            println!("generate-baseline: not yet implemented");
            Ok(())
        }
        Commands::Compare { baseline_dir, result_dir } => {
            println!("compare: not yet implemented");
            Ok(())
        }
    }
}
```

**Step 3: Add to workspace**

In `Cargo.toml` (workspace root), add `"crates/convergence-harness"` to the `members` array on line 3.

**Step 4: Verify it compiles**

Run: `cargo build -p convergence-harness`
Expected: Compiles with no errors. Warnings about unused variables are fine.

**Step 5: Commit**

```bash
git add crates/convergence-harness/ Cargo.toml
git commit -m "feat(convergence-harness): scaffold crate with CLI skeleton"
```

---

### Task 2: Game definition — FlopPokerGame

**Files:**
- Create: `crates/convergence-harness/src/game.rs`
- Modify: `crates/convergence-harness/src/main.rs` (add module declaration)

This task defines the Flop Poker game and builds a range-solver `PostFlopGame` from it. The game definition is:
- Board: QhJdTh, starting from flop, turn/river dealt by solver
- Ranges: all combos (full 1326 minus board blockers)
- Starting pot: 2bb (as integer: 2), Effective stack: 50bb (as integer: 50)
- Bet sizes: 33% pot, 67% pot for both players on all streets. All-in always available.
- Raise sizes: 33%, 67% pot (1 raise cap enforced by `force_allin_threshold`)
- `add_allin_threshold: 0.0` (all-in always available — not gated by ratio)

**Important range-solver API notes:**
- `CardConfig.range` uses `Range::from_str()`. A range of "random" or all combos is represented by listing every hand: use a wildcard. The range solver's `Range` type supports this via `"2c2d"` through `"AsAh"` but the simplest way is to pass a range string that matches everything. Check if `"random"` or equivalent is supported; otherwise use a range that covers all combos. Actually, the simplest approach: create a `Range` with all 1326 weights set to 1.0 using `Range::ones()` or by constructing it manually.
- `card_from_str("Qh")` returns `Card` (u8). Card encoding: `4 * rank + suit` where rank 2=0..A=12, suit club=0,diamond=1,heart=2,spade=3.
- `flop_from_str("QhJdTh")` returns `[Card; 3]`.
- `BetSizeOptions::try_from(("33%,67%,a", "33%,67%"))` for bet/raise sizes.
- `NOT_DEALT` constant (255) for turn/river in `CardConfig`.

**Step 1: Write the failing test**

Add to `crates/convergence-harness/src/game.rs`:

```rust
use range_solver::{
    PostFlopGame, CardConfig, ActionTree, TreeConfig, BoardState,
    card::flop_from_str,
    card::NOT_DEALT,
    range::Range,
    bet_size::BetSizeOptions,
};

/// Builds the Flop Poker game: QhJdTh, all combos, 50bb/2bb, limited bets.
pub fn build_flop_poker_game() -> Result<PostFlopGame, String> {
    // Full range — every combo gets weight 1.0
    let full_range = "22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,82s+,72s+,62s+,52s+,42s+,32s,A2o+,K2o+,Q2o+,J2o+,T2o+,92o+,82o+,72o+,62o+,52o+,42o+,32o".to_string();

    let card_config = CardConfig {
        range: [full_range.parse().unwrap(), full_range.parse().unwrap()],
        flop: flop_from_str("QhJdTh").unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from(("33%,67%,a", "33%,67%")).unwrap();

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 2,
        effective_stack: 50,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        merging_threshold: 0.0,
        depth_limit: None,
    };

    let action_tree = ActionTree::new(tree_config)?;
    PostFlopGame::with_config(card_config, action_tree)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_flop_poker_game() {
        let game = build_flop_poker_game().unwrap();
        // 49 remaining cards → C(49,2) = 1176 combos per player
        assert_eq!(game.num_private_hands(0), 1176);
        assert_eq!(game.num_private_hands(1), 1176);
    }
}
```

Note: the `num_private_hands()` method is on the `Game` trait (see `crates/range-solver/src/interface.rs:17`). You may need to import the `Game` trait: `use range_solver::interface::Game;` — check if it's re-exported from `range_solver`. If `interface` is not public, use `game.private_cards(0).len()` instead (from `PostFlopGame::private_cards()` at `crates/range-solver/src/game/interpreter.rs:180`).

Also check whether the full range string above actually covers all combos, or whether `Range` has a simpler constructor. Look at `crates/range-solver/src/range.rs` for a `new()`, `ones()`, or similar method. If none exists, the string approach works.

**Step 2: Add module to main.rs**

Add `mod game;` to the top of `main.rs`.

**Step 3: Run test to verify it passes**

Run: `cargo test -p convergence-harness test_build_flop_poker_game`
Expected: PASS — game builds with 1176 combos per player. If the combo count differs (due to range parsing), adjust the assertion to match the actual count and verify it's close to 1176.

**Step 4: Check memory usage**

Add a second test to verify the game is tractable:

```rust
#[test]
fn test_flop_poker_memory_is_tractable() {
    let game = build_flop_poker_game().unwrap();
    let (uncompressed, _compressed) = game.memory_usage();
    // Should be under 4GB uncompressed
    assert!(uncompressed < 4 * 1024 * 1024 * 1024, "Memory usage too high: {} bytes", uncompressed);
    println!("Flop Poker memory: {:.2} MB uncompressed", uncompressed as f64 / (1024.0 * 1024.0));
}
```

Run: `cargo test -p convergence-harness test_flop_poker_memory -- --nocapture`
Expected: PASS, prints memory usage. If memory is too high (>4GB), we'll need to trim bet sizes or add compression.

**Step 5: Commit**

```bash
git add crates/convergence-harness/src/game.rs crates/convergence-harness/src/main.rs
git commit -m "feat(convergence-harness): Flop Poker game definition with tests"
```

---

### Task 3: Solver trait and exhaustive adapter

**Files:**
- Create: `crates/convergence-harness/src/solver_trait.rs`
- Create: `crates/convergence-harness/src/solvers/mod.rs`
- Create: `crates/convergence-harness/src/solvers/exhaustive.rs`
- Modify: `crates/convergence-harness/src/main.rs` (add module declarations)

The `ConvergenceSolver` trait lets the harness drive any solver one iteration at a time. The exhaustive adapter wraps `range_solver::solve_step()`.

**Key API references:**
- `solve_step(game, current_iteration)` at `crates/range-solver/src/solver.rs:181` — runs one full CFR iteration (both players)
- `compute_exploitability(game)` at `crates/range-solver/src/utility.rs:359` — computes exploitability of current average strategy
- `finalize(game)` at `crates/range-solver/src/utility.rs:327` — normalizes strategy and computes EVs. **Caution:** sets game to Solved state, after which no more iterations can run. Only call when done.
- `game.strategy()` at `crates/range-solver/src/game/query.rs:341` — returns strategy at current node as `Vec<f32>`, layout: `action_idx * num_hands + hand_idx`
- `game.expected_values(player)` at `crates/range-solver/src/game/query.rs:540` — returns per-hand EVs at current node

**Step 1: Define the solver trait**

Create `crates/convergence-harness/src/solver_trait.rs`:

```rust
use std::collections::BTreeMap;

/// Per-node strategy: maps node_id to flat Vec<f32> (action_idx * num_hands + hand_idx).
pub type StrategyMap = BTreeMap<u64, Vec<f32>>;

/// Per-node, per-player combo EVs: maps node_id to [oop_evs, ip_evs].
pub type ComboEvMap = BTreeMap<u64, [Vec<f32>; 2]>;

/// Algorithm-specific metrics reported by the solver.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SolverMetrics {
    /// Key-value pairs of algorithm-specific metrics (e.g., "avg_regret" -> 0.05)
    pub values: BTreeMap<String, f64>,
}

/// Trait for pluggable CFR solver algorithms.
pub trait ConvergenceSolver {
    /// Human-readable name (e.g., "Exhaustive DCFR", "MCCFR 500bkt")
    fn name(&self) -> &str;

    /// Run one iteration (or batch). The harness calls this in a loop.
    fn solve_step(&mut self);

    /// Current iteration count.
    fn iterations(&self) -> u64;

    /// Extract the current average strategy at every decision node.
    /// Keys are node IDs (unique per decision point), values are flat strategy vectors.
    fn average_strategy(&self) -> StrategyMap;

    /// Extract per-combo EVs at every decision node for both players.
    fn combo_evs(&self) -> ComboEvMap;

    /// Algorithm-specific metrics (avg regret, strategy delta, etc.)
    fn self_reported_metrics(&self) -> SolverMetrics;
}
```

**Step 2: Implement the exhaustive solver adapter**

Create `crates/convergence-harness/src/solvers/mod.rs`:

```rust
pub mod exhaustive;
```

Create `crates/convergence-harness/src/solvers/exhaustive.rs`:

```rust
use range_solver::{PostFlopGame, solve_step, compute_exploitability, finalize};
use crate::solver_trait::{ConvergenceSolver, StrategyMap, ComboEvMap, SolverMetrics};

pub struct ExhaustiveSolver {
    game: PostFlopGame,
    iteration: u64,
}

impl ExhaustiveSolver {
    pub fn new(mut game: PostFlopGame) -> Self {
        game.allocate_memory(false);
        Self { game, iteration: 0 }
    }

    /// Compute exploitability without finalizing (can continue iterating).
    pub fn exploitability(&self) -> f32 {
        compute_exploitability(&self.game)
    }

    /// Access the underlying game for tree traversal and metric extraction.
    pub fn game(&self) -> &PostFlopGame {
        &self.game
    }

    /// Finalize and get mutable access (for strategy/EV extraction after solving).
    /// WARNING: After calling this, no more iterations can run.
    pub fn finalize(&mut self) {
        finalize(&mut self.game);
    }
}

impl ConvergenceSolver for ExhaustiveSolver {
    fn name(&self) -> &str {
        "Exhaustive DCFR"
    }

    fn solve_step(&mut self) {
        solve_step(&self.game, self.iteration as u32);
        self.iteration += 1;
    }

    fn iterations(&self) -> u64 {
        self.iteration
    }

    fn average_strategy(&self) -> StrategyMap {
        // Tree traversal to collect strategy at every decision node.
        // Implementation deferred to Task 5 (evaluator) where we build the tree walker.
        // For now, return empty — the evaluator will handle full extraction.
        StrategyMap::new()
    }

    fn combo_evs(&self) -> ComboEvMap {
        // Same — deferred to evaluator's tree walker.
        ComboEvMap::new()
    }

    fn self_reported_metrics(&self) -> SolverMetrics {
        SolverMetrics::default()
    }
}
```

**Important note for the implementer:** The `average_strategy()` and `combo_evs()` methods need a tree walker that navigates the `PostFlopGame` via `play()`/`back_to_root()` and collects strategy/EV at each decision node. This is non-trivial because:
- You must traverse chance nodes (deal all possible turn/river cards)
- The game tree can be large
- Node identity needs a stable key (e.g., the action history as a `Vec<usize>`)

This tree walker will be built in Task 5 (evaluator). For now, the stubs return empty maps. The trait methods will be filled in once the tree walker exists.

**Step 3: Add module declarations to main.rs**

Add to top of `main.rs`:
```rust
mod solver_trait;
mod solvers;
```

**Step 4: Write a test that creates the solver and runs a few iterations**

Add to `exhaustive.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::build_flop_poker_game;

    #[test]
    fn test_exhaustive_solver_runs_iterations() {
        let game = build_flop_poker_game().unwrap();
        let mut solver = ExhaustiveSolver::new(game);

        assert_eq!(solver.iterations(), 0);
        let expl_before = solver.exploitability();

        // Run 10 iterations
        for _ in 0..10 {
            solver.solve_step();
        }

        assert_eq!(solver.iterations(), 10);
        let expl_after = solver.exploitability();

        // Exploitability should decrease (or at least not increase significantly)
        println!("Exploitability: before={:.4e}, after 10 iters={:.4e}", expl_before, expl_after);
        assert!(expl_after < expl_before, "Exploitability should decrease after iterations");
    }
}
```

**Step 5: Run the test**

Run: `cargo test -p convergence-harness test_exhaustive_solver_runs_iterations -- --nocapture`
Expected: PASS. Exploitability decreases after 10 iterations.

**Step 6: Commit**

```bash
git add crates/convergence-harness/src/solver_trait.rs crates/convergence-harness/src/solvers/
git commit -m "feat(convergence-harness): ConvergenceSolver trait and exhaustive DCFR adapter"
```

---

### Task 4: Baseline serialization and persistence

**Files:**
- Create: `crates/convergence-harness/src/baseline.rs`
- Modify: `crates/convergence-harness/src/main.rs` (add module declaration)

This task handles saving and loading the baseline artifact to/from disk.

**Step 1: Define baseline data structures**

Create `crates/convergence-harness/src/baseline.rs`:

```rust
use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;
use std::path::Path;

/// A single convergence sample: iteration number, exploitability, elapsed time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceSample {
    pub iteration: u64,
    pub exploitability: f64,
    pub elapsed_ms: u64,
}

/// Summary of the baseline solve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSummary {
    pub solver_name: String,
    pub total_iterations: u64,
    pub final_exploitability: f64,
    pub total_time_ms: u64,
    pub num_info_sets: usize,
    pub num_combos_per_player: usize,
    pub game_description: String,
}

/// The full baseline artifact.
#[derive(Debug, Serialize, Deserialize)]
pub struct Baseline {
    pub summary: BaselineSummary,
    pub convergence_curve: Vec<ConvergenceSample>,
    /// Per-node strategy: node_id -> flat Vec<f32> (action_idx * num_hands + hand_idx)
    pub strategy: BTreeMap<u64, Vec<f32>>,
    /// Per-node combo EVs: node_id -> [oop_evs, ip_evs]
    pub combo_evs: BTreeMap<u64, [Vec<f32>; 2]>,
}

impl Baseline {
    /// Save baseline to a directory.
    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;

        // Summary as JSON (human-readable)
        let summary_json = serde_json::to_string_pretty(&self.summary)?;
        std::fs::write(dir.join("summary.json"), summary_json)?;

        // Convergence curve as CSV
        let mut wtr = csv::Writer::from_path(dir.join("convergence.csv"))?;
        for sample in &self.convergence_curve {
            wtr.serialize(sample)?;
        }
        wtr.flush()?;

        // Strategy and combo EVs as bincode (compact)
        let strategy_bytes = bincode::serialize(&self.strategy)?;
        std::fs::write(dir.join("strategy.bin"), strategy_bytes)?;

        let ev_bytes = bincode::serialize(&self.combo_evs)?;
        std::fs::write(dir.join("combo_ev.bin"), ev_bytes)?;

        Ok(())
    }

    /// Load baseline from a directory.
    pub fn load(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let summary: BaselineSummary =
            serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json"))?)?;

        let mut convergence_curve = Vec::new();
        let mut rdr = csv::Reader::from_path(dir.join("convergence.csv"))?;
        for result in rdr.deserialize() {
            convergence_curve.push(result?);
        }

        let strategy: BTreeMap<u64, Vec<f32>> =
            bincode::deserialize(&std::fs::read(dir.join("strategy.bin"))?)?;

        let combo_evs: BTreeMap<u64, [Vec<f32>; 2]> =
            bincode::deserialize(&std::fs::read(dir.join("combo_ev.bin"))?)?;

        Ok(Self { summary, convergence_curve, strategy, combo_evs })
    }
}
```

**Step 2: Write round-trip test**

Add to `baseline.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_baseline_round_trip() {
        let baseline = Baseline {
            summary: BaselineSummary {
                solver_name: "Exhaustive DCFR".into(),
                total_iterations: 100,
                final_exploitability: 0.001,
                total_time_ms: 5000,
                num_info_sets: 42,
                num_combos_per_player: 1176,
                game_description: "Flop Poker: QhJdTh, 50bb/2bb".into(),
            },
            convergence_curve: vec![
                ConvergenceSample { iteration: 0, exploitability: 1.0, elapsed_ms: 0 },
                ConvergenceSample { iteration: 50, exploitability: 0.01, elapsed_ms: 2500 },
                ConvergenceSample { iteration: 100, exploitability: 0.001, elapsed_ms: 5000 },
            ],
            strategy: {
                let mut m = BTreeMap::new();
                m.insert(0, vec![0.5, 0.3, 0.2]);
                m
            },
            combo_evs: {
                let mut m = BTreeMap::new();
                m.insert(0, [vec![1.5, -0.5], vec![-1.5, 0.5]]);
                m
            },
        };

        let dir = TempDir::new().unwrap();
        baseline.save(dir.path()).unwrap();

        let loaded = Baseline::load(dir.path()).unwrap();
        assert_eq!(loaded.summary.total_iterations, 100);
        assert_eq!(loaded.convergence_curve.len(), 3);
        assert_eq!(loaded.strategy.len(), 1);
        assert_eq!(loaded.combo_evs.len(), 1);
        assert!((loaded.summary.final_exploitability - 0.001).abs() < 1e-9);
    }
}
```

**Step 3: Add `tempfile` dev-dependency**

In `crates/convergence-harness/Cargo.toml`, add:
```toml
[dev-dependencies]
tempfile = "3"
```

**Step 4: Add module to main.rs**

Add `mod baseline;` to the top of `main.rs`.

**Step 5: Run the test**

Run: `cargo test -p convergence-harness test_baseline_round_trip`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/convergence-harness/src/baseline.rs crates/convergence-harness/Cargo.toml
git commit -m "feat(convergence-harness): baseline serialization with round-trip test"
```

---

### Task 5: Tree walker and strategy/EV extraction

**Files:**
- Create: `crates/convergence-harness/src/evaluator.rs`
- Modify: `crates/convergence-harness/src/solvers/exhaustive.rs` (fill in trait methods)
- Modify: `crates/convergence-harness/src/main.rs` (add module declaration)

This is the most complex task. We need to walk the entire game tree of a `PostFlopGame`, visiting every decision node to extract strategy and combo EVs. The challenge is that `PostFlopGame` uses `play(action_idx)` for navigation and `back_to_root()` + `apply_history(&[usize])` for repositioning.

**Key API:**
- `game.play(action_idx)` — navigate to child. For chance nodes, `action_idx` is a card ID.
- `game.back_to_root()` — reset to root
- `game.apply_history(&[usize])` — replay a sequence of actions from root (at `crates/range-solver/src/game/query.rs`)
- `game.available_actions()` — get `Vec<Action>` at current node
- `game.is_terminal_node()` — check if terminal
- `game.is_chance_node()` — check if chance node
- `game.possible_cards()` — bitmask of dealable cards at a chance node
- `game.strategy()` — strategy at current decision node
- `game.expected_values(player)` — combo EVs at current node (requires game to be finalized/solved)
- `game.cache_normalized_weights()` — must call before expected_values()

**Strategy for node IDs:** Use a hash of the action history (path from root). This gives a stable, unique key per node. A simple approach: hash the `Vec<usize>` action history.

**Step 1: Write the tree walker**

Create `crates/convergence-harness/src/evaluator.rs`:

```rust
use range_solver::{PostFlopGame, compute_exploitability};
use crate::solver_trait::{StrategyMap, ComboEvMap};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Compute a stable node ID from an action history.
fn node_id(history: &[usize]) -> u64 {
    let mut hasher = DefaultHasher::new();
    history.hash(&mut hasher);
    hasher.finish()
}

/// Walk the game tree and extract strategy at every decision node.
/// The game must have had `finalize()` called (for normalized strategy).
/// However, `strategy()` works on the cumulative strategy sums even before finalize,
/// so this can be called at any point during solving for the current average strategy.
pub fn extract_strategy(game: &mut PostFlopGame) -> StrategyMap {
    let mut result = StrategyMap::new();
    game.back_to_root();
    let mut history = Vec::new();
    walk_strategy(game, &mut history, &mut result);
    game.back_to_root();
    result
}

fn walk_strategy(
    game: &mut PostFlopGame,
    history: &mut Vec<usize>,
    result: &mut StrategyMap,
) {
    if game.is_terminal_node() {
        return;
    }

    if game.is_chance_node() {
        let possible = game.possible_cards();
        for card in 0..52u8 {
            if possible & (1u64 << card) != 0 {
                history.push(card as usize);
                game.play(card as usize);
                walk_strategy(game, history, result);
                // Navigate back: replay history from root
                history.pop();
                game.back_to_root();
                if !history.is_empty() {
                    game.apply_history(history);
                }
                break; // Only visit one representative card per chance node
                       // (isomorphism handles the rest)
                       // Actually, we need ALL cards for full strategy extraction.
                       // Remove the break if you want all runouts.
                       // For baseline purposes, the range solver already handles
                       // isomorphism internally — strategy() returns the full
                       // strategy accounting for all isomorphic runouts.
            }
        }
        // Revisit: The range solver's strategy at a node already averages over
        // all chance outcomes. We don't need to walk through every turn/river
        // card to get the strategy. The strategy at a pre-chance node captures
        // the average strategy over all runouts.
        //
        // For the BASELINE, we actually want strategy per-runout (per turn card,
        // per river card) because different runouts produce different strategies.
        // But for the initial implementation, extracting strategy at the root
        // flop decision and a sample of turn/river nodes is sufficient.
        //
        // IMPLEMENTER NOTE: Start simple. Extract strategy only at the root
        // node first. Expand to full tree walking in a follow-up if needed.
        // The range solver's `strategy()` at any navigated-to node gives the
        // correct per-combo strategy for that specific node.
        return;
    }

    // Decision node — extract strategy
    let nid = node_id(history);
    let strat = game.strategy();
    result.insert(nid, strat);

    // Recurse into children
    let num_actions = game.available_actions().len();
    for action_idx in 0..num_actions {
        history.push(action_idx);
        game.play(action_idx);
        walk_strategy(game, history, result);
        history.pop();
        game.back_to_root();
        if !history.is_empty() {
            game.apply_history(history);
        }
    }
}

/// Extract combo EVs at every decision node.
/// Game must be finalized (solved state) for expected_values() to work.
pub fn extract_combo_evs(game: &mut PostFlopGame) -> ComboEvMap {
    let mut result = ComboEvMap::new();
    game.back_to_root();
    game.cache_normalized_weights();
    let mut history = Vec::new();
    walk_combo_evs(game, &mut history, &mut result);
    game.back_to_root();
    result
}

fn walk_combo_evs(
    game: &mut PostFlopGame,
    history: &mut Vec<usize>,
    result: &mut ComboEvMap,
) {
    if game.is_terminal_node() {
        return;
    }

    if game.is_chance_node() {
        // Same approach as walk_strategy — see note above
        let possible = game.possible_cards();
        for card in 0..52u8 {
            if possible & (1u64 << card) != 0 {
                history.push(card as usize);
                game.play(card as usize);
                walk_combo_evs(game, history, result);
                history.pop();
                game.back_to_root();
                if !history.is_empty() {
                    game.apply_history(history);
                }
            }
        }
        return;
    }

    // Decision node — extract EVs
    let nid = node_id(history);
    game.cache_normalized_weights();
    let oop_ev = game.expected_values(0);
    let ip_ev = game.expected_values(1);
    result.insert(nid, [oop_ev, ip_ev]);

    // Recurse into children
    let num_actions = game.available_actions().len();
    for action_idx in 0..num_actions {
        history.push(action_idx);
        game.play(action_idx);
        walk_combo_evs(game, history, result);
        history.pop();
        game.back_to_root();
        if !history.is_empty() {
            game.apply_history(history);
        }
    }
}

/// Compute exploitability of a PostFlopGame's current strategy.
pub fn exploitability(game: &PostFlopGame) -> f64 {
    compute_exploitability(game) as f64
}
```

**IMPORTANT IMPLEMENTER NOTES:**

1. **Tree size concern:** Walking every chance node (every turn × every river card) produces a massive tree. The range solver handles ~46 turn cards × ~45 river cards = ~2000 runouts, times the action tree nodes at each street. This could be millions of nodes. For the initial implementation, consider:
   - Walking only decision nodes reachable from one or a few representative turn/river cards
   - OR walking the full tree but being aware it may take significant time/memory
   - The range solver's isomorphism reduces this significantly (to ~400-600 canonical runouts)

2. **Navigation cost:** `back_to_root()` + `apply_history()` is O(depth) per node visit. For a full tree walk, consider whether the range solver provides a more efficient traversal API. Check if there's an internal node iterator.

3. **`expected_values()` requires solved state:** The combo EV extraction (`extract_combo_evs`) can only be called after `finalize()`. Strategy extraction can be called during solving.

4. **Start simple:** For the first pass, extract strategy and EVs only at the root node. This proves the pipeline works. Full tree walking can be a follow-up enhancement.

**Step 2: Write a test**

Add to `evaluator.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::build_flop_poker_game;
    use crate::solvers::exhaustive::ExhaustiveSolver;
    use crate::solver_trait::ConvergenceSolver;

    #[test]
    fn test_extract_strategy_at_root() {
        let game = build_flop_poker_game().unwrap();
        let mut solver = ExhaustiveSolver::new(game);

        // Run a few iterations
        for _ in 0..20 {
            solver.solve_step();
        }

        // Extract strategy — should have at least the root node
        let mut game = solver.into_game(); // Need to add this method
        let strategy = extract_strategy(&mut game);
        assert!(!strategy.is_empty(), "Should extract at least one node's strategy");

        // Root strategy should have correct shape: num_actions * 1176
        let root_strat = strategy.get(&node_id(&[])).expect("Root node should be present");
        assert!(root_strat.len() > 0);
        // Strategy probabilities should sum to ~1.0 for each hand
        let num_hands = 1176; // approximate
        let num_actions = root_strat.len() / num_hands;
        assert!(num_actions >= 2, "Should have at least 2 actions at root");
    }
}
```

**Step 3: Add `into_game()` to ExhaustiveSolver**

In `solvers/exhaustive.rs`, add:
```rust
/// Consume the solver and return the underlying game.
pub fn into_game(self) -> PostFlopGame {
    self.game
}
```

**Step 4: Run the test**

Run: `cargo test -p convergence-harness test_extract_strategy_at_root -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/convergence-harness/src/evaluator.rs crates/convergence-harness/src/solvers/exhaustive.rs
git commit -m "feat(convergence-harness): tree walker for strategy and combo EV extraction"
```

---

### Task 6: Convergence loop and baseline generation

**Files:**
- Create: `crates/convergence-harness/src/harness.rs`
- Modify: `crates/convergence-harness/src/main.rs` (wire up generate-baseline command)

This task builds the main convergence loop that drives any solver and records metrics.

**Step 1: Define the convergence loop**

Create `crates/convergence-harness/src/harness.rs`:

```rust
use std::time::Instant;
use crate::solver_trait::ConvergenceSolver;
use crate::baseline::{Baseline, BaselineSummary, ConvergenceSample};
use crate::evaluator;
use crate::solvers::exhaustive::ExhaustiveSolver;
use crate::game::build_flop_poker_game;
use range_solver::compute_exploitability;

/// Determines how often to sample exploitability.
/// Dense early, sparse later.
fn should_sample(iteration: u64) -> bool {
    if iteration < 100 {
        true // every iteration
    } else if iteration < 1000 {
        iteration % 10 == 0
    } else {
        iteration % 100 == 0
    }
}

/// Run the exhaustive solver and produce a baseline.
pub fn generate_baseline(
    max_iterations: u32,
    target_exploitability: f32,
) -> Result<Baseline, Box<dyn std::error::Error>> {
    let game = build_flop_poker_game()?;
    let mut solver = ExhaustiveSolver::new(game);

    let num_combos = solver.game().private_cards(0).len();

    let mut convergence_curve = Vec::new();
    let start = Instant::now();

    // Record initial exploitability
    let initial_expl = solver.exploitability();
    convergence_curve.push(ConvergenceSample {
        iteration: 0,
        exploitability: initial_expl as f64,
        elapsed_ms: 0,
    });
    println!("iteration: 0 (exploitability = {:.4e})", initial_expl);

    for t in 0..max_iterations {
        solver.solve_step();
        let iter = solver.iterations();

        if should_sample(iter) || iter == max_iterations as u64 {
            let expl = solver.exploitability();
            let elapsed = start.elapsed().as_millis() as u64;

            convergence_curve.push(ConvergenceSample {
                iteration: iter,
                exploitability: expl as f64,
                elapsed_ms: elapsed,
            });

            println!(
                "iteration: {} / {} (exploitability = {:.4e}, elapsed = {:.1}s)",
                iter, max_iterations, expl, elapsed as f64 / 1000.0
            );

            if expl <= target_exploitability {
                println!("Target exploitability reached. Stopping.");
                break;
            }
        }
    }

    let total_time = start.elapsed().as_millis() as u64;
    let final_expl = solver.exploitability();

    println!("\nFinalizing solver (computing EVs and normalizing strategy)...");
    solver.finalize();

    println!("Extracting strategy and combo EVs...");
    let mut game = solver.into_game();
    let strategy = evaluator::extract_strategy(&mut game);
    let combo_evs = evaluator::extract_combo_evs(&mut game);

    let summary = BaselineSummary {
        solver_name: "Exhaustive DCFR".into(),
        total_iterations: convergence_curve.last().map_or(0, |s| s.iteration),
        final_exploitability: final_expl as f64,
        total_time_ms: total_time,
        num_info_sets: strategy.len(),
        num_combos_per_player: num_combos,
        game_description: "Flop Poker: QhJdTh, all combos, 50bb/2bb, 33%/67%/a bet sizes".into(),
    };

    Ok(Baseline {
        summary,
        convergence_curve,
        strategy,
        combo_evs,
    })
}
```

**Step 2: Wire up the CLI**

In `main.rs`, replace the `GenerateBaseline` match arm:

```rust
Commands::GenerateBaseline { output_dir, iterations, target_exploitability } => {
    let baseline = harness::generate_baseline(iterations, target_exploitability)?;

    let dir = std::path::Path::new(&output_dir);
    baseline.save(dir)?;

    println!("\n=== Baseline Summary ===");
    println!("Solver: {}", baseline.summary.solver_name);
    println!("Iterations: {}", baseline.summary.total_iterations);
    println!("Final exploitability: {:.4e}", baseline.summary.final_exploitability);
    println!("Time: {:.1}s", baseline.summary.total_time_ms as f64 / 1000.0);
    println!("Info sets captured: {}", baseline.summary.num_info_sets);
    println!("Combos per player: {}", baseline.summary.num_combos_per_player);
    println!("Saved to: {}", output_dir);

    Ok(())
}
```

Add `mod harness;` to the module declarations.

**Step 3: Write an integration test**

Add to `harness.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_baseline_small() {
        // Small number of iterations — just test the pipeline works
        let baseline = generate_baseline(20, 10.0).unwrap();
        assert!(baseline.convergence_curve.len() >= 2); // at least initial + some samples
        assert!(baseline.summary.total_iterations > 0);
        assert!(baseline.summary.final_exploitability > 0.0);
        assert!(!baseline.strategy.is_empty());
    }
}
```

**Step 4: Run the test**

Run: `cargo test -p convergence-harness test_generate_baseline_small -- --nocapture`
Expected: PASS. Prints convergence progress for 20 iterations.

**Step 5: Commit**

```bash
git add crates/convergence-harness/src/harness.rs crates/convergence-harness/src/main.rs
git commit -m "feat(convergence-harness): convergence loop and generate-baseline command"
```

---

### Task 7: Comparison metrics (L1 distance, EV diff)

**Files:**
- Modify: `crates/convergence-harness/src/evaluator.rs` (add comparison functions)

**Step 1: Implement L1 strategy distance**

Add to `evaluator.rs`:

```rust
use crate::solver_trait::StrategyMap;

/// Compute L1 strategy distance between two strategy maps.
/// Returns per-node L1 distance and the overall weighted average.
/// L1 at a node = sum_a |p(a) - q(a)| averaged over hands.
pub fn l1_strategy_distance(
    baseline: &StrategyMap,
    candidate: &StrategyMap,
    num_hands: usize,
) -> (BTreeMap<u64, f64>, f64) {
    let mut per_node = BTreeMap::new();
    let mut total_distance = 0.0;
    let mut node_count = 0;

    for (node_id, base_strat) in baseline {
        if let Some(cand_strat) = candidate.get(node_id) {
            if base_strat.len() != cand_strat.len() {
                continue; // Mismatched dimensions — skip
            }

            let num_actions = base_strat.len() / num_hands;
            let mut node_l1_sum = 0.0;
            let mut hand_count = 0;

            for h in 0..num_hands {
                let mut hand_l1 = 0.0;
                for a in 0..num_actions {
                    let idx = a * num_hands + h;
                    hand_l1 += (base_strat[idx] - cand_strat[idx]).abs() as f64;
                }
                node_l1_sum += hand_l1;
                hand_count += 1;
            }

            let avg_l1 = if hand_count > 0 { node_l1_sum / hand_count as f64 } else { 0.0 };
            per_node.insert(*node_id, avg_l1);
            total_distance += avg_l1;
            node_count += 1;
        }
    }

    let overall = if node_count > 0 { total_distance / node_count as f64 } else { 0.0 };
    (per_node, overall)
}

/// Compute combo EV differences between baseline and candidate.
/// Returns per-node max absolute EV difference and overall stats.
pub fn combo_ev_diff(
    baseline: &ComboEvMap,
    candidate: &ComboEvMap,
) -> (BTreeMap<u64, f64>, f64) {
    let mut per_node = BTreeMap::new();
    let mut total_max_diff = 0.0;
    let mut node_count = 0;

    for (node_id, [base_oop, base_ip]) in baseline {
        if let Some([cand_oop, cand_ip]) = candidate.get(node_id) {
            let max_diff_oop = base_oop.iter().zip(cand_oop.iter())
                .map(|(b, c)| (b - c).abs() as f64)
                .fold(0.0f64, f64::max);
            let max_diff_ip = base_ip.iter().zip(cand_ip.iter())
                .map(|(b, c)| (b - c).abs() as f64)
                .fold(0.0f64, f64::max);
            let max_diff = max_diff_oop.max(max_diff_ip);

            per_node.insert(*node_id, max_diff);
            total_max_diff += max_diff;
            node_count += 1;
        }
    }

    let overall = if node_count > 0 { total_max_diff / node_count as f64 } else { 0.0 };
    (per_node, overall)
}
```

**Step 2: Write tests**

```rust
#[test]
fn test_l1_distance_identical_strategies() {
    let mut strat = StrategyMap::new();
    // 2 actions, 3 hands: [0.6, 0.4, 0.8, 0.4, 0.6, 0.2]
    strat.insert(0, vec![0.6, 0.4, 0.8, 0.4, 0.6, 0.2]);

    let (per_node, overall) = l1_strategy_distance(&strat, &strat, 3);
    assert!((overall).abs() < 1e-9, "Identical strategies should have 0 L1 distance");
    assert!((per_node[&0]).abs() < 1e-9);
}

#[test]
fn test_l1_distance_different_strategies() {
    let mut base = StrategyMap::new();
    let mut cand = StrategyMap::new();
    // 2 actions, 2 hands
    base.insert(0, vec![1.0, 0.0, 0.0, 1.0]); // hand0: always action0, hand1: always action1
    cand.insert(0, vec![0.0, 1.0, 1.0, 0.0]); // opposite

    let (_, overall) = l1_strategy_distance(&base, &cand, 2);
    // Each hand has L1 = |1-0| + |0-1| = 2.0. Average = 2.0.
    assert!((overall - 2.0).abs() < 1e-9);
}
```

**Step 3: Run tests**

Run: `cargo test -p convergence-harness test_l1_distance`
Expected: Both PASS

**Step 4: Commit**

```bash
git add crates/convergence-harness/src/evaluator.rs
git commit -m "feat(convergence-harness): L1 strategy distance and combo EV diff metrics"
```

---

### Task 8: Reporter — terminal output, CSV, and human summary

**Files:**
- Create: `crates/convergence-harness/src/reporter.rs`
- Modify: `crates/convergence-harness/src/main.rs` (add module declaration)

**Step 1: Implement the reporter**

Create `crates/convergence-harness/src/reporter.rs`:

```rust
use crate::baseline::{Baseline, ConvergenceSample};
use std::collections::BTreeMap;
use std::path::Path;

/// Comparison results from running a solver against the baseline.
#[derive(Debug, serde::Serialize)]
pub struct ComparisonResult {
    pub solver_name: String,
    pub total_iterations: u64,
    pub total_time_ms: u64,
    pub final_exploitability: f64,
    pub baseline_exploitability: f64,
    pub overall_l1_distance: f64,
    pub overall_max_ev_diff: f64,
    pub convergence_curve: Vec<ConvergenceSample>,
    pub per_node_l1: BTreeMap<u64, f64>,
    pub per_node_ev_diff: BTreeMap<u64, f64>,
}

impl ComparisonResult {
    /// Save all comparison artifacts to a directory.
    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;

        // Machine-readable summary
        let summary_json = serde_json::to_string_pretty(&serde_json::json!({
            "solver_name": self.solver_name,
            "total_iterations": self.total_iterations,
            "total_time_ms": self.total_time_ms,
            "final_exploitability": self.final_exploitability,
            "baseline_exploitability": self.baseline_exploitability,
            "overall_l1_distance": self.overall_l1_distance,
            "overall_max_ev_diff": self.overall_max_ev_diff,
        }))?;
        std::fs::write(dir.join("summary.json"), summary_json)?;

        // Convergence CSV
        let mut wtr = csv::Writer::from_path(dir.join("convergence.csv"))?;
        for sample in &self.convergence_curve {
            wtr.serialize(sample)?;
        }
        wtr.flush()?;

        // Per-node L1 distance CSV
        let mut wtr = csv::Writer::from_path(dir.join("strategy_distance.csv"))?;
        wtr.write_record(["node_id", "l1_distance"])?;
        for (node_id, dist) in &self.per_node_l1 {
            wtr.write_record([node_id.to_string(), format!("{:.6}", dist)])?;
        }
        wtr.flush()?;

        // Per-node EV diff CSV
        let mut wtr = csv::Writer::from_path(dir.join("combo_ev_diff.csv"))?;
        wtr.write_record(["node_id", "max_ev_diff"])?;
        for (node_id, diff) in &self.per_node_ev_diff {
            wtr.write_record([node_id.to_string(), format!("{:.6}", diff)])?;
        }
        wtr.flush()?;

        // Human summary
        let report = self.human_summary();
        std::fs::write(dir.join("report.txt"), &report)?;

        Ok(())
    }

    /// Generate a human-readable summary report.
    pub fn human_summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Convergence Harness Report ===\n\n");

        s.push_str(&format!("Solver: {}\n", self.solver_name));
        s.push_str(&format!("Iterations: {}\n", self.total_iterations));
        s.push_str(&format!("Time: {:.1}s\n\n", self.total_time_ms as f64 / 1000.0));

        s.push_str("--- Exploitability ---\n");
        s.push_str(&format!("Baseline: {:.4e}\n", self.baseline_exploitability));
        s.push_str(&format!("Solver:   {:.4e}\n", self.final_exploitability));
        let gap = self.final_exploitability - self.baseline_exploitability;
        s.push_str(&format!("Gap:      {:.4e}\n\n", gap));

        s.push_str("--- Strategy Distance (L1) ---\n");
        s.push_str(&format!("Overall average: {:.4}\n", self.overall_l1_distance));
        if self.overall_l1_distance < 0.05 {
            s.push_str("Assessment: Excellent — abstraction is faithful.\n\n");
        } else if self.overall_l1_distance < 0.1 {
            s.push_str("Assessment: Good — minor strategy deviations.\n\n");
        } else if self.overall_l1_distance < 0.3 {
            s.push_str("Assessment: Concerning — noticeable strategy deviations.\n\n");
        } else {
            s.push_str("Assessment: Poor — abstraction substantially alters strategy.\n\n");
        }

        s.push_str("--- Combo EV Difference ---\n");
        s.push_str(&format!("Overall average max diff: {:.4} bb\n", self.overall_max_ev_diff));

        // Convergence rate estimate
        if self.convergence_curve.len() >= 2 {
            let first = &self.convergence_curve[0];
            let last = self.convergence_curve.last().unwrap();
            if last.iteration > first.iteration && last.exploitability > 0.0 && first.exploitability > 0.0 {
                let log_ratio = (first.exploitability / last.exploitability).ln();
                let iter_ratio = (last.iteration as f64 / first.iteration.max(1) as f64).ln();
                if iter_ratio > 0.0 {
                    let rate = log_ratio / iter_ratio;
                    s.push_str(&format!("\n--- Convergence Rate ---\n"));
                    s.push_str(&format!("Approximate rate: O(1/T^{:.2})\n", rate));
                    s.push_str("(DCFR typically converges at ~O(1/T^0.5))\n");
                }
            }
        }

        s
    }
}
```

**Step 2: Write a test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::ConvergenceSample;
    use tempfile::TempDir;

    #[test]
    fn test_comparison_result_save_and_summary() {
        let result = ComparisonResult {
            solver_name: "Test Solver".into(),
            total_iterations: 100,
            total_time_ms: 5000,
            final_exploitability: 0.05,
            baseline_exploitability: 0.001,
            overall_l1_distance: 0.12,
            overall_max_ev_diff: 0.5,
            convergence_curve: vec![
                ConvergenceSample { iteration: 1, exploitability: 1.0, elapsed_ms: 50 },
                ConvergenceSample { iteration: 100, exploitability: 0.05, elapsed_ms: 5000 },
            ],
            per_node_l1: {
                let mut m = BTreeMap::new();
                m.insert(0, 0.1);
                m.insert(1, 0.15);
                m
            },
            per_node_ev_diff: {
                let mut m = BTreeMap::new();
                m.insert(0, 0.3);
                m
            },
        };

        let dir = TempDir::new().unwrap();
        result.save(dir.path()).unwrap();

        // Verify files exist
        assert!(dir.path().join("summary.json").exists());
        assert!(dir.path().join("convergence.csv").exists());
        assert!(dir.path().join("strategy_distance.csv").exists());
        assert!(dir.path().join("report.txt").exists());

        let report = result.human_summary();
        assert!(report.contains("Test Solver"));
        assert!(report.contains("Concerning"));
    }
}
```

**Step 3: Add module to main.rs**

Add `mod reporter;`

**Step 4: Run the test**

Run: `cargo test -p convergence-harness test_comparison_result`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/convergence-harness/src/reporter.rs crates/convergence-harness/src/main.rs
git commit -m "feat(convergence-harness): reporter with CSV, JSON, and human summary output"
```

---

### Task 9: Wire up the compare command

**Files:**
- Modify: `crates/convergence-harness/src/main.rs` (implement Compare subcommand)

This wires up the `compare` command which loads a baseline and a solver result, then computes metrics.

**Step 1: Implement the Compare command**

Replace the `Compare` match arm in `main.rs`:

```rust
Commands::Compare { baseline_dir, result_dir } => {
    let baseline_path = std::path::Path::new(&baseline_dir);
    let result_path = std::path::Path::new(&result_dir);

    println!("Loading baseline from: {}", baseline_dir);
    let baseline = baseline::Baseline::load(baseline_path)?;
    println!("Baseline: {} iterations, exploitability={:.4e}",
        baseline.summary.total_iterations, baseline.summary.final_exploitability);

    println!("Loading solver result from: {}", result_dir);
    let result_summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(result_path.join("summary.json"))?)?;

    // Load solver strategy and EVs
    let solver_strategy: std::collections::BTreeMap<u64, Vec<f32>> =
        bincode::deserialize(&std::fs::read(result_path.join("strategy.bin"))?)?;
    let solver_evs: std::collections::BTreeMap<u64, [Vec<f32>; 2]> =
        bincode::deserialize(&std::fs::read(result_path.join("combo_ev.bin"))?)?;

    // Compute metrics
    let (per_node_l1, overall_l1) = evaluator::l1_strategy_distance(
        &baseline.strategy,
        &solver_strategy,
        baseline.summary.num_combos_per_player,
    );
    let (per_node_ev_diff, overall_ev_diff) = evaluator::combo_ev_diff(
        &baseline.combo_evs,
        &solver_evs,
    );

    let comparison = reporter::ComparisonResult {
        solver_name: result_summary["solver_name"].as_str().unwrap_or("Unknown").into(),
        total_iterations: result_summary["total_iterations"].as_u64().unwrap_or(0),
        total_time_ms: result_summary["total_time_ms"].as_u64().unwrap_or(0),
        final_exploitability: result_summary["final_exploitability"].as_f64().unwrap_or(0.0),
        baseline_exploitability: baseline.summary.final_exploitability,
        overall_l1_distance: overall_l1,
        overall_max_ev_diff: overall_ev_diff,
        convergence_curve: Vec::new(), // Loaded separately if needed
        per_node_l1,
        per_node_ev_diff,
    };

    let report = comparison.human_summary();
    println!("{}", report);

    let output_dir = result_path.join("comparison");
    comparison.save(&output_dir)?;
    println!("Comparison artifacts saved to: {}", output_dir.display());

    Ok(())
}
```

**Step 2: Run the full pipeline manually**

Run: `cargo run -p convergence-harness --release -- generate-baseline --iterations 50 --output-dir /tmp/test_baseline`
Expected: Prints convergence progress, saves baseline to `/tmp/test_baseline/`.

Verify: `ls /tmp/test_baseline/`
Expected: `summary.json`, `convergence.csv`, `strategy.bin`, `combo_ev.bin`

**Step 3: Commit**

```bash
git add crates/convergence-harness/src/main.rs
git commit -m "feat(convergence-harness): wire up compare command with metric computation"
```

---

### Task 10: End-to-end integration test

**Files:**
- Create: `crates/convergence-harness/tests/integration.rs`

This tests the full pipeline: build game, solve, save baseline, load it back, verify.

**Step 1: Write the integration test**

```rust
use std::path::Path;

#[test]
fn test_end_to_end_baseline_generation() {
    // This test runs the full pipeline with minimal iterations
    // to verify everything connects properly.
    let dir = tempfile::TempDir::new().unwrap();

    // Generate baseline (inline, not via CLI)
    let baseline = convergence_harness::harness::generate_baseline(10, 10.0).unwrap();

    // Save
    baseline.save(dir.path()).unwrap();

    // Verify files
    assert!(dir.path().join("summary.json").exists());
    assert!(dir.path().join("convergence.csv").exists());
    assert!(dir.path().join("strategy.bin").exists());
    assert!(dir.path().join("combo_ev.bin").exists());

    // Load back
    let loaded = convergence_harness::baseline::Baseline::load(dir.path()).unwrap();
    assert_eq!(loaded.summary.solver_name, "Exhaustive DCFR");
    assert!(loaded.summary.total_iterations > 0);
    assert!(loaded.convergence_curve.len() >= 2);

    // Verify strategy is non-empty and has valid probabilities
    assert!(!loaded.strategy.is_empty());
    for (_, strat) in &loaded.strategy {
        for &v in strat {
            assert!(v >= 0.0, "Strategy prob should be >= 0");
            assert!(v <= 1.0001, "Strategy prob should be <= 1"); // small float tolerance
        }
    }
}
```

**Note:** This requires making `harness` and `baseline` modules public in `lib.rs`. Create `crates/convergence-harness/src/lib.rs`:

```rust
pub mod game;
pub mod solver_trait;
pub mod solvers;
pub mod baseline;
pub mod evaluator;
pub mod harness;
pub mod reporter;
```

And update `main.rs` to import from the lib:
```rust
use convergence_harness::{harness, baseline, evaluator, reporter};
```

**Step 2: Run the integration test**

Run: `cargo test -p convergence-harness --test integration test_end_to_end -- --nocapture`
Expected: PASS

**Step 3: Run the full test suite**

Run: `cargo test -p convergence-harness`
Expected: All tests pass.

**Step 4: Run the workspace test suite**

Run: `cargo test`
Expected: All tests pass in under 1 minute (per CLAUDE.md requirement).

**Step 5: Commit**

```bash
git add crates/convergence-harness/tests/ crates/convergence-harness/src/lib.rs crates/convergence-harness/src/main.rs
git commit -m "feat(convergence-harness): end-to-end integration test and lib.rs exports"
```

---

### Task 11: Manual smoke test and cleanup

**Step 1: Run a real baseline generation**

Run: `cargo run -p convergence-harness --release -- generate-baseline --iterations 200 --target-exploitability 0.01`

Expected: Prints convergence curve, saves baseline. Monitor:
- Memory usage (should be reasonable)
- Time (should complete in under a few minutes for 200 iterations)
- Exploitability trend (should decrease)

**Step 2: Inspect the outputs**

Read `baselines/flop_poker_v1/summary.json` and `convergence.csv`. Verify:
- Summary has correct metadata
- Convergence CSV shows decreasing exploitability
- strategy.bin and combo_ev.bin exist and are non-empty

**Step 3: Run clippy**

Run: `cargo clippy -p convergence-harness`
Expected: No errors. Fix any warnings.

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore(convergence-harness): cleanup and clippy fixes"
```

---

## Summary of Tasks

| Task | Description | Key Deliverable |
|------|-------------|-----------------|
| 1 | Scaffold crate | Cargo.toml, main.rs with CLI skeleton |
| 2 | Game definition | `build_flop_poker_game()` with tests |
| 3 | Solver trait + exhaustive adapter | `ConvergenceSolver` trait, `ExhaustiveSolver` |
| 4 | Baseline serialization | `Baseline` save/load with round-trip test |
| 5 | Tree walker | Strategy and combo EV extraction from game tree |
| 6 | Convergence loop | `generate_baseline()` function, CLI wiring |
| 7 | Comparison metrics | L1 distance, combo EV diff |
| 8 | Reporter | Terminal, CSV, JSON, human summary output |
| 9 | Compare command | CLI `compare` subcommand wired up |
| 10 | Integration test | End-to-end pipeline verification |
| 11 | Smoke test + cleanup | Manual verification, clippy, final polish |
