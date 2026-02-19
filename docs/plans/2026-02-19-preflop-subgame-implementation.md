# Preflop Solver + Subgame Solving Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the three-layer solving architecture from `docs/plans/2026-02-19-preflop-subgame-architecture.md`: exact preflop solver, coarse postflop blueprint, real-time subgame solving, with explorer integration.

**Architecture:** Linear CFR preflop solver (standalone module), coarse LCFR postflop blueprint (existing `SequenceCfrSolver` with config), real-time CFR+ subgame solver (new module in `blueprint/`), all loadable into the Tauri explorer via the existing `StrategySource` pattern.

**Tech Stack:** Rust, existing `crates/core` infrastructure, Tauri explorer, `bincode`/`serde_yaml` for persistence.

---

## Phase 1: Preflop Solver

### Task 1: Preflop Config and Tree Types

**Files:**
- Create: `crates/core/src/preflop/mod.rs`
- Create: `crates/core/src/preflop/config.rs`
- Create: `crates/core/src/preflop/tree.rs`
- Modify: `crates/core/src/lib.rs` (add `pub mod preflop;`)

**Step 1: Write the failing test for `PreflopConfig`**

In `crates/core/src/preflop/config.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn hu_config_has_two_positions() {
        let config = PreflopConfig::heads_up(100);
        assert_eq!(config.positions.len(), 2);
        assert_eq!(config.stacks.len(), 2);
        assert_eq!(config.stacks[0], 200); // 100BB * 2 internal units
        assert_eq!(config.stacks[1], 200);
    }

    #[timed_test]
    fn hu_config_blinds_are_sb_bb() {
        let config = PreflopConfig::heads_up(100);
        // SB = position 0, BB = position 1
        assert_eq!(config.blinds, vec![(0, 1), (1, 2)]);
    }

    #[timed_test]
    fn six_max_config_has_six_positions() {
        let config = PreflopConfig::six_max(100);
        assert_eq!(config.positions.len(), 6);
        assert_eq!(config.stacks.len(), 6);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core --lib preflop::config::tests -- --no-run 2>&1 | head -20`
Expected: Compilation error — module `preflop` not found.

**Step 3: Implement `PreflopConfig`**

`crates/core/src/preflop/mod.rs`:

```rust
pub mod config;
pub mod tree;

pub use config::PreflopConfig;
pub use tree::{PreflopTree, PreflopNode, PreflopAction};
```

`crates/core/src/preflop/config.rs`:

```rust
use serde::{Deserialize, Serialize};

/// Position label for display and identification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionInfo {
    pub name: String,
    pub short_name: String,
}

/// Configuration for a preflop game.
///
/// Supports any number of players (2-9), any stack depth, and
/// configurable raise sizes per raise depth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflopConfig {
    pub positions: Vec<PositionInfo>,
    pub blinds: Vec<(usize, u32)>,
    pub antes: Vec<(usize, u32)>,
    pub stacks: Vec<u32>,
    pub raise_sizes: Vec<Vec<f64>>,
    pub raise_cap: u8,
}

impl PreflopConfig {
    /// Standard heads-up config with given stack depth in BB.
    #[must_use]
    pub fn heads_up(stack_depth_bb: u32) -> Self {
        let stack = stack_depth_bb * 2; // internal units
        Self {
            positions: vec![
                PositionInfo { name: "Small Blind".into(), short_name: "SB".into() },
                PositionInfo { name: "Big Blind".into(), short_name: "BB".into() },
            ],
            blinds: vec![(0, 1), (1, 2)],
            antes: vec![],
            stacks: vec![stack, stack],
            raise_sizes: vec![
                vec![2.5],  // open raise
                vec![3.0],  // 3-bet
                vec![2.5],  // 4-bet
            ],
            raise_cap: 4,
        }
    }

    /// Standard 6-max config.
    #[must_use]
    pub fn six_max(stack_depth_bb: u32) -> Self {
        let stack = stack_depth_bb * 2;
        Self {
            positions: vec![
                PositionInfo { name: "Under the Gun".into(), short_name: "UTG".into() },
                PositionInfo { name: "Hijack".into(), short_name: "HJ".into() },
                PositionInfo { name: "Cutoff".into(), short_name: "CO".into() },
                PositionInfo { name: "Button".into(), short_name: "BTN".into() },
                PositionInfo { name: "Small Blind".into(), short_name: "SB".into() },
                PositionInfo { name: "Big Blind".into(), short_name: "BB".into() },
            ],
            blinds: vec![(4, 1), (5, 2)], // SB=pos 4, BB=pos 5
            antes: vec![],
            stacks: vec![stack; 6],
            raise_sizes: vec![
                vec![2.5],
                vec![3.0],
                vec![2.5],
            ],
            raise_cap: 4,
        }
    }

    /// Number of players.
    #[must_use]
    pub fn num_players(&self) -> u8 {
        self.positions.len() as u8
    }

    /// Total pot after blinds and antes are posted.
    #[must_use]
    pub fn initial_pot(&self) -> u32 {
        let blind_total: u32 = self.blinds.iter().map(|(_, amt)| amt).sum();
        let ante_total: u32 = self.antes.iter().map(|(_, amt)| amt).sum();
        blind_total + ante_total
    }
}
```

Add `pub mod preflop;` to `crates/core/src/lib.rs`.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core --lib preflop::config::tests -v`
Expected: 3 tests PASS.

**Step 5: Commit**

```bash
git add crates/core/src/preflop/ crates/core/src/lib.rs
git commit -m "feat(preflop): add PreflopConfig with HU and 6-max constructors"
```

---

### Task 2: Preflop Tree Builder

**Files:**
- Create: `crates/core/src/preflop/tree.rs`
- Test in: `crates/core/src/preflop/tree.rs` (inline `#[cfg(test)]`)

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::preflop::PreflopConfig;
    use test_macros::timed_test;

    #[timed_test]
    fn hu_tree_has_root_decision() {
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);
        assert!(tree.nodes.len() > 1);
        assert!(matches!(tree.nodes[0], PreflopNode::Decision { .. }));
    }

    #[timed_test]
    fn hu_tree_root_is_sb_to_act() {
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);
        match &tree.nodes[0] {
            PreflopNode::Decision { position, .. } => assert_eq!(*position, 0, "SB acts first HU"),
            _ => panic!("root should be Decision"),
        }
    }

    #[timed_test]
    fn hu_tree_fold_produces_terminal() {
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);
        // Root actions: fold, call, raise(s), all-in
        match &tree.nodes[0] {
            PreflopNode::Decision { children, action_labels, .. } => {
                assert_eq!(action_labels[0], PreflopAction::Fold);
                let fold_child = children[0] as usize;
                assert!(matches!(tree.nodes[fold_child], PreflopNode::Terminal { .. }));
            }
            _ => panic!("root should be Decision"),
        }
    }

    #[timed_test]
    fn hu_tree_call_then_check_is_showdown() {
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);
        // SB calls (limps) → BB can check → showdown
        let sb_call_idx = find_action_child(&tree, 0, PreflopAction::Call);
        let bb_check_idx = find_action_child(&tree, sb_call_idx, PreflopAction::Call);
        // BB checking after SB limp = showdown
        assert!(
            matches!(tree.nodes[bb_check_idx], PreflopNode::Terminal { .. }),
            "call-check should reach showdown"
        );
    }

    #[timed_test]
    fn hu_tree_respects_raise_cap() {
        let mut config = PreflopConfig::heads_up(100);
        config.raise_cap = 2;
        let tree = PreflopTree::build(&config);
        let max_raises = count_max_raise_depth(&tree, 0, 0);
        assert!(max_raises <= 2, "raise depth {max_raises} exceeds cap 2");
    }

    fn find_action_child(tree: &PreflopTree, node_idx: usize, target: PreflopAction) -> usize {
        match &tree.nodes[node_idx] {
            PreflopNode::Decision { children, action_labels, .. } => {
                let pos = action_labels.iter().position(|a| *a == target)
                    .unwrap_or_else(|| panic!("action {target:?} not found at node {node_idx}"));
                children[pos] as usize
            }
            _ => panic!("node {node_idx} is not a Decision"),
        }
    }

    fn count_max_raise_depth(tree: &PreflopTree, node_idx: usize, raises: usize) -> usize {
        match &tree.nodes[node_idx] {
            PreflopNode::Terminal { .. } => raises,
            PreflopNode::Decision { children, action_labels, .. } => {
                children.iter().zip(action_labels.iter()).map(|(&child, label)| {
                    let new_raises = if matches!(label, PreflopAction::Raise(_) | PreflopAction::AllIn) {
                        raises + 1
                    } else {
                        raises
                    };
                    count_max_raise_depth(tree, child as usize, new_raises)
                }).max().unwrap_or(raises)
            }
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core --lib preflop::tree::tests -- --no-run 2>&1 | head -20`
Expected: Compilation error — `PreflopTree`, `PreflopNode` not defined.

**Step 3: Implement the tree builder**

`crates/core/src/preflop/tree.rs`:

```rust
use super::config::PreflopConfig;

/// An action in the preflop tree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreflopAction {
    Fold,
    Call,
    Raise(f64),
    AllIn,
}

/// How a hand ended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TerminalType {
    Fold { folder: u8 },
    Showdown,
}

/// A node in the preflop game tree.
#[derive(Debug, Clone)]
pub enum PreflopNode {
    Decision {
        position: u8,
        children: Vec<u32>,
        action_labels: Vec<PreflopAction>,
    },
    Terminal {
        terminal_type: TerminalType,
        pot: u32,
    },
}

/// Arena-allocated preflop game tree.
#[derive(Debug, Clone)]
pub struct PreflopTree {
    pub nodes: Vec<PreflopNode>,
}

impl PreflopTree {
    /// Build the full preflop tree from config.
    #[must_use]
    pub fn build(config: &PreflopConfig) -> Self {
        let mut nodes = Vec::new();
        let stacks = post_blind_stacks(config);
        let pot = config.initial_pot();
        let to_call = biggest_blind_diff(config);
        // SB acts first in HU, UTG in multiway
        let first_to_act = if config.num_players() == 2 { 0 } else { 0 };
        build_node(
            config, &mut nodes, &stacks, pot, to_call,
            first_to_act, 0, /* raise_count */ 0,
        );
        Self { nodes }
    }

    /// Number of nodes in the tree.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}
```

The `build_node` function recursively builds the tree. It handles:
- Fold → terminal
- Call → if all players have acted and pot is level → showdown; else next player
- Raise → next player with new to_call, increment raise count
- All-in → next player with max to_call
- Raise cap enforcement

Full implementation: recursive DFS that pushes nodes into the arena, returns the node index. Similar to `build_recursive` in `game_tree.rs` but simpler (no street transitions, no board).

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core --lib preflop::tree::tests -v`
Expected: 5 tests PASS.

**Step 5: Commit**

```bash
git add crates/core/src/preflop/tree.rs
git commit -m "feat(preflop): add PreflopTree builder with fold/call/raise/all-in"
```

---

### Task 3: HU Equity Table

**Files:**
- Create: `crates/core/src/preflop/equity.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn equity_table_is_169x169() {
        let table = EquityTable::compute_hu();
        assert_eq!(table.equities.len(), 169);
        assert_eq!(table.equities[0].len(), 169);
    }

    #[timed_test]
    fn aa_vs_kk_equity_is_above_80_pct() {
        let table = EquityTable::compute_hu();
        let aa = 0;  // AA = index 0
        let kk = 1;  // KK = index 1
        assert!(table.equity(aa, kk) > 0.80, "AA vs KK should be >80%");
    }

    #[timed_test]
    fn equity_is_symmetric() {
        let table = EquityTable::compute_hu();
        for i in 0..169 {
            for j in 0..169 {
                let diff = (table.equity(i, j) + table.equity(j, i) - 1.0).abs();
                assert!(diff < 0.001, "equity({i},{j}) + equity({j},{i}) should ≈ 1.0");
            }
        }
    }

    #[timed_test]
    fn combo_weights_sum_correctly() {
        let table = EquityTable::compute_hu();
        // AA vs KK: 6 combos of AA * 6 combos of KK = 36 matchups, no card overlap
        assert_eq!(table.weight(0, 1), 36.0);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core --lib preflop::equity::tests -- --no-run 2>&1 | head -20`
Expected: Compilation error.

**Step 3: Implement equity table**

`crates/core/src/preflop/equity.rs`:

The table precomputes equity for all 169x169 matchups. For each pair of canonical hands, enumerate all specific combo pairs (accounting for card removal — a shared card means 0 weight), and average equities weighted by combo counts.

Use existing `showdown_equity::compute_equity` for each specific matchup, or better: precompute the full 1326x1326 matrix and aggregate into 169x169. The 1326x1326 approach iterates all ~880K valid matchups once.

```rust
/// Precomputed equity table for all 169 canonical hand matchups.
#[derive(Debug, Clone)]
pub struct EquityTable {
    /// equities[i][j] = P(hand i beats hand j) including ties as 0.5
    equities: Vec<Vec<f64>>,
    /// weights[i][j] = number of valid combo pairs (card removal adjusted)
    weights: Vec<Vec<f64>>,
}

impl EquityTable {
    /// Compute the full 169x169 HU equity table.
    /// This is a one-time cost (~10-30 seconds in release mode).
    pub fn compute_hu() -> Self { /* ... */ }

    /// Look up precomputed equity.
    pub fn equity(&self, hand_i: usize, hand_j: usize) -> f64 {
        self.equities[hand_i][hand_j]
    }

    /// Look up combo weight for card-removal correction.
    pub fn weight(&self, hand_i: usize, hand_j: usize) -> f64 {
        self.weights[hand_i][hand_j]
    }
}
```

Implementation strategy: iterate all C(52,2)=1326 combos for player 1, all non-overlapping C(50,2)=1225 combos for player 2, compute `rs_poker` rank comparison (much faster than full board enumeration — just compare 2-card ranks for preflop all-in equity via Monte Carlo sampling of boards, or use existing `showdown_equity::compute_equity`).

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core --lib preflop::equity::tests -v --release`
Expected: 4 tests PASS (release mode for speed).

**Step 5: Commit**

```bash
git add crates/core/src/preflop/equity.rs crates/core/src/preflop/mod.rs
git commit -m "feat(preflop): add HU 169x169 equity table computation"
```

---

### Task 4: Linear CFR Preflop Solver

**Files:**
- Create: `crates/core/src/preflop/solver.rs`
- Modify: `crates/core/src/preflop/mod.rs` (add `pub mod solver;`)

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::preflop::PreflopConfig;
    use test_macros::timed_test;

    #[timed_test]
    fn solver_creates_from_config() {
        let config = PreflopConfig::heads_up(20);
        let solver = PreflopSolver::new(config);
        assert_eq!(solver.iteration(), 0);
    }

    #[timed_test]
    fn solver_runs_one_iteration() {
        let config = PreflopConfig::heads_up(20);
        let mut solver = PreflopSolver::new(config);
        solver.train(1);
        assert_eq!(solver.iteration(), 1);
    }

    #[timed_test]
    fn solver_strategy_is_valid_distribution() {
        let config = PreflopConfig::heads_up(20);
        let mut solver = PreflopSolver::new(config);
        solver.train(100);
        let strategy = solver.strategy();
        // Check a random hand's strategy sums to ~1.0
        let probs = strategy.get_root_probs(0); // hand 0 (AA) at root
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "strategy should sum to 1.0, got {sum}");
    }

    #[timed_test]
    fn solver_converges_aa_never_folds_preflop() {
        let config = PreflopConfig::heads_up(20);
        let mut solver = PreflopSolver::new(config);
        solver.train(1000);
        let strategy = solver.strategy();
        let aa_probs = strategy.get_root_probs(0); // AA at root
        // AA should essentially never fold at the root
        assert!(aa_probs[0] < 0.01, "AA fold probability should be <1%, got {}", aa_probs[0]);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core --lib preflop::solver::tests -- --no-run 2>&1 | head -20`
Expected: Compilation error.

**Step 3: Implement solver**

`crates/core/src/preflop/solver.rs`:

```rust
use rustc_hash::FxHashMap;
use super::config::PreflopConfig;
use super::tree::{PreflopTree, PreflopNode, PreflopAction};
use super::equity::EquityTable;

/// Extracted average strategy from the solver.
pub struct PreflopStrategy {
    /// (node_id, hand_index) → action probabilities
    strategies: FxHashMap<(u32, u16), Vec<f64>>,
    pub config: PreflopConfig,
    pub tree: PreflopTree,
}

pub struct PreflopSolver {
    tree: PreflopTree,
    config: PreflopConfig,
    equity: EquityTable,
    /// Cumulative regrets: (node_id, hand_index) → [regret per action]
    regret_sum: FxHashMap<(u32, u16), Vec<f64>>,
    /// Cumulative strategy: (node_id, hand_index) → [weighted strategy per action]
    strategy_sum: FxHashMap<(u32, u16), Vec<f64>>,
    iteration: u64,
}
```

The core loop: for each iteration `t`, traverse the tree for all 169 hands simultaneously. At each decision node for position `p`, compute counterfactual regrets by considering all 169 opponent hands (weighted by equity table and card removal). Apply Linear CFR weighting: multiply regret and strategy updates by `t`.

Key methods:
- `train(iterations)` — runs the LCFR loop
- `cfr_traverse(node_id, hand_idx, reach_probs)` — recursive CFR on the preflop tree
- `strategy()` — extract average strategy from `strategy_sum`

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core --lib preflop::solver::tests -v --release`
Expected: 4 tests PASS.

**Step 5: Commit**

```bash
git add crates/core/src/preflop/solver.rs crates/core/src/preflop/mod.rs
git commit -m "feat(preflop): add Linear CFR solver with convergence tests"
```

---

### Task 5: Preflop Strategy Persistence

**Files:**
- Create: `crates/core/src/preflop/bundle.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::preflop::PreflopConfig;
    use tempfile::TempDir;
    use test_macros::timed_test;

    #[timed_test]
    fn preflop_bundle_roundtrip() {
        let config = PreflopConfig::heads_up(20);
        let strategy = create_dummy_strategy(&config);
        let bundle = PreflopBundle::new(config, strategy);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("preflop_test");
        bundle.save(&path).unwrap();

        assert!(PreflopBundle::exists(&path));
        let loaded = PreflopBundle::load(&path).unwrap();
        assert_eq!(loaded.config.num_players(), 2);
    }

    #[timed_test]
    fn preflop_bundle_files_on_disk() {
        let config = PreflopConfig::heads_up(20);
        let strategy = create_dummy_strategy(&config);
        let bundle = PreflopBundle::new(config, strategy);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("preflop_test");
        bundle.save(&path).unwrap();

        assert!(path.join("config.yaml").exists());
        assert!(path.join("strategy.bin").exists());
    }
}
```

**Step 2: Run to verify failure, Step 3: Implement**

```rust
/// A saved preflop solution bundle.
pub struct PreflopBundle {
    pub config: PreflopConfig,
    pub strategy: PreflopStrategy,
}

impl PreflopBundle {
    pub fn save(&self, dir: &Path) -> Result<(), PreflopError>;
    pub fn load(dir: &Path) -> Result<Self, PreflopError>;
    pub fn exists(dir: &Path) -> bool;
}
```

Format:
```
preflop_solve/
  config.yaml
  strategy.bin    # bincode-serialized PreflopStrategy
```

**Step 4: Run tests, Step 5: Commit**

```bash
git commit -m "feat(preflop): add PreflopBundle save/load persistence"
```

---

### Task 6: Preflop Convergence Integration Test

**Files:**
- Create: `crates/core/tests/preflop_convergence.rs`

**Step 1: Write the test**

```rust
//! Integration test: verify the preflop solver converges to reasonable strategies.

use poker_solver_core::preflop::{PreflopConfig, PreflopSolver};

#[test]
fn hu_preflop_converges_in_1000_iterations() {
    let config = PreflopConfig::heads_up(20);
    let mut solver = PreflopSolver::new(config);
    solver.train(1000);
    let strategy = solver.strategy();

    // AA should raise, never fold
    let aa_root = strategy.get_root_probs(0);
    assert!(aa_root[0] < 0.01, "AA should not fold");

    // 72o should fold most of the time from SB
    let seven_two_off = 168; // last hand index
    let junk_root = strategy.get_root_probs(seven_two_off);
    assert!(junk_root[0] > 0.50, "72o should fold >50% from SB");

    // Overall strategy should sum to 1.0 for every hand
    for hand_idx in 0..169 {
        let probs = strategy.get_root_probs(hand_idx);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "hand {hand_idx}: sum = {sum}");
    }
}
```

**Step 2: Run:** `cargo test -p poker-solver-core --test preflop_convergence -v --release`

Expected: PASS.

**Step 3: Commit**

```bash
git commit -m "test(preflop): add convergence integration test for HU solver"
```

---

## Phase 2: Explorer Preflop Integration

### Task 7: Generalize `ExplorationPosition` to N Players

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs`

This is a refactor of the existing types. The `ExplorationPosition` struct changes from `stack_p1`/`stack_p2` to `stacks: Vec<u32>`, and `compute_position_state` is updated to handle N players.

**Step 1: Write tests for the new position types**

Add to the `exploration.rs` test module:

```rust
#[timed_test]
fn exploration_position_default_has_two_stacks() {
    let pos = ExplorationPosition::default();
    assert_eq!(pos.stacks.len(), 2);
    assert_eq!(pos.num_players, 2);
}

#[timed_test]
fn compute_position_state_matches_legacy() {
    let pos = ExplorationPosition {
        board: vec![],
        history: vec![],
        pot: 3,
        stacks: vec![199, 198],
        to_act: 0,
        num_players: 2,
        active_players: vec![true, true],
    };
    let state = compute_position_state(&[0.5, 1.0], &pos);
    assert_eq!(state.pot, 3);
    assert_eq!(state.stacks[0], 199);
    assert_eq!(state.stacks[1], 198);
}
```

**Step 2-4: Refactor, run tests, verify existing tests still pass**

Key changes:
- `ExplorationPosition`: replace `stack_p1`/`stack_p2` with `stacks: Vec<u32>`, add `num_players: u8`, `active_players: Vec<bool>`
- `PositionState`: replace `stack_p1`/`stack_p2` with `stacks: Vec<u32>`
- `compute_position_state`: index into `stacks` vec
- `get_strategy_matrix`: use `position.stacks[position.to_act as usize]` instead of if/else on `to_act`
- Keep backward compatibility: `Default` for `ExplorationPosition` produces the same 2-player state

Run: `cargo test -p poker-solver-tauri -v`
Expected: All existing tests PASS.

**Step 5: Commit**

```bash
git commit -m "refactor(explorer): generalize ExplorationPosition to N players"
```

---

### Task 8: Add `PreflopSolve` Strategy Source to Explorer

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs`

**Step 1: Write tests**

```rust
#[timed_test]
fn load_preflop_solve_sets_source() {
    // Test that loading a preflop bundle sets the strategy source
    // This requires a saved preflop bundle on disk (use tempdir)
}
```

**Step 2-3: Implement**

Add the new variant to `StrategySource`:

```rust
enum StrategySource {
    Bundle { config: BundleConfig, blueprint: BlueprintStrategy },
    Agent(AgentConfig),
    PreflopSolve {
        config: PreflopConfig,
        strategy: PreflopStrategy,
    },
}
```

New Tauri commands:

```rust
#[tauri::command]
pub async fn load_preflop_solve(
    state: State<'_, ExplorationState>,
    path: String,
) -> Result<BundleInfo, String>;

#[tauri::command]
pub async fn solve_preflop(
    state: State<'_, ExplorationState>,
    stack_depth: u32,
    num_players: u8,
) -> Result<BundleInfo, String>;
```

Update `get_strategy_matrix` dispatch:

```rust
match source {
    StrategySource::Bundle { .. } => { /* existing */ }
    StrategySource::Agent(_) => { /* existing */ }
    StrategySource::PreflopSolve { config, strategy } => {
        get_strategy_matrix_preflop(config, strategy, &position)
    }
}
```

The `get_strategy_matrix_preflop` function looks up each of the 169 hands in the `PreflopStrategy` at the current tree node (determined by replaying the action history).

Register new commands in `main.rs`.

**Step 4: Run tests, Step 5: Commit**

```bash
git commit -m "feat(explorer): add PreflopSolve source with load and on-the-fly solve"
```

---

## Phase 3: Coarse Postflop Blueprint

### Task 9: Verify LCFR Config Works with Existing Solver

**Files:**
- Create: `crates/core/tests/lcfr_convergence.rs`

This task confirms that setting `dcfr_alpha=1, dcfr_beta=1, dcfr_gamma=1` on the existing `SequenceCfrSolver` produces valid LCFR behavior. No new code — just a test.

**Step 1: Write the test**

```rust
use poker_solver_core::cfr::{SequenceCfrConfig, SequenceCfrSolver};
// Use Kuhn poker as a small testbed

#[test]
fn lcfr_config_converges_on_kuhn() {
    let config = SequenceCfrConfig {
        dcfr_alpha: 1.0,
        dcfr_beta: 1.0,
        dcfr_gamma: 1.0,
    };
    // Build Kuhn poker solver with LCFR config, train 10K iterations
    // Verify exploitability < 0.01
}
```

**Step 2: Run, verify, commit**

```bash
git commit -m "test: verify LCFR config converges on Kuhn poker"
```

---

### Task 10: Add `is_subgame_base` and `num_players` to BundleConfig

**Files:**
- Modify: `crates/core/src/blueprint/bundle.rs`

**Step 1: Write tests**

```rust
#[timed_test]
fn bundle_config_defaults_to_hu() {
    let config: BundleConfig = serde_yaml::from_str("game:\n  stack_depth: 20\n  bet_sizes: [0.5]").unwrap();
    assert_eq!(config.num_players, 2);
    assert!(!config.is_subgame_base);
}

#[timed_test]
fn bundle_config_roundtrips_new_fields() {
    let config = BundleConfig {
        num_players: 6,
        is_subgame_base: true,
        ..Default::default()
    };
    let yaml = serde_yaml::to_string(&config).unwrap();
    let loaded: BundleConfig = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(loaded.num_players, 6);
    assert!(loaded.is_subgame_base);
}
```

**Step 2-4: Implement, run tests, verify no regressions**

Add fields with `#[serde(default)]` for backward compatibility.

**Step 5: Commit**

```bash
git commit -m "feat(blueprint): add num_players and is_subgame_base to BundleConfig"
```

---

### Task 11: Add `linear_cfr()` Constructor to `SequenceCfrConfig`

**Files:**
- Modify: `crates/core/src/cfr/sequence_cfr.rs`

**Step 1: Write test**

```rust
#[timed_test]
fn linear_cfr_config_has_all_ones() {
    let config = SequenceCfrConfig::linear_cfr();
    assert_eq!(config.dcfr_alpha, 1.0);
    assert_eq!(config.dcfr_beta, 1.0);
    assert_eq!(config.dcfr_gamma, 1.0);
}
```

**Step 2-4: Implement (3-line constructor), run tests**

**Step 5: Commit**

```bash
git commit -m "feat(cfr): add SequenceCfrConfig::linear_cfr() constructor"
```

---

## Phase 4: Blueprint Reach and Value Queries

### Task 12: Opponent Reach Computation

**Files:**
- Modify: `crates/core/src/blueprint/strategy.rs`

**Step 1: Write tests**

```rust
#[timed_test]
fn opponent_reach_uniform_when_no_actions() {
    let mut blueprint = BlueprintStrategy::new();
    // With no actions taken, all hands should have reach = 1.0
    let reach = blueprint.opponent_reach_preflop(&[], 169);
    assert_eq!(reach.len(), 169);
    for &r in &reach {
        assert!((r - 1.0).abs() < 0.001);
    }
}

#[timed_test]
fn opponent_reach_narrows_after_raise() {
    let mut blueprint = BlueprintStrategy::new();
    // Insert strategy: hand 0 (AA) raises 100%, hand 168 (72o) folds 100%
    // After observing a raise, opponent reach for AA should be 1.0
    // and reach for 72o should be 0.0
    // ... setup keys and strategies ...
    let reach = blueprint.opponent_reach_preflop(&actions, 169);
    assert!((reach[0] - 1.0).abs() < 0.01, "AA should have full reach after raise");
    assert!(reach[168] < 0.01, "72o should have zero reach after raise");
}
```

**Step 2-4: Implement, run tests**

```rust
impl BlueprintStrategy {
    /// Compute opponent reach probabilities for all hands at a given node.
    ///
    /// Walks the action history, multiplying the blueprint probability of
    /// each action the opponent took. Returns a vector of length `num_hands`.
    pub fn opponent_reach_preflop(
        &self,
        actions: &[(u8, Action)], // (position, action) pairs
        num_hands: usize,
    ) -> Vec<f64>;
}
```

**Step 5: Commit**

```bash
git commit -m "feat(blueprint): add opponent_reach_preflop computation"
```

---

### Task 13: Leaf Continuation Value Computation

**Files:**
- Modify: `crates/core/src/blueprint/strategy.rs`

This is the most complex new query. Given a node at a street boundary, compute the expected value of each hand by doing a forward pass through the blueprint's subtree below that node, weighted by opponent reach.

**Step 1: Write tests**

```rust
#[timed_test]
fn continuation_value_returns_per_hand_values() {
    // Setup: blueprint with known strategies for a simple subtree
    // Verify that continuation values are in a reasonable range
    // (between -pot/2 and +pot/2 for zero-sum game)
}
```

**Step 2-4: Implement**

```rust
impl BlueprintStrategy {
    pub fn continuation_values(
        &self,
        hand_bits_fn: impl Fn(usize) -> u32,  // hand_idx → hand_bits for next street
        street: u8,
        spr: u32,
        action_codes: &[u8],
        opponent_reach: &[f64],
        num_hands: usize,
    ) -> Vec<f64>;
}
```

This traverses the blueprint subtree (all action sequences from the boundary node), computing expected values. The implementation is a depth-first walk of possible action sequences, multiplying reach probabilities and summing terminal values.

**Step 5: Commit**

```bash
git commit -m "feat(blueprint): add continuation_values for depth-limited subgame leaves"
```

---

## Phase 5: Real-Time Subgame Solver

### Task 14: Subgame Tree Builder

**Files:**
- Create: `crates/core/src/blueprint/subgame_tree.rs`
- Modify: `crates/core/src/blueprint/mod.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn subgame_tree_for_river_has_no_depth_boundary() {
        let board = parse_test_board("As Kh 7d 4c Tc");
        let tree = SubgameTreeBuilder::new()
            .board(&board)
            .bet_sizes(&[0.5, 1.0, 2.0])
            .pot(100)
            .stacks(&[200, 200])
            .build();
        // River: no depth boundary needed (final street)
        assert!(!tree.nodes.iter().any(|n| matches!(n, SubgameNode::DepthBoundary { .. })));
    }

    #[timed_test]
    fn subgame_tree_for_flop_has_depth_boundaries() {
        let board = parse_test_board("As Kh 7d");
        let tree = SubgameTreeBuilder::new()
            .board(&board)
            .bet_sizes(&[0.5, 1.0])
            .pot(100)
            .stacks(&[200, 200])
            .depth_limit(1) // only solve flop
            .build();
        // Should have DepthBoundary nodes where flop action completes
        let boundaries = tree.nodes.iter()
            .filter(|n| matches!(n, SubgameNode::DepthBoundary { .. }))
            .count();
        assert!(boundaries > 0, "depth-limited flop tree should have boundary nodes");
    }

    #[timed_test]
    fn subgame_hands_excludes_board_blockers() {
        let board = parse_test_board("As Kh 7d 4c Tc");
        let hands = SubgameHands::enumerate(&board);
        // 52 - 5 = 47 remaining cards, C(47,2) = 1081
        assert_eq!(hands.combos.len(), 1081);
    }
}
```

**Step 2-4: Implement**

```rust
pub struct SubgameTreeBuilder { /* ... */ }

impl SubgameTreeBuilder {
    pub fn new() -> Self;
    pub fn board(self, board: &[Card]) -> Self;
    pub fn bet_sizes(self, sizes: &[f32]) -> Self;
    pub fn pot(self, pot: u32) -> Self;
    pub fn stacks(self, stacks: &[u32]) -> Self;
    pub fn depth_limit(self, streets: usize) -> Self;
    pub fn build(self) -> SubgameTree;
}
```

The builder creates a concrete action tree for the given board and sizing. At depth boundaries (when the street would advance past the limit), it inserts `DepthBoundary` nodes.

**Step 5: Commit**

```bash
git commit -m "feat(subgame): add SubgameTreeBuilder with depth-limited boundaries"
```

---

### Task 15: Subgame CFR+ Solver

**Files:**
- Create: `crates/core/src/blueprint/subgame_cfr.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn subgame_solver_produces_valid_strategies() {
        // River subgame with known hands and equities
        let board = parse_test_board("As Kh 7d 4c Tc");
        let result = SubgameCfrSolver::solve(
            &board,
            100, // pot
            &[200, 200], // stacks
            &uniform_reach(1081),
            &[], // no leaf values (river)
            &SubgameConfig::default(),
        );
        // Every combo's strategy should sum to 1.0
        for combo_idx in 0..result.num_combos() {
            let probs = result.root_probs(combo_idx);
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 0.01);
        }
    }

    #[timed_test]
    fn subgame_solver_nuts_always_bets() {
        // Setup a river where one combo is the absolute nuts
        // It should bet/raise at very high frequency
    }
}
```

**Step 2-4: Implement**

The solver runs CFR+ (non-negative regret accumulation) on the subgame tree. Each iteration:
1. For each combo, compute reach probabilities (forward pass)
2. Compute counterfactual values (backward pass)
3. Update regrets (clamped to non-negative)
4. Update weighted strategy sums

```rust
pub struct SubgameCfrSolver { /* regrets, strategy sums, tree, hands */ }

impl SubgameCfrSolver {
    pub fn solve(
        board: &[Card],
        pot: u32,
        stacks: &[u32],
        opponent_reach: &[f64],
        leaf_values: &[f64],
        config: &SubgameConfig,
    ) -> SubgameStrategy;
}

pub struct SubgameStrategy {
    /// combo_idx → node_id → action probabilities
    strategies: Vec<Vec<Vec<f64>>>,
}
```

**Step 5: Commit**

```bash
git commit -m "feat(subgame): add CFR+ real-time solver with depth-limited support"
```

---

### Task 16: Subgame Solver Integration Test

**Files:**
- Create: `crates/core/tests/subgame_integration.rs`

**Step 1: Write test**

```rust
#[test]
fn river_subgame_solve_produces_reasonable_strategy() {
    // Load a test blueprint, pick a specific river board,
    // compute opponent reach, solve the subgame,
    // verify that strong hands bet and weak hands check/fold
}

#[test]
fn depth_limited_flop_solve_uses_leaf_values() {
    // Flop subgame with depth limit = 1 street
    // Verify that DepthBoundary nodes use continuation values
    // from the blueprint rather than continuing to turn
}
```

**Step 2: Run, commit**

```bash
git commit -m "test(subgame): add integration tests for river and depth-limited solves"
```

---

## Phase 6: Explorer Subgame Integration

### Task 17: Add `SubgameSolve` Strategy Source

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs`
- Modify: `crates/tauri-app/src/main.rs`

**Step 1: Add new source variant and commands**

```rust
enum StrategySource {
    // ... existing ...
    SubgameSolve {
        blueprint: Arc<BlueprintStrategy>,
        blueprint_config: BundleConfig,
        subgame_config: SubgameConfig,
        /// Cache of solved subgames: board+history hash → SubgameStrategy
        solve_cache: Arc<RwLock<HashMap<u64, SubgameStrategy>>>,
    },
}
```

New commands:

```rust
#[tauri::command]
pub async fn load_subgame_source(
    state: State<'_, ExplorationState>,
    blueprint_path: String,
    subgame_config: SubgameConfig,
) -> Result<BundleInfo, String>;

#[tauri::command]
pub fn get_subgame_status(
    state: State<'_, ExplorationState>,
) -> SubgameStatus;
```

**Step 2: Implement async subgame dispatch in `get_strategy_matrix`**

When source is `SubgameSolve` and the position is postflop:
1. Hash (board, history) as cache key
2. If cached, return immediately
3. If not, start background solve, return placeholder (or error with "solving in progress")
4. Emit `subgame-solving` / `subgame-solved` events

**Step 3: Register new commands in `main.rs`**

**Step 4: Run full test suite**

Run: `cargo test -p poker-solver-tauri -v`

**Step 5: Commit**

```bash
git commit -m "feat(explorer): add SubgameSolve source with async real-time CFR+"
```

---

### Task 18: Add Subgame Progress Events

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs`

Add `SubgameProgressEvent` (mirrors existing `BucketProgressEvent`):

```rust
#[derive(Debug, Clone, Serialize)]
pub struct SubgameProgressEvent {
    pub iteration: u32,
    pub max_iterations: u32,
    pub board_key: String,
    pub elapsed_ms: u64,
}
```

Emit every 100 iterations during the background solve thread.

```bash
git commit -m "feat(explorer): add subgame solve progress events"
```

---

## Phase 7: End-to-End and Cleanup

### Task 19: Trainer CLI for Coarse Blueprint

**Files:**
- Modify: `crates/trainer/src/main.rs`

Add a training config option for coarse blueprint mode:

```yaml
# training_coarse_blueprint.yaml
algorithm: sequence_cfr
cfr_variant: linear  # NEW: uses alpha=1, beta=1, gamma=1
abstraction_mode: hand_class_v2
strength_bits: 2
equity_bits: 2
bet_sizes: [0.5, 1.0, 2.0]
stack_depth: 100
iterations: 10000
```

The trainer reads `cfr_variant: linear` and constructs `SequenceCfrConfig::linear_cfr()`.

```bash
git commit -m "feat(trainer): support cfr_variant: linear for LCFR blueprint training"
```

---

### Task 20: Preflop Trainer CLI Command

**Files:**
- Modify: `crates/trainer/src/main.rs`

Add a `solve-preflop` subcommand:

```bash
cargo run -p poker-solver-trainer -- solve-preflop \
    --stack-depth 100 \
    --players 2 \
    --iterations 5000 \
    --output preflop_hu_100bb/
```

```bash
git commit -m "feat(trainer): add solve-preflop CLI command"
```

---

### Task 21: Full Pipeline Integration Test

**Files:**
- Create: `crates/core/tests/full_pipeline.rs`

End-to-end test that:
1. Solves preflop (HU, 20BB, 1000 iterations)
2. Trains a coarse blueprint (20BB, few iterations)
3. Runs a subgame solve on a specific river board
4. Verifies strategies are valid distributions
5. Verifies the whole pipeline completes without errors

```bash
git commit -m "test: add full pipeline integration test (preflop → blueprint → subgame)"
```

---

## Summary

| Phase | Tasks | New files | Modified files |
|-|-|-|-|
| 1: Preflop solver | 1-6 | 5 new in `preflop/` | `lib.rs` |
| 2: Explorer preflop | 7-8 | 0 | `exploration.rs`, `main.rs` |
| 3: Coarse blueprint | 9-11 | 1 test | `bundle.rs`, `sequence_cfr.rs` |
| 4: Reach/value queries | 12-13 | 0 | `strategy.rs` |
| 5: Subgame solver | 14-16 | 2 new in `blueprint/` | `blueprint/mod.rs` |
| 6: Explorer subgame | 17-18 | 0 | `exploration.rs`, `main.rs` |
| 7: End-to-end | 19-21 | 2 tests | `trainer/main.rs` |

Phases 1-2 and 3-4 are independent and can be parallelized.
