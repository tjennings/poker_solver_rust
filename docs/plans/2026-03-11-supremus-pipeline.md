# Supremus Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Implement the full Supremus training pipeline — street-boundary CFV nets for turn and flop (river is done), DCFR+ Supremus variant, street-generic datagen, then decision-point value nets for instant strategy serving.

**Architecture:** Bottom-up neural net pipeline. Street-boundary CFV nets provide leaf values for one-street-ahead solving via the existing `LeafEvaluator` trait and `CfvSubgameSolver`. Decision-point value nets (Layer 2) distill solved strategies into instant-lookup networks. `RiverNetEvaluator` already bridges the river CfvNet to the `LeafEvaluator` trait — we extend this pattern to turn and flop.

**Tech Stack:** Rust, burn 0.16 (ML framework), range-solver (exact postflop solver), poker-solver-core (game tree, CFR infrastructure)

**Reference:** `docs/supremus-pipeline.md` for the full design and Supremus paper details.

---

## What Already Exists

Before starting, know what's already built:

| Component | Status | Key Files |
|-|-|-|
| River CFV net (training, inference, eval) | Done | `crates/cfvnet/` |
| `LeafEvaluator` trait | Done | `crates/core/src/blueprint_v2/cfv_subgame_solver.rs:30-55` |
| `CfvSubgameSolver` (DCFR with dynamic leaf eval) | Done | `crates/core/src/blueprint_v2/cfv_subgame_solver.rs:123-444` |
| `RiverNetEvaluator` (river CfvNet → LeafEvaluator) | Done | `crates/cfvnet/src/eval/river_net_evaluator.rs` |
| `GameTree::build_subgame()` with depth_limit | Done | `crates/core/src/blueprint_v2/game_tree.rs:733` |
| `DcfrParams` (Vanilla/DCFR/CFR+/Linear variants) | Done | `crates/core/src/cfr/dcfr.rs` |
| Street-aware config (`DatagenConfig.street`) | Done | `crates/cfvnet/src/config.rs` |
| Model path config (`GameConfig.river_model_path`) | Done | `crates/cfvnet/src/config.rs:38` |

---

## Milestone 1: DCFR+ Supremus Variant

### Task 1: Add `strategy_delay` to DcfrParams

The Supremus paper uses DCFR+ with a d=100 delay before accumulating the average strategy, and linear weighting t/(t+1) after the delay. The current `DcfrParams` has `warmup` (delays discounting) but no concept of delaying strategy accumulation.

**Files:**
- Modify: `crates/core/src/cfr/dcfr.rs`
- Test: inline `#[cfg(test)]` in same file

**Step 1: Write the failing tests**

Add to the existing `mod tests` block in `crates/core/src/cfr/dcfr.rs`:

```rust
#[timed_test]
fn supremus_constructor_values() {
    let p = DcfrParams::supremus();
    assert_eq!(p.variant, CfrVariant::Linear);
    assert!((p.alpha - 1.0).abs() < f64::EPSILON);
    assert!((p.beta - 1.0).abs() < f64::EPSILON);
    assert!((p.gamma - 1.0).abs() < f64::EPSILON);
    assert_eq!(p.warmup, 0);
    assert_eq!(p.strategy_delay, 100);
}

#[timed_test]
fn should_accumulate_strategy_respects_delay() {
    let p = DcfrParams::supremus();
    assert!(!p.should_accumulate_strategy(0));
    assert!(!p.should_accumulate_strategy(99));
    assert!(!p.should_accumulate_strategy(100));
    assert!(p.should_accumulate_strategy(101));
    assert!(p.should_accumulate_strategy(1000));
}

#[timed_test]
fn should_accumulate_strategy_no_delay() {
    let p = DcfrParams::default();
    assert!(p.should_accumulate_strategy(0));
    assert!(p.should_accumulate_strategy(1));
}

#[timed_test]
fn strategy_weight_linear_after_delay() {
    let p = DcfrParams::supremus();
    // Before delay: weight should be 0 (no accumulation)
    assert!(!p.should_accumulate_strategy(50));
    // After delay: linear weight t/(t+1)
    let (_, sw) = p.iteration_weights(200);
    assert!((sw - 200.0).abs() < f64::EPSILON);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core dcfr::tests::supremus -- --no-run 2>&1 | head -5`
Expected: compilation error — `supremus()` and `strategy_delay` don't exist

**Step 3: Implement**

In `crates/core/src/cfr/dcfr.rs`, add `strategy_delay` field to `DcfrParams`:

```rust
pub struct DcfrParams {
    pub variant: CfrVariant,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub warmup: u64,
    /// Delay before accumulating average strategy (Supremus DCFR+ d parameter).
    /// Strategy sums are only updated when `iteration > strategy_delay`.
    pub strategy_delay: u64,
}
```

Update `Default`:
```rust
impl Default for DcfrParams {
    fn default() -> Self {
        Self {
            variant: CfrVariant::Dcfr,
            alpha: 1.5,
            beta: 0.5,
            gamma: 2.0,
            warmup: 0,
            strategy_delay: 0,
        }
    }
}
```

Add `strategy_delay: 0` to all existing constructors (`linear()`, `vanilla()`, `from_config()`).

Add new constructor and method:
```rust
/// Supremus DCFR+: linear weighting with d=100 strategy delay.
///
/// - Linear discounting (α=β=γ=1.0)
/// - Strategy accumulation delayed by 100 iterations
/// - Simultaneous updates (handled by caller)
#[must_use]
pub fn supremus() -> Self {
    Self {
        variant: CfrVariant::Linear,
        alpha: 1.0,
        beta: 1.0,
        gamma: 1.0,
        warmup: 0,
        strategy_delay: 100,
    }
}

/// Whether strategy sums should be accumulated at this iteration.
///
/// Returns `false` for the first `strategy_delay` iterations.
#[must_use]
pub fn should_accumulate_strategy(&self, iteration: u64) -> bool {
    iteration > self.strategy_delay
}
```

Update `from_config` to accept a `strategy_delay` parameter (add it as the last arg with a default-compatible overload or just extend the signature).

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core dcfr::tests`
Expected: all tests pass, including the new ones

**Step 5: Wire strategy_delay into CfvSubgameSolver**

In `crates/core/src/blueprint_v2/cfv_subgame_solver.rs`, the `train()` method currently always accumulates strategy sums. Add a check:

In the `traverse_as_traverser` method (line ~604), the strategy delta is always written. Instead, condition strategy accumulation in the `train()` method where `add_into` is called:

```rust
// In train(), after parallel_traverse:
add_into(&mut self.regret_sum, &regret_delta);
if self.dcfr.should_accumulate_strategy(u64::from(self.iteration)) {
    add_into(&mut self.strategy_sum, &strategy_delta);
}
```

**Step 6: Add a method to set DCFR params on CfvSubgameSolver**

```rust
/// Set the DCFR parameters (e.g., for Supremus variant).
pub fn set_dcfr_params(&mut self, dcfr: DcfrParams) {
    self.dcfr = dcfr;
}
```

**Step 7: Run full test suite**

Run: `cargo test -p poker-solver-core`
Expected: all tests pass

**Step 8: Commit**

```bash
git add crates/core/src/cfr/dcfr.rs crates/core/src/blueprint_v2/cfv_subgame_solver.rs
git commit -m "feat: add DCFR+ Supremus variant with strategy_delay=100"
```

---

## Milestone 2: Street-Generic Datagen with Net-at-Leaf

### Task 2: Add model path config for turn and flop

**Files:**
- Modify: `crates/cfvnet/src/config.rs`

**Step 1: Write the failing test**

Add to `mod tests` in `crates/cfvnet/src/config.rs`:

```rust
#[test]
fn parse_config_with_model_paths() {
    let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
  river_model_path: "/models/river/model"
  turn_model_path: "/models/turn/model"
datagen:
  num_samples: 100
  street: flop
"#;
    let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(config.game.river_model_path.as_deref(), Some("/models/river/model"));
    assert_eq!(config.game.turn_model_path.as_deref(), Some("/models/turn/model"));
    assert_eq!(config.datagen.street, "flop");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet config::tests::parse_config_with_model_paths`
Expected: FAIL — `turn_model_path` field doesn't exist

**Step 3: Implement**

Add to `GameConfig` in `crates/cfvnet/src/config.rs`:

```rust
/// Path to a trained turn CFV model (used by flop datagen for leaf evaluation).
#[serde(default)]
pub turn_model_path: Option<String>,
```

Add `turn_model_path: None` to the `Default` impl.

**Step 4: Run test**

Run: `cargo test -p cfvnet config::tests`
Expected: all pass

**Step 5: Commit**

```bash
git add crates/cfvnet/src/config.rs
git commit -m "feat: add turn_model_path to GameConfig for multi-street pipeline"
```

---

### Task 3: Turn datagen using CfvSubgameSolver + RiverNetEvaluator

This is the core of the pipeline: generate turn training data by solving turn subgames with the river net at leaf (river transition) nodes.

The existing `generate_training_data()` uses `range-solver` directly. For turn datagen, we need to:
1. Load the trained river model
2. Build a one-street lookahead tree (turn only, depth_limit=Some(1))
3. Solve with `CfvSubgameSolver` using `RiverNetEvaluator` at leaves
4. Extract CFVs from the solution
5. Write training records in the same format

**Files:**
- Create: `crates/cfvnet/src/datagen/cfv_solver.rs`
- Modify: `crates/cfvnet/src/datagen/mod.rs`
- Modify: `crates/cfvnet/src/datagen/generate.rs`
- Test: inline in `cfv_solver.rs` + integration test

**Step 1: Write the failing test**

Create `crates/cfvnet/src/datagen/cfv_solver.rs` with:

```rust
//! Solve subgames using CfvSubgameSolver with a LeafEvaluator.
//!
//! Used for turn/flop datagen where leaf values come from a trained
//! CFV network instead of exact showdown equity.

use poker_solver_core::blueprint_v2::cfv_subgame_solver::LeafEvaluator;
use poker_solver_core::blueprint_v2::game_tree::GameTree;
use poker_solver_core::blueprint_v2::subgame_cfr::SubgameHands;
use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::cfr::dcfr::DcfrParams;
use poker_solver_core::poker::Card;
use range_solver::card::card_pair_to_index;

use super::range_gen::NUM_COMBOS;

/// Result of solving a subgame with CfvSubgameSolver.
pub struct CfvSolveResult {
    /// Pot-relative CFVs for OOP, indexed by canonical combo (0..1326).
    pub oop_cfvs: [f32; NUM_COMBOS],
    /// Pot-relative CFVs for IP, indexed by canonical combo (0..1326).
    pub ip_cfvs: [f32; NUM_COMBOS],
    /// Weighted game value for OOP.
    pub oop_game_value: f32,
    /// Weighted game value for IP.
    pub ip_game_value: f32,
    /// Which combos were present (not board-blocked).
    pub valid_mask: [bool; NUM_COMBOS],
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal test that the module compiles and types are correct
    #[test]
    fn cfv_solve_result_has_correct_sizes() {
        let result = CfvSolveResult {
            oop_cfvs: [0.0; NUM_COMBOS],
            ip_cfvs: [0.0; NUM_COMBOS],
            oop_game_value: 0.0,
            ip_game_value: 0.0,
            valid_mask: [false; NUM_COMBOS],
        };
        assert_eq!(result.oop_cfvs.len(), 1326);
        assert_eq!(result.valid_mask.len(), 1326);
    }
}
```

**Step 2: Run test to verify it passes (just types)**

Run: `cargo test -p cfvnet datagen::cfv_solver::tests`
Expected: PASS

**Step 3: Implement `solve_with_evaluator`**

Add to `cfv_solver.rs`:

```rust
use crate::eval::river_net_evaluator::card_to_u8;

/// Solve a one-street subgame using CfvSubgameSolver with a dynamic leaf evaluator.
///
/// # Arguments
/// * `board` - Board cards in cfvnet u8 encoding (3 for flop, 4 for turn)
/// * `pot` - Pot size in chips
/// * `effective_stack` - Remaining effective stack in chips
/// * `oop_range` - OOP range as 1326-element array (cfvnet canonical ordering)
/// * `ip_range` - IP range as 1326-element array
/// * `bet_sizes` - Bet sizes as pot fractions (e.g., [0.5, 1.0])
/// * `evaluator` - LeafEvaluator for depth boundary nodes
/// * `iterations` - Number of DCFR iterations
/// * `dcfr` - DCFR parameters (use `DcfrParams::supremus()` for Supremus)
pub fn solve_with_evaluator(
    board: &[u8],
    pot: i32,
    effective_stack: i32,
    oop_range: &[f32; NUM_COMBOS],
    ip_range: &[f32; NUM_COMBOS],
    bet_sizes: &[f64],
    evaluator: Box<dyn LeafEvaluator>,
    iterations: u32,
    dcfr: DcfrParams,
) -> CfvSolveResult {
    let street = match board.len() {
        3 => Street::Flop,
        4 => Street::Turn,
        5 => Street::River,
        n => panic!("invalid board length {n}"),
    };

    // Convert board from cfvnet u8 to rs_poker Card for SubgameHands.
    let board_cards: Vec<Card> = board.iter().map(|&b| u8_to_rs_poker_card(b)).collect();

    // Build one-street lookahead tree with depth_limit=1.
    let invested = [pot as f64 / 2.0, pot as f64 / 2.0];
    let starting_stack = (pot as f64 / 2.0) + effective_stack as f64;
    let tree = GameTree::build_subgame(
        street,
        pot as f64,
        invested,
        starting_stack,
        &[bet_sizes.to_vec()],
        Some(1), // depth_limit = 1 street
    );

    let hands = SubgameHands::enumerate(&board_cards);
    let num_combos = hands.combos.len();

    // Map ranges from 1326-canonical to combo-indexed for the solver.
    // (The solver works with combo-indexed ranges internally.)

    let mut solver = poker_solver_core::blueprint_v2::CfvSubgameSolver::new(
        tree,
        hands.clone(),
        board_cards.clone(),
        evaluator,
        starting_stack,
    );
    solver.set_dcfr_params(dcfr);
    solver.train(iterations);

    // Extract average strategy and compute CFVs.
    // The strategy is what we need to derive CFVs from.
    // Actually, CfvSubgameSolver doesn't directly expose per-combo CFVs.
    // We need to run one more traversal to extract them.
    // For training data, we need the counterfactual values, not the strategy.
    //
    // Alternative: after training, do a final traversal with the converged
    // strategy to compute the expected value for each combo.
    let strategy = solver.strategy();

    // Extract CFVs by computing expected values under the converged strategy.
    // For each combo, the CFV is the expected value when playing according to
    // the average strategy.
    let mut oop_cfvs = [0.0f32; NUM_COMBOS];
    let mut ip_cfvs = [0.0f32; NUM_COMBOS];
    let mut valid_mask = [false; NUM_COMBOS];

    for (combo_idx, &combo) in hands.combos.iter().enumerate() {
        let c0 = card_to_u8(combo[0]);
        let c1 = card_to_u8(combo[1]);
        let canonical_idx = card_pair_to_index(c0, c1);
        valid_mask[canonical_idx] = true;

        // Get root strategy for this combo
        let probs = strategy.root_probs(combo_idx);
        if probs.is_empty() {
            continue;
        }

        // TODO: Extract actual CFVs from the solver.
        // For now, store 0.0 — the real implementation needs
        // a final CFR traversal to compute per-combo expected values.
    }

    let oop_game_value = weighted_sum_f32(oop_range, &oop_cfvs);
    let ip_game_value = weighted_sum_f32(ip_range, &ip_cfvs);

    CfvSolveResult {
        oop_cfvs,
        ip_cfvs,
        oop_game_value,
        ip_game_value,
        valid_mask,
    }
}

/// Convert cfvnet u8 encoding to rs_poker Card.
///
/// cfvnet encoding: rank * 4 + suit (club=0, diamond=1, heart=2, spade=3)
/// rs_poker: Value::from(rank), Suit mapping via lookup
fn u8_to_rs_poker_card(v: u8) -> Card {
    use poker_solver_core::poker::{Suit, Value};
    let rank = v / 4;
    let suit_id = v & 3;
    // cfvnet suit order: club=0, diamond=1, heart=2, spade=3
    // rs_poker suit order: Spade=0, Club=1, Heart=2, Diamond=3
    let suit = match suit_id {
        0 => Suit::Club,
        1 => Suit::Diamond,
        2 => Suit::Heart,
        3 => Suit::Spade,
        _ => unreachable!(),
    };
    Card::new(Value::from(rank), suit)
}

fn weighted_sum_f32(range: &[f32; NUM_COMBOS], values: &[f32; NUM_COMBOS]) -> f32 {
    range.iter().zip(values.iter()).map(|(&r, &v)| r * v).sum()
}
```

**Important implementation note:** The `CfvSubgameSolver` computes CFVs internally during traversal but doesn't expose them directly. The implementer needs to add a `cfvs()` method to `CfvSubgameSolver` that returns the per-combo expected values from the last traversal, OR compute them via a final evaluation pass. Check `crates/core/src/blueprint_v2/cfv_subgame_solver.rs` for the best approach — likely adding a method that does one final traversal with the average strategy and returns the resulting CFVs.

**Step 4: Add `cfv_solver` to mod.rs**

In `crates/cfvnet/src/datagen/mod.rs`, add:

```rust
pub mod cfv_solver;
```

**Step 5: Run tests**

Run: `cargo test -p cfvnet datagen::cfv_solver`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/cfvnet/src/datagen/cfv_solver.rs crates/cfvnet/src/datagen/mod.rs
git commit -m "feat: add CfvSubgameSolver-based datagen for turn/flop streets"
```

---

### Task 4: Add `compute_cfvs()` to CfvSubgameSolver

The solver needs to expose the counterfactual values for training data extraction.

**Files:**
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs`

**Step 1: Write the failing test**

Add to the existing `mod tests` in `cfv_subgame_solver.rs`:

```rust
#[timed_test(5)]
fn compute_cfvs_returns_both_players() {
    let board = turn_board();
    let tree = GameTree::build_subgame(
        Street::Turn, 100.0, [50.0, 50.0], 250.0,
        &[vec![1.0]], Some(1),
    );
    let hands = small_hands(&board, 30);
    let n = hands.combos.len();
    let mut solver = CfvSubgameSolver::new(
        tree, hands, board, Box::new(ConstantEvaluator), 250.0
    );
    solver.train(200);

    let cfvs = solver.compute_cfvs();
    assert_eq!(cfvs.len(), 2); // [OOP, IP]
    assert_eq!(cfvs[0].len(), n);
    assert_eq!(cfvs[1].len(), n);

    // CFVs should be finite
    for player in 0..2 {
        for &v in &cfvs[player] {
            assert!(v.is_finite(), "CFV should be finite");
        }
    }
}

#[timed_test(5)]
fn compute_cfvs_zero_sum_property() {
    let board = vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Heart),
        Card::new(Value::Seven, Suit::Diamond),
        Card::new(Value::Four, Suit::Club),
        Card::new(Value::Ten, Suit::Club),
    ];
    // Full-depth river: no boundaries, pure showdown.
    let tree = GameTree::build_subgame(
        Street::River, 100.0, [50.0, 50.0], 250.0,
        &[vec![1.0]], None,
    );
    let hands = small_hands(&board, 30);
    let n = hands.combos.len();
    let mut solver = CfvSubgameSolver::new(
        tree, hands, board, Box::new(ConstantEvaluator), 250.0
    );
    solver.train(300);

    let cfvs = solver.compute_cfvs();
    // In a zero-sum game with uniform ranges, sum of all CFVs should be ~0
    let total: f64 = cfvs[0].iter().zip(cfvs[1].iter()).map(|(a, b)| a + b).sum();
    let avg = total / (n as f64 * 2.0);
    assert!(
        avg.abs() < 5.0,
        "average CFV sum should be near 0, got {avg}"
    );
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core cfv_subgame_solver::tests::compute_cfvs -- --no-run 2>&1 | head -5`
Expected: compilation error — `compute_cfvs()` doesn't exist

**Step 3: Implement `compute_cfvs()`**

Add to `impl CfvSubgameSolver`:

```rust
/// Compute per-combo counterfactual values under the average strategy.
///
/// Returns `[oop_cfvs, ip_cfvs]` where each is a Vec<f64> indexed by
/// combo position. These are the training targets for CFV networks.
///
/// Performs a final traversal using the average strategy snapshot,
/// evaluating leaf nodes with the attached evaluator.
#[must_use]
pub fn compute_cfvs(&self) -> [Vec<f64>; 2] {
    let snapshot = self.build_average_strategy_snapshot();
    let num_combos = self.hands.combos.len();
    let mut result = [
        vec![0.0; num_combos],
        vec![0.0; num_combos],
    ];

    for traverser in 0..2u8 {
        for combo_idx in 0..num_combos {
            result[traverser as usize][combo_idx] = self.eval_cfv(
                &snapshot,
                self.tree.root as usize,
                combo_idx,
                traverser,
            );
        }
    }

    result
}

/// Build strategy snapshot from average strategy sums (not regret sums).
fn build_average_strategy_snapshot(&self) -> Vec<f64> {
    let mut snapshot = vec![0.0; self.layout.total_size];
    let num_combos = self.hands.combos.len();

    for (node_idx, node) in self.tree.nodes.iter().enumerate() {
        let num_actions = match node {
            GameNode::Decision { actions, .. } => actions.len(),
            _ => continue,
        };

        for combo_idx in 0..num_combos {
            let (base, _) = self.layout.slot(node_idx, combo_idx);
            let sums = &self.strategy_sum[base..base + num_actions];
            let total: f64 = sums.iter().sum();
            if total > 0.0 {
                for a in 0..num_actions {
                    snapshot[base + a] = sums[a] / total;
                }
            } else {
                // Uniform if no strategy accumulated
                let uniform = 1.0 / num_actions as f64;
                for a in 0..num_actions {
                    snapshot[base + a] = uniform;
                }
            }
        }
    }

    snapshot
}

/// Evaluate the counterfactual value of a single combo under a fixed strategy.
fn eval_cfv(
    &self,
    snapshot: &[f64],
    node_idx: usize,
    combo_idx: usize,
    traverser: u8,
) -> f64 {
    match &self.tree.nodes[node_idx] {
        GameNode::Terminal { kind, pot, .. } => {
            let half_pot = *pot / 2.0;
            match kind {
                TerminalKind::Fold { winner } => {
                    if *winner == traverser { half_pot } else { -half_pot }
                }
                TerminalKind::Showdown => {
                    // Use same showdown logic as CFR traversal
                    self.showdown_value_for_eval(combo_idx, half_pot, traverser)
                }
                TerminalKind::DepthBoundary => {
                    let b = self.node_to_boundary[node_idx];
                    self.leaf_cfvs[traverser as usize][b][combo_idx] * half_pot
                }
            }
        }
        GameNode::Chance { child, .. } => {
            self.eval_cfv(snapshot, *child as usize, combo_idx, traverser)
        }
        GameNode::Decision { player, children, .. } => {
            let (base, num_actions) = self.layout.slot(node_idx, combo_idx);
            let strategy = &snapshot[base..base + num_actions];
            let mut value = 0.0;
            for (a, &child_idx) in children.iter().enumerate() {
                value += strategy[a] * self.eval_cfv(
                    snapshot, child_idx as usize, combo_idx, traverser
                );
            }
            value
        }
    }
}
```

Note: `showdown_value_for_eval` may need to be factored out from the existing `showdown_value` on `CfvCfrCtx`. The implementer should check whether the existing private `showdown_value` can be reused or needs to be extracted into a shared helper.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core cfv_subgame_solver::tests`
Expected: all pass

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cfv_subgame_solver.rs
git commit -m "feat: add compute_cfvs() to CfvSubgameSolver for training data extraction"
```

---

### Task 5: Wire turn datagen into generate_training_data

Now connect everything: when `street == "turn"`, load the river model, create `RiverNetEvaluator`, and use `solve_with_evaluator` instead of the range-solver path.

**Files:**
- Modify: `crates/cfvnet/src/datagen/generate.rs`
- Modify: `crates/cfvnet/src/datagen/cfv_solver.rs` (finalize solve_with_evaluator)
- Test: integration test in `crates/cfvnet/tests/`

**Step 1: Write the failing integration test**

Create or add to `crates/cfvnet/tests/turn_datagen_test.rs`:

```rust
//! Integration test: generate turn training data using a (random) river model.
use cfvnet::config::{CfvnetConfig, DatagenConfig, GameConfig, TrainingConfig, EvaluationConfig};
use cfvnet::datagen::storage;
use tempfile::TempDir;

#[test]
fn turn_datagen_with_river_model() {
    // Step 1: Train a tiny river model
    let tmp = TempDir::new().unwrap();
    let river_data_path = tmp.path().join("river.bin");
    let river_model_dir = tmp.path().join("river_model");

    let river_config = CfvnetConfig {
        game: GameConfig {
            initial_stack: 200,
            bet_sizes: vec!["50%".into(), "a".into()],
            ..Default::default()
        },
        datagen: DatagenConfig {
            num_samples: 4,
            street: "river".into(),
            solver_iterations: 100,
            target_exploitability: 0.05,
            threads: 1,
            seed: 42,
            ..Default::default()
        },
        training: TrainingConfig {
            hidden_layers: 1,
            hidden_size: 8,
            epochs: 2,
            batch_size: 2,
            ..Default::default()
        },
        evaluation: EvaluationConfig::default(),
    };

    // Generate river data
    cfvnet::datagen::generate::generate_training_data(&river_config, &river_data_path).unwrap();

    // Train river model
    use burn::backend::{Autodiff, NdArray};
    type B = Autodiff<NdArray>;
    let device = Default::default();
    let board_cards = 5;
    let dataset = cfvnet::model::dataset::CfvDataset::from_file(&river_data_path, board_cards).unwrap();
    let train_config = cfvnet::model::training::TrainConfig {
        hidden_layers: 1,
        hidden_size: 8,
        epochs: 2,
        batch_size: 2,
        ..Default::default()
    };
    cfvnet::model::training::train::<B>(&device, &dataset, &train_config, Some(&river_model_dir));

    // Step 2: Generate turn data using the river model
    let turn_data_path = tmp.path().join("turn.bin");
    let turn_config = CfvnetConfig {
        game: GameConfig {
            initial_stack: 200,
            bet_sizes: vec!["50%".into(), "a".into()],
            river_model_path: Some(river_model_dir.join("model").to_string_lossy().into()),
            ..Default::default()
        },
        datagen: DatagenConfig {
            num_samples: 2,
            street: "turn".into(),
            solver_iterations: 50,
            target_exploitability: 0.1,
            threads: 1,
            seed: 99,
            ..Default::default()
        },
        training: TrainingConfig::default(),
        evaluation: EvaluationConfig::default(),
    };

    cfvnet::datagen::generate::generate_training_data(&turn_config, &turn_data_path).unwrap();

    // Verify output
    let mut file = std::fs::File::open(&turn_data_path).unwrap();
    let count = storage::count_records(&mut file, 4).unwrap(); // 4-card turn boards
    assert!(count > 0, "should have generated turn records");
    assert_eq!(count % 2, 0, "records come in OOP/IP pairs");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet turn_datagen_test -- --no-run 2>&1 | head -10`
Expected: fails because `generate_training_data` doesn't support turn street yet

**Step 3: Implement the turn datagen path**

In `crates/cfvnet/src/datagen/generate.rs`, modify `generate_training_data()`:

```rust
pub fn generate_training_data(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
    let street = config.datagen.street.as_str();
    let board_size = config.datagen.board_cards();

    match street {
        "river" => generate_river_data(config, output_path),
        "turn" => generate_with_leaf_evaluator(config, output_path, board_size),
        "flop" => generate_with_leaf_evaluator(config, output_path, board_size),
        _ => Err(format!("unsupported street: {street}")),
    }
}
```

Extract the current river logic into `generate_river_data()`, then implement `generate_with_leaf_evaluator()` that:

1. Loads the appropriate model based on street:
   - Turn datagen → load `river_model_path`
   - Flop datagen → load `turn_model_path`
2. Creates a `RiverNetEvaluator` (or future `TurnNetEvaluator`)
3. For each sample, calls `solve_with_evaluator()` from `cfv_solver.rs`
4. Writes training records in the same format

```rust
fn generate_with_leaf_evaluator(
    config: &CfvnetConfig,
    output_path: &Path,
    board_size: usize,
) -> Result<(), String> {
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
    use crate::eval::river_net_evaluator::RiverNetEvaluator;
    use crate::model::network::{CfvNet, input_size};

    let street = config.datagen.street.as_str();

    // Determine which model to load based on street
    let model_path = match street {
        "turn" => config.game.river_model_path.as_ref()
            .ok_or("turn datagen requires game.river_model_path")?,
        "flop" => config.game.turn_model_path.as_ref()
            .ok_or("flop datagen requires game.turn_model_path")?,
        _ => return Err(format!("unsupported street for leaf evaluator: {street}")),
    };

    // Load the trained model
    type B = NdArray;
    let device = Default::default();
    let leaf_board_cards = board_size + 1; // next street's board size
    let in_size = input_size(leaf_board_cards);
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

    let model = CfvNet::<B>::new(&device, 7, 500, in_size)
        .load_file(std::path::Path::new(model_path), &recorder, &device)
        .map_err(|e| format!("failed to load model from {model_path}: {e}"))?;

    println!("Loaded leaf evaluator model from {model_path}");

    // ... rest follows same pattern as generate_river_data but uses
    // solve_with_evaluator instead of solve_situation
    // ... (implementer fills in the parallel solve + write loop)

    Ok(())
}
```

**Step 4: Run integration test**

Run: `cargo test -p cfvnet turn_datagen_test -- --ignored` (may need `--release` for speed)
Expected: PASS

**Step 5: Run full test suite**

Run: `cargo test`
Expected: all pass, < 60s

**Step 6: Commit**

```bash
git add crates/cfvnet/src/datagen/
git commit -m "feat: street-generic datagen with net-at-leaf for turn/flop"
```

---

## Milestone 3: Turn Net End-to-End

### Task 6: Turn training config and CLI

**Files:**
- Create: `sample_configurations/turn_cfvnet.yaml`
- Modify: `crates/cfvnet/src/main.rs` (if needed — should already work since train is street-agnostic)

**Step 1: Create turn config**

```yaml
# Turn CFVnet training configuration
# Requires a trained river model at the path specified below.
game:
  initial_stack: 200
  bet_sizes: ["33%", "50%", "100%", "a"]
  river_model_path: "models/river/model"

datagen:
  num_samples: 20000000  # 20M (Supremus target)
  street: turn
  pot_intervals: [[4, 20], [20, 80], [80, 200], [200, 400]]
  solver_iterations: 1000
  target_exploitability: 0.005
  threads: 8
  seed: 42

training:
  hidden_layers: 7
  hidden_size: 500
  batch_size: 2048
  epochs: 200
  learning_rate: 0.001
  lr_min: 0.00001
  huber_delta: 1.0
  aux_loss_weight: 1.0
  validation_split: 0.05
  checkpoint_every_n_epochs: 10
```

**Step 2: Verify CLI handles turn training**

The existing `cmd_train` in `main.rs` should already work because it reads `board_cards` from config. Verify:

Run: `cargo run -p cfvnet -- train --help`
Expected: shows train subcommand options

**Step 3: Commit**

```bash
git add sample_configurations/turn_cfvnet.yaml
git commit -m "docs: add turn CFVnet training config (Supremus 20M samples)"
```

---

### Task 7: TurnNetEvaluator for flop datagen

Once the turn net is trained, we need a `TurnNetEvaluator` that implements `LeafEvaluator` for flop datagen. This is structurally identical to `RiverNetEvaluator` but operates on 3-card boards and averages over all possible turn cards.

**Files:**
- Create: `crates/cfvnet/src/eval/turn_net_evaluator.rs`
- Modify: `crates/cfvnet/src/eval/mod.rs`

**Step 1: Write the test**

Same pattern as `river_net_evaluator.rs` tests but with a 3-card board:

```rust
#[test]
fn turn_net_evaluator_returns_correct_shape() {
    let device = Default::default();
    let in_size = input_size(4); // turn = 4 cards
    let model = CfvNet::<NdArray>::new(&device, 1, 8, in_size);

    let board = vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Heart),
        Card::new(Value::Seven, Suit::Diamond),
    ];
    let hands = SubgameHands::enumerate(&board);
    let n = hands.combos.len();
    let oop_range = vec![1.0 / n as f64; n];
    let ip_range = vec![1.0 / n as f64; n];

    let evaluator = TurnNetEvaluator::new(model, device);
    let cfvs = evaluator.evaluate(&hands.combos, &board, 100.0, 200.0, &oop_range, &ip_range, 0);
    assert_eq!(cfvs.len(), n);
}
```

**Step 2: Implement**

The `TurnNetEvaluator` is nearly identical to `RiverNetEvaluator`:
- Input: 3-card board (flop)
- For each of 49 possible turn cards (52 - 3 board):
  - Build 4-card board
  - Map ranges to 1326-indexed arrays
  - Run turn net forward pass
  - Accumulate and average CFVs

**Step 3: Commit**

```bash
git add crates/cfvnet/src/eval/turn_net_evaluator.rs crates/cfvnet/src/eval/mod.rs
git commit -m "feat: add TurnNetEvaluator for flop datagen"
```

---

## Milestone 4: Flop Net (follows same pattern)

### Task 8: Flop datagen + training config

Same as turn but:
- `street: flop`, `turn_model_path` required in config
- 5M samples (Supremus target)
- Uses `TurnNetEvaluator` at turn transition leaves
- Huber target: ≤ 0.0092

Create `sample_configurations/flop_cfvnet.yaml` following the turn config pattern.

---

## Milestone 5: Decision-Point Value Net Architecture

### Task 9: Extended network for action-value prediction

**Files:**
- Create: `crates/cfvnet/src/model/action_value_network.rs`
- Modify: `crates/cfvnet/src/model/mod.rs`

The decision-point net takes the same base input as the CFV net plus an action-history encoding, and outputs `1326 × num_actions` values.

**Input encoding** (~2670 floats for turn):
- Same as CFV net: OOP range (1326) + IP range (1326) + board cards + pot/stack + player
- **New**: action history vector (fixed-length, ~10 floats):
  - `depth_in_street` (0-based, normalized)
  - `facing_bet_size` (pot-relative, 0 if no bet)
  - `pot_after_action` (normalized)
  - `spr_after_action`
  - One-hot action type of last action (fold/check/call/bet/raise/allin = 6 floats)

**Output**: `1326 × num_actions` (e.g., 1326 × 9 = 11,934 for Supremus action abstraction)

**Architecture**: Same 7×500 MLP backbone, wider output layer. Or use the existing `CfvNet` with a parameterized output size:

```rust
impl<B: Backend> CfvNet<B> {
    pub fn new(device: &B::Device, num_layers: usize, hidden_size: usize, in_size: usize) -> Self {
        // ... existing code, OUTPUT_SIZE is currently hardcoded to 1326
    }

    /// Create a network with custom output size (for decision-point nets).
    pub fn with_output_size(
        device: &B::Device,
        num_layers: usize,
        hidden_size: usize,
        in_size: usize,
        out_size: usize,
    ) -> Self {
        // Same as new() but uses out_size instead of OUTPUT_SIZE
    }
}
```

### Task 10: Decision-point datagen

**Files:**
- Create: `crates/cfvnet/src/datagen/decision_point.rs`

Extract per-node action values from solved trees:

```rust
/// Record for decision-point training: action values at a single game tree node.
pub struct DecisionPointRecord {
    pub board: Vec<u8>,
    pub pot: f32,
    pub effective_stack: f32,
    pub player: u8,
    /// Action history encoding for this node
    pub action_history: Vec<f32>,
    /// Per-combo, per-action counterfactual values: [num_actions][1326]
    pub action_values: Vec<[f32; 1326]>,
    /// Valid combo mask
    pub valid_mask: [u8; 1326],
}
```

The datagen loop wraps the existing solve infrastructure:

```
for each random deal:
    build one-street lookahead
    attach next-street CFV net at leaves
    run DCFR+ to convergence
    walk the solved tree:
        for each decision node:
            encode action history
            for each action, compute the CFV of taking that action
            emit DecisionPointRecord
```

One solve yields ~10-50 records (one per decision node in the tree).

---

## Milestone 6: Preflop Decision Net

### Task 11: Preflop decision-point datagen + training

- Generate preflop subgames using flop CFV net at leaves
- 10M samples (Supremus target)
- Validate against published GTO preflop charts (e.g., open-raise frequencies from SB/BTN)
- Huber target: ≤ 0.000069

### Task 12: Preflop decision net validation

Compare network predictions against:
1. Known GTO preflop opening frequencies
2. Exact preflop solver (existing LCFR) results
3. Head-to-head exploitability measurement

---

## Milestone 7: Postflop Decision Nets + Serving

### Task 13: Flop/Turn/River decision-point nets

Same pattern as preflop but for each postflop street. Train bottom-up:
- Flop decision net (uses turn CFV net at leaves)
- Turn decision net (uses river CFV net at leaves)
- River decision net (uses showdown equity at leaves)

### Task 14: Product serving layer

Simple inference API:

```rust
pub fn query_strategy(
    board: &[u8],
    pot: f32,
    stack: f32,
    action_history: &[ActionHistoryEntry],
    player: u8,
) -> HashMap<usize, Vec<(String, f64)>>  // combo_idx -> [(action_name, probability)]
```

Internally: encode input → forward pass through appropriate street's decision-point net → regret match action values → return strategy.

---

## Execution Notes

- **Milestones 1-3 are the critical path** — getting to a trained turn net validates the entire architecture
- Milestone 4 (flop) is the same pattern as Milestone 3, just different street
- Milestones 5-7 are a second phase that can be planned in more detail once Layer 1 is proven
- **Compute budget**: Turn datagen (20M samples × 50 iterations each) is significant. Start with small test runs (1K-10K samples) to validate correctness, then scale up
- **Release mode is essential** for any datagen or training beyond test scale
