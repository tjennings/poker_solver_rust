# Multi-Valued Leaf Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace single-valued leaf evaluation with multi-valued continuation strategy rollouts at depth boundaries, using a virtual opponent choice node (Pluribus approach).

**Architecture:** Add `RolloutLeafEvaluator` that walks the abstract tree with biased blueprint strategies using concrete cards. Add a virtual choice node (K=4 branches) to `CfvSubgameSolver` where the opponent selects a continuation strategy. The subgame solver runs 4 traversals per iteration (one per strategy), accumulating choice-node regrets alongside the real tree regrets.

**Tech Stack:** Rust, rayon (parallelism), `BlueprintV2Strategy`, `AllBuckets`, `CfvSubgameSolver`

**Design doc:** `docs/plans/2026-03-21-multi-valued-leaf-evaluation-design.md`

---

### Task 1: Strategy Biasing Functions

Pure domain functions for biasing blueprint action probabilities. No I/O, no solver integration.

**Files:**
- Create: `crates/core/src/blueprint_v2/continuation.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs` (add `pub mod continuation;`)

**Step 1: Write the failing test**

```rust
// In continuation.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_fold_multiplies_fold_actions() {
        // Actions: [Fold, Call, Bet(x)] with probs [0.2, 0.5, 0.3]
        let probs = vec![0.2_f32, 0.5, 0.3];
        let actions = vec![ActionClass::Fold, ActionClass::Call, ActionClass::Raise];
        let biased = bias_strategy(&probs, &actions, BiasType::Fold, 10.0);
        // Fold: 0.2*10=2.0, Call: 0.5, Raise: 0.3. Sum=2.8
        // Normalized: [2.0/2.8, 0.5/2.8, 0.3/2.8] = [0.714, 0.179, 0.107]
        assert!((biased[0] - 0.714).abs() < 0.01);
        assert!((biased[1] - 0.179).abs() < 0.01);
        assert!((biased[2] - 0.107).abs() < 0.01);
    }

    #[test]
    fn bias_unbiased_returns_original() {
        let probs = vec![0.2_f32, 0.5, 0.3];
        let actions = vec![ActionClass::Fold, ActionClass::Call, ActionClass::Raise];
        let biased = bias_strategy(&probs, &actions, BiasType::Unbiased, 10.0);
        for (a, b) in probs.iter().zip(biased.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn classify_tree_actions() {
        use crate::blueprint_v2::game_tree::TreeAction;
        assert_eq!(classify_action(&TreeAction::Fold), ActionClass::Fold);
        assert_eq!(classify_action(&TreeAction::Check), ActionClass::Fold); // check = passive
        assert_eq!(classify_action(&TreeAction::Call), ActionClass::Call);
        assert_eq!(classify_action(&TreeAction::Bet(5.0)), ActionClass::Raise);
        assert_eq!(classify_action(&TreeAction::Raise(10.0)), ActionClass::Raise);
        assert_eq!(classify_action(&TreeAction::AllIn), ActionClass::Raise);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core continuation`
Expected: FAIL — module doesn't exist.

**Step 3: Implement**

```rust
//! Continuation strategy biasing for multi-valued leaf evaluation.
//!
//! Generates biased variants of the blueprint strategy for use at
//! depth boundaries in subgame solving (Pluribus approach).

use crate::blueprint_v2::game_tree::TreeAction;

/// Classification of poker actions for biasing purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionClass {
    Fold,  // includes Check (passive)
    Call,
    Raise, // includes Bet, Raise, AllIn (aggressive)
}

/// Which bias to apply to a continuation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasType {
    Unbiased,
    Fold,
    Call,
    Raise,
}

/// Classify a tree action into Fold/Call/Raise for biasing.
pub fn classify_action(action: &TreeAction) -> ActionClass {
    match action {
        TreeAction::Fold | TreeAction::Check => ActionClass::Fold,
        TreeAction::Call => ActionClass::Call,
        TreeAction::Bet(_) | TreeAction::Raise(_) | TreeAction::AllIn => ActionClass::Raise,
    }
}

/// Bias a strategy probability vector by multiplying the target action
/// class by `factor` and renormalizing.
///
/// `actions` must be the same length as `probs` and gives the action
/// class for each entry.
pub fn bias_strategy(
    probs: &[f32],
    actions: &[ActionClass],
    bias: BiasType,
    factor: f64,
) -> Vec<f32> {
    if bias == BiasType::Unbiased {
        return probs.to_vec();
    }
    let target = match bias {
        BiasType::Fold => ActionClass::Fold,
        BiasType::Call => ActionClass::Call,
        BiasType::Raise => ActionClass::Raise,
        BiasType::Unbiased => unreachable!(),
    };
    let mut biased: Vec<f32> = probs
        .iter()
        .zip(actions)
        .map(|(&p, &a)| if a == target { p * factor as f32 } else { p })
        .collect();
    let sum: f32 = biased.iter().sum();
    if sum > 0.0 {
        for p in &mut biased { *p /= sum; }
    }
    biased
}
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core continuation`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/continuation.rs crates/core/src/blueprint_v2/mod.rs
git commit -m "feat: strategy biasing functions for multi-valued leaf evaluation"
```

---

### Task 2: Rollout Evaluator — Fixed-Strategy Tree Walk

Implement the core rollout function that walks the abstract tree with a (biased) blueprint strategy using concrete cards.

**Files:**
- Modify: `crates/core/src/blueprint_v2/continuation.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn rollout_simple_tree_returns_terminal_value() {
    // Build a minimal abstract tree: Decision -> Terminal(Showdown)
    // With a known hand and board, verify the rollout returns the
    // correct EV based on blueprint strategy weights × terminal payoffs.
    // (Details: build a 1-street tree, set up strategy, verify EV)
}
```

**Step 2: Implement `rollout_from_boundary`**

The function:
1. Starts at `abstract_node` in the abstract tree
2. If at a Decision node: look up the acting player's bucket via `AllBuckets::get_bucket()`, get blueprint probs via `BlueprintV2Strategy::get_action_probs()`, apply bias, compute EV = Σ p_action × rollout(child)
3. If at a Terminal (fold/showdown): compute payoff using concrete hand evaluation
4. If at a Chance node: deal a random next-street card from the remaining deck, extend the board, recurse. Repeat `num_rollouts` times, average.

Key: uses the abstract tree's `decision_index_map` to look up strategy, and `AllBuckets` for bucket mapping on each new board.

```rust
use rand::Rng;
use crate::blueprint_v2::bundle::BlueprintV2Strategy;
use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind, TreeAction};
use crate::blueprint_v2::mccfr::AllBuckets;
use crate::blueprint_v2::Street;
use crate::poker::Card;

/// Roll out from a depth boundary using a (biased) blueprint strategy.
///
/// Returns the expected value for `player` in the abstract tree's units (BB).
/// Uses Monte Carlo sampling at Chance nodes (`num_rollouts` samples).
pub fn rollout_from_boundary(
    hero_hand: [Card; 2],
    opponent_hand: [Card; 2],  // sampled by caller
    board: &[Card],
    abstract_tree: &GameTree,
    abstract_node: u32,
    decision_idx_map: &[u32],
    strategy: &BlueprintV2Strategy,
    buckets: &AllBuckets,
    bias: BiasType,
    bias_factor: f64,
    player: u8,
    num_rollouts: u32,
    rng: &mut impl Rng,
) -> f64 {
    // Implementation: recursive tree walk with strategy-weighted EV
    todo!()
}
```

Note: the caller (LeafEvaluator) will sample opponent hands from the reaching range and call this function per sample. Each call walks the abstract tree mechanically — no CFR, no iteration.

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core rollout`
Expected: PASS.

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_v2/continuation.rs
git commit -m "feat: rollout_from_boundary — fixed-strategy tree walk"
```

---

### Task 3: RolloutLeafEvaluator implementing LeafEvaluator

Wrap the rollout function into a `LeafEvaluator` implementation that the `CfvSubgameSolver` can use.

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs` (add `RolloutLeafEvaluator`)

**Step 1: Design**

The `RolloutLeafEvaluator` holds:
- `Arc<BlueprintV2Strategy>` — the blueprint
- `Arc<GameTree>` — the abstract tree
- `Arc<AllBuckets>` — bucket files
- `decision_idx_map: Vec<u32>` — precomputed from abstract tree
- `abstract_start_node: u32` — where in the abstract tree this subgame starts
- `bias: BiasType` — which continuation strategy to use
- `bias_factor: f64`
- `num_rollouts: u32`

The `evaluate()` method: for each combo, sample opponent hands from `reach_opponent`, run `rollout_from_boundary` for each sample, average. Return per-combo CFVs.

Unlike `BlueprintCbvEvaluator`, this evaluator IS range-sensitive — it samples opponents from the reaching range, so values adapt as the opponent's strategy evolves during CFR.

**Step 2: Implement**

```rust
struct RolloutLeafEvaluator {
    strategy: Arc<BlueprintV2Strategy>,
    abstract_tree: Arc<GameTree>,
    all_buckets: Arc<AllBuckets>,
    decision_idx_map: Vec<u32>,
    abstract_start_node: u32,
    bias: BiasType,
    bias_factor: f64,
    num_rollouts: u32,
}

impl LeafEvaluator for RolloutLeafEvaluator {
    fn evaluate(&self, combos, board, pot, eff_stack, oop_range, ip_range, traverser) -> Vec<f64> {
        // For each combo i:
        //   1. Build remaining deck (exclude board + combo cards)
        //   2. Sample opponent hands weighted by reach
        //   3. For each opponent sample, call rollout_from_boundary
        //   4. Average rollout values
        //   5. Normalize to pot-fraction units
        // Parallelize across combos with rayon
    }
}
```

**Step 3: Commit**

```bash
git add crates/tauri-app/src/postflop.rs
git commit -m "feat: RolloutLeafEvaluator implementing LeafEvaluator"
```

---

### Task 4: Virtual Choice Node in CfvSubgameSolver

Add the opponent's virtual choice node with K=4 branches to the solver. This is the core architectural change.

**Files:**
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs`

**Step 1: Add choice node state to CfvSubgameSolver**

```rust
// New fields in CfvSubgameSolver:
/// Number of continuation strategies (K=4 for Pluribus).
num_continuation_strategies: u32,
/// Regret sums for the virtual choice node (length K).
choice_regret_sum: Vec<f64>,
/// Strategy sums for the virtual choice node (length K).
choice_strategy_sum: Vec<f64>,
```

**Step 2: Modify `train_with_leaf_interval`**

The iteration loop changes from:
```
for each traverser:
    evaluate leaves
    run CFR traversal
```

To:
```
for each traverser:
    build choice strategy from choice_regret_sum (regret matching)
    for each k in 0..K:
        evaluate leaves using continuation strategy k
        run CFR traversal → get root CFV for this k
    compute choice node regrets: for each k, regret += (cfv[k] - weighted_avg)
    accumulate choice strategy sums
```

The key: each k produces a different set of leaf values, which produces a different root CFV. The choice node's regret tracks which k gives the opponent the best result.

**Step 3: Pass `bias_type` to the evaluator**

The `LeafEvaluator` needs to know which continuation strategy to use. Options:
- (A) Create K separate `LeafEvaluator` instances (one per bias type)
- (B) Add a `set_bias` method to the evaluator
- (C) Pass `continuation_k` as a parameter

Option (A) is cleanest — create 4 `RolloutLeafEvaluator` instances at construction time, each with a different `BiasType`. Store as `evaluators: Vec<Box<dyn LeafEvaluator>>`.

**Step 4: Update constructor**

```rust
pub fn new(
    tree: GameTree,
    hands: SubgameHands,
    board: &[Card],
    evaluators: Vec<Box<dyn LeafEvaluator>>,  // K evaluators, one per strategy
    starting_stack: f64,
    oop_reach: Vec<f64>,
    ip_reach: Vec<f64>,
) -> Self
```

Note: this changes the constructor signature. All callers must be updated:
- `postflop.rs` `build_subgame_solver()` — create 4 evaluators
- Test call sites — create 1 evaluator wrapped in a vec

For backward compatibility with K=1 (single evaluator), the code should work with `evaluators.len() == 1` by skipping the choice node.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cfv_subgame_solver.rs
git commit -m "feat: virtual choice node with K continuation strategies"
```

---

### Task 5: Analytics — Choice Node Diagnostics

Add logging for choice node action frequencies and regrets.

**Files:**
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs`
- Modify: `crates/tauri-app/src/postflop.rs`

**Step 1: Add `choice_strategy()` method to CfvSubgameSolver**

Returns the averaged choice node strategy (normalized strategy sums):

```rust
pub fn choice_strategy(&self) -> Vec<f64> {
    let total: f64 = self.choice_strategy_sum.iter().sum();
    if total > 0.0 {
        self.choice_strategy_sum.iter().map(|&s| s / total).collect()
    } else {
        vec![1.0 / self.num_continuation_strategies as f64; self.num_continuation_strategies as usize]
    }
}
```

**Step 2: Log during training**

In `train_with_leaf_interval`, at snapshot intervals:
```rust
if is_snapshot_interval {
    let mix = self.choice_strategy();
    eprintln!("[choice node] iter={}: unbiased={:.1}% fold={:.1}% call={:.1}% raise={:.1}%",
        self.iteration, mix[0]*100.0, mix[1]*100.0, mix[2]*100.0, mix[3]*100.0);
}
```

**Step 3: Log after training in postflop.rs**

After the solve loop completes, log the final choice strategy and regrets:
```rust
let choice_mix = solver.choice_strategy();
let choice_regrets = solver.choice_regrets();
eprintln!("[choice audit] final mix: unbiased={:.1}% fold={:.1}% call={:.1}% raise={:.1}%",
    choice_mix[0]*100.0, choice_mix[1]*100.0, choice_mix[2]*100.0, choice_mix[3]*100.0);
eprintln!("[choice regrets] {:?}", choice_regrets);
```

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_v2/cfv_subgame_solver.rs crates/tauri-app/src/postflop.rs
git commit -m "feat: choice node analytics — action frequencies and regrets"
```

---

### Task 6: Wire Into TUI — Add BlueprintV2Strategy to CbvContext

Plumb the blueprint strategy through to the solver so `RolloutLeafEvaluator` can access it.

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs` (`CbvContext`)
- Modify: `crates/tauri-app/src/exploration.rs` (`populate_cbv_context`)

**Step 1: Add strategy to CbvContext**

```rust
pub struct CbvContext {
    pub cbv_table: CbvTable,
    pub abstract_tree: GameTree,
    pub all_buckets: AllBuckets,
    pub strategy: BlueprintV2Strategy,  // NEW
}
```

**Step 2: Load strategy in `populate_cbv_context`**

The strategy is already loaded in `StrategySource::BlueprintV2 { strategy, .. }`. Clone it into `CbvContext`.

**Step 3: Create 4 RolloutLeafEvaluators in `build_subgame_solver`**

When `cbv_context` is available, create 4 evaluators:
```rust
let evaluators: Vec<Box<dyn LeafEvaluator>> = vec![
    Box::new(RolloutLeafEvaluator::new(ctx, BiasType::Unbiased, 10.0, 3)),
    Box::new(RolloutLeafEvaluator::new(ctx, BiasType::Fold, 10.0, 3)),
    Box::new(RolloutLeafEvaluator::new(ctx, BiasType::Call, 10.0, 3)),
    Box::new(RolloutLeafEvaluator::new(ctx, BiasType::Raise, 10.0, 3)),
];
```

When no CBV context: create 1 evaluator (EquityLeafEvaluator), no choice node.

**Step 4: Commit**

```bash
git add crates/tauri-app/src/postflop.rs crates/tauri-app/src/exploration.rs
git commit -m "feat: wire rollout evaluators into TUI subgame solver"
```

---

### Task 7: Full Integration Test and Cleanup

Run the full test suite, verify the solver produces mixed strategies (not pure shove), clean up diagnostics.

**Step 1: Run full test suite**

Run: `cargo test`
Expected: ALL PASS in under 1 minute.

**Step 2: Run clippy**

Run: `cargo clippy`
Expected: No new warnings.

**Step 3: Manual test**

Load a blueprint, navigate to a flop spot, solve with 500 iterations. Verify:
- `[choice node]` diagnostics appear in logs
- AK on QJT uses mixed bet sizes (not pure shove)
- 45o on QJT mostly checks (not pure shove)
- Flush draws bet at higher frequency than non-flush-draw suited combos

**Step 4: Commit**

```bash
git commit -m "chore: integration test and cleanup for multi-valued leaf evaluation"
```
