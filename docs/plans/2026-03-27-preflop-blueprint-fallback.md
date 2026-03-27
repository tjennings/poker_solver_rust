# Preflop Blueprint Fallback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Use blueprint strategy for preflop in self-play and orchestration, starting subgame solving from flop onward.

**Architecture:** Extract a `play_preflop_under_blueprint()` function from the existing `traverse()` logic in `blueprint_sampler.rs`. Wire it into `self_play.rs` so hands begin with blueprint preflop, then solve flop/turn/river. Remove preflop from orchestration's street order.

**Tech Stack:** Rust, existing `BlueprintV2Strategy`/`GameTree`/`AllBuckets` APIs.

---

### Task 1: Add `PreFlopResult` and `play_preflop_under_blueprint` to blueprint_sampler

**Files:**
- Modify: `crates/rebel/src/blueprint_sampler.rs`

**Step 1: Write the failing test**

Add to the existing `#[cfg(test)] mod tests` block:

```rust
#[test]
fn test_play_preflop_result_has_updated_reach() {
    // PreFlopResult should exist and have the right fields
    let result = PreFlopResult {
        reach_probs: Box::new([[1.0f32; 1326]; 2]),
        pot: 3,
        effective_stack: 47,
    };
    assert_eq!(result.pot, 3);
    assert_eq!(result.effective_stack, 47);
    assert_eq!(result.reach_probs[0].len(), 1326);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rebel blueprint_sampler::tests::test_play_preflop_result_has_updated_reach`
Expected: FAIL — `PreFlopResult` not defined

**Step 3: Implement**

Add to `crates/rebel/src/blueprint_sampler.rs`:

```rust
/// Result of playing preflop under blueprint policy.
///
/// Contains the reach probabilities, pot, and effective stack
/// at the flop entry point (after all preflop betting).
pub struct PreFlopResult {
    /// Reach probabilities for both players after preflop actions.
    pub reach_probs: Box<[[f32; NUM_COMBOS]; 2]>,
    /// Pot size entering the flop (sum of all preflop investments).
    pub pot: i32,
    /// Effective stack remaining after preflop betting.
    pub effective_stack: i32,
}

/// Play preflop under blueprint policy, stopping at the flop boundary.
///
/// Traverses the blueprint game tree from root through all preflop
/// Decision nodes, sampling actions and updating reach probabilities
/// for all 1326 combos. Stops at the first Chance node (preflop→flop).
///
/// Returns updated reach, pot, and effective stack entering the flop.
/// If the hand terminates during preflop (e.g., fold), returns `None`.
pub fn play_preflop_under_blueprint<R: Rng>(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    deal: &Deal,
    initial_stack: i32,
    small_blind: i32,
    big_blind: i32,
    rng: &mut R,
) -> Option<PreFlopResult> {
    let starting_stack = f64::from(initial_stack);
    let decision_idx_map = tree.decision_index_map();
    let mut reach = [[1.0f32; NUM_COMBOS]; 2];
    let invested = [f64::from(small_blind), f64::from(big_blind)];

    traverse_preflop(
        strategy,
        tree,
        buckets,
        deal,
        &decision_idx_map,
        &mut reach,
        tree.root,
        invested,
        starting_stack,
        initial_stack,
        rng,
    )
}
```

Then implement `traverse_preflop` — a variant of `traverse` that:
- At `Terminal` nodes: returns `None` (hand ended preflop, e.g., fold)
- At `Chance` nodes: computes pot/effective_stack, returns `Some(PreFlopResult)` — does NOT recurse further
- At `Decision` nodes: same logic as `traverse` (sample action, update reach, recurse into child)

The key difference from `traverse`: it does NOT push PBS snapshots, and it STOPS at the first Chance node instead of recursing into postflop streets.

```rust
#[allow(clippy::too_many_arguments)]
fn traverse_preflop<R: Rng>(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    deal: &Deal,
    decision_idx_map: &[u32],
    reach: &mut [[f32; NUM_COMBOS]; 2],
    node_idx: u32,
    invested: [f64; 2],
    starting_stack: f64,
    initial_stack: i32,
    rng: &mut R,
) -> Option<PreFlopResult> {
    match &tree.nodes[node_idx as usize] {
        GameNode::Terminal { .. } => {
            // Hand ended during preflop (fold)
            None
        }

        GameNode::Chance { .. } => {
            // Reached flop boundary — return result
            let pot = (invested[0] + invested[1]).round() as i32;
            let max_invested = invested[0].max(invested[1]);
            let effective_stack = initial_stack - max_invested.round() as i32;

            Some(PreFlopResult {
                reach_probs: Box::new(*reach),
                pot,
                effective_stack,
            })
        }

        GameNode::Decision {
            player,
            street,
            actions,
            children,
            ..
        } => {
            // Same action sampling + reach update logic as traverse()
            let player = *player;
            let street = *street;
            let decision_idx = decision_idx_map[node_idx as usize];

            let hole = deal.hole_cards[player as usize];
            let board_slice = board_for_street(&deal.board, street);
            let rs_hole = [rs_id_to_card(hole[0]), rs_id_to_card(hole[1])];
            let rs_board: Vec<Card> = board_slice.iter().map(|&c| rs_id_to_card(c)).collect();
            let actual_bucket = buckets.get_bucket(street, rs_hole, &rs_board);

            let action_probs = strategy.get_action_probs(decision_idx as usize, actual_bucket);
            if action_probs.is_empty() || actions.is_empty() {
                return None;
            }

            let chosen_action_idx = sample_action(action_probs, rng);

            // Update reach for acting player
            let combo_buckets = compute_combo_buckets(buckets, street, board_slice);
            let num_buckets = buckets.bucket_counts[street as usize] as usize;
            let num_actions = actions.len();

            let mut action_probs_per_bucket: Vec<Vec<f32>> = Vec::with_capacity(num_buckets);
            for b in 0..num_buckets {
                let probs = strategy.get_action_probs(decision_idx as usize, b as u16);
                if probs.len() == num_actions {
                    action_probs_per_bucket.push(probs.to_vec());
                } else {
                    let uniform = 1.0 / num_actions as f32;
                    action_probs_per_bucket.push(vec![uniform; num_actions]);
                }
            }

            update_reach(
                &mut reach[player as usize],
                &combo_buckets,
                &action_probs_per_bucket,
                chosen_action_idx,
            );

            // Update invested based on action
            let new_invested = apply_action(invested, &actions[chosen_action_idx], player, starting_stack);

            // Recurse into chosen child
            traverse_preflop(
                strategy, tree, buckets, deal, decision_idx_map, reach,
                children[chosen_action_idx], new_invested, starting_stack, initial_stack, rng,
            )
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p rebel blueprint_sampler::tests::test_play_preflop_result_has_updated_reach`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/rebel/src/blueprint_sampler.rs
git commit -m "feat(rebel): add play_preflop_under_blueprint — preflop traversal stopping at flop boundary"
```

---

### Task 2: Wire preflop blueprint into self_play

**Files:**
- Modify: `crates/rebel/src/self_play.rs`

**Step 1: Write the failing test**

Update existing test `test_street_order` (or add new):

```rust
#[test]
fn test_street_order_starts_at_flop() {
    assert_eq!(STREET_ORDER[0], Street::Flop);
    assert_eq!(STREET_ORDER[1], Street::Turn);
    assert_eq!(STREET_ORDER[2], Street::River);
    assert_eq!(STREET_ORDER.len(), 3);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rebel self_play::tests::test_street_order_starts_at_flop`
Expected: FAIL — STREET_ORDER still has 4 elements starting with Preflop

**Step 3: Implement**

Make these changes to `crates/rebel/src/self_play.rs`:

**3a.** Add imports:

```rust
use crate::blueprint_sampler::{play_preflop_under_blueprint, Deal};
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::game_tree::GameTree;
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
```

**3b.** Change `STREET_ORDER` from 4 streets to 3 (remove Preflop):

```rust
const STREET_ORDER: [Street; 3] = [
    Street::Flop,
    Street::Turn,
    Street::River,
];
```

**3c.** Update `street_index` to match (Flop=0, Turn=1, River=2):

```rust
fn street_index(street: Street) -> usize {
    match street {
        Street::Preflop => panic!("preflop not in STREET_ORDER"),
        Street::Flop => 0,
        Street::Turn => 1,
        Street::River => 2,
    }
}
```

**3d.** Update `play_self_play_hand` signature to accept blueprint params:

```rust
pub fn play_self_play_hand<R: Rng>(
    evaluator: &dyn LeafEvaluator,
    solve_config: &SolveConfig,
    sp_config: &SelfPlayConfig,
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    rng: &mut R,
) -> Vec<TrainingExample> {
```

**3e.** At the top of `play_self_play_hand`, before the street loop, play preflop:

```rust
    let mut examples = Vec::new();
    let deal = deal_hand(rng);

    // --- Play preflop under blueprint policy ---
    let preflop_result = match play_preflop_under_blueprint(
        strategy, tree, buckets, &deal,
        sp_config.initial_stack, sp_config.small_blind, sp_config.big_blind,
        rng,
    ) {
        Some(r) => r,
        None => return examples, // hand ended preflop (fold)
    };

    // Use preflop-updated state for postflop solving
    let pot = preflop_result.pot;
    let effective_stack = preflop_result.effective_stack;
    let mut reach_probs = preflop_result.reach_probs;
```

Remove the old `initial_pot`, `initial_eff_stack`, `pot`, `effective_stack`, and `reach_probs` initializations that precede the loop.

**3f.** Update `self_play_training_loop` to accept and pass blueprint params:

```rust
pub fn self_play_training_loop(
    evaluator: &dyn LeafEvaluator,
    solve_config: &SolveConfig,
    sp_config: &SelfPlayConfig,
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    buffer: &mut DiskBuffer,
) -> usize {
```

And update the inner call:

```rust
let examples = play_self_play_hand(
    evaluator, solve_config, sp_config,
    strategy, tree, buckets,
    &mut rng,
);
```

**3g.** Update existing tests that reference STREET_ORDER or street_index to match the new 3-element array.

**Step 4: Run tests**

Run: `cargo test -p rebel self_play`
Expected: PASS (fix any tests that assumed 4 streets)

Run: `cargo test -p rebel`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add crates/rebel/src/self_play.rs
git commit -m "feat(rebel): wire preflop blueprint into self-play — solve from flop onward"
```

---

### Task 3: Remove preflop from orchestration

**Files:**
- Modify: `crates/rebel/src/orchestration.rs`

**Step 1: Write the failing test**

Update existing test:

```rust
#[test]
fn test_street_order_no_preflop() {
    assert_eq!(STREET_ORDER.len(), 3);
    assert_eq!(STREET_ORDER[0], Street::River);
    assert_eq!(STREET_ORDER[1], Street::Turn);
    assert_eq!(STREET_ORDER[2], Street::Flop);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rebel orchestration::tests::test_street_order_no_preflop`
Expected: FAIL

**Step 3: Implement**

Change `STREET_ORDER` in `orchestration.rs`:

```rust
const STREET_ORDER: [Street; 3] = [Street::River, Street::Turn, Street::Flop];
```

Update any code that references the old 4-element array or handles `Street::Preflop` in the seeding loop. Add a comment explaining why preflop is excluded.

**Step 4: Run tests**

Run: `cargo test -p rebel orchestration`
Expected: PASS

Run: `cargo test -p rebel`
Expected: All PASS

**Step 5: Verify trainer builds**

Run: `cargo build -p poker-solver-trainer`
Expected: compiles (the trainer's `run_rebel_train` calls orchestration — make sure it still works)

If the trainer references `self_play_training_loop` with the old signature, update those call sites too.

**Step 6: Commit**

```bash
git add crates/rebel/src/orchestration.rs crates/trainer/src/main.rs
git commit -m "feat(rebel): remove preflop from orchestration — blueprint handles preflop"
```
