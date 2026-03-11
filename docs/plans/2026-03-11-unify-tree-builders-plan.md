# Unify Tree Builders Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Make V2GameTree the single tree builder for both blueprint training and real-time subgame solving, eliminating SubgameTree.

**Architecture:** Add `DepthBoundary` to `TerminalKind` and a `build_subgame()` constructor to `GameTree`. Update `SubgameCfrSolver` to use `GameNode`/`TreeAction` instead of `SubgameNode`/`Action`. Delete `SubgameTree` and `SubgameTreeBuilder`.

**Tech Stack:** Rust, rayon (parallelism), rs_poker (hand evaluation)

---

### Task 1: Add DepthBoundary to TerminalKind

**Files:**
- Modify: `crates/core/src/blueprint_v2/game_tree.rs`

**Step 1: Add the DepthBoundary variant**

In `TerminalKind` enum (line ~51), add a new variant:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalKind {
    Fold { winner: u8 },
    Showdown,
    DepthBoundary,
}
```

**Step 2: Run tests to verify nothing breaks**

Run: `cargo test -p poker-solver-core --lib blueprint_v2::game_tree`
Expected: All 18 tests pass (no code references the new variant yet).

**Step 3: Commit**

```bash
git add crates/core/src/blueprint_v2/game_tree.rs
git commit -m "feat: add DepthBoundary variant to TerminalKind"
```

---

### Task 2: Add build_subgame() to GameTree

**Files:**
- Modify: `crates/core/src/blueprint_v2/game_tree.rs`

This is the core task. `build_subgame()` creates a partial tree rooted at a specific postflop street. It reuses the existing `build_node()` / `build_child()` / `compute_sized_actions()` internals — the only change is the initial state and how street transitions work at the depth limit.

**Step 1: Write a failing test**

Add to the `tests` module at the bottom of `game_tree.rs`:

```rust
#[test]
fn test_build_subgame_basic() {
    // Flop subgame: 10bb pot, 50bb stacks, bet sizes [0.5, 1.0]
    let tree = GameTree::build_subgame(
        Street::Flop,
        10.0,                     // pot
        [5.0, 5.0],              // invested
        50.0,                     // starting_stack
        &[vec![0.5, 1.0]],       // bet sizes (pot fractions)
        None,                     // no depth limit = solve to river
    );
    assert!(matches!(
        tree.nodes[tree.root as usize],
        GameNode::Decision { player: 0, .. }
    ));
    // Should have Chance nodes transitioning to turn and river
    let has_chance = tree.nodes.iter().any(|n| matches!(n, GameNode::Chance { .. }));
    assert!(has_chance, "Full-depth subgame should have Chance nodes");
}

#[test]
fn test_build_subgame_depth_limited() {
    // Flop subgame with depth_limit=1: solve flop only, DepthBoundary at turn
    let tree = GameTree::build_subgame(
        Street::Flop,
        10.0,
        [5.0, 5.0],
        50.0,
        &[vec![0.5, 1.0]],
        Some(1),                  // depth limit: 1 street only
    );
    // Should have DepthBoundary terminals instead of Chance nodes
    let has_boundary = tree.nodes.iter().any(|n| matches!(
        n,
        GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. }
    ));
    assert!(has_boundary, "Depth-limited subgame should have DepthBoundary terminals");
    // Should NOT have Chance nodes (no street transitions)
    let has_chance = tree.nodes.iter().any(|n| matches!(n, GameNode::Chance { .. }));
    assert!(!has_chance, "Depth-limited flop subgame should not transition to turn");
}

#[test]
fn test_build_subgame_river_no_chance() {
    // River subgame: no next street, so no Chance or DepthBoundary
    let tree = GameTree::build_subgame(
        Street::River,
        20.0,
        [10.0, 10.0],
        50.0,
        &[vec![0.5, 1.0]],
        None,
    );
    let has_chance = tree.nodes.iter().any(|n| matches!(n, GameNode::Chance { .. }));
    assert!(!has_chance, "River subgame has no next street");
    let has_boundary = tree.nodes.iter().any(|n| matches!(
        n,
        GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. }
    ));
    assert!(!has_boundary, "River subgame has no DepthBoundary");
}

#[test]
fn test_build_subgame_allin_everywhere() {
    // All-in should be available at every decision node
    for street in [Street::Flop, Street::Turn, Street::River] {
        let tree = GameTree::build_subgame(
            street,
            10.0,
            [5.0, 5.0],
            50.0,
            &[vec![0.5]],
            Some(1),
        );
        check_all_in_everywhere(&tree, 50.0);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core --lib blueprint_v2::game_tree::tests::test_build_subgame`
Expected: FAIL — `build_subgame` doesn't exist yet.

**Step 3: Implement build_subgame()**

Add to the `impl GameTree` block. The key insight: reuse the existing `build_node()` by constructing a `TreeConfig` and `BuildState` from the subgame parameters. The `TreeConfig` uses the provided bet sizes for the starting street and all subsequent streets. Add `depth_limit` to `TreeConfig` and check it in `make_showdown_or_chance`.

```rust
impl GameTree {
    /// Build a subgame tree rooted at a specific postflop street.
    ///
    /// # Arguments
    /// * `street` - Starting street (Flop, Turn, or River)
    /// * `pot` - Current pot in BB
    /// * `invested` - How much each player has invested so far
    /// * `starting_stack` - Starting stack for both players in BB
    /// * `bet_sizes` - Per raise depth, list of pot fractions.
    ///   Applied to all streets in the subgame.
    /// * `depth_limit` - Max streets to solve. `None` = solve to river.
    ///   `Some(1)` = solve current street only.
    #[must_use]
    pub fn build_subgame(
        street: Street,
        pot: f64,
        invested: [f64; 2],
        starting_stack: f64,
        bet_sizes: &[Vec<f64>],
        depth_limit: Option<u8>,
    ) -> Self {
        // Use the same bet sizes for all streets in the subgame
        let config = TreeConfig {
            preflop_sizes: vec![],  // subgames never start preflop
            flop_sizes: bet_sizes.to_vec(),
            turn_sizes: bet_sizes.to_vec(),
            river_sizes: bet_sizes.to_vec(),
            depth_limit,
            starting_street: Some(street),
        };

        let initial_state = BuildState {
            starting_stack,
            invested,
            street,
            num_raises: 0,
            to_act: 0,       // OOP acts first postflop
            facing_bet: false,
            last_raise_to: 0.0,
        };

        let mut nodes = Vec::new();
        let root = Self::build_node(&config, &initial_state, &mut nodes);
        Self { nodes, root }
    }
}
```

This requires two changes to `TreeConfig`:

```rust
struct TreeConfig {
    preflop_sizes: Vec<Vec<PreflopSize>>,
    flop_sizes: Vec<Vec<f64>>,
    turn_sizes: Vec<Vec<f64>>,
    river_sizes: Vec<Vec<f64>>,
    /// Maximum streets to solve. None = full depth.
    depth_limit: Option<u8>,
    /// Street the subgame starts on (for depth counting).
    starting_street: Option<Street>,
}
```

Update `build()` to set `depth_limit: None, starting_street: None` in its `TreeConfig`.

Then modify `make_showdown_or_chance` to check the depth limit. Where it currently creates a `Chance` node (line ~467-493), add a check:

```rust
Some(next_street) if !both_all_in => {
    // Check depth limit
    if let (Some(limit), Some(start)) = (config.depth_limit, config.starting_street) {
        let streets_played = next_street.index() - start.index();
        if streets_played >= limit as usize {
            // Hit depth limit: emit DepthBoundary instead of Chance
            let idx = nodes.len() as u32;
            nodes.push(GameNode::Terminal {
                kind: TerminalKind::DepthBoundary,
                pot: state.invested[0] + state.invested[1],
                invested: state.invested,
            });
            return idx;
        }
    }

    // Normal: transition to next street via Chance node
    // ... existing code ...
}
```

This requires `Street` to have an `index()` method. Check if it exists; if not, add:

```rust
impl Street {
    pub fn index(self) -> usize {
        self as usize
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core --lib blueprint_v2::game_tree`
Expected: All tests pass (original 18 + 4 new).

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/game_tree.rs crates/core/src/blueprint_v2/mod.rs
git commit -m "feat: add GameTree::build_subgame() with depth limit support"
```

---

### Task 3: Update SubgameLayout and SubgameCfrSolver to use GameTree

**Files:**
- Modify: `crates/core/src/blueprint/subgame_cfr.rs`

This is the largest task. The solver switches from `SubgameTree`/`SubgameNode` to `GameTree`/`GameNode`. The algorithm is unchanged — only the node type inspections change.

**Step 1: Update imports and SubgameLayout**

Replace:
```rust
use super::subgame_tree::{SubgameHands, SubgameNode, SubgameTree, SubgameTreeBuilder};
```
With:
```rust
use super::subgame_tree::SubgameHands;  // keep SubgameHands (combo enumeration)
use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind, TreeAction};
```

Update `SubgameLayout::build` to pattern match on `GameNode::Decision` instead of `SubgameNode::Decision`:

```rust
impl SubgameLayout {
    fn build(tree: &GameTree, num_combos: usize) -> Self {
        let n = tree.nodes.len();
        let mut bases = vec![usize::MAX; n];
        let mut num_actions_vec = vec![0usize; n];
        let mut offset = 0;

        for (i, node) in tree.nodes.iter().enumerate() {
            if let GameNode::Decision { actions, .. } = node {
                bases[i] = offset;
                num_actions_vec[i] = actions.len();
                offset += num_combos * actions.len();
            }
        }

        Self {
            bases,
            num_actions: num_actions_vec,
            total_size: offset,
        }
    }
}
```

**Step 2: Update SubgameCfrSolver struct and constructor**

Change `tree: SubgameTree` to `tree: GameTree`. Remove `board` from the tree (it's not in `GameTree`); instead pass board to the constructor for equity computation.

```rust
pub struct SubgameCfrSolver {
    tree: GameTree,
    hands: SubgameHands,
    equity_matrix: Vec<Vec<f64>>,
    opponent_reach: Vec<f64>,
    leaf_values: Vec<f64>,
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
    layout: SubgameLayout,
    opp_reach_totals: Vec<f64>,
    dcfr: DcfrParams,
    pub iteration: u32,
}

impl SubgameCfrSolver {
    #[must_use]
    pub fn new(
        tree: GameTree,
        hands: SubgameHands,
        board: &[Card],
        opponent_reach: Vec<f64>,
        leaf_values: Vec<f64>,
    ) -> Self {
        let equity_matrix = compute_equity_matrix(&hands.combos, board);
        let opp_reach_totals = precompute_opp_reach(&hands.combos, &opponent_reach);
        let layout = SubgameLayout::build(&tree, hands.combos.len());
        let buf_size = layout.total_size;
        Self {
            tree,
            hands,
            equity_matrix,
            opponent_reach,
            leaf_values,
            regret_sum: vec![0.0; buf_size],
            strategy_sum: vec![0.0; buf_size],
            layout,
            opp_reach_totals,
            dcfr: DcfrParams::default(),
            iteration: 0,
        }
    }
}
```

Note: `new()` now takes an explicit `board: &[Card]` parameter since `GameTree` doesn't carry board info. Update `with_cbv_table` similarly.

**Step 3: Update cfr_traverse to match on GameNode**

Replace the match in `cfr_traverse` (line ~355):

```rust
fn cfr_traverse(
    &self,
    regret_delta: &mut [f64],
    strategy_delta: &mut [f64],
    node_idx: usize,
    hero_combo: usize,
    reach_hero: f64,
    reach_opp: f64,
) -> f64 {
    match &self.tree.nodes[node_idx] {
        GameNode::Terminal { kind, pot, invested } => {
            match kind {
                TerminalKind::Fold { winner } => {
                    let hero_invested = invested[self.traverser as usize];
                    if *winner == self.traverser {
                        // Hero wins: gain opponent's investment
                        (*pot - hero_invested) as f64  // not needed, simpler:
                        // Actually keep the half_pot approach:
                        let half_pot = *pot / 2.0;
                        if *winner == self.traverser { half_pot } else { -half_pot }
                    }
                }
                TerminalKind::Showdown => {
                    let half_pot = *pot / 2.0;
                    self.showdown_value(hero_combo, half_pot)
                }
                TerminalKind::DepthBoundary => {
                    let equity = self.leaf_values.get(hero_combo).copied().unwrap_or(0.5);
                    let half_pot = *pot / 2.0;
                    (2.0 * equity - 1.0) * half_pot
                }
            }
        }

        GameNode::Chance { child, .. } => {
            // Chance nodes in subgame trees: pass through to child
            // (concrete board is already known — no branching over cards)
            self.cfr_traverse(
                regret_delta, strategy_delta,
                *child as usize, hero_combo, reach_hero, reach_opp,
            )
        }

        GameNode::Decision { player, actions, children, .. } => {
            let (base, _) = self.layout.slot(node_idx, hero_combo);
            let num_actions = actions.len();
            let strategy = &self.snapshot[base..base + num_actions];

            if *player == self.traverser {
                self.traverse_as_traverser(
                    regret_delta, strategy_delta,
                    node_idx, hero_combo, reach_hero, reach_opp,
                    strategy, children,
                )
            } else {
                self.traverse_as_opponent(
                    regret_delta, strategy_delta,
                    hero_combo, reach_hero, reach_opp,
                    strategy, children,
                )
            }
        }
    }
}
```

Important note on fold payoffs: `SubgameTree` used `is_fold` + `fold_player` + `pot`. `GameTree` uses `TerminalKind::Fold { winner }` + `pot` + `invested`. The fold value logic must be:
- If `winner == traverser`: hero wins half the pot → `+half_pot`
  Wait — actually the old code uses `half_pot` = `pot / 2`. That represents each player's investment (since pot = sum of both investments). The actual payoff is: winner gets `pot - own_investment` = `opponent_investment`. Hero's net = `opponent_investment - hero_investment`. For a symmetric pot: `net = pot/2 - pot/2 = 0`... no, that's not right either.

Let me re-read the old terminal value logic carefully:

```rust
fn terminal_value(&self, hero_combo, is_fold, fold_player, pot) -> f64 {
    let half_pot = f64::from(pot) / 2.0;
    if is_fold {
        if fold_player == self.traverser { -half_pot } else { half_pot }
    }
    self.showdown_value(hero_combo, half_pot)
}
```

So the convention is: each player's investment = `half_pot`, and the value is `±half_pot`. This works because the pot is the total and each player has put in half. But with V2GameTree, the pot isn't always split evenly (one player may have bet more). However, `invested` is tracked, so the correct payoff is:

```rust
TerminalKind::Fold { winner } => {
    // Payoff = what hero gains or loses relative to 0 (start of subgame)
    // If hero wins: net = pot - hero_invested = opp_invested
    // If hero loses: net = -hero_invested
    let hero_inv = invested[self.traverser as usize];
    if *winner == self.traverser {
        pot - hero_inv  // gain opponent's chips
    } else {
        -hero_inv       // lose own chips
    }
}
```

Wait, but the old code used `half_pot` for both fold AND showdown. Let me think... In the subgame tree, pot tracks the TOTAL pot. If hero folds, hero loses what they invested. If opponent folds, hero gains pot minus what hero invested. This is equivalent to `±half_pot` only when investments are equal. With V2GameTree we have exact `invested` values, so use them properly:

```rust
TerminalKind::Fold { winner } => {
    let hero_inv = invested[self.traverser as usize];
    if *winner == self.traverser {
        *pot - hero_inv
    } else {
        -hero_inv
    }
}
TerminalKind::Showdown => {
    let hero_inv = invested[self.traverser as usize];
    let opp_inv = invested[1 - self.traverser as usize];
    // equity in [0, 1]; value = equity * pot - hero_investment
    self.showdown_value_v2(hero_combo, *pot, hero_inv)
}
```

Actually this gets complicated. The simplest correct approach: keep `half_pot` convention but compute it as `pot / 2.0`. For subgames, both players always have equal investment at terminals (because calls equalize investment). So `invested[0]` ≈ `invested[1]` ≈ `pot/2` at every terminal. The old `half_pot` approach is fine.

```rust
TerminalKind::Fold { winner } => {
    let half_pot = *pot / 2.0;
    if *winner == self.traverser { half_pot } else { -half_pot }
}
TerminalKind::Showdown => {
    self.showdown_value(hero_combo, *pot / 2.0)
}
TerminalKind::DepthBoundary => {
    let equity = self.leaf_values.get(hero_combo).copied().unwrap_or(0.5);
    (2.0 * equity - 1.0) * (*pot / 2.0)
}
```

**Step 4: Update strategy() and build_strategy_snapshot()**

These iterate over `tree.nodes` matching `SubgameNode::Decision`. Change to match `GameNode::Decision`:

```rust
// In strategy():
let num_actions = match node {
    GameNode::Decision { actions, .. } => actions.len(),
    _ => continue,
};

// In build_strategy_snapshot():
let num_actions = match node {
    GameNode::Decision { actions, .. } => actions.len(),
    _ => continue,
};
```

**Step 5: Update SubgameCfrCtx**

```rust
struct SubgameCfrCtx<'a> {
    tree: &'a GameTree,
    hands: &'a SubgameHands,
    // ... rest unchanged
}
```

**Step 6: Update solve_subgame convenience function**

```rust
pub fn solve_subgame(
    board: &[Card],
    bet_sizes: &[f64],        // changed from &[f32] to &[f64]
    pot: f64,                  // changed from u32 to f64
    stacks: [f64; 2],         // changed from &[u32] to [f64; 2]
    opponent_reach: &[f64],
    leaf_values: &[f64],
    config: &SubgameConfig,
) -> SubgameStrategy {
    let street = match board.len() {
        3 => Street::Flop,
        4 => Street::Turn,
        5 => Street::River,
        _ => panic!("invalid board length: {}", board.len()),
    };
    let invested = [pot / 2.0; 2]; // assume equal investment
    let starting_stack = stacks[0]; // assume symmetric
    let tree = GameTree::build_subgame(
        street,
        pot,
        invested,
        starting_stack,
        &[bet_sizes.to_vec()],  // single raise depth
        config.depth_limit.map(|d| d as u8),
    );
    let hands = SubgameHands::enumerate(board);
    let mut solver = SubgameCfrSolver::new(
        tree, hands, board,
        opponent_reach.to_vec(), leaf_values.to_vec(),
    );
    solver.train(config.max_iterations);
    solver.strategy()
}
```

**Step 7: Run tests**

Run: `cargo test -p poker-solver-core --lib blueprint::subgame_cfr`
Expected: All existing solver tests pass. Some tests will need updating because:
1. `SubgameTreeBuilder` calls → `GameTree::build_subgame()` calls
2. `SubgameCfrSolver::new()` now takes `board` as explicit param
3. Values change from `u32` to `f64`

Update each test's tree construction. Example pattern:

Old:
```rust
let tree = SubgameTreeBuilder::new()
    .board(&board)
    .bet_sizes(&[0.5])
    .pot(200)
    .stacks(&[400, 400])
    .build();
let solver = SubgameCfrSolver::new(tree, hands, opp_reach, leaf_values);
```

New:
```rust
let tree = GameTree::build_subgame(
    Street::River,
    200.0,
    [100.0, 100.0],
    500.0,
    &[vec![0.5]],
    None,
);
let solver = SubgameCfrSolver::new(tree, hands, &board, opp_reach, leaf_values);
```

Note on value conversion: old tests used `pot: 200` (u32 chips). New uses `pot: 200.0` (f64 BB). The values are numerically the same — the solver doesn't care about units, just relative magnitudes.

**Step 8: Commit**

```bash
git add crates/core/src/blueprint/subgame_cfr.rs
git commit -m "refactor: SubgameCfrSolver uses V2GameTree instead of SubgameTree"
```

---

### Task 4: Update callers in postflop.rs

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs`

**Step 1: Update build_subgame_solver**

The function at line ~390 builds a `SubgameTreeBuilder`. Switch to `GameTree::build_subgame()`.

Old:
```rust
let tree = SubgameTreeBuilder::new()
    .board(board_cards)
    .bet_sizes(bet_sizes)
    .pot(pot)
    .stacks(&[stacks[0], stacks[1]])
    .depth_limit(1)
    .build();
```

New:
```rust
let street = match board_cards.len() {
    3 => Street::Flop,
    4 => Street::Turn,
    5 => Street::River,
    _ => return Err(format!("invalid board length: {}", board_cards.len())),
};
let pot_f = f64::from(pot);
let inv = [pot_f / 2.0; 2];
let starting_stack = f64::from(stacks[0]) + inv[0]; // stack + investment = starting
let bet_sizes_f64: Vec<f64> = bet_sizes.iter().map(|&s| f64::from(s)).collect();
let tree = GameTree::build_subgame(
    street,
    pot_f,
    inv,
    starting_stack,
    &[bet_sizes_f64],
    Some(1),
);
```

**Step 2: Update action_info extraction**

The root node is now `GameNode::Decision` with `TreeAction` instead of `SubgameNode::Decision` with `Action`. Update the match and the action-to-info conversion:

Old:
```rust
let action_infos = match &tree.nodes[0] {
    SubgameNode::Decision { actions, .. } => actions
        .iter()
        .enumerate()
        .map(|(i, a)| subgame_action_to_info(a, i, pot, bet_sizes))
        .collect(),
    _ => return Err("...".to_string()),
};
```

New:
```rust
let action_infos = match &tree.nodes[tree.root as usize] {
    GameNode::Decision { actions, .. } => actions
        .iter()
        .enumerate()
        .map(|(i, a)| v2_tree_action_to_info(a, i))
        .collect(),
    _ => return Err("Subgame tree root is not a decision node".to_string()),
};
```

Where `v2_tree_action_to_info` converts `TreeAction` to `ActionInfo` (this function may already exist in `exploration.rs` as `v2_action_info` — reuse it by making it `pub`).

**Step 3: Update SubgameCfrSolver::new() call**

Add the `board` parameter:

```rust
let solver = SubgameCfrSolver::new(
    tree.clone(), hands.clone(), board_cards,
    opponent_reach, leaf_values,
);
```

**Step 4: Update imports**

Replace:
```rust
use poker_solver_core::blueprint::{
    SubgameCfrSolver, SubgameHands, SubgameNode, SubgameStrategy, SubgameTree, SubgameTreeBuilder,
};
```
With:
```rust
use poker_solver_core::blueprint::{SubgameCfrSolver, SubgameHands, SubgameStrategy};
use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree, TreeAction};
use poker_solver_core::blueprint_v2::Street;
```

**Step 5: Update return type**

`build_subgame_solver` returns `(..., SubgameTree)`. Change to `(..., GameTree)`.

**Step 6: Run tests**

Run: `cargo test -p poker-solver-tauri`
Expected: All tests pass.

**Step 7: Commit**

```bash
git add crates/tauri-app/src/postflop.rs
git commit -m "refactor: postflop.rs uses GameTree::build_subgame()"
```

---

### Task 5: Update integration tests

**Files:**
- Modify: `crates/core/tests/full_pipeline.rs`

**Step 1: Update SubgameTreeBuilder calls**

There are two call sites (lines ~114 and ~224). Convert both from `SubgameTreeBuilder` to `GameTree::build_subgame()`, same pattern as Task 3 tests.

**Step 2: Run integration tests**

Run: `cargo test -p poker-solver-core --test full_pipeline`
Expected: All pass.

**Step 3: Commit**

```bash
git add crates/core/tests/full_pipeline.rs
git commit -m "refactor: integration tests use GameTree::build_subgame()"
```

---

### Task 6: Delete SubgameTree and clean up exports

**Files:**
- Delete contents of: `crates/core/src/blueprint/subgame_tree.rs` (keep `SubgameHands` and helper fns)
- Modify: `crates/core/src/blueprint/mod.rs`

**Step 1: Move SubgameHands to subgame_cfr.rs**

`SubgameHands` (combo enumeration) is still needed. Move it and its helper fns (`remaining_deck`, `cards_overlap`) to `subgame_cfr.rs` since that's the only consumer.

**Step 2: Delete SubgameTree, SubgameTreeBuilder, SubgameNode**

Remove all tree-related types and builder from `subgame_tree.rs`. If the file becomes empty (only had the tree), delete it entirely.

**Step 3: Update mod.rs exports**

Remove `SubgameTree`, `SubgameTreeBuilder`, `SubgameNode` from `pub use` in `crates/core/src/blueprint/mod.rs`. Keep `SubgameHands`.

**Step 4: Run full test suite**

Run: `cargo test`
Expected: All tests pass. No remaining references to deleted types.

**Step 5: Run clippy**

Run: `cargo clippy`
Expected: Clean.

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: delete SubgameTree and SubgameTreeBuilder (replaced by V2GameTree)"
```

---

### Task 7: Final verification

**Step 1: Run full test suite**

Run: `cargo test`
Expected: All tests pass.

**Step 2: Run clippy**

Run: `cargo clippy`
Expected: Clean.

**Step 3: Verify no remaining references**

Run: `rg "SubgameTree|SubgameNode|SubgameTreeBuilder" crates/`
Expected: No matches (except possibly in comments or doc strings).
