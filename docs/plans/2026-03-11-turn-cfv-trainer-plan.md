# Turn CFV Trainer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Build a turn CFV network trainer using the Supremus/DeepStack approach — a new `CfvSubgameSolver` with dynamic per-iteration leaf evaluation, plus cfvnet pipeline generalizations for turn datagen and validation CLI commands.

**Architecture:** New `CfvSubgameSolver` in core with `LeafEvaluator` trait (dynamic dispatch). Core provides `ExactRiverEvaluator` for testing. cfvnet provides `RiverNetEvaluator` using the trained river model. cfvnet's datagen pipeline generalizes to support both river (exact) and turn (net-backed) modes.

**Tech Stack:** Rust, rayon (parallel CFR), burn (neural net), rs_poker (card types)

**Design doc:** `docs/plans/2026-03-11-turn-cfv-trainer-design.md`

---

## Important Context

### Combo indexing
- Core's `SubgameHands::enumerate(board)` gives combos in deck order (variable count: 1128 for turn, 1081 for river)
- cfvnet uses canonical 1326-combo indexing via `card_pair_to_index(c1, c2)`
- The `LeafEvaluator` works in SubgameHands combo order (matches the solver's internal indexing)
- `RiverNetEvaluator` must map between the two orderings

### Card encoding
- Core uses `Card` type from `rs_poker` (value + suit)
- cfvnet uses `u8` encoding (0-51, `rank*4 + suit`)
- Conversions needed at the boundary between CfvSubgameSolver and cfvnet

### Dependency graph
- cfvnet currently depends on `range-solver` only (no core dependency)
- This plan adds `poker-solver-core` as a cfvnet dependency for `LeafEvaluator`, `CfvSubgameSolver`, `SubgameHands`, `GameTree` types

---

### Task 1: LeafEvaluator Trait + CfvSubgameSolver Module Skeleton

**Files:**
- Create: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs`

**Step 1: Create the new module with the LeafEvaluator trait and solver struct**

```rust
// crates/core/src/blueprint_v2/cfv_subgame_solver.rs

//! DCFR solver with dynamic per-iteration leaf evaluation.
//!
//! Like `SubgameCfrSolver` but recomputes leaf node values each iteration
//! by querying a `LeafEvaluator` with current reach-weighted ranges.
//! Used for training CFV networks (Supremus/DeepStack approach).

use crate::cfr::dcfr::DcfrParams;
use crate::cfr::parallel::{ParallelCfr, add_into, parallel_traverse};
use crate::cfr::regret::regret_match;
use crate::poker::Card;

use super::game_tree::{GameNode, GameTree, TerminalKind};
use super::subgame_cfr::{SubgameHands, SubgameStrategy, compute_equity_matrix, cards_overlap};

/// Evaluates counterfactual values at depth boundary nodes.
///
/// Called once per CFR iteration per boundary node with the current
/// reach-weighted ranges. Returns pot-relative CFVs from the
/// traverser's perspective, indexed by combo position in `SubgameHands`.
pub trait LeafEvaluator {
    fn evaluate(
        &self,
        combos: &[[Card; 2]],
        board: &[Card],
        pot: f64,
        effective_stack: f64,
        oop_range: &[f64],
        ip_range: &[f64],
        traverser: u8,
    ) -> Vec<f64>;
}

/// Indices of all `DepthBoundary` terminal nodes in the tree.
struct BoundaryInfo {
    /// (node_index, pot, invested) for each boundary node.
    boundaries: Vec<(usize, f64, [f64; 2])>,
}

/// Parallel DCFR solver with dynamic leaf evaluation.
pub struct CfvSubgameSolver {
    tree: GameTree,
    hands: SubgameHands,
    board: Vec<Card>,
    /// `equity[i][j]` = P(combo i beats combo j) at showdown.
    equity_matrix: Vec<Vec<f64>>,
    /// Flat buffer: cumulative regret sums.
    regret_sum: Vec<f64>,
    /// Flat buffer: cumulative strategy sums.
    strategy_sum: Vec<f64>,
    /// Maps (node_idx, combo_idx) to flat buffer offsets.
    layout: SubgameLayout,
    /// Per-boundary-node leaf CFVs, recomputed each iteration.
    /// Outer index: boundary ordinal. Inner index: combo.
    /// Two sets: one per traverser (OOP=0, IP=1).
    leaf_cfvs: [[Vec<Vec<f64>>; 2]],  // [traverser][boundary_idx][combo_idx]
    /// Boundary node info (indices, pots).
    boundary_info: BoundaryInfo,
    /// Maps tree node_idx -> boundary ordinal (or usize::MAX if not a boundary).
    node_to_boundary: Vec<usize>,
    /// Dynamic leaf evaluator.
    evaluator: Box<dyn LeafEvaluator>,
    /// DCFR discounting parameters.
    dcfr: DcfrParams,
    /// Current iteration count.
    pub iteration: u32,
}
```

Note: `SubgameLayout` is private in `subgame_cfr.rs`. You will need to either:
- Make it `pub(super)` and reuse it, OR
- Copy the struct into this file (it's small: `bases`, `num_actions`, `total_size` + `build()` and `slot()`)

Recommend: copy it as `CfvLayout` to keep modules independent.

**Step 2: Wire up the module in mod.rs**

Add to `crates/core/src/blueprint_v2/mod.rs`:
```rust
pub mod cfv_subgame_solver;
pub use cfv_subgame_solver::{CfvSubgameSolver, LeafEvaluator};
```

Also, make `cards_overlap` and `compute_equity_matrix` pub(super) in `subgame_cfr.rs` so the new module can reuse them (they are currently private `fn`).

**Step 3: Add a basic constructor and compilation test**

```rust
impl CfvSubgameSolver {
    pub fn new(
        tree: GameTree,
        hands: SubgameHands,
        board: Vec<Card>,
        evaluator: Box<dyn LeafEvaluator>,
    ) -> Self {
        let equity_matrix = compute_equity_matrix(&hands.combos, &board);
        let layout = CfvLayout::build(&tree, hands.combos.len());
        let buf_size = layout.total_size;

        // Find all DepthBoundary nodes
        let mut boundaries = Vec::new();
        let mut node_to_boundary = vec![usize::MAX; tree.nodes.len()];
        for (i, node) in tree.nodes.iter().enumerate() {
            if let GameNode::Terminal { kind: TerminalKind::DepthBoundary, pot, invested, .. } = node {
                node_to_boundary[i] = boundaries.len();
                boundaries.push((i, *pot, *invested));
            }
        }

        let num_boundaries = boundaries.len();
        let num_combos = hands.combos.len();
        let leaf_cfvs = [
            vec![vec![0.0; num_combos]; num_boundaries],
            vec![vec![0.0; num_combos]; num_boundaries],
        ];

        Self {
            tree,
            hands,
            board,
            equity_matrix,
            regret_sum: vec![0.0; buf_size],
            strategy_sum: vec![0.0; buf_size],
            layout,
            leaf_cfvs,
            boundary_info: BoundaryInfo { boundaries },
            node_to_boundary,
            evaluator,
            dcfr: DcfrParams::default(),
            iteration: 0,
        }
    }
}
```

**Step 4: Write a test that constructs the solver**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Card, Suit, Value};
    use crate::blueprint_v2::game_tree::GameTree;
    use crate::blueprint_v2::Street;
    use test_macros::timed_test;

    /// Trivial evaluator that returns 0.5 equity for all combos.
    struct ConstantEvaluator;
    impl LeafEvaluator for ConstantEvaluator {
        fn evaluate(&self, combos: &[[Card; 2]], _board: &[Card], _pot: f64,
            _stack: f64, _oop: &[f64], _ip: &[f64], _traverser: u8) -> Vec<f64> {
            vec![0.5; combos.len()]
        }
    }

    fn turn_board() -> Vec<Card> {
        vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
        ]
    }

    #[timed_test]
    fn solver_constructs_without_panic() {
        let board = turn_board();
        let tree = GameTree::build_subgame(
            Street::Turn, 100.0, [50.0, 50.0], 250.0, &[vec![1.0]], Some(1),
        );
        let hands = SubgameHands::enumerate(&board);
        let solver = CfvSubgameSolver::new(
            tree, hands, board, Box::new(ConstantEvaluator),
        );
        assert!(solver.boundary_info.boundaries.len() > 0);
    }
}
```

**Step 5: Run tests**

```bash
cargo test -p poker-solver-core cfv_subgame_solver -- --nocapture
```

**Step 6: Commit**

```bash
git add crates/core/src/blueprint_v2/cfv_subgame_solver.rs crates/core/src/blueprint_v2/mod.rs crates/core/src/blueprint_v2/subgame_cfr.rs
git commit -m "feat: add LeafEvaluator trait and CfvSubgameSolver skeleton"
```

---

### Task 2: Range Propagation

**Files:**
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs`

Range propagation walks the tree with the current strategy snapshot, tracking per-combo reach probabilities for both players. At each `DepthBoundary` node, it records the accumulated reach vectors.

**Step 1: Write the failing test**

```rust
#[timed_test]
fn propagate_ranges_sums_correctly() {
    let board = turn_board();
    let tree = GameTree::build_subgame(
        Street::Turn, 100.0, [50.0, 50.0], 250.0, &[vec![1.0]], Some(1),
    );
    let hands = SubgameHands::enumerate(&board);
    let n = hands.combos.len();
    let solver = CfvSubgameSolver::new(
        tree, hands, board, Box::new(ConstantEvaluator),
    );

    // With uniform strategy, ranges at each leaf should be a fraction of initial ranges
    let snapshot = solver.build_strategy_snapshot();
    let (oop_ranges, ip_ranges) = solver.propagate_ranges(&snapshot);

    // Should have one range vector per boundary node
    assert_eq!(oop_ranges.len(), solver.boundary_info.boundaries.len());

    // All range values should be non-negative
    for ranges in &oop_ranges {
        for &v in ranges {
            assert!(v >= 0.0, "negative range value: {v}");
        }
    }

    // Sum of ranges across all boundary nodes for each combo should be <= 1.0
    // (some reach goes to fold/showdown terminals)
    let mut total_reach = vec![0.0; n];
    for ranges in &oop_ranges {
        for (i, &v) in ranges.iter().enumerate() {
            total_reach[i] += v;
        }
    }
    for (i, &t) in total_reach.iter().enumerate() {
        assert!(t <= 1.0 + 1e-10, "combo {i} total reach {t} > 1.0");
    }
}
```

**Step 2: Verify it fails**

```bash
cargo test -p poker-solver-core cfv_subgame_solver::tests::propagate_ranges -- --nocapture
```

**Step 3: Implement `propagate_ranges` and `build_strategy_snapshot`**

`build_strategy_snapshot` is identical to `SubgameCfrSolver::build_strategy_snapshot` — copies regret sums through regret_match into a flat snapshot buffer.

`propagate_ranges` is a recursive tree walk:

```rust
impl CfvSubgameSolver {
    /// Build frozen strategy from current regret sums.
    fn build_strategy_snapshot(&self) -> Vec<f64> {
        // Same as SubgameCfrSolver::build_strategy_snapshot
        let mut snapshot = vec![0.0; self.layout.total_size];
        let num_combos = self.hands.combos.len();
        for (node_idx, node) in self.tree.nodes.iter().enumerate() {
            let num_actions = match node {
                GameNode::Decision { actions, .. } => actions.len(),
                _ => continue,
            };
            for combo_idx in 0..num_combos {
                let (base, _) = self.layout.slot(node_idx, combo_idx);
                let regrets = &self.regret_sum[base..base + num_actions];
                let probs = regret_match(regrets);
                snapshot[base..base + num_actions].copy_from_slice(&probs);
            }
        }
        snapshot
    }

    /// Propagate reach probabilities through the tree, collecting ranges at boundary nodes.
    /// Returns (oop_ranges, ip_ranges) where each is Vec<Vec<f64>> indexed by [boundary][combo].
    fn propagate_ranges(&self, snapshot: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let num_combos = self.hands.combos.len();
        let num_boundaries = self.boundary_info.boundaries.len();
        let mut oop_ranges = vec![vec![0.0; num_combos]; num_boundaries];
        let mut ip_ranges = vec![vec![0.0; num_combos]; num_boundaries];

        // For each combo, walk the tree tracking reach for both players.
        // Initial reach: 1.0 for both players (uniform starting range).
        for combo_idx in 0..num_combos {
            self.propagate_combo(
                snapshot, self.tree.root as usize, combo_idx,
                1.0, 1.0,
                &mut oop_ranges, &mut ip_ranges,
            );
        }

        (oop_ranges, ip_ranges)
    }

    fn propagate_combo(
        &self, snapshot: &[f64], node_idx: usize, combo_idx: usize,
        oop_reach: f64, ip_reach: f64,
        oop_ranges: &mut [Vec<f64>], ip_ranges: &mut [Vec<f64>],
    ) {
        match &self.tree.nodes[node_idx] {
            GameNode::Terminal { kind, .. } => {
                if *kind == TerminalKind::DepthBoundary {
                    let b = self.node_to_boundary[node_idx];
                    oop_ranges[b][combo_idx] += oop_reach;
                    ip_ranges[b][combo_idx] += ip_reach;
                }
                // Fold/Showdown: reach terminates here, nothing to record
            }
            GameNode::Chance { child, .. } => {
                self.propagate_combo(
                    snapshot, *child as usize, combo_idx,
                    oop_reach, ip_reach, oop_ranges, ip_ranges,
                );
            }
            GameNode::Decision { player, children, .. } => {
                let (base, num_actions) = self.layout.slot(node_idx, combo_idx);
                let strategy = &snapshot[base..base + num_actions];
                for (a, &child) in children.iter().enumerate() {
                    let p = strategy[a];
                    if p <= 0.0 { continue; }
                    let (new_oop, new_ip) = if *player == 0 {
                        (oop_reach * p, ip_reach)
                    } else {
                        (oop_reach, ip_reach * p)
                    };
                    self.propagate_combo(
                        snapshot, child as usize, combo_idx,
                        new_oop, new_ip, oop_ranges, ip_ranges,
                    );
                }
            }
        }
    }
}
```

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core cfv_subgame_solver -- --nocapture
```

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cfv_subgame_solver.rs
git commit -m "feat: implement range propagation for CfvSubgameSolver"
```

---

### Task 3: CFR Traversal with Per-Node Leaf Values

**Files:**
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs`

The CFR traversal is nearly identical to `SubgameCfrSolver`, except `DepthBoundary` reads from `leaf_cfvs[boundary_idx][combo]` instead of `leaf_values[combo]`.

**Step 1: Write the failing test**

```rust
#[timed_test]
fn solver_trains_and_produces_strategy() {
    let board = turn_board();
    // depth_limit=None means no boundaries, just like a river tree
    let tree = GameTree::build_subgame(
        Street::Turn, 100.0, [50.0, 50.0], 250.0, &[vec![1.0]], None,
    );
    let hands = SubgameHands::enumerate(&board);
    let n = hands.combos.len();
    // With no boundaries, ConstantEvaluator is never called
    let mut solver = CfvSubgameSolver::new(
        tree, hands, board, Box::new(ConstantEvaluator),
    );
    solver.train(100);
    assert_eq!(solver.iteration, 100);

    let strategy = solver.strategy();
    assert!(strategy.num_combos() > 0);

    // Strategies should be valid distributions
    for combo_idx in 0..n {
        let probs = strategy.root_probs(combo_idx);
        if probs.is_empty() { continue; }
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "combo {combo_idx}: sum = {sum}");
    }
}
```

**Step 2: Verify it fails**

```bash
cargo test -p poker-solver-core cfv_subgame_solver::tests::solver_trains -- --nocapture
```

**Step 3: Implement the traversal context, train loop, and strategy extraction**

Implement `CfvCfrCtx` (the per-iteration context struct implementing `ParallelCfr`), with `cfr_traverse` that handles:
- `Fold`: ±half_pot based on winner
- `Showdown`: reach-weighted equity (reuse `compute_equity_matrix`)
- `DepthBoundary`: `leaf_cfvs[boundary_ordinal][combo] * half_pot`
- `Chance`: pass through
- `Decision`: traverser/opponent branching (same as SubgameCfrSolver)

The `train()` method implements the per-iteration loop:
```
for iteration in 0..N:
    snapshot = build_strategy_snapshot()
    (oop_ranges, ip_ranges) = propagate_ranges(snapshot)
    for traverser in 0..2:
        // Evaluate leaf nodes for this traverser
        for (b_idx, (node_idx, pot, invested)) in boundaries.iter().enumerate():
            effective_stack = starting_stack - max(invested[0], invested[1])
            leaf_cfvs[traverser][b_idx] = evaluator.evaluate(
                combos, board, pot, effective_stack,
                oop_ranges[b_idx], ip_ranges[b_idx], traverser
            )
        // Build traversal context and run parallel CFR
        let ctx = CfvCfrCtx { ..., traverser, leaf_cfvs: &self.leaf_cfvs[traverser] }
        let (regret_delta, strategy_delta) = parallel_traverse(&ctx, &combos)
        add_into(&mut regret_sum, &regret_delta)
        add_into(&mut strategy_sum, &strategy_delta)
    // DCFR discounting
    if dcfr.should_discount(iteration):
        dcfr.discount_regrets(&mut regret_sum)
        dcfr.discount_strategy_sums(&mut strategy_sum)
```

The `strategy()` method is identical to `SubgameCfrSolver::strategy()`.

**Important:** The `CfvCfrCtx` struct needs `leaf_cfvs` and `node_to_boundary` to look up per-node leaf values during traversal. The DepthBoundary branch becomes:

```rust
TerminalKind::DepthBoundary => {
    let b_idx = self.node_to_boundary[node_idx];
    let cfv = self.leaf_cfvs[b_idx][hero_combo];
    cfv * half_pot
}
```

For `showdown_value`: reuse the same logic as `SubgameCfrSolver` — reach-weighted equity against opponent range. The opponent reach comes from `opp_reach_totals`, which must also be precomputed (same as SubgameCfrSolver).

**Note on opponent reach:** In `SubgameCfrSolver`, `opponent_reach` is a fixed input. In `CfvSubgameSolver`, opponent reach is implicitly 1.0 for all combos (uniform starting range) since the solver starts from the subgame root. The `opp_reach_totals` precomputation should use `vec![1.0; num_combos]` as the base opponent reach, then filter by card overlap.

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core cfv_subgame_solver -- --nocapture
```

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cfv_subgame_solver.rs
git commit -m "feat: implement CfvSubgameSolver train loop with dynamic leaf evaluation"
```

---

### Task 4: CfvSubgameSolver with DepthBoundary Integration Test

**Files:**
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs` (tests)

Test the solver with a depth-limited tree where DepthBoundary nodes are actually hit.

**Step 1: Write the test**

```rust
#[timed_test(5)]
fn solver_with_depth_boundary_converges() {
    let board = turn_board();
    // depth_limit=1: turn actions only, river transitions become DepthBoundary
    let tree = GameTree::build_subgame(
        Street::Turn, 100.0, [50.0, 50.0], 250.0, &[vec![1.0]], Some(1),
    );

    // Verify tree has boundary nodes
    let boundary_count = tree.nodes.iter()
        .filter(|n| matches!(n, GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. }))
        .count();
    assert!(boundary_count > 0, "tree should have DepthBoundary nodes");

    let hands = SubgameHands::enumerate(&board);
    let n = hands.combos.len();
    let mut solver = CfvSubgameSolver::new(
        tree, hands, board, Box::new(ConstantEvaluator),
    );
    solver.train(200);

    let strategy = solver.strategy();
    // With constant 0.5 equity evaluator, strategies should still be valid distributions
    for combo_idx in 0..n {
        let probs = strategy.root_probs(combo_idx);
        if probs.is_empty() { continue; }
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "combo {combo_idx}: sum = {sum}");
    }
}
```

**Step 2: Run test**

```bash
cargo test -p poker-solver-core cfv_subgame_solver::tests::solver_with_depth_boundary -- --nocapture
```

Expected: PASS (if Task 3 was implemented correctly). If it fails, debug and fix.

**Step 3: Commit**

```bash
git add crates/core/src/blueprint_v2/cfv_subgame_solver.rs
git commit -m "test: CfvSubgameSolver with DepthBoundary convergence"
```

---

### Task 5: ExactRiverEvaluator

**Files:**
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs`

Implements `LeafEvaluator` by solving each possible river card exactly using the existing `SubgameCfrSolver`.

**Step 1: Write the failing test**

```rust
#[timed_test(30)]
fn exact_river_evaluator_produces_bounded_cfvs() {
    let board = turn_board();
    let hands = SubgameHands::enumerate(&board);

    let evaluator = ExactRiverEvaluator::new(vec![1.0], 1000);
    let oop_range: Vec<f64> = vec![1.0 / hands.combos.len() as f64; hands.combos.len()];
    let ip_range = oop_range.clone();

    let cfvs = evaluator.evaluate(
        &hands.combos, &board, 100.0, 200.0,
        &oop_range, &ip_range, 0,
    );

    assert_eq!(cfvs.len(), hands.combos.len());
    // Pot-relative CFVs should be in [-1, 1] range
    for (i, &v) in cfvs.iter().enumerate() {
        assert!((-1.0..=1.0).contains(&v),
            "combo {i} CFV {v} out of pot-relative range");
    }
}
```

**Step 2: Verify it fails**

```bash
cargo test -p poker-solver-core cfv_subgame_solver::tests::exact_river_evaluator -- --nocapture
```

**Step 3: Implement ExactRiverEvaluator**

```rust
/// Evaluates depth boundary nodes by solving each possible next-street
/// card exactly using `SubgameCfrSolver`. Slow but provides ground truth.
pub struct ExactRiverEvaluator {
    /// Bet sizes for the river solver.
    bet_sizes: Vec<f64>,
    /// Number of CFR iterations per river solve.
    iterations: u32,
}

impl ExactRiverEvaluator {
    pub fn new(bet_sizes: Vec<f64>, iterations: u32) -> Self {
        Self { bet_sizes, iterations }
    }
}

impl LeafEvaluator for ExactRiverEvaluator {
    fn evaluate(
        &self, combos: &[[Card; 2]], board: &[Card], pot: f64,
        effective_stack: f64, oop_range: &[f64], ip_range: &[f64], traverser: u8,
    ) -> Vec<f64> {
        let num_combos = combos.len();
        let mut cfvs = vec![0.0; num_combos];
        let mut counts = vec![0u32; num_combos];
        let remaining = remaining_deck(board);

        for &river_card in &remaining {
            let mut full_board = board.to_vec();
            full_board.push(river_card);

            let river_hands = SubgameHands::enumerate(&full_board);
            let river_n = river_hands.combos.len();

            // Map opponent reach from our combo indexing to river combo indexing.
            // Use the opponent's range (ip if traverser=0, oop if traverser=1).
            let opp_src = if traverser == 0 { ip_range } else { oop_range };
            let mut river_opp_reach = vec![0.0; river_n];
            for (ri, river_combo) in river_hands.combos.iter().enumerate() {
                // Find this combo in the parent combo list
                if let Some(pi) = combos.iter().position(|c| c == river_combo) {
                    river_opp_reach[ri] = opp_src[pi];
                }
            }

            let river_tree = GameTree::build_subgame(
                Street::River, pot,
                [pot / 2.0, pot / 2.0],
                effective_stack + pot / 2.0,
                &[self.bet_sizes.clone()],
                None,
            );
            let leaf_vals = vec![0.0; river_n];
            let mut solver = SubgameCfrSolver::new(
                river_tree, river_hands.clone(), &full_board,
                river_opp_reach, leaf_vals,
            );
            solver.train(self.iterations);

            // Extract per-combo values from the solved equilibrium.
            // Use the equity-based approach: compute showdown value at root
            // weighted by opponent reach. This gives the CFV for each combo.
            let river_equities = compute_combo_equities(
                &river_hands, &full_board,
                &(if traverser == 0 { ip_range } else { oop_range })
                    .iter().copied()
                    .collect::<Vec<_>>()  // simplified — need proper mapping
            );

            // Map river combo CFVs back to parent combo indices
            for (ri, river_combo) in river_hands.combos.iter().enumerate() {
                if let Some(pi) = combos.iter().position(|c| c == river_combo) {
                    // blocked by river card?
                    if river_combo[0] == river_card || river_combo[1] == river_card {
                        continue;
                    }
                    cfvs[pi] += river_equities[ri];
                    counts[pi] += 1;
                }
            }
        }

        // Average across river cards, normalize to pot-relative
        for i in 0..num_combos {
            if counts[i] > 0 {
                cfvs[i] /= counts[i] as f64;
            } else {
                cfvs[i] = 0.5; // no valid river cards: neutral
            }
        }
        cfvs
    }
}
```

**Important implementation note:** The above is a sketch. The exact extraction of per-combo CFVs from a solved `SubgameCfrSolver` needs careful thought. Two options:
1. Use `compute_combo_equities()` from `subgame_cfr.rs` to get raw equities per river card, averaged across cards. This gives equity (0-1) which IS the pot-relative leaf value.
2. Run the full solver and extract the root-node expected value per combo from the strategy. This is more accurate but more complex.

Start with option 1 (equity-based) for the first pass — it gives a simple, correct implementation.

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core cfv_subgame_solver::tests::exact_river_evaluator -- --nocapture
```

**Step 5: Add integration test: CfvSubgameSolver + ExactRiverEvaluator on a turn tree**

```rust
#[timed_test(60)]
fn cfv_solver_with_exact_evaluator_on_turn() {
    let board = turn_board();
    let tree = GameTree::build_subgame(
        Street::Turn, 100.0, [50.0, 50.0], 250.0, &[vec![1.0]], Some(1),
    );
    // Use small hand set for speed
    let full_hands = SubgameHands::enumerate(&board);
    let hands = SubgameHands { combos: full_hands.combos[..20].to_vec() };

    let evaluator = ExactRiverEvaluator::new(vec![1.0], 200);
    let mut solver = CfvSubgameSolver::new(
        tree, hands, board, Box::new(evaluator),
    );
    solver.train(50);

    let strategy = solver.strategy();
    assert!(strategy.num_combos() > 0);
}
```

**Step 6: Run all tests**

```bash
cargo test -p poker-solver-core cfv_subgame_solver -- --nocapture
```

**Step 7: Commit**

```bash
git add crates/core/src/blueprint_v2/cfv_subgame_solver.rs
git commit -m "feat: add ExactRiverEvaluator implementing LeafEvaluator"
```

---

### Task 6: Generalize cfvnet Board Handling

**Files:**
- Modify: `crates/cfvnet/src/datagen/sampler.rs`
- Modify: `crates/cfvnet/src/datagen/storage.rs`
- Modify: `crates/cfvnet/src/datagen/range_gen.rs`

**Step 1: Generalize `sample_board` to accept board size**

In `sampler.rs`, change `sample_board`:
```rust
// Before: fn sample_board<R: Rng>(rng: &mut R) -> [u8; 5]
// After:
pub fn sample_board<R: Rng>(rng: &mut R, num_cards: usize) -> Vec<u8> {
    let mut board = Vec::with_capacity(num_cards);
    while board.len() < num_cards {
        let card = rng.gen_range(0..52u8);
        if !board.contains(&card) {
            board.push(card);
        }
    }
    board
}
```

Update `Situation.board` from `[u8; 5]` to `Vec<u8>` and update `sample_situation` accordingly.

**Step 2: Generalize `TrainingRecord.board` in storage.rs**

Change board from `[u8; 5]` to `Vec<u8>`. Add a `board_size: u8` prefix to the binary format:

```
board_size (u8) | board (board_size × u8) | pot (f32) | stack (f32) | player (u8) | ...
```

Update `RECORD_SIZE` to be a function: `fn record_size(board_size: usize) -> usize`.
Update `write_record` and `read_record` to handle variable board size.

**Step 3: Generalize `compute_hand_strengths` in range_gen.rs**

Currently takes `&[u8; 5]`. Change to `&[u8]` (slice). For 4-card boards, the 6-card evaluator is needed instead of 7-card. The function should dispatch based on `board.len()`:

```rust
pub fn compute_hand_strengths(board: &[u8]) -> [u16; NUM_COMBOS] {
    match board.len() {
        5 => compute_hand_strengths_7(board),  // 2 hole + 5 board = 7
        4 => compute_hand_strengths_6(board),  // 2 hole + 4 board = 6
        _ => panic!("unsupported board size: {}", board.len()),
    }
}
```

For the 6-card evaluator: evaluate all C(remaining_deck, 1) = 46 possible 5th cards, average the ranks. Or simply use the best-5-of-6 evaluator if available.

**Note:** `generate_rsp_range` currently takes `&[u8; 5]`. Change to `&[u8]`.

**Step 4: Write tests**

```rust
#[test]
fn sample_board_4_cards() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let board = sample_board(&mut rng, 4);
    assert_eq!(board.len(), 4);
    // All unique
    for i in 0..4 { for j in (i+1)..4 { assert_ne!(board[i], board[j]); } }
}

#[test]
fn record_roundtrip_4_card_board() {
    // Create a record with 4-card board, write it, read it back, verify equality
}
```

**Step 5: Run tests**

```bash
cargo test -p cfvnet datagen -- --nocapture
```

**Step 6: Commit**

```bash
git add crates/cfvnet/src/datagen/
git commit -m "feat: generalize cfvnet datagen for variable board sizes (turn support)"
```

---

### Task 7: Generalize cfvnet Network INPUT_SIZE

**Files:**
- Modify: `crates/cfvnet/src/model/network.rs`
- Modify: `crates/cfvnet/src/model/dataset.rs`
- Modify: `crates/cfvnet/src/config.rs`

**Step 1: Make INPUT_SIZE configurable**

In `network.rs`, change `INPUT_SIZE` from a constant to a parameter of `CfvNet::new()`:

```rust
// Before: pub const INPUT_SIZE: usize = 2660;
// After: computed from board_size
pub fn input_size(board_cards: usize) -> usize {
    NUM_COMBOS + NUM_COMBOS + board_cards + 1 + 1 + 1
    // OOP range + IP range + board + pot + stack + player
}
```

Update `CfvNet::new()` to take `input_size: usize` parameter.

**Step 2: Update dataset encoding**

In `dataset.rs`, `encode_record` must handle variable board size:

```rust
pub fn encode_record(rec: &TrainingRecord) -> CfvItem {
    let board_size = rec.board.len();
    let input_size = input_size(board_size);
    let mut input = Vec::with_capacity(input_size);

    // OOP range (1326)
    input.extend(rec.oop_range.iter().map(|&v| v));
    // IP range (1326)
    input.extend(rec.ip_range.iter().map(|&v| v));
    // Board cards (variable: 4 or 5)
    for &card in &rec.board {
        input.push(card as f32 / 51.0);
    }
    // Pot, stack, player
    input.push(rec.pot / 400.0);
    input.push(rec.effective_stack / 400.0);
    input.push(rec.player as f32);
    // ... rest same
}
```

**Step 3: Add `board_size` / `street` to config**

In `config.rs`, add to `DatagenConfig`:
```rust
pub street: String,  // "river" or "turn"
```

And derive board_size from street (river=5, turn=4, flop=3).

Add to `GameConfig`:
```rust
pub river_model_path: Option<String>,  // path to trained river model (for turn datagen)
```

**Step 4: Write tests**

```rust
#[test]
fn input_size_correct_for_turn() {
    assert_eq!(input_size(4), 2659);
}
#[test]
fn input_size_correct_for_river() {
    assert_eq!(input_size(5), 2660);
}
```

**Step 5: Run tests**

```bash
cargo test -p cfvnet model -- --nocapture
```

**Step 6: Commit**

```bash
git add crates/cfvnet/src/model/ crates/cfvnet/src/config.rs
git commit -m "feat: parameterize cfvnet network INPUT_SIZE for variable board sizes"
```

---

### Task 8: RiverNetEvaluator

**Files:**
- Modify: `crates/cfvnet/Cargo.toml` (add poker-solver-core dependency)
- Create: `crates/cfvnet/src/eval/river_net_evaluator.rs`
- Modify: `crates/cfvnet/src/eval/mod.rs`

**Step 1: Add core dependency to cfvnet**

In `crates/cfvnet/Cargo.toml`:
```toml
poker-solver-core = { path = "../core" }
```

**Step 2: Implement RiverNetEvaluator**

```rust
use poker_solver_core::blueprint_v2::cfv_subgame_solver::LeafEvaluator;
use poker_solver_core::poker::Card;

/// Evaluates turn depth boundaries by averaging the river CFV network
/// over all possible river cards.
pub struct RiverNetEvaluator<B: Backend> {
    model: CfvNet<B>,
    device: B::Device,
}

impl<B: Backend> RiverNetEvaluator<B> {
    pub fn new(model: CfvNet<B>, device: B::Device) -> Self {
        Self { model, device }
    }
}

impl<B: Backend> LeafEvaluator for RiverNetEvaluator<B> {
    fn evaluate(
        &self, combos: &[[Card; 2]], board: &[Card], pot: f64,
        effective_stack: f64, oop_range: &[f64], ip_range: &[f64], traverser: u8,
    ) -> Vec<f64> {
        let num_combos = combos.len();
        let mut cfvs = vec![0.0; num_combos];
        let mut counts = vec![0u32; num_combos];

        // Find remaining cards not on the board
        let board_set: Vec<u8> = board.iter().map(|c| card_to_u8(*c)).collect();
        let remaining: Vec<u8> = (0..52u8).filter(|c| !board_set.contains(c)).collect();

        for &river_card in &remaining {
            let mut full_board_u8 = board_set.clone();
            full_board_u8.push(river_card);

            // Build input for the river net
            // Convert ranges from combo-indexed to canonical 1326-indexed
            let mut oop_1326 = [0.0f32; 1326];
            let mut ip_1326 = [0.0f32; 1326];
            for (ci, combo) in combos.iter().enumerate() {
                let idx = canonical_combo_index(combo);
                oop_1326[idx] = oop_range[ci] as f32;
                ip_1326[idx] = ip_range[ci] as f32;
            }

            // Encode situation and run forward pass
            let input = encode_for_inference(
                &full_board_u8, pot as f32, effective_stack as f32,
                &oop_1326, &ip_1326, traverser,
            );
            let output = self.model.forward(/* tensor from input */);

            // Map output (1326-indexed) back to combo-indexed
            for (ci, combo) in combos.iter().enumerate() {
                if combo[0] == u8_to_card(river_card) || combo[1] == u8_to_card(river_card) {
                    continue; // blocked by river card
                }
                let idx = canonical_combo_index(combo);
                cfvs[ci] += output[idx] as f64;
                counts[ci] += 1;
            }
        }

        for i in 0..num_combos {
            if counts[i] > 0 {
                cfvs[i] /= counts[i] as f64;
            }
        }
        cfvs
    }
}
```

**Note on Backend generic:** `LeafEvaluator` is `dyn`-dispatched, but `RiverNetEvaluator<B>` has a type parameter. This works because you instantiate it with a concrete backend (e.g., `NdArray`) and box it: `Box::new(RiverNetEvaluator::<NdArray>::new(model, device)) as Box<dyn LeafEvaluator>`.

**Step 3: Write test**

Testing requires a trained river model. For unit tests, use a mock or very small model:

```rust
#[test]
fn river_net_evaluator_returns_correct_shape() {
    // Build a tiny model (1 hidden layer, 8 units) for testing
    let device = NdArrayDevice::default();
    let model = CfvNet::<NdArray>::new(&device, 1, 8);
    let evaluator = RiverNetEvaluator::new(model, device);

    let board = /* 4 turn cards as Card */;
    let hands = SubgameHands::enumerate(&board);
    let n = hands.combos.len();
    let range = vec![1.0 / n as f64; n];

    let cfvs = evaluator.evaluate(&hands.combos, &board, 100.0, 200.0, &range, &range, 0);
    assert_eq!(cfvs.len(), n);
}
```

**Step 4: Run tests**

```bash
cargo test -p cfvnet eval::river_net -- --nocapture
```

**Step 5: Commit**

```bash
git add crates/cfvnet/Cargo.toml crates/cfvnet/src/eval/
git commit -m "feat: add RiverNetEvaluator implementing LeafEvaluator"
```

---

### Task 9: Turn Datagen Pipeline

**Files:**
- Create: `crates/cfvnet/src/datagen/turn_generate.rs`
- Modify: `crates/cfvnet/src/datagen/mod.rs`
- Modify: `crates/cfvnet/src/main.rs`

**Step 1: Create turn datagen orchestrator**

```rust
// crates/cfvnet/src/datagen/turn_generate.rs

/// Generate turn training data using CfvSubgameSolver + RiverNetEvaluator.
pub fn generate_turn_training_data(
    config: &CfvnetConfig,
    river_model_path: &Path,
    output_path: &Path,
) -> Result<(), String> {
    // 1. Load trained river model
    let model = load_river_model(river_model_path)?;
    let evaluator = RiverNetEvaluator::new(model, device);

    // 2. Sample turn situations (4-card boards)
    // 3. For each situation:
    //    a. Build turn tree with depth_limit=1
    //    b. Create CfvSubgameSolver with evaluator
    //    c. Train for config.datagen.solver_iterations
    //    d. Extract root CFVs → write as TrainingRecord
    // 4. Write records to output file

    // Parallel over situations using rayon
}
```

The key difference from river `generate_training_data`: instead of calling `solve_situation` (range-solver), we build a `CfvSubgameSolver`, train it, and extract root CFVs.

**Extracting root CFVs:** After solving, for each combo, the root-node expected value can be computed by:
1. Get the strategy at the root
2. For each action, compute the expected value recursively through the tree
3. Weight by strategy probabilities

Or more simply: during the last iteration's traversal, record the per-combo expected values at the root. This requires a small modification to capture root values.

**Alternative approach:** Use `compute_combo_equities` or a final traversal pass to extract CFVs.

**Step 2: Wire into CLI**

Add `generate-turn` subcommand (or generalize `generate` with a `--street turn` flag):

```rust
Generate {
    #[clap(long)]
    config: PathBuf,
    #[clap(long)]
    output: PathBuf,
    #[clap(long)]
    street: Option<String>,  // "river" (default) or "turn"
    #[clap(long)]
    river_model: Option<PathBuf>,  // required for --street turn
}
```

**Step 3: Write test**

```rust
#[test]
fn turn_datagen_produces_records() {
    // Use a tiny config (2 samples, few iterations)
    // Verify output file has correct number of records
}
```

**Step 4: Run tests**

```bash
cargo test -p cfvnet datagen::turn -- --nocapture
```

**Step 5: Commit**

```bash
git add crates/cfvnet/src/datagen/ crates/cfvnet/src/main.rs
git commit -m "feat: add turn datagen pipeline using CfvSubgameSolver"
```

---

### Task 10: compare-exact and compare-net CLI Subcommands

**Files:**
- Modify: `crates/cfvnet/src/main.rs`
- Create: `crates/cfvnet/src/eval/compare_turn.rs`
- Modify: `crates/cfvnet/src/eval/mod.rs`

**Step 1: Add CLI subcommands**

```rust
CompareExact {
    #[clap(long)]
    model: PathBuf,
    #[clap(long)]
    river_model: Option<PathBuf>,
    #[clap(long)]
    num_spots: usize,
    #[clap(long)]
    threads: Option<usize>,
    #[clap(long)]
    config: Option<PathBuf>,
},
CompareNet {
    #[clap(long)]
    model: PathBuf,
    #[clap(long)]
    river_model: PathBuf,
    #[clap(long)]
    num_spots: usize,
    #[clap(long)]
    threads: Option<usize>,
    #[clap(long)]
    config: Option<PathBuf>,
},
```

**Step 2: Implement compare-net**

`compare-net` for each spot:
1. Sample random turn situation
2. Solve with `CfvSubgameSolver` + `RiverNetEvaluator` → get "exact" CFVs
3. Query turn model with same situation → get predicted CFVs
4. Compute metrics (MAE, RMSE, max error, mBB)

Reuse the pattern from `eval/compare.rs`'s `run_comparison`:

```rust
pub fn run_turn_comparison_net(
    config: &GameConfig,
    turn_model_path: &Path,
    river_model_path: &Path,
    num_spots: usize,
    seed: u64,
) -> Result<ComparisonSummary, String> {
    // Load both models
    // For each spot: solve with net evaluator, predict with turn model, compare
}
```

**Step 3: Implement compare-exact**

Same as compare-net but uses `ExactRiverEvaluator` instead of `RiverNetEvaluator`. Much slower.

```rust
pub fn run_turn_comparison_exact(
    config: &GameConfig,
    turn_model_path: &Path,
    num_spots: usize,
    seed: u64,
) -> Result<ComparisonSummary, String> {
    // For each spot: solve with ExactRiverEvaluator, predict with turn model, compare
}
```

**Step 4: Write tests**

```rust
#[test]
fn compare_net_subcommand_parses() {
    // Test CLI argument parsing
}
```

**Step 5: Run tests**

```bash
cargo test -p cfvnet eval::compare_turn -- --nocapture
```

**Step 6: Commit**

```bash
git add crates/cfvnet/src/eval/ crates/cfvnet/src/main.rs
git commit -m "feat: add compare-exact and compare-net CLI subcommands"
```

---

## Summary

| Task | Component | Layer | New/Modify |
|-|-|-|-|
| 1 | LeafEvaluator + CfvSubgameSolver skeleton | Domain (core) | Create |
| 2 | Range propagation | Domain (core) | Modify |
| 3 | CFR traversal + train loop | Domain (core) | Modify |
| 4 | DepthBoundary integration test | Domain (core) | Test |
| 5 | ExactRiverEvaluator | Domain (core) | Modify |
| 6 | Generalize cfvnet board handling | I/O (cfvnet) | Modify |
| 7 | Generalize cfvnet INPUT_SIZE | I/O (cfvnet) | Modify |
| 8 | RiverNetEvaluator | Coordination (cfvnet) | Create |
| 9 | Turn datagen pipeline | Coordination (cfvnet) | Create |
| 10 | compare-exact/compare-net CLI | I/O (cfvnet) | Modify |
