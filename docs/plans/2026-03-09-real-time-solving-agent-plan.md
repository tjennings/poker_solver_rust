# Real-Time Solving Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Build a Pluribus-inspired `RealTimeSolvingAgent` that uses blueprint strategy for preflop + real-time subgame solving for postflop decisions, plugging into the existing simulation framework.

**Architecture:** The agent narrows the opponent's range via blueprint action probabilities, then dispatches to either the range-solver crate (full-depth, for river or narrow ranges) or SubgameCfrSolver (depth-limited with precomputed CBVs, for wide ranges on flop/turn). Implements the existing `Agent` trait so it works with `run_simulation()`.

**Tech Stack:** Rust, rs_poker arena Agent trait, range-solver crate, existing SubgameCfrSolver, bincode serialization for CBV tables.

**Design doc:** `docs/plans/2026-03-09-real-time-solving-agent-design.md`

---

### Task 1: RangeNarrower — combo weight tracking

Core domain component: tracks 1326 opponent combo weights, updates them by multiplying blueprint action probabilities, applies card removal.

**Files:**
- Create: `crates/core/src/blueprint/range_narrower.rs`
- Modify: `crates/core/src/blueprint/mod.rs` (add `pub mod range_narrower;`)
- Test: inline `#[cfg(test)] mod tests` in `range_narrower.rs`

**Step 1: Write failing tests for RangeNarrower**

```rust
// crates/core/src/blueprint/range_narrower.rs

/// Maps canonical 169 hand indices to all corresponding 1326 combos.
/// Each combo is an ordered pair (c1, c2) where c1 < c2, indexed as c1*52+c2
/// minus the diagonal. Standard 1326 = C(52,2) indexing.

#[cfg(test)]
mod tests {
    use super::*;
    use rs_poker::core::Card;

    #[test]
    fn test_new_starts_uniform() {
        let rn = RangeNarrower::new();
        assert_eq!(rn.weights().len(), 1326);
        assert!(rn.weights().iter().all(|&w| (w - 1.0).abs() < 1e-9));
    }

    #[test]
    fn test_update_multiplies_weights() {
        let mut rn = RangeNarrower::new();
        // Action probs: combo 0 had 0.5 prob of taking this action
        let mut action_probs = vec![1.0; 1326];
        action_probs[0] = 0.5;
        action_probs[1] = 0.0;
        rn.update(&action_probs);
        assert!((rn.weights()[0] - 0.5).abs() < 1e-9);
        assert!((rn.weights()[1] - 0.0).abs() < 1e-9);
        assert!((rn.weights()[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_card_removal_zeros_blocked_combos() {
        let mut rn = RangeNarrower::new();
        let board = vec![
            Card::new(rs_poker::core::Value::Ace, rs_poker::core::Suit::Spade),
        ];
        let hero = [
            Card::new(rs_poker::core::Value::King, rs_poker::core::Suit::Spade),
            Card::new(rs_poker::core::Value::King, rs_poker::core::Suit::Heart),
        ];
        rn.apply_card_removal(&board, &hero);
        // Any combo containing As, Ks, or Kh should be zero
        let live = rn.live_combo_count();
        // 1326 - combos containing any of 3 blocked cards
        // 3 cards block: 3*51 - C(3,2) = 153 - 3 = 150 combos
        assert_eq!(live, 1326 - 150);
    }

    #[test]
    fn test_live_combo_count() {
        let mut rn = RangeNarrower::new();
        assert_eq!(rn.live_combo_count(), 1326);
        let mut action_probs = vec![1.0; 1326];
        action_probs[0] = 0.0;
        rn.update(&action_probs);
        assert_eq!(rn.live_combo_count(), 1325);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core range_narrower -- --nocapture`
Expected: FAIL — module doesn't exist yet.

**Step 3: Implement RangeNarrower**

```rust
// crates/core/src/blueprint/range_narrower.rs

use rs_poker::core::Card;

/// Combo index: for cards c1 < c2 (0-indexed, 0..52),
/// index = c1 * (103 - c1) / 2 + c2 - c1 - 1
/// Total: C(52,2) = 1326 combos.
pub const NUM_COMBOS: usize = 1326;

pub struct RangeNarrower {
    weights: Vec<f64>,
}

impl RangeNarrower {
    pub fn new() -> Self {
        Self {
            weights: vec![1.0; NUM_COMBOS],
        }
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Multiply each combo's weight by the blueprint's probability
    /// of the observed action for that combo.
    pub fn update(&mut self, action_probs: &[f64]) {
        assert_eq!(action_probs.len(), NUM_COMBOS);
        for (w, &p) in self.weights.iter_mut().zip(action_probs.iter()) {
            *w *= p;
        }
    }

    /// Zero out combos containing any card in board or hero's hand.
    pub fn apply_card_removal(&mut self, board: &[Card], hero: &[Card; 2]) {
        let mut blocked = [false; 52];
        for c in board.iter().chain(hero.iter()) {
            blocked[card_index(c)] = true;
        }
        for (idx, w) in self.weights.iter_mut().enumerate() {
            let (c1, c2) = combo_cards(idx);
            if blocked[c1] || blocked[c2] {
                *w = 0.0;
            }
        }
    }

    pub fn live_combo_count(&self) -> usize {
        self.weights.iter().filter(|&&w| w > 0.0).count()
    }

    pub fn reset(&mut self) {
        self.weights.fill(1.0);
    }
}

/// Convert a Card to a 0..52 index.
pub fn card_index(card: &Card) -> usize {
    let suit = match card.suit {
        rs_poker::core::Suit::Spade => 0,
        rs_poker::core::Suit::Heart => 1,
        rs_poker::core::Suit::Diamond => 2,
        rs_poker::core::Suit::Club => 3,
    };
    let rank = card.value as usize; // Value enum starts at 2
    // Map to 0..12: Two=0, Three=1, ..., Ace=12
    // rs_poker Value enum: Two=2, Three=3, ..., Ace=14
    let rank_idx = rank - 2;
    rank_idx * 4 + suit
}

/// Convert combo index to (c1, c2) card indices where c1 < c2.
pub fn combo_cards(combo_idx: usize) -> (usize, usize) {
    // Enumerate: (0,1), (0,2), ..., (0,51), (1,2), ..., (50,51)
    let mut idx = combo_idx;
    let mut c1 = 0usize;
    loop {
        let remaining = 51 - c1;
        if idx < remaining {
            return (c1, c1 + 1 + idx);
        }
        idx -= remaining;
        c1 += 1;
    }
}

/// Convert (c1, c2) card indices to combo index. Requires c1 < c2.
pub fn combo_index(c1: usize, c2: usize) -> usize {
    debug_assert!(c1 < c2 && c2 < 52);
    c1 * (103 - c1) / 2 + c2 - c1 - 1
}
```

**Step 4: Register the module**

Add `pub mod range_narrower;` to `crates/core/src/blueprint/mod.rs`.

**Step 5: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core range_narrower -- --nocapture`
Expected: All 4 tests PASS.

**Step 6: Commit**

```bash
git add crates/core/src/blueprint/range_narrower.rs crates/core/src/blueprint/mod.rs
git commit -m "feat: add RangeNarrower for 1326-combo opponent range tracking"
```

---

### Task 2: Bucket-to-combo expansion

Bridge between the blueprint's abstract bucket space and the 1326-combo space needed by solvers.

**Files:**
- Modify: `crates/core/src/blueprint/range_narrower.rs` (add expansion functions)
- Test: inline tests in same file

**Step 1: Write failing tests**

```rust
#[test]
fn test_expand_buckets_to_combos() {
    // 169 canonical hands → 1326 combos
    // Bucket 0 (hand AA) has weight 0.5, everything else 1.0
    let mut bucket_weights = vec![1.0f64; 169];
    bucket_weights[0] = 0.5; // AA pocket pair
    let result = expand_buckets_to_combos(&bucket_weights);
    assert_eq!(result.len(), NUM_COMBOS);
    // AA has 6 combos (C(4,2)), all should have weight 0.5
    // Verify at least one AA combo has weight 0.5
    let aa_combo = combo_index(card_index_from_rank_suit(12, 0), card_index_from_rank_suit(12, 1));
    assert!((result[aa_combo] - 0.5).abs() < 1e-9);
}

#[test]
fn test_combo_to_canonical_hand_roundtrip() {
    // Every combo maps to exactly one of 169 canonical hands
    for idx in 0..NUM_COMBOS {
        let hand = combo_to_canonical_hand(idx);
        assert!(hand < 169, "combo {} mapped to invalid hand {}", idx, hand);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core range_narrower -- --nocapture`
Expected: FAIL — functions don't exist.

**Step 3: Implement bucket expansion**

```rust
/// Map a combo index to its canonical 169-hand index.
/// 169 hands: 13 pocket pairs + 78 suited + 78 offsuit.
/// Layout: pairs at 0..13, suited at 13..91, offsuit at 91..169.
/// Or use the project's existing canonical hand indexing.
pub fn combo_to_canonical_hand(combo_idx: usize) -> usize {
    let (c1, c2) = combo_cards(combo_idx);
    let rank1 = c1 / 4; // 0=Two .. 12=Ace
    let suit1 = c1 % 4;
    let rank2 = c2 / 4;
    let suit2 = c2 % 4;
    let (hi, lo) = if rank1 >= rank2 { (rank1, rank2) } else { (rank2, rank1) };
    if rank1 == rank2 {
        // Pocket pair: index by rank (Ace=12 → index 12, Two=0 → index 0)
        hi
    } else if suit1 == suit2 {
        // Suited: 13 + triangular index
        13 + suited_index(hi, lo)
    } else {
        // Offsuit: 13 + 78 + triangular index
        91 + offsuit_index(hi, lo)
    }
}

/// Triangular index for hi > lo, both in 0..13
fn suited_index(hi: usize, lo: usize) -> usize {
    // Map (hi, lo) where hi > lo to 0..78
    // (1,0)=0, (2,0)=1, (2,1)=2, (3,0)=3, ...
    hi * (hi - 1) / 2 + lo
}

fn offsuit_index(hi: usize, lo: usize) -> usize {
    suited_index(hi, lo)
}

/// Expand 169 bucket weights to 1326 combo weights.
pub fn expand_buckets_to_combos(bucket_weights: &[f64]) -> Vec<f64> {
    assert_eq!(bucket_weights.len(), 169);
    let mut result = vec![0.0f64; NUM_COMBOS];
    for idx in 0..NUM_COMBOS {
        let hand = combo_to_canonical_hand(idx);
        result[idx] = bucket_weights[hand];
    }
    result
}

/// Helper: card index from rank (0=Two..12=Ace) and suit (0..4)
pub fn card_index_from_rank_suit(rank: usize, suit: usize) -> usize {
    rank * 4 + suit
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core range_narrower -- --nocapture`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint/range_narrower.rs
git commit -m "feat: add bucket-to-combo expansion for range narrowing"
```

---

### Task 3: CbvTable data structure

Pure domain type: stores precomputed continuation values, supports lookup and serialization.

**Files:**
- Create: `crates/core/src/blueprint/cbv.rs`
- Modify: `crates/core/src/blueprint/mod.rs` (add `pub mod cbv;`)
- Test: inline tests

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cbv_lookup() {
        // 2 boundary nodes, 3 buckets each
        let table = CbvTable {
            values: vec![
                // node 0: buckets 0,1,2
                1.0, 2.0, 3.0,
                // node 1: buckets 0,1,2
                4.0, 5.0, 6.0,
            ],
            node_offsets: vec![0, 3],
            buckets_per_node: vec![3, 3],
        };
        assert!((table.lookup(0, 0) - 1.0).abs() < 1e-9);
        assert!((table.lookup(0, 2) - 3.0).abs() < 1e-9);
        assert!((table.lookup(1, 1) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_cbv_roundtrip_serialization() {
        let table = CbvTable {
            values: vec![1.0, 2.0, 3.0],
            node_offsets: vec![0],
            buckets_per_node: vec![3],
        };
        let mut buf = Vec::new();
        table.save_to_writer(&mut buf).unwrap();
        let loaded = CbvTable::load_from_reader(&buf[..]).unwrap();
        assert_eq!(loaded.values, table.values);
        assert_eq!(loaded.node_offsets, table.node_offsets);
        assert_eq!(loaded.buckets_per_node, table.buckets_per_node);
    }

    #[test]
    fn test_cbv_num_boundary_nodes() {
        let table = CbvTable {
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            node_offsets: vec![0, 3],
            buckets_per_node: vec![3, 2],
        };
        assert_eq!(table.num_boundary_nodes(), 2);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core cbv:: -- --nocapture`
Expected: FAIL.

**Step 3: Implement CbvTable**

```rust
// crates/core/src/blueprint/cbv.rs

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CbvTable {
    pub values: Vec<f32>,
    pub node_offsets: Vec<usize>,
    pub buckets_per_node: Vec<u16>,
}

impl CbvTable {
    pub fn lookup(&self, boundary_node: usize, bucket: usize) -> f32 {
        let offset = self.node_offsets[boundary_node];
        self.values[offset + bucket]
    }

    pub fn num_boundary_nodes(&self) -> usize {
        self.node_offsets.len()
    }

    pub fn save_to_writer<W: Write>(&self, writer: &mut W) -> Result<(), bincode::Error> {
        bincode::serialize_into(writer, self)
    }

    pub fn load_from_reader<R: Read>(reader: R) -> Result<Self, bincode::Error> {
        bincode::deserialize_from(reader)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        Ok(bincode::deserialize_from(reader)?)
    }
}
```

**Step 4: Register module, run tests**

Add `pub mod cbv;` to `crates/core/src/blueprint/mod.rs`.

Run: `cargo test -p poker-solver-core cbv:: -- --nocapture`
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint/cbv.rs crates/core/src/blueprint/mod.rs
git commit -m "feat: add CbvTable for precomputed continuation values"
```

---

### Task 4: CBV precomputation during training

Hook into `save_snapshot()` (trainer.rs:903-930) to compute and save CBVs.

**Files:**
- Create: `crates/core/src/blueprint/cbv_compute.rs` (computation logic, in core so it's testable)
- Modify: `crates/core/src/blueprint/mod.rs` (add module)
- Modify: `crates/core/src/blueprint_v2/trainer.rs:903-930` (call CBV computation after saving strategy)
- Test: inline tests in `cbv_compute.rs`

**Step 1: Write failing test for CBV computation**

```rust
// crates/core/src/blueprint/cbv_compute.rs
#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal tree: root decision (2 actions: check, bet) → terminal nodes.
    /// No depth boundaries in a 1-street tree, so we test the forward-pass logic
    /// by building a 2-street tree where street 2 is the boundary.
    #[test]
    fn test_compute_cbvs_simple_tree() {
        // Build a tiny game tree with known structure:
        // Decision(check, bet) → check leads to DepthBoundary, bet leads to Terminal(fold)
        // At the boundary, CBV should equal the blueprint's continuation value
        // For a terminal fold: loser loses their contribution to pot
        //
        // This test verifies the forward pass correctly computes values
        // at depth boundary nodes from the blueprint strategy.
        let cbv = compute_cbvs_for_test_tree();
        assert!(cbv.num_boundary_nodes() > 0);
        // Verify boundary values are non-zero (not the old stub)
        for i in 0..cbv.num_boundary_nodes() {
            for b in 0..cbv.buckets_per_node[i] as usize {
                let val = cbv.lookup(i, b);
                // Values should be finite and reasonable
                assert!(val.is_finite(), "CBV at node {} bucket {} is not finite", i, b);
            }
        }
    }
}
```

Note: The exact test will depend on the tree structure available from the BlueprintV2 game tree. The implementer should study `crates/core/src/blueprint_v2/game_tree.rs` to understand the node structure, then build a minimal tree with at least one `DepthBoundary` node. The test tree should have:
- 2-4 buckets
- 2 bet sizes
- 1 decision node leading to both a terminal and a boundary
- Hand-computable expected values

**Step 2: Implement CBV computation**

The computation is a backward pass (dynamic programming) through the abstract game tree:

```rust
// crates/core/src/blueprint/cbv_compute.rs

use super::cbv::CbvTable;

/// Compute CBVs for all boundary nodes in the blueprint game tree.
///
/// For each boundary node, for each bucket, compute the expected value
/// of continuing play under the blueprint strategy to showdown.
///
/// Algorithm: backward induction through the abstract tree.
/// - Terminal nodes: payoff from pot and fold/showdown rules
/// - Decision nodes: sum over actions weighted by blueprint action probs
/// - Chance nodes: average over possible next cards
/// - Boundary nodes: these ARE the values we're computing
pub fn compute_cbvs(
    strategy: &BlueprintV2Strategy,
    game_tree: &GameTree,   // the abstract game tree from training
    bucket_counts: &[u16; 4],
) -> CbvTable {
    // 1. Identify all boundary nodes (nodes at street transitions)
    // 2. For each boundary node, for each bucket:
    //    - Walk the subtree from that node forward in the abstract tree
    //    - Compute EV using blueprint action probs and chance probs
    //    - Store the result
    //
    // Implementation detail: the BlueprintV2 tree stores decision nodes
    // with node indices. The boundary nodes are at street transitions
    // where depth_limit would apply in the subgame tree.
    todo!("Implementer: study game_tree.rs node structure, then implement backward induction")
}
```

The implementer should:
1. Read `crates/core/src/blueprint_v2/game_tree.rs` to understand the tree structure
2. Identify which nodes are at street boundaries (flop→turn, turn→river transitions)
3. Implement backward induction from terminals back to boundaries
4. Test with a small hand-crafted tree

**Step 3: Hook into snapshot saving**

Modify `crates/core/src/blueprint_v2/trainer.rs` around line 920 (after strategy save):

```rust
// After saving strategy.bin, compute and save CBVs
let cbv_table = cbv_compute::compute_cbvs(&strategy, &self.game_tree, &self.config.bucket_counts);
let cbv_path = snapshot_dir.join("cbv.bin");
cbv_table.save(&cbv_path)?;
log::info!("Saved CBV table: {} boundary nodes", cbv_table.num_boundary_nodes());
```

**Step 4: Run tests, commit**

Run: `cargo test -p poker-solver-core cbv_compute -- --nocapture`
Expected: PASS.

```bash
git add crates/core/src/blueprint/cbv_compute.rs crates/core/src/blueprint/mod.rs crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat: precompute CBVs during blueprint training snapshots"
```

---

### Task 5: Wire CBVs into SubgameCfrSolver

Replace the leaf_values stub with CbvTable lookups at DepthBoundary nodes.

**Files:**
- Modify: `crates/core/src/blueprint/subgame_cfr.rs:60-77` (constructor), `L165-167` (boundary handling)
- Test: existing tests + new test with non-zero leaf values

**Step 1: Write failing test**

```rust
// Add to subgame_cfr.rs tests
#[test]
fn test_depth_boundary_uses_leaf_values() {
    // Build a tiny subgame tree with 1 decision → 2 children:
    //   child 0: Terminal (fold)
    //   child 1: DepthBoundary
    // Set leaf_values for the boundary such that continuing is +EV
    // Verify the solver's strategy prefers the action leading to the boundary
    // over folding (because the boundary value is positive for hero)

    let tree = build_test_tree_with_boundary();
    let hands = SubgameHands { combos: vec![/* 2-3 test hands */] };
    let leaf_values = vec![10.0; hands.combos.len()]; // +10 for continuing
    let mut solver = SubgameCfrSolver::new(tree, hands, equity_matrix, opponent_reach, leaf_values);
    solver.train(100);
    let strategy = solver.strategy();
    // Hero should prefer the action leading to boundary (positive value)
    // over folding (loses current pot contribution)
}
```

**Step 2: Verify test fails meaningfully**

Run: `cargo test -p poker-solver-core test_depth_boundary -- --nocapture`
Expected: Should compile but may show unexpected strategy (if leaf_values were being ignored).

**Step 3: Modify SubgameCfrSolver to accept CbvTable**

At `subgame_cfr.rs:60-77`, the solver already accepts `leaf_values: Vec<f64>`. The current code at L165-167 correctly returns `leaf_values[combo_idx]`. The change here is ensuring the caller correctly populates `leaf_values` from the `CbvTable` — this is wiring, not solver changes.

Add a convenience constructor:

```rust
impl SubgameCfrSolver {
    /// Create solver with leaf values populated from a CbvTable.
    /// Maps each combo to its bucket, looks up the CBV.
    pub fn with_cbv_table(
        tree: SubgameTree,
        hands: SubgameHands,
        equity_matrix: Vec<Vec<f64>>,
        opponent_reach: Vec<f64>,
        cbv_table: &CbvTable,
        boundary_node_indices: &[usize],
        combo_to_bucket: &dyn Fn(usize) -> usize,
    ) -> Self {
        // For each boundary node in the tree, for each combo,
        // look up cbv_table.lookup(boundary_node, bucket)
        let leaf_values: Vec<f64> = hands.combos.iter().enumerate()
            .map(|(combo_idx, _)| {
                let bucket = combo_to_bucket(combo_idx);
                // Use the first boundary node for now — will be extended
                // when multiple boundaries exist
                cbv_table.lookup(0, bucket) as f64
            })
            .collect();
        Self::new(tree, hands, equity_matrix, opponent_reach, leaf_values)
    }
}
```

**Step 4: Run tests, commit**

Run: `cargo test -p poker-solver-core subgame_cfr -- --nocapture`
Expected: All tests PASS.

```bash
git add crates/core/src/blueprint/subgame_cfr.rs
git commit -m "feat: wire CbvTable into SubgameCfrSolver for depth-limited solving"
```

---

### Task 6: Full-depth solve bridge (range-solver integration)

Create a wrapper that translates game state into range-solver inputs and extracts the strategy.

**Files:**
- Create: `crates/core/src/blueprint/full_depth_solver.rs`
- Modify: `crates/core/src/blueprint/mod.rs`
- Test: inline tests

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_depth_solve_river_spot() {
        // River board: As Kh Qd Jc 2s
        // Hero range: just AA (weight 1.0), everything else 0.0
        // Villain range: just 72o (weight 1.0), everything else 0.0
        // Pot: 100, stacks: 100
        // 1 bet size: 1.0x pot
        //
        // AA should bet or check (dominates), never fold.
        let board = [/* As, Kh, Qd, Jc, 2s */];
        let hero_combo = /* AsAh combo index */;
        let mut hero_range = vec![0.0f64; 1326];
        hero_range[hero_combo] = 1.0;
        let mut villain_range = vec![0.0f64; 1326];
        villain_range[/* 7s2h combo */] = 1.0;

        let config = FullDepthConfig {
            board: board.to_vec(),
            pot: 100,
            effective_stack: 100,
            bet_sizes: vec![1.0],
            iterations: 100,
        };

        let result = solve_full_depth(&config, &hero_range, &villain_range)?;
        let strategy = result.strategy_for_combo(hero_combo);
        // AA with nuts should never fold
        assert!(strategy.fold_prob < 0.01);
    }
}
```

**Step 2: Implement the bridge**

```rust
// crates/core/src/blueprint/full_depth_solver.rs

use range_solver::{PostFlopGame, CardConfig, TreeConfig, BoardState, Range, Action, BetSizeOptions};

pub struct FullDepthConfig {
    pub board: Vec<rs_poker::core::Card>,
    pub pot: u32,
    pub effective_stack: u32,
    pub bet_sizes: Vec<f64>,
    pub iterations: u32,
    pub target_exploitability: f64,
}

pub struct SolveResult {
    game: PostFlopGame,
}

impl SolveResult {
    /// Get action probabilities for a specific combo.
    /// Returns (action_label, probability) pairs.
    pub fn strategy_for_combo(&self, combo_idx: usize) -> Vec<(Action, f64)> {
        let strategy = self.game.strategy();
        let num_actions = self.game.available_actions().len();
        let mut result = Vec::with_capacity(num_actions);
        for (i, action) in self.game.available_actions().iter().enumerate() {
            let prob = strategy[i * self.game.num_private_hands() + combo_idx] as f64;
            result.push((action.clone(), prob));
        }
        result
    }
}

/// Convert 1326-weight vector to range-solver Range.
fn weights_to_range(weights: &[f64]) -> Range {
    let mut range = Range::new();
    for (idx, &w) in weights.iter().enumerate() {
        let (c1, c2) = super::range_narrower::combo_cards(idx);
        range.set_weight_by_cards(c1, c2, w as f32);
    }
    range
}

/// Solve a postflop spot to full depth using the range-solver crate.
pub fn solve_full_depth(
    config: &FullDepthConfig,
    hero_range: &[f64],
    villain_range: &[f64],
) -> Result<SolveResult, Box<dyn std::error::Error>> {
    let board_state = match config.board.len() {
        3 => BoardState::Flop,
        4 => BoardState::Turn,
        5 => BoardState::River,
        _ => return Err("Invalid board length".into()),
    };

    let bet_sizes = BetSizeOptions::new()
        .with_pot_fractions(&config.bet_sizes);

    let tree_config = TreeConfig {
        initial_state: board_state,
        starting_pot: config.pot,
        effective_stack: config.effective_stack,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        ..Default::default()
    };

    let card_config = CardConfig {
        range: [weights_to_range(hero_range), weights_to_range(villain_range)],
        flop: config.board[..3].try_into()?,
        turn: if config.board.len() > 3 { Some(config.board[3]) } else { None },
        river: if config.board.len() > 4 { Some(config.board[4]) } else { None },
    };

    let mut game = PostFlopGame::new(&tree_config, &card_config)?;
    range_solver::solve(&mut game, config.iterations, config.target_exploitability, false);
    game.finalize();

    Ok(SolveResult { game })
}
```

Note: The implementer must check the exact range-solver API (structs, method signatures, card types) in `crates/range-solver/src/`. The above is a sketch — adapt to the actual API. Key files:
- `crates/range-solver/src/lib.rs` — public exports
- `crates/range-solver/src/game/mod.rs` — PostFlopGame
- `crates/range-solver/src/range.rs` — Range type

**Step 3: Run tests, commit**

Run: `cargo test -p poker-solver-core full_depth_solver -- --nocapture`
Expected: PASS.

```bash
git add crates/core/src/blueprint/full_depth_solver.rs crates/core/src/blueprint/mod.rs
git commit -m "feat: add full-depth solve bridge to range-solver crate"
```

---

### Task 7: Hybrid solver dispatch

Coordination layer that decides which solver to use based on street and combo count.

**Files:**
- Create: `crates/core/src/blueprint/solver_dispatch.rs`
- Modify: `crates/core/src/blueprint/mod.rs`
- Test: inline tests

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_river_always_full() {
        let config = SolverConfig {
            flop_combo_threshold: 200,
            turn_combo_threshold: 300,
            depth_limited_iterations: 200,
            full_solve_iterations: 1000,
            target_exploitability: 0.005,
        };
        // River with 1000 combos should still dispatch to full solve
        assert_eq!(dispatch_decision(&config, Street::River, 1000), SolverChoice::FullDepth);
    }

    #[test]
    fn test_dispatch_flop_wide_range_depth_limited() {
        let config = SolverConfig {
            flop_combo_threshold: 200,
            turn_combo_threshold: 300,
            depth_limited_iterations: 200,
            full_solve_iterations: 1000,
            target_exploitability: 0.005,
        };
        assert_eq!(dispatch_decision(&config, Street::Flop, 500), SolverChoice::DepthLimited);
    }

    #[test]
    fn test_dispatch_flop_narrow_range_full() {
        let config = SolverConfig {
            flop_combo_threshold: 200,
            turn_combo_threshold: 300,
            depth_limited_iterations: 200,
            full_solve_iterations: 1000,
            target_exploitability: 0.005,
        };
        assert_eq!(dispatch_decision(&config, Street::Flop, 100), SolverChoice::FullDepth);
    }

    #[test]
    fn test_dispatch_turn_boundary() {
        let config = SolverConfig {
            flop_combo_threshold: 200,
            turn_combo_threshold: 300,
            depth_limited_iterations: 200,
            full_solve_iterations: 1000,
            target_exploitability: 0.005,
        };
        assert_eq!(dispatch_decision(&config, Street::Turn, 300), SolverChoice::FullDepth);
        assert_eq!(dispatch_decision(&config, Street::Turn, 301), SolverChoice::DepthLimited);
    }
}
```

**Step 2: Implement dispatch**

```rust
// crates/core/src/blueprint/solver_dispatch.rs

#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub flop_combo_threshold: usize,
    pub turn_combo_threshold: usize,
    pub depth_limited_iterations: u32,
    pub full_solve_iterations: u32,
    pub target_exploitability: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverChoice {
    FullDepth,
    DepthLimited,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Street {
    Flop,
    Turn,
    River,
}

pub fn dispatch_decision(config: &SolverConfig, street: Street, live_combos: usize) -> SolverChoice {
    match street {
        Street::River => SolverChoice::FullDepth,
        Street::Turn => {
            if live_combos <= config.turn_combo_threshold {
                SolverChoice::FullDepth
            } else {
                SolverChoice::DepthLimited
            }
        }
        Street::Flop => {
            if live_combos <= config.flop_combo_threshold {
                SolverChoice::FullDepth
            } else {
                SolverChoice::DepthLimited
            }
        }
    }
}
```

**Step 3: Run tests, commit**

Run: `cargo test -p poker-solver-core solver_dispatch -- --nocapture`
Expected: All 4 tests PASS.

```bash
git add crates/core/src/blueprint/solver_dispatch.rs crates/core/src/blueprint/mod.rs
git commit -m "feat: add hybrid solver dispatch logic (per-street combo thresholds)"
```

---

### Task 8: RealTimeSolvingAgent

The main agent struct that implements `Agent` and orchestrates range narrowing + solver dispatch.

**Files:**
- Modify: `crates/core/src/simulation.rs` (add agent + generator, near `BlueprintAgent` at L113)
- Test: inline test or in `crates/core/tests/`

**Step 1: Write failing test**

```rust
#[test]
fn test_real_time_agent_returns_valid_action() {
    // Load a tiny/mock blueprint
    // Create RealTimeSolvingAgent
    // Simulate a single postflop decision via act()
    // Verify it returns Fold, Call, Bet, or AllIn (not panic)
    //
    // This test needs a minimal blueprint bundle. The implementer should
    // either create a mock BlueprintV2Strategy or use a tiny trained one.
    // Key: the test must complete in < 10s with a trivial tree.
}
```

**Step 2: Implement RealTimeSolvingAgent**

Add to `crates/core/src/simulation.rs` after `BlueprintAgent` (around L250):

```rust
pub struct RealTimeSolvingAgent {
    /// Loaded blueprint for preflop play and range narrowing
    blueprint: Arc<BlueprintV2Strategy>,
    /// Precomputed continuation values
    cbv_table: Arc<CbvTable>,
    /// Solver configuration (thresholds, iterations)
    solver_config: SolverConfig,
    /// Opponent range tracker (1326 combo weights)
    range_narrower: RangeNarrower,
    /// Hero's hole cards (set on generation)
    hero_cards: [Card; 2],
    /// Actions seen so far (to diff against ACTION_LOG)
    last_action_count: usize,
    /// Bet sizes from blueprint config
    bet_sizes: Vec<f64>,
    /// Game config (stack depth, etc.)
    stack_depth: u32,
}

impl Agent for RealTimeSolvingAgent {
    fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
        let round = game_state.round();

        // Preflop: use blueprint lookup (same as BlueprintAgent)
        if round == Round::Preflop {
            return self.blueprint_action(game_state);
        }

        // Postflop: update range from new opponent actions
        self.update_opponent_range();

        // Apply card removal for current board
        let board = self.extract_board(game_state);
        self.range_narrower.apply_card_removal(&board, &self.hero_cards);

        // Dispatch to appropriate solver
        let street = match round {
            Round::Flop => Street::Flop,
            Round::Turn => Street::Turn,
            Round::River => Street::River,
            _ => unreachable!(),
        };
        let live_combos = self.range_narrower.live_combo_count();
        let choice = dispatch_decision(&self.solver_config, street, live_combos);

        match choice {
            SolverChoice::FullDepth => self.solve_full_depth(game_state, &board),
            SolverChoice::DepthLimited => self.solve_depth_limited(game_state, &board),
        }
    }
}
```

The `blueprint_action`, `update_opponent_range`, `solve_full_depth`, and `solve_depth_limited` methods are private helpers on the impl block. The implementer should model these after `BlueprintAgent::act()` (simulation.rs:L128-240) for the action log reading pattern.

**Step 3: Implement RealTimeSolvingAgentGenerator**

```rust
pub struct RealTimeSolvingAgentGenerator {
    blueprint: Arc<BlueprintV2Strategy>,
    cbv_table: Arc<CbvTable>,
    solver_config: SolverConfig,
    bet_sizes: Vec<f64>,
    stack_depth: u32,
}

impl AgentGenerator for RealTimeSolvingAgentGenerator {
    fn generate(&mut self, _hand: &Hand) -> Box<dyn Agent> {
        let hero_cards = extract_hole_cards(hand);
        Box::new(RealTimeSolvingAgent {
            blueprint: Arc::clone(&self.blueprint),
            cbv_table: Arc::clone(&self.cbv_table),
            solver_config: self.solver_config.clone(),
            range_narrower: RangeNarrower::new(),
            hero_cards,
            last_action_count: 0,
            bet_sizes: self.bet_sizes.clone(),
            stack_depth: self.stack_depth,
        })
    }
}
```

**Step 4: Run tests, commit**

Run: `cargo test -p poker-solver-core real_time -- --nocapture`
Expected: PASS.

```bash
git add crates/core/src/simulation.rs
git commit -m "feat: add RealTimeSolvingAgent with hybrid postflop solving"
```

---

### Task 9: CLI integration

Wire the new agent type into the simulation CLI and the Tauri exploration app.

**Files:**
- Modify: `crates/tauri-app/src/simulation.rs` (~L163, `build_agent_generator()`)
- Modify: `crates/trainer/src/main.rs` (if simulate subcommand lives here)
- Test: manual — run a sim with the new agent type

**Step 1: Add agent type to build_agent_generator**

In `crates/tauri-app/src/simulation.rs`, extend the pattern matching in `build_agent_generator()`:

```rust
// Existing patterns:
// "builtin:calling" → CallingAgentGenerator
// "builtin:folding" → FoldingAgentGenerator
// *.toml → RuleBasedAgentGenerator
// bundle paths → BlueprintAgentGenerator

// Add new pattern:
// "solver:<bundle_path>" → RealTimeSolvingAgentGenerator
if source.starts_with("solver:") {
    let bundle_path = &source[7..];
    let bundle = load_blueprint_v2(bundle_path)?;
    let cbv_path = Path::new(bundle_path).join("cbv.bin");
    let cbv_table = CbvTable::load(&cbv_path)?;
    let solver_config = SolverConfig {
        flop_combo_threshold: 200,  // defaults, make configurable later
        turn_combo_threshold: 300,
        depth_limited_iterations: 200,
        full_solve_iterations: 1000,
        target_exploitability: 0.005,
    };
    return Ok(Box::new(RealTimeSolvingAgentGenerator::new(
        bundle, cbv_table, solver_config, bet_sizes, stack_depth,
    )));
}
```

**Step 2: Test manually**

```bash
# Train a small blueprint first (if not already available)
# Then run simulation:
cargo run -p poker-solver-trainer --release -- simulate \
  --p1 "solver:/path/to/blueprint/snapshot" \
  --p2 "/path/to/blueprint/snapshot" \
  --hands 1000
```

Expected: Simulation runs, prints mbb/h. Solve times logged per decision.

**Step 3: Commit**

```bash
git add crates/tauri-app/src/simulation.rs
git commit -m "feat: wire RealTimeSolvingAgent into simulation CLI"
```

---

### Task 10: Integration test — full pipeline

End-to-end test with a tiny blueprint to verify all components wire together.

**Files:**
- Create: `crates/core/tests/real_time_solving.rs`
- Test: integration test

**Step 1: Write the test**

```rust
// crates/core/tests/real_time_solving.rs

/// Integration test: run a short simulation with RealTimeSolvingAgent vs BlueprintAgent.
/// Uses a minimal blueprint with small bucket counts and trivial bet tree.
/// Must complete in < 10 seconds.
#[test]
fn test_real_time_vs_blueprint_simulation() {
    // 1. Create or load a tiny blueprint (2 buckets, 2 bet sizes, 1 street)
    // 2. Create a CbvTable with dummy values
    // 3. Set up RealTimeSolvingAgentGenerator + BlueprintAgentGenerator
    // 4. Run simulation for 100 hands
    // 5. Assert: simulation completes, result has finite mbb/h

    let result = run_test_simulation();
    assert!(result.mbbh.is_finite());
    assert!(result.hands_played >= 100);
}
```

The implementer should create a test helper that builds a minimal in-memory blueprint (not loaded from disk) to keep the test self-contained and fast.

**Step 2: Run test, commit**

Run: `cargo test -p poker-solver-core --test real_time_solving -- --nocapture`
Expected: PASS in < 10 seconds.

```bash
git add crates/core/tests/real_time_solving.rs
git commit -m "test: add integration test for real-time solving pipeline"
```
