# Blueprint CBV Evaluator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace raw showdown equity at depth boundaries with blueprint counterfactual boundary values (CBVs) in the TUI subgame solver.

**Architecture:** Fix `SubgameCfrSolver::with_cbv_table()` to support per-boundary CBV normalization, add a parallel tree walk that maps subgame `DepthBoundary` nodes to abstract-tree `Chance` node ordinals, load CBV tables + bucket files in the bundle loader, and wire it all through the TUI solve dispatch.

**Tech Stack:** Rust, `SubgameCfrSolver`, `CbvTable`, `AllBuckets`, `GameTree`

**Design doc:** `docs/plans/2026-03-20-blueprint-cbv-evaluator-design.md`

---

### Task 1: Per-boundary leaf values in `SubgameCfrSolver`

Change `leaf_values` from `Vec<f64>` (one value per combo, shared across all boundaries) to `Vec<Vec<f64>>` (one vec per boundary × per combo). Add `node_to_boundary` mapping. Update traversal to index by boundary ordinal.

**Files:**
- Modify: `crates/core/src/blueprint_v2/subgame_cfr.rs:156-178` (struct fields)
- Modify: `crates/core/src/blueprint_v2/subgame_cfr.rs:180-235` (constructors)
- Modify: `crates/core/src/blueprint_v2/subgame_cfr.rs:343-355` (SubgameCfrCtx)
- Modify: `crates/core/src/blueprint_v2/subgame_cfr.rs:392-410` (cfr_traverse DepthBoundary branch)
- Test: `crates/core/src/blueprint_v2/subgame_cfr.rs` (existing tests in same file)

**Step 1: Write the failing test**

Add a test in the `mod tests` block of `subgame_cfr.rs` that verifies per-boundary leaf values produce different payoffs at boundaries with different pots.

```rust
#[timed_test]
fn per_boundary_leaf_values_differ_by_pot() {
    // Build a depth-limited flop tree with bet sizes that create
    // multiple DepthBoundary nodes at different pot sizes.
    let board = vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Heart),
        Card::new(Value::Seven, Suit::Diamond),
    ];
    let tree = GameTree::build_subgame(
        Street::Flop,
        100.0,
        [50.0, 50.0],
        250.0,
        &[vec![1.0]],  // pot-size bet
        Some(1),
    );

    // Count DepthBoundary nodes — should be > 1 (check-check and bet-call paths).
    let boundary_count = tree.nodes.iter().filter(|n| matches!(n,
        GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. }
    )).count();
    assert!(boundary_count >= 2, "need multiple boundaries, got {boundary_count}");

    // Collect boundary pots.
    let boundary_pots: Vec<f64> = tree.nodes.iter().filter_map(|n| match n {
        GameNode::Terminal { kind: TerminalKind::DepthBoundary, pot, .. } => Some(*pot),
        _ => None,
    }).collect();

    // Verify boundaries have different pots.
    assert!(
        boundary_pots.windows(2).any(|w| (w[0] - w[1]).abs() > 1.0),
        "boundaries should have different pots: {boundary_pots:?}"
    );

    let hands = small_hands(&board, 10);
    let n = hands.combos.len();

    // Create per-boundary leaf values: all combos get equity 0.7 at every boundary.
    let leaf_values: Vec<Vec<f64>> = vec![vec![0.7; n]; boundary_count];

    let solver = SubgameCfrSolver::new(
        tree, hands, &board, vec![1.0; n], leaf_values,
    );

    // Train briefly and verify it doesn't panic.
    // (Actual value correctness is tested by the CBV normalization test.)
    solver.train(5);
    assert_eq!(solver.iteration, 5);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core per_boundary_leaf_values_differ_by_pot`
Expected: FAIL — `SubgameCfrSolver::new` doesn't accept `Vec<Vec<f64>>` yet.

**Step 3: Implement per-boundary leaf values**

In `SubgameCfrSolver` struct (line 156-178), change:

```rust
// OLD
/// Per-combo leaf equity (0.0 to 1.0) for `DepthBoundary` nodes.
/// Converted to chip values via `(2 * equity - 1) * half_pot`.
leaf_values: Vec<f64>,

// NEW
/// Per-boundary, per-combo leaf equity (0.0 to 1.0) for `DepthBoundary` nodes.
/// Outer index = boundary ordinal, inner index = combo index.
/// Converted to chip values via `(2 * equity - 1) * half_pot`.
leaf_values: Vec<Vec<f64>>,
/// Maps tree node index → boundary ordinal (`usize::MAX` if not a boundary).
node_to_boundary: Vec<usize>,
```

Update `SubgameCfrSolver::new()` (line 183-207):

```rust
pub fn new(
    tree: GameTree,
    hands: SubgameHands,
    board: &[Card],
    opponent_reach: Vec<f64>,
    leaf_values: Vec<Vec<f64>>,
) -> Self {
    let equity_matrix = compute_equity_matrix(&hands.combos, board);
    let opp_reach_totals = precompute_opp_reach(&hands.combos, &opponent_reach);
    let layout = SubgameLayout::build(&tree, hands.combos.len());
    let buf_size = layout.total_size;

    // Build node_to_boundary mapping.
    let mut node_to_boundary = vec![usize::MAX; tree.nodes.len()];
    let mut boundary_ord = 0;
    for (idx, node) in tree.nodes.iter().enumerate() {
        if matches!(node, GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. }) {
            node_to_boundary[idx] = boundary_ord;
            boundary_ord += 1;
        }
    }
    assert_eq!(
        boundary_ord, leaf_values.len(),
        "leaf_values length ({}) must match DepthBoundary count ({boundary_ord})",
        leaf_values.len(),
    );

    Self {
        tree,
        hands,
        equity_matrix,
        opponent_reach,
        leaf_values,
        node_to_boundary,
        regret_sum: vec![0.0; buf_size],
        strategy_sum: vec![0.0; buf_size],
        layout,
        opp_reach_totals,
        dcfr: DcfrParams::default(),
        iteration: 0,
    }
}
```

Update `SubgameCfrCtx` (line 343-355) — change `leaf_values` field type:

```rust
struct SubgameCfrCtx<'a> {
    tree: &'a GameTree,
    hands: &'a SubgameHands,
    equity_matrix: &'a [Vec<f64>],
    opponent_reach: &'a [f64],
    leaf_values: &'a [Vec<f64>],          // was &'a [f64]
    node_to_boundary: &'a [usize],        // NEW
    opp_reach_totals: &'a [f64],
    layout: &'a SubgameLayout,
    snapshot: &'a [f64],
    traverser: u8,
}
```

Update `SubgameCfrCtx` construction in `train()` (line 253-263) — add the new field:

```rust
let ctx = SubgameCfrCtx {
    tree: &self.tree,
    hands: &self.hands,
    equity_matrix: &self.equity_matrix,
    opponent_reach: &self.opponent_reach,
    leaf_values: &self.leaf_values,
    node_to_boundary: &self.node_to_boundary,
    opp_reach_totals: &self.opp_reach_totals,
    layout: &self.layout,
    snapshot: &snapshot,
    traverser,
};
```

Update the `DepthBoundary` branch in `cfr_traverse` (line 404-408):

```rust
TerminalKind::DepthBoundary => {
    let boundary_ord = self.node_to_boundary[node_idx];
    assert_ne!(boundary_ord, usize::MAX, "DepthBoundary node {node_idx} has no boundary ordinal");
    let equity = self.leaf_values[boundary_ord][hero_combo];
    (2.0 * equity - 1.0) * half_pot
}
```

**Step 4: Fix all existing callers**

Every existing call to `SubgameCfrSolver::new()` passes `leaf_values: Vec<f64>` (flat). These must be wrapped in a single-element outer vec if only one boundary exists, or converted to per-boundary format.

In `build_subgame_solver()` (`crates/tauri-app/src/postflop.rs:366-385`), change:

```rust
// OLD
let leaf_values = compute_combo_equities(&hands, board_cards, &opponent_reach);
let solver = SubgameCfrSolver::new(tree.clone(), hands.clone(), board_cards, opponent_reach, leaf_values);

// NEW
let equities = compute_combo_equities(&hands, board_cards, &opponent_reach);
let boundary_count = tree.nodes.iter().filter(|n| matches!(n,
    GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. }
)).count();
let leaf_values: Vec<Vec<f64>> = vec![equities; boundary_count];
let solver = SubgameCfrSolver::new(tree.clone(), hands.clone(), board_cards, opponent_reach, leaf_values);
```

Similarly fix all test call sites in `subgame_cfr.rs::tests` — each test that calls `SubgameCfrSolver::new()` with a flat `Vec<f64>` needs to wrap it. For river trees (no boundaries), use `vec![]` (empty). For depth-limited trees, count boundaries and replicate.

Helper for tests:

```rust
/// Wrap flat leaf values for all boundaries (test helper).
fn flat_leaf_values(tree: &GameTree, values: Vec<f64>) -> Vec<Vec<f64>> {
    let count = tree.nodes.iter().filter(|n| matches!(n,
        GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. }
    )).count();
    if count == 0 { vec![] } else { vec![values; count] }
}
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core subgame_cfr`
Expected: ALL PASS (existing + new test).

**Step 6: Also run cfv_subgame_solver tests** since it imports from subgame_cfr.

Run: `cargo test -p poker-solver-core cfv_subgame_solver`
Expected: ALL PASS.

**Step 7: Commit**

```bash
git add crates/core/src/blueprint_v2/subgame_cfr.rs crates/tauri-app/src/postflop.rs
git commit -m "refactor: per-boundary leaf values in SubgameCfrSolver"
```

---

### Task 2: Replace `with_cbv_table()` with `with_cbv_tables()`

Remove the old single-boundary `with_cbv_table()` method. Replace with `with_cbv_tables()` that accepts per-boundary mapping and normalizes CBV chip values to equity.

**Files:**
- Modify: `crates/core/src/blueprint_v2/subgame_cfr.rs:209-235` (replace method)
- Test: `crates/core/src/blueprint_v2/subgame_cfr.rs` (replace `with_cbv_table_populates_leaf_values` test)

**Step 1: Write the failing test**

Replace the existing `with_cbv_table_populates_leaf_values` test with one that tests the new per-boundary normalization:

```rust
#[timed_test]
fn with_cbv_tables_normalizes_per_boundary() {
    use super::super::cbv::CbvTable;

    let board = vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Heart),
        Card::new(Value::Seven, Suit::Diamond),
    ];
    let tree = GameTree::build_subgame(
        Street::Flop,
        100.0,
        [50.0, 50.0],
        250.0,
        &[vec![1.0]],
        Some(1),
    );

    let hands = small_hands(&board, 6);
    let n = hands.combos.len();

    // Count boundaries and their pots.
    let boundary_pots: Vec<f64> = tree.nodes.iter().filter_map(|n| match n {
        GameNode::Terminal { kind: TerminalKind::DepthBoundary, pot, .. } => Some(*pot),
        _ => None,
    }).collect();
    let num_boundaries = boundary_pots.len();
    assert!(num_boundaries >= 2, "need multiple boundaries");

    // CBV table with `num_boundaries` chance nodes, 4 buckets each.
    // Use known chip values so we can verify the normalization.
    let cbv_chip_value = 15.0_f32; // 15 chips
    let values: Vec<f32> = vec![cbv_chip_value; num_boundaries * 4];
    let node_offsets: Vec<usize> = (0..num_boundaries).map(|i| i * 4).collect();
    let buckets_per_node: Vec<u16> = vec![4; num_boundaries];
    let cbv_table = CbvTable { values, node_offsets, buckets_per_node };

    // boundary_mapping: identity (boundary 0 → chance 0, boundary 1 → chance 1, etc.)
    let boundary_mapping: Vec<usize> = (0..num_boundaries).collect();

    let mut solver = SubgameCfrSolver::with_cbv_tables(
        tree,
        hands,
        &board,
        vec![1.0; n],
        [&cbv_table, &cbv_table],
        &boundary_mapping,
        |combo_idx| combo_idx % 4,
    );

    // Verify normalization: equity = (cbv / half_pot + 1) / 2
    // Round-trip check: (2 * equity - 1) * half_pot should recover cbv.
    for (b, &pot) in boundary_pots.iter().enumerate() {
        let half_pot = pot / 2.0;
        let expected_equity = (f64::from(cbv_chip_value) / half_pot + 1.0) / 2.0;
        for combo_idx in 0..n {
            let actual = solver.leaf_values[b][combo_idx];
            assert!(
                (actual - expected_equity).abs() < 1e-10,
                "boundary {b} combo {combo_idx}: expected {expected_equity}, got {actual}"
            );
            // Round-trip: traversal would compute (2*eq - 1) * half_pot = cbv
            let round_trip = (2.0 * actual - 1.0) * half_pot;
            assert!(
                (round_trip - f64::from(cbv_chip_value)).abs() < 1e-6,
                "round-trip failed: {round_trip} != {cbv_chip_value}"
            );
        }
    }

    // Train to verify no panics.
    solver.train(10);
    assert_eq!(solver.iteration, 10);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core with_cbv_tables_normalizes_per_boundary`
Expected: FAIL — `with_cbv_tables` doesn't exist.

**Step 3: Implement `with_cbv_tables()`**

Replace the old `with_cbv_table()` method (lines 209-235) with:

```rust
/// Create a solver with per-boundary leaf values from [`CbvTable`]s.
///
/// For each `DepthBoundary` node, `boundary_mapping[ordinal]` gives the
/// corresponding Chance node ordinal in the CBV table. Each combo's CBV
/// is looked up via `combo_to_bucket`, then normalized from chip units
/// to equity [0, 1] using the boundary's pot.
///
/// # Panics
///
/// Panics if `boundary_mapping` length doesn't match the number of
/// `DepthBoundary` nodes, or if any CBV/bucket lookup is out of range.
#[must_use]
pub fn with_cbv_tables(
    tree: GameTree,
    hands: SubgameHands,
    board: &[Card],
    opponent_reach: Vec<f64>,
    cbv_tables: [&CbvTable; 2],
    boundary_mapping: &[usize],
    combo_to_bucket: impl Fn(usize) -> u16,
) -> Self {
    let n = hands.combos.len();

    // Collect boundary pots in ordinal order.
    let mut boundary_pots = Vec::new();
    for node in &tree.nodes {
        if let GameNode::Terminal { kind: TerminalKind::DepthBoundary, pot, .. } = node {
            boundary_pots.push(*pot);
        }
    }
    assert_eq!(
        boundary_pots.len(), boundary_mapping.len(),
        "boundary_mapping length ({}) must match DepthBoundary count ({})",
        boundary_mapping.len(), boundary_pots.len(),
    );

    // Build per-boundary, per-combo leaf values.
    // Use player 0's CBV table (both players' CBVs are symmetric in
    // the abstract tree; the traversal handles player perspective).
    let leaf_values: Vec<Vec<f64>> = boundary_mapping
        .iter()
        .zip(boundary_pots.iter())
        .map(|(&chance_ord, &pot)| {
            let half_pot = pot / 2.0;
            assert!(half_pot > 0.0, "boundary pot must be positive, got {pot}");
            (0..n)
                .map(|combo_idx| {
                    let bucket = combo_to_bucket(combo_idx) as usize;
                    let cbv = f64::from(cbv_tables[0].lookup(chance_ord, bucket));
                    // Normalize: equity = (cbv / half_pot + 1) / 2
                    (cbv / half_pot + 1.0) / 2.0
                })
                .collect()
        })
        .collect();

    Self::new(tree, hands, board, opponent_reach, leaf_values)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core subgame_cfr`
Expected: ALL PASS.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/subgame_cfr.rs
git commit -m "feat: with_cbv_tables() — per-boundary CBV normalization"
```

---

### Task 3: `build_boundary_mapping()` — parallel tree walk

Walk the subgame tree and abstract tree in lockstep, mapping each subgame `DepthBoundary` to the abstract tree's corresponding `Chance` node ordinal.

**Files:**
- Modify: `crates/core/src/blueprint_v2/subgame_cfr.rs` (add free function + tests)

**Step 1: Write the failing test**

```rust
#[timed_test]
fn build_boundary_mapping_matches_trees() {
    // Build abstract full-depth flop tree.
    let abstract_tree = GameTree::build_subgame(
        Street::Flop,
        100.0,
        [50.0, 50.0],
        250.0,
        &[vec![1.0]],
        None, // full depth
    );

    // Build subgame depth-limited flop tree with SAME bet sizes.
    let subgame_tree = GameTree::build_subgame(
        Street::Flop,
        100.0,
        [50.0, 50.0],
        250.0,
        &[vec![1.0]],
        Some(1), // depth limit = 1 street
    );

    // Count expected boundaries.
    let boundary_count = subgame_tree.nodes.iter().filter(|n| matches!(n,
        GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. }
    )).count();
    assert!(boundary_count > 0, "subgame should have boundaries");

    // Count chance nodes in abstract tree.
    let chance_count = abstract_tree.nodes.iter().filter(|n| matches!(n,
        GameNode::Chance { .. }
    )).count();

    let mapping = build_boundary_mapping(&subgame_tree, &abstract_tree);

    assert_eq!(mapping.len(), boundary_count);
    // Every mapped ordinal must be valid.
    for &ord in &mapping {
        assert!(ord < chance_count, "ordinal {ord} out of range (chance_count={chance_count})");
    }
}

#[timed_test]
#[should_panic(expected = "no matching action")]
fn build_boundary_mapping_panics_on_mismatch() {
    let tree_a = GameTree::build_subgame(
        Street::Flop, 100.0, [50.0, 50.0], 250.0,
        &[vec![1.0]],
        Some(1),
    );
    // Different bet sizes → different actions → panic.
    let tree_b = GameTree::build_subgame(
        Street::Flop, 100.0, [50.0, 50.0], 250.0,
        &[vec![0.5]],
        None,
    );
    build_boundary_mapping(&tree_a, &tree_b);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core build_boundary_mapping`
Expected: FAIL — function doesn't exist.

**Step 3: Implement `build_boundary_mapping()`**

Add as a public free function in `subgame_cfr.rs`:

```rust
/// Map each `DepthBoundary` in `subgame_tree` to the corresponding `Chance`
/// node ordinal in `abstract_tree` by walking both trees in lockstep.
///
/// Both trees must share the same action set. Returns a `Vec<usize>` of length
/// equal to the number of `DepthBoundary` nodes, where each value is the
/// ordinal position of the matching `Chance` node among all `Chance` nodes
/// in the abstract tree (matching `CbvTable` indexing).
///
/// # Panics
///
/// Panics if any subgame action has no match in the abstract tree, or if a
/// `DepthBoundary` doesn't correspond to a `Chance` node.
#[must_use]
pub fn build_boundary_mapping(
    subgame_tree: &GameTree,
    abstract_tree: &GameTree,
) -> Vec<usize> {
    // Precompute chance node ordinals in the abstract tree.
    let mut chance_ordinals = vec![usize::MAX; abstract_tree.nodes.len()];
    let mut ord = 0;
    for (idx, node) in abstract_tree.nodes.iter().enumerate() {
        if matches!(node, GameNode::Chance { .. }) {
            chance_ordinals[idx] = ord;
            ord += 1;
        }
    }

    let mut mapping = Vec::new();
    walk_trees(
        subgame_tree, abstract_tree, &chance_ordinals,
        subgame_tree.root as usize, abstract_tree.root as usize,
        &mut mapping,
    );
    mapping
}

fn walk_trees(
    sub: &GameTree,
    abs: &GameTree,
    chance_ordinals: &[usize],
    sub_idx: usize,
    abs_idx: usize,
    mapping: &mut Vec<usize>,
) {
    match (&sub.nodes[sub_idx], &abs.nodes[abs_idx]) {
        // Subgame hits DepthBoundary → abstract should be at Chance.
        (
            GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. },
            GameNode::Chance { .. },
        ) => {
            let ord = chance_ordinals[abs_idx];
            assert_ne!(ord, usize::MAX, "abstract Chance node {abs_idx} has no ordinal");
            mapping.push(ord);
        }

        // Both are terminals (Fold or Showdown) — nothing to map.
        (
            GameNode::Terminal { .. },
            GameNode::Terminal { .. },
        ) => {}

        // Both are Chance nodes — recurse into children.
        (
            GameNode::Chance { child: sub_child, .. },
            GameNode::Chance { child: abs_child, .. },
        ) => {
            walk_trees(sub, abs, chance_ordinals, *sub_child as usize, *abs_child as usize, mapping);
        }

        // Both are Decision nodes — match actions and recurse.
        (
            GameNode::Decision { actions: sub_actions, children: sub_children, .. },
            GameNode::Decision { actions: abs_actions, children: abs_children, .. },
        ) => {
            for (sub_a_idx, sub_action) in sub_actions.iter().enumerate() {
                let abs_a_idx = abs_actions.iter().position(|a| a == sub_action)
                    .unwrap_or_else(|| panic!(
                        "no matching action for {sub_action:?} in abstract tree at node {abs_idx}"
                    ));
                walk_trees(
                    sub, abs, chance_ordinals,
                    sub_children[sub_a_idx] as usize,
                    abs_children[abs_a_idx] as usize,
                    mapping,
                );
            }
        }

        (sub_node, abs_node) => {
            panic!(
                "tree structure mismatch at sub={sub_idx} abs={abs_idx}: \
                 sub={sub_node:?}, abs={abs_node:?}"
            );
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core build_boundary_mapping`
Expected: ALL PASS.

**Step 5: Export from module**

In `crates/core/src/blueprint_v2/mod.rs` line 25, add `build_boundary_mapping` to the re-export:

```rust
pub use subgame_cfr::{SubgameCfrSolver, SubgameHands, SubgameStrategy, compute_combo_equities, build_boundary_mapping};
```

**Step 6: Run full core test suite**

Run: `cargo test -p poker-solver-core`
Expected: ALL PASS.

**Step 7: Commit**

```bash
git add crates/core/src/blueprint_v2/subgame_cfr.rs crates/core/src/blueprint_v2/mod.rs
git commit -m "feat: build_boundary_mapping() — parallel tree walk for CBV integration"
```

---

### Task 4: Load CBV tables and bucket files with blueprint bundle

Add CBV table and bucket file loading to `load_blueprint_v2_core()`. Store them in `StrategySource::BlueprintV2`.

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs:75-84` (StrategySource enum)
- Modify: `crates/tauri-app/src/exploration.rs:272-349` (load_blueprint_v2_core)

**Step 1: Add CBV + bucket fields to `StrategySource::BlueprintV2`**

In `exploration.rs` line 75-84, add fields:

```rust
enum StrategySource {
    Agent(AgentConfig),
    BlueprintV2 {
        config: Box<BlueprintV2Config>,
        strategy: Box<BlueprintV2Strategy>,
        tree: Box<V2GameTree>,
        decision_map: Vec<u32>,
        cbv_tables: Option<[CbvTable; 2]>,
    },
}
```

Add the necessary import at the top of `exploration.rs`:

```rust
use poker_solver_core::blueprint_v2::cbv::CbvTable;
```

**Step 2: Load CBV files in `load_blueprint_v2_core()`**

Inside the `spawn_blocking` closure (line 277-311), after loading strategy, add CBV loading:

```rust
// Load CBV tables if present.
let cbv_dir = if dir.join("final/cbv_p0.bin").exists() {
    dir.join("final")
} else if dir.join("cbv_p0.bin").exists() {
    dir.clone()
} else {
    // Check latest snapshot.
    let mut snapshots: Vec<_> = std::fs::read_dir(&dir)
        .map_err(|e| format!("Cannot read directory: {e}"))?
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_str().is_some_and(|n| n.starts_with("snapshot_")))
        .collect();
    snapshots.sort_by_key(|e| e.file_name());
    snapshots.last()
        .map(|e| e.path())
        .unwrap_or_else(|| dir.clone())
};

let cbv_tables = if cbv_dir.join("cbv_p0.bin").exists() && cbv_dir.join("cbv_p1.bin").exists() {
    let p0 = CbvTable::load(&cbv_dir.join("cbv_p0.bin"))
        .map_err(|e| format!("Failed to load cbv_p0.bin: {e}"))?;
    let p1 = CbvTable::load(&cbv_dir.join("cbv_p1.bin"))
        .map_err(|e| format!("Failed to load cbv_p1.bin: {e}"))?;
    Some([p0, p1])
} else {
    None
};
```

Return `cbv_tables` from the closure alongside `config` and `strategy`. Update the closure return type and destructuring.

Store in StrategySource (line 340):

```rust
*state.source.write() = Some(StrategySource::BlueprintV2 {
    config: Box::new(config),
    strategy: Box::new(strategy),
    tree: Box::new(tree),
    decision_map,
    cbv_tables,
});
```

**Step 3: Verify it compiles and existing tests pass**

Run: `cargo test -p tauri-app` (or `cargo build -p tauri-app` if no unit tests)
Run: `cargo clippy -p tauri-app`
Expected: PASS / no errors.

**Step 4: Commit**

```bash
git add crates/tauri-app/src/exploration.rs
git commit -m "feat: load CBV tables alongside blueprint bundle"
```

---

### Task 5: Wire CBV-based solving into the TUI dispatch

Replace `compute_combo_equities()` in `build_subgame_solver()` with `with_cbv_tables()` when CBV data is available. Pass CBV tables and abstract tree through the solve dispatch chain.

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:322-388` (build_subgame_solver)
- Modify: `crates/tauri-app/src/postflop.rs:1085-1203` (solve_depth_limited)
- Modify: `crates/tauri-app/src/postflop.rs:931-991` (postflop_solve_street_impl)
- Modify: `crates/tauri-app/src/postflop.rs:401-418` (PostflopState)

**Step 1: Add CBV data to `PostflopState`**

In `PostflopState` struct (line 401-418), add:

```rust
/// CBV tables + abstract tree for depth-limited solving.
/// Loaded from the blueprint bundle. If present, subgame solver
/// uses CBVs at depth boundaries instead of raw equity.
pub cbv_context: RwLock<Option<CbvContext>>,
```

Add the supporting struct:

```rust
/// Blueprint data needed for CBV-based depth-limited solving.
pub struct CbvContext {
    pub cbv_tables: [CbvTable; 2],
    pub abstract_tree: GameTree,
    pub all_buckets: AllBuckets,
}
```

Add necessary imports:

```rust
use poker_solver_core::blueprint_v2::cbv::CbvTable;
use poker_solver_core::blueprint_v2::game_tree::GameTree as V2GameTree;
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::blueprint_v2::build_boundary_mapping;
```

Initialize with `None` in `Default` impl (line 420-440):

```rust
cbv_context: RwLock::new(None),
```

**Step 2: Populate `CbvContext` when loading a bundle with CBVs**

This happens in the exploration layer. When `load_blueprint_v2_core` finishes, the TUI's postflop state needs the CBV data. The simplest approach: after loading the bundle in `exploration.rs`, expose a method or command that sets up the `CbvContext` on `PostflopState`.

Alternatively, `solve_depth_limited` can read `cbv_tables` from the `ExplorationState`'s `StrategySource`. Check how `solve_depth_limited` currently accesses state — it takes `&Arc<PostflopState>` but not `ExplorationState`.

The cleanest approach: pass the CBV context into `postflop_solve_street_impl` → `solve_depth_limited` → `build_subgame_solver`. Add an optional parameter to `postflop_solve_street_core` that accepts CBV context.

Update `build_subgame_solver` signature (line 322-330):

```rust
pub fn build_subgame_solver(
    board_cards: &[RsPokerCard],
    bet_sizes: &[f32],
    pot: u32,
    stacks: [u32; 2],
    oop_weights: &[f32],
    ip_weights: &[f32],
    player: usize,
    cbv_context: Option<&CbvContext>,
) -> Result<(SubgameCfrSolver, SubgameHands, Vec<ActionInfo>, GameTree, f64, f64), String>
```

Inside `build_subgame_solver`, after tree construction (line 342-367), replace the leaf value computation:

```rust
let leaf_values = if let Some(ctx) = cbv_context {
    let street = match board_cards.len() {
        3 => V2Street::Flop,
        4 => V2Street::Turn,
        5 => V2Street::River,
        _ => unreachable!(),
    };
    let boundary_mapping = build_boundary_mapping(&tree, &ctx.abstract_tree);
    // Convert rs_poker Cards to core Cards for bucket lookup.
    let core_board: Vec<CoreCard> = board_cards.iter().map(|c| to_core_card(*c)).collect();
    let leaf_values_per_boundary: Vec<Vec<f64>> = boundary_mapping
        .iter()
        .zip(tree.nodes.iter().filter_map(|n| match n {
            GameNode::Terminal { kind: TerminalKind::DepthBoundary, pot, .. } => Some(*pot),
            _ => None,
        }))
        .map(|(&chance_ord, pot)| {
            let half_pot = pot / 2.0;
            (0..hands.combos.len())
                .map(|combo_idx| {
                    let hole = hands.combos[combo_idx];
                    let bucket = ctx.all_buckets.get_bucket(street, hole, &core_board) as usize;
                    let cbv = f64::from(ctx.cbv_tables[0].lookup(chance_ord, bucket));
                    (cbv / half_pot + 1.0) / 2.0
                })
                .collect()
        })
        .collect();
    leaf_values_per_boundary
} else {
    let equities = compute_combo_equities(&hands, board_cards, &opponent_reach);
    let boundary_count = tree.nodes.iter().filter(|n| matches!(n,
        GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. }
    )).count();
    vec![equities; boundary_count]
};

let solver = SubgameCfrSolver::new(
    tree.clone(), hands.clone(), board_cards, opponent_reach, leaf_values,
);
```

**Step 3: Thread `cbv_context` through the call chain**

In `solve_depth_limited` (line 1085), add `cbv_context: Option<&CbvContext>` parameter. Pass it through to `build_subgame_solver` (line 1139-1147).

In `postflop_solve_street_impl` (line 931), read `cbv_context` from `PostflopState` and pass to `solve_depth_limited`:

```rust
let cbv_ctx = state.cbv_context.read().clone();
// In the DepthLimited branch:
solve_depth_limited(state, &config, board, max_iterations, &solver_config, &filtered_oop, &filtered_ip, cbv_ctx.as_ref())
```

Note: `CbvContext` must implement `Clone` or be wrapped in `Arc`. Using `Arc<CbvContext>` in `PostflopState` is cleanest since it avoids copying large CBV tables.

**Step 4: Verify compilation and run existing tests**

Run: `cargo build -p tauri-app`
Run: `cargo test -p poker-solver-core`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tauri-app/src/postflop.rs crates/tauri-app/src/exploration.rs
git commit -m "feat: wire CBV-based leaf values into TUI subgame solver"
```

---

### Task 6: Full test suite and cleanup

Run the entire test suite, fix any breakage, run clippy.

**Step 1: Run full test suite**

Run: `cargo test`
Expected: ALL PASS in under 1 minute.

**Step 2: Run clippy**

Run: `cargo clippy`
Expected: No new warnings.

**Step 3: Fix any issues found**

**Step 4: Final commit**

```bash
git commit -m "chore: fix any test/clippy issues from CBV evaluator integration"
```
