# Blueprint CBV Evaluator — Use Blueprint CFVs at Depth Boundaries

**Goal:** When the TUI subgame solver hits a `DepthBoundary` node, use the blueprint's precomputed counterfactual boundary values (CBVs) instead of raw showdown equity. This gives strategically informed leaf values that encode fold equity, future street potential, and bluff/value frequency from the blueprint solution.

**Problem:** The current `build_subgame_solver()` in `postflop.rs` calls `compute_combo_equities()` at depth boundaries — pure showdown equity that ignores all strategic information the blueprint already computed.

## Design

### Overview

1. Fix the unit-conversion bug in `SubgameCfrSolver::with_cbv_table()`
2. Build a boundary mapping from subgame `DepthBoundary` nodes to abstract-tree `Chance` nodes
3. Load abstract tree + CBV tables + bucket files with the blueprint bundle
4. Replace `compute_combo_equities()` with `with_cbv_table()` in the TUI solve path

### Why static CBVs are correct

Blueprint CBVs are properties of the fixed blueprint strategy, computed via backward induction. They do not depend on the subgame solver's evolving ranges. Re-evaluating them each iteration (as `CfvSubgameSolver` + `LeafEvaluator` would) is wasteful — the values never change. The existing `SubgameCfrSolver` with static `leaf_values` is the right target.

### Component 1: Fix `with_cbv_table()` unit conversion

**File:** `crates/core/src/blueprint_v2/subgame_cfr.rs`

CBVs from `cbv_compute.rs` are in absolute chip units. The `DepthBoundary` traversal interprets `leaf_values` as equity in [0, 1] and converts via `(2 * equity - 1) * half_pot`. To make this round-trip correctly:

```
equity = (cbv / half_pot_at_boundary + 1.0) / 2.0
```

Verification: `(2 * ((cbv/hp + 1)/2) - 1) * hp = cbv` ✓

Change `with_cbv_table()` to:
- Accept a per-boundary mapping `&[usize]` (one abstract Chance node ordinal per subgame DepthBoundary)
- Accept the subgame tree reference to read each boundary's pot
- Normalize each combo's CBV using that boundary's `half_pot`
- Panic if any boundary or bucket lookup fails — no fallbacks

New signature:
```rust
pub fn with_cbv_tables(
    tree: GameTree,
    hands: SubgameHands,
    board: &[Card],
    opponent_reach: Vec<f64>,
    cbv_tables: [&CbvTable; 2],         // [player_0, player_1]
    boundary_mapping: &[usize],          // subgame boundary ordinal → abstract chance ordinal
    combo_to_bucket: impl Fn(usize) -> u16,
) -> Self
```

Leaf value computation per boundary per combo:
```rust
for (boundary_ord, &chance_ord) in boundary_mapping.iter().enumerate() {
    let half_pot = boundary_pots[boundary_ord] / 2.0;
    for combo_idx in 0..num_combos {
        let bucket = combo_to_bucket(combo_idx) as usize;
        let cbv = cbv_table.lookup(chance_ord, bucket);
        let equity = (f64::from(cbv) / half_pot + 1.0) / 2.0;
        leaf_values[boundary_ord][combo_idx] = equity;
    }
}
```

This means `leaf_values` becomes per-boundary, not a single flat vec. The `DepthBoundary` traversal code must index by boundary ordinal.

### Component 2: Boundary mapping via parallel tree walk

**New function:** `build_boundary_mapping(subgame_tree: &GameTree, abstract_tree: &GameTree) -> Vec<usize>`

Both trees share the same action set (same bet sizes). Walk them in lockstep:

1. Start at both roots
2. At `Decision` nodes: match actions exactly (Check↔Check, Bet(x)↔Bet(x), Fold↔Fold, Call↔Call)
3. When subgame reaches `DepthBoundary` and abstract reaches `Chance`: record `(subgame_boundary_ordinal, abstract_chance_ordinal)`
4. Panic if any action in the subgame tree has no match in the abstract tree
5. Panic if a subgame `DepthBoundary` doesn't correspond to an abstract `Chance` node

Returns a `Vec<usize>` of length `num_subgame_boundaries`, where each entry is the abstract-tree Chance node's ordinal in the CBV table.

The abstract tree's Chance node ordinal is its position among all Chance nodes (matching how `cbv_compute.rs` indexes `CbvTable`).

### Component 3: Load CBV tables + bucket files + abstract tree with bundle

**File:** `crates/tauri-app/src/exploration.rs`

When loading a blueprint bundle for subgame solving:
- Load `cbv_p0.bin` and `cbv_p1.bin` → `[CbvTable; 2]`
- Load bucket files (already available as `BucketFile` per street)
- Load the abstract `GameTree` (already available from bundle)
- Store all three in the strategy source / postflop state

Panic if CBV files or bucket files are missing from the bundle.

### Component 4: Wire into `build_subgame_solver()`

**File:** `crates/tauri-app/src/postflop.rs`

Replace:
```rust
let leaf_values = compute_combo_equities(&hands, board_cards, &opponent_reach);
let solver = SubgameCfrSolver::new(tree, hands, board, opponent_reach, leaf_values);
```

With:
```rust
let boundary_mapping = build_boundary_mapping(&subgame_tree, &abstract_tree);
let solver = SubgameCfrSolver::with_cbv_tables(
    tree, hands, board, opponent_reach,
    [&cbv_p0, &cbv_p1],
    &boundary_mapping,
    |combo_idx| all_buckets.get_bucket(street, hands.combos[combo_idx], board),
);
```

Additional parameters flow from the loaded bundle through the solve dispatch.

### Per-boundary leaf values

The current `SubgameCfrSolver` stores `leaf_values: Vec<f64>` — one value per combo, shared across all boundaries. This must change to `leaf_values: Vec<Vec<f64>>` — one vec per boundary, each of length `num_combos`. The `DepthBoundary` traversal uses `self.node_to_boundary[node_idx]` (which already exists in `CfvSubgameSolver`) to index into the correct boundary's values.

### Error handling

No fallbacks. All lookups must succeed:
- `AllBuckets::get_bucket()` → panic if bucket file missing (already panics in production)
- `CbvTable::lookup()` → panic if boundary node or bucket out of range
- `build_boundary_mapping()` → panic if trees don't align
- CBV/bucket file loading → panic if files missing from bundle

## Testing

- **Unit:** `with_cbv_tables` normalizes CBVs correctly (known CBV + known pot → expected equity)
- **Unit:** `build_boundary_mapping` correctly maps a small hand-crafted tree pair
- **Unit:** `build_boundary_mapping` panics on mismatched action sets
- **Unit:** Per-boundary leaf values produce different CFVs at different pot-size boundaries
- **Integration:** Subgame solver with CBV leaf values converges and produces valid strategy distributions

## Files Changed

| File | Change |
|------|--------|
| `crates/core/src/blueprint_v2/subgame_cfr.rs` | Fix `with_cbv_table()` → `with_cbv_tables()`, per-boundary leaf values, boundary ordinal indexing |
| `crates/core/src/blueprint_v2/subgame_cfr.rs` | Add `build_boundary_mapping()` |
| `crates/tauri-app/src/exploration.rs` | Load CBV tables + bucket files with bundle |
| `crates/tauri-app/src/postflop.rs` | Wire CBV-based solver into `build_subgame_solver()` |
