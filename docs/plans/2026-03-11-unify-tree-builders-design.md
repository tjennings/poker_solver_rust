# Unify Tree Builders: V2GameTree as Canonical Tree

**Goal:** Eliminate SubgameTree by making V2GameTree the single tree builder for both blueprint training and real-time subgame solving.

**Motivation:** Four independent tree builders each implement their own action generation logic. The same all-in bug (missing all-in when raise cap is reached) was fixed in PreflopTree (58f100c) but lurked undetected in V2GameTree until today. One tree builder = one place to fix bugs.

## Architecture

V2GameTree gets a new `build_subgame()` constructor that creates a partial tree rooted at a specific street/state. `TerminalKind` gains a `DepthBoundary` variant for depth-limited solving. SubgameCfrSolver switches from SubgameTree to V2GameTree. SubgameTree is deleted.

## Core Changes

### 1. Extend TerminalKind

```rust
enum TerminalKind {
    Fold { winner: u8 },
    Showdown,
    DepthBoundary,  // new: depth-limited solver leaf
}
```

### 2. Add build_subgame() to GameTree

```rust
impl GameTree {
    pub fn build_subgame(
        street: Street,
        pot: f64,
        invested: [f64; 2],
        starting_stack: f64,
        bet_sizes: &[Vec<f64>],      // pot fractions per raise depth
        depth_limit: Option<u8>,      // None = full depth to river
    ) -> Self;
}
```

When `depth_limit` is reached at a street boundary, emit `Terminal { kind: DepthBoundary }` instead of a `Chance` node. The existing `build()` never hits this path.

### 3. Update SubgameCfrSolver

Change tree type from `SubgameTree` to `GameTree`. Key mappings:

| SubgameNode | GameNode |
|-|-|
| `Decision { position, actions, children, pot, stacks }` | `Decision { player, street, actions, children }` |
| `Terminal { is_fold, fold_player, pot, stacks }` | `Terminal { kind: Fold/Showdown, pot, invested }` |
| `DepthBoundary { pot, stacks }` | `Terminal { kind: DepthBoundary, pot, invested }` |

Action type changes from `Action::Bet(u32_index)` / `Action::Raise(u32_index)` to `TreeAction::Bet(f64)` / `TreeAction::Raise(f64)`. The solver's hot loop only indexes by action position, so this doesn't affect performance.

Value representation changes from `u32` chips to `f64` BB. Solver computes `pot = invested[0] + invested[1]`.

`SubgameLayout` flat-buffer indexing is unchanged — it maps `(node_idx, combo_idx)` to buffer offsets, which works with any arena-allocated tree.

### 4. Deletions

- `SubgameTree`, `SubgameTreeBuilder`, `SubgameNode` (subgame_tree.rs) — replaced by V2GameTree
- All SubgameTreeBuilder call sites switch to `GameTree::build_subgame()`

### 5. Kept As-Is

- `SubgameCfrSolver` — algorithm unchanged, just new tree type
- `SubgameStrategy` — output format unchanged
- Range-solver (`ActionTree`) — separate crate, separate concerns
- PreflopTree — separate bean (g2gm) for removal

## Testing

- Existing V2GameTree tests cover action generation (including all-in-everywhere)
- Existing SubgameCfrSolver convergence tests re-targeted at `build_subgame()`
- New test: `build_subgame` with `depth_limit: Some(1)` produces DepthBoundary terminals
