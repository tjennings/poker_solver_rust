# Subgame Solver Dispatch in PostflopExplorer

**Goal:** When solving spots in the PostflopExplorer with a blueprint loaded, dispatch to either the range-solver (full-depth) or `SubgameCfrSolver` (depth-limited) based on the existing combo threshold config. River always uses range-solver. Ranges propagate correctly between streets.

**Problem:** The PostflopExplorer currently always builds a full remaining-streets tree via the range-solver, which OOMs on large models (flop with wide ranges = massive tree).

## Architecture

### Solver Dispatch

Uses the existing `dispatch_decision()` from `solver_dispatch.rs`:

- **River** → always `FullDepth` → range-solver (manageable tree, exact solution)
- **Turn** → range-solver if `live_combos <= 300`, else `SubgameCfrSolver` with `depth_limit(1)`
- **Flop** → range-solver if `live_combos <= 200`, else `SubgameCfrSolver` with `depth_limit(1)`

Live combo count = max(oop_combos, ip_combos) from the filtered weights already set by the PostflopExplorer on launch.

### Data Flow

```
Explorer loads bundle → config.yaml + blueprint.bin + cbv_p0.bin + cbv_p1.bin
                                                       ↓
PostflopExplorer launches with blueprintConfig (ranges, pot, stacks, bet sizes, cbv tables)
                                                       ↓
User clicks SOLVE → backend dispatch_decision(street, live_combos)
                     ├─ FullDepth → range-solver (existing path, unchanged)
                     └─ DepthLimited → SubgameCfrSolver with depth_limit(1) + CBV leaf values
                                                       ↓
Both paths → PostflopProgress atomics → frontend polls → matrix display
```

### Components Modified

1. **`exploration.rs`** — `SubgameSolve` variant: add `cbv_p0: Arc<CbvTable>`, `cbv_p1: Arc<CbvTable>`. Load from bundle dir on `load_blueprint_for_subgame`.

2. **`postflop.rs`** — New `postflop_solve_dispatched` command (or modify `postflop_solve_street` to accept a solver mode):
   - Accepts optional `cbv_table`, `blueprint`, `solver_config` from the PostflopExplorer context
   - Computes live_combos from range weights
   - Calls `dispatch_decision()`
   - `FullDepth` → existing `build_game()` + range-solver path
   - `DepthLimited` → builds `SubgameTreeBuilder` with `.depth_limit(1)`, converts range weights to opponent_reach, populates leaf_values from CbvTable, runs `SubgameCfrSolver.train(iterations)`

3. **`PostflopProgress`** — Add `solver_name: String` field (`"range"` or `"subgame"`)

4. **`PostflopExplorer.tsx`** — `handleSolve` dispatches to `postflop_solve_dispatched` when `blueprintConfig` is present, passing CBV table reference. Shows `progress.solver_name` in progress bar.

### Range Propagation Between Streets

When a street solve completes and the user picks a turn/river card:

1. **Range-solver path**: `postflop_close_street` already extracts reaching ranges by weighting the initial range through the strategy along the action path. These become the starting ranges for the next street.

2. **Subgame solver path**: Same principle — extract per-combo reaching weights from `SubgameStrategy` by multiplying initial weights by strategy probabilities along the chosen action path. These become the starting ranges for the next street's solve.

Both paths feed into `postflop_set_filtered_weights` before the next street's solve, so the dispatch logic runs again with the narrowed ranges.

### SubgameCfrSolver ↔ Strategy Matrix

The existing matrix display expects per-hand-class action probabilities. The `SubgameCfrSolver` produces per-combo (1326) probabilities. Need a translation layer:

- For each matrix cell (e.g., "AKs"), enumerate the concrete combos, look up their strategy from `SubgameStrategy`, and average the probabilities weighted by reaching weight.
- This is analogous to what the range-solver path does via `game.strategy()`.

### Memory Budget

- Range-solver river tree: ~100-500 MB (acceptable)
- `SubgameCfrSolver` single-street tree: hundreds of nodes, ~1 MB equity matrix for ~1000 combos (tiny)
- CBV tables: small (a few MB per player)

No OOM risk with this design.

## Testing

- Unit test: dispatch decision returns correct solver choice for various combo counts
- Unit test: SubgameCfrSolver produces valid strategy with CBV leaf values
- Integration: PostflopExplorer flop solve with >200 combos uses subgame solver (check solver_name in progress)
- Integration: Multi-street range propagation (solve flop → pick turn → solve turn with narrowed ranges)
