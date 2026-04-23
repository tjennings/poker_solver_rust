# DeepStack Range Gadget for Safe Subgame Solving

## Goal

Implement a DeepStack-style range gadget (Moravcik et al. 2017; Brown et al. 2020 ReBeL)
to constrain opponent strategy at subgame boundaries. The gadget gives the opponent a
per-hand choice between entering the subtree or taking an opt-out value equal to their
counterfactual best-response value from the blueprint. This produces subgame solutions
that are **safe** -- the opponent's subgame CFV is guaranteed to be at least as good as
the blueprint's, preventing the strategy inversions observed in iterations 1-7 of the
subgame-exact-parity investigation.

## Architecture

```
Parent DCFR solver
  |
  v  (reaches boundary node)
  |
  +--[GadgetEvaluator]--+
  |                      |
  |  For each opponent   |
  |  hand h:             |
  |                      |
  |  subtree_cfv[h]      |     opt_out_cfv[h]
  |  (from inner eval)   |     (from OptOutProvider)
  |                      |
  |  gadget_cfv[h] =     |
  |    max(subtree_cfv,  |
  |        opt_out_cfv)  |
  |                      |
  +----------------------+
  |
  v  (returns adjusted CFVs to parent)
```

The gadget node conceptually sits between the parent's boundary and the subtree:

```
Parent game tree:
  ... -> [boundary] -> (subtree evaluator returns CFVs)

With gadget:
  ... -> [boundary] -> [gadget: opt-out vs enter] -> (subtree)

  At the gadget, opponent chooses per-hand:
    ENTER:   play subtree, get subtree_cfv[h]
    OPT-OUT: take opt_out_cfv[h] (from blueprint CBV)

  Regret matching at gadget ensures opponent never does worse
  than opt-out. This constrains the subtree solution to be safe.
```

## Implementation Path Comparison

### Option A: Tree-extension (add explicit gadget node to ActionTree)

**Description:** Insert a new decision node into the range-solver's `ActionTree` at each
boundary. The node has K+1 children: one ENTER action leading to the subtree, and one
OPT-OUT action per hand (or a single OPT-OUT action with per-hand payoffs). The DCFR
solver's standard regret matching naturally handles the gadget choice.

**Advantages:**
- Theoretically cleanest: the gadget is a real game node, DCFR discounting and strategy
  averaging apply naturally
- Per-iteration reach evolution is handled automatically by the solver
- No special-case code in the evaluation path

**Disadvantages:**
- Requires modifying `ActionTree` construction and `PostFlopGame`'s node layout -- these
  are the most performance-critical and delicate structures in range-solver
- The gadget node is NOT a standard poker action (check/bet/fold/call), so the node type
  system needs extension
- The OPT-OUT payoff is per-hand (not per-pot), requiring a new terminal evaluation mode
- Tight coupling between gadget logic and range-solver internals makes it fragile to
  refactor
- Substantial code surface: new node type, new action type, new terminal evaluator,
  modified tree-build, modified traversal

### Option B: Per-call CFV adjustment (wrapper evaluator) -- CHOSEN

**Description:** Wrapper `BoundaryEvaluator` that adjusts CFVs after the inner evaluator
computes them. The gadget logic lives entirely outside range-solver.

**Justification:** This avoids modifying range-solver's ActionTree or evaluation engine
entirely. The gadget logic lives in a thin wrapper that:
1. Delegates to the inner evaluator (e.g. `SubtreeExactEvaluator`)
2. For the opponent's CFVs, clamps each hand's value upward to the opt-out
3. Returns the clamped values through the standard `BoundaryEvaluator` interface

This is the least invasive approach -- no changes to range-solver at all. It also
composes cleanly: any `BoundaryEvaluator` can be wrapped (cfvnet, rollout, exact subtree).

**Advantages:**
- Zero changes to range-solver
- Composable: wraps any inner evaluator
- Small code surface (~100 lines)
- Easy to test with stub evaluators

**Disadvantages:**
- The clamp is applied once per boundary visit, not iterated within the subtree solve.
  This means the gadget doesn't benefit from DCFR's convergence properties at the
  gadget node itself. In practice, the upward clamp is a sufficient approximation
  because the inner evaluator already produces converged CFVs.
- The traversing player's CFV adjustment (zero-sum accounting) requires care -- currently
  we leave player CFVs unchanged since the bcfv interface treats per-hand values
  independently.

**Per-hand gadget math:**
For each opponent hand h at boundary ordinal b:
- `inner_cfv[h]` = CFV from subtree evaluator
- `opt_out[h]` = CFV from OptOutProvider (blueprint CBV)
- `gadget_cfv[h]` = max(inner_cfv[h], opt_out[h])

For the traversing player, the adjustment is the negative of the opponent's gain:
- `player_adjustment[h]` = -(gadget_cfv_opp[h] - inner_cfv_opp[h]) weighted by card overlap

Since the bcfv interface already handles reach weighting, we apply the clamp directly
to the bcfv values before they are stored in the boundary cache.

## Interface Surface

### `OptOutProvider` trait (`gadget.rs`)

```rust
pub trait OptOutProvider: Send + Sync {
    /// Returns per-hand opt-out CFVs for the OPPONENT at this boundary.
    /// Values are in pot-normalised bcfv units (1.0 = one half-pot).
    /// Vec length must equal `opponent_private_cards.len()`.
    fn opt_out_cfvs(
        &self,
        boundary_ordinal: usize,
        opponent: usize,
        pot: i32,
        effective_stack: i32,
        board: &[u8],
        opponent_private_cards: &[(u8, u8)],
    ) -> Vec<f32>;
}
```

### `ConstantOptOut` (testing)

Returns the same value for every hand at every boundary. Used to verify gadget behavior
in unit tests without needing a real blueprint.

```rust
pub struct ConstantOptOut(pub f32);
```

### `BlueprintCbvOptOut` (production)

Queries the loaded blueprint strategy to compute per-hand counterfactual best-response
values at the boundary. This is the production implementation that makes re-solving safe.

```rust
pub struct BlueprintCbvOptOut {
    /// Reference to the loaded blueprint strategy bundle.
    blueprint: Arc<BlueprintV2Strategy>,
    /// Board cards for the current subgame.
    board: Vec<u8>,
    /// Pre-computed per-boundary CBVs, indexed by boundary_ordinal.
    /// Each entry is a Vec<f32> with one value per opponent hand.
    cached_cbvs: Vec<Vec<f32>>,
}
```

**How CBVs are computed:** For each boundary, walk the blueprint game tree to the
boundary node, then compute the opponent's counterfactual best-response value by
maximising over available actions weighted by the blueprint's reach probabilities. This
is the value the opponent *could* achieve by deviating from the blueprint at this node.

### `GadgetEvaluator` (boundary evaluator wrapper)

Wraps an inner `BoundaryEvaluator` + `OptOutProvider`. Implements `BoundaryEvaluator`.

```rust
pub struct GadgetEvaluator {
    inner: Arc<dyn BoundaryEvaluator>,
    opt_out: Arc<dyn OptOutProvider>,
    board: Vec<u8>,
    private_cards: [Vec<(u8, u8)>; 2],
}
```

## Tests

### Test 1: Huge positive opt-out forces all opt-out

`ConstantOptOut(1000.0)` -- every hand's opt-out value far exceeds any realistic subtree
CFV. The opponent should always opt out. The returned opponent CFVs should all equal
the opt-out value (1000.0 in bcfv units).

### Test 2: Very negative opt-out is dominated

`ConstantOptOut(-1000.0)` -- opt-out is so bad that the opponent always enters the
subtree. The gadget-wrapped CFVs should match the non-gadget inner evaluator exactly.

### Test 3: Integration -- opponent CFV >= opt-out per hand

On the small test game (AA,KK,QQ vs TT,99,88), verify that with a moderate opt-out
value (0.0 = break-even), every opponent hand's CFV from the gadget evaluator is >= the
opt-out value. Also verify that at least one hand was actually clamped (i.e., the gadget
had an effect, not a vacuous pass-through).
