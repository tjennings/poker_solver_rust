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

## Implementation Path: Per-call CFV adjustment (wrapper evaluator)

**Chosen approach:** Wrapper `BoundaryEvaluator` that adjusts CFVs after the inner
evaluator computes them.

**Justification:** This avoids modifying range-solver's ActionTree or evaluation engine
entirely. The gadget logic lives in a thin wrapper that:
1. Delegates to the inner evaluator (e.g. `SubtreeExactEvaluator`) 
2. For the opponent's CFVs, clamps each hand's value upward to the opt-out
3. Adjusts the player's CFVs to maintain zero-sum (player gets -opt_out when opponent opts out)

This is the least invasive approach -- no changes to range-solver at all.

**Per-hand gadget math:**
For each opponent hand h at boundary ordinal b:
- `inner_cfv[h]` = CFV from subtree evaluator
- `opt_out[h]` = CFV from OptOutProvider (blueprint CBV)
- `gadget_cfv[h]` = max(inner_cfv[h], opt_out[h])

For the traversing player, the adjustment is the negative of the opponent's gain:
- `player_adjustment[h]` = -(gadget_cfv_opp[h] - inner_cfv_opp[h]) weighted by card overlap

Since the bcfv interface already handles reach weighting, we apply the clamp directly
to the bcfv values before they are stored in the boundary cache.

## Key Types

### `OptOutProvider` trait (gadget.rs)
```rust
pub trait OptOutProvider: Send + Sync {
    fn opt_out_cfvs(&self, boundary_ordinal: usize, opponent: usize,
                     pot: i32, effective_stack: i32, board: &[u8],
                     opponent_private_cards: &[(u8, u8)]) -> Vec<f32>;
}
```

### `ConstantOptOut` (gadget.rs, testing)
Returns the same value for every hand. Used to verify gadget behavior in unit tests.

### `GadgetEvaluator` (gadget.rs)
Wraps an inner `BoundaryEvaluator` + `OptOutProvider`. Implements `BoundaryEvaluator`.

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
value, every opponent hand's CFV from the gadget evaluator is >= the opt-out value.
