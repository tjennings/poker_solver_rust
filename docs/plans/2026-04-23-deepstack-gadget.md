# Libratus-style Safe Re-solving Gadget — Validated Design

> **Terminology note.** The original draft called this a "DeepStack gadget." After
> brainstorming the two lineages apart, this design is **Libratus-style** (static
> opt-out values from a pre-computed blueprint CBV table). DeepStack-proper uses
> opt-out bounds as a runtime parameter that the CFV network is trained to
> consume, which we cannot do without retraining. That retrain is tracked as
> bean [`poker_solver_rust-akg3`](../../.beans/poker_solver_rust-akg3--retrain-cfvnet-with-opt-out-input-channel-for-deep.md).
>
> The module name in code remains `gadget` (generic) rather than renaming to
> `libratus_safe_resolve`. The design doc is the place we state which flavour.

Parent bean: [`poker_solver_rust-lay5`](../../.beans/poker_solver_rust-lay5--libratus-style-safe-re-solving-gadget-mvp.md).

## Goal

Make cfvnet-based and other boundary-evaluator-based subgame solves **safe** — the
opponent's reported per-hand CFV at a boundary is no worse than the blueprint's
counterfactual best-response value for that hand. This prevents the strategy
inversions observed across 8 iterations of subgame-exact-parity investigation
(`docs/progress/2026-04-22-subgame-exact-parity.md`).

**Primary consumer:** cfvnet (`NeuralBoundaryEvaluator`) in both the
compare-solve harness and the Tauri game_solve command.

**Secondary consumer:** `SubtreeExactEvaluator` (diagnostic oracle). Already wired
by the skeleton agent.

**Not in scope:** DeepStack-proper continual re-solving with trained opt-out
input channel (bean `akg3`).

## Architecture

The gadget is a `BoundaryEvaluator` **wrapper** — it lives outside range-solver
entirely, consumes any inner `BoundaryEvaluator`, and clamps the opponent's
per-hand CFV upward to a pre-computed opt-out floor.

```
┌────────────────────────────────────────────────────────────┐
│ parent solver                                               │
│  │                                                          │
│  ▼  (reaches boundary)                                      │
│ ┌────────────────────────────────────────┐                  │
│ │ GadgetEvaluator                        │                  │
│ │                                        │                  │
│ │   inner.compute_cfvs_both(...)         │                  │
│ │   │                                    │                  │
│ │   ▼                                    │                  │
│ │   (oop_cfv, ip_cfv)                    │                  │
│ │   │                                    │                  │
│ │   ▼ clamp opp side up to opt_out       │                  │
│ │   opp_cfv[h] ← max(inner[h], opt[h])   │                  │
│ └────────────────────────────────────────┘                  │
│  │                                                          │
│  ▼ adjusted CFVs                                            │
└────────────────────────────────────────────────────────────┘
```

### Per-hand clamp

For the opponent's hand `h` at boundary ordinal `b`:

```
opt_out[h] = blueprint_cbv[h] / half_pot      // chip CBV → bcfv units
gadget_cfv[h] = max(inner_cfv[h], opt_out[h])
```

The traversing player's CFVs pass through unchanged. Both `compute_cfvs` and
`compute_cfvs_both` apply the clamp on the opponent side of the call.

### Safety property (honest)

- **Reported-CFV safety:** the opponent's CFV as returned is ≥ opt-out. The
  parent solver therefore gets a boundary value that is at least as good for
  the opponent as the blueprint.
- **Not full strategy-level safety:** in DeepStack's paper, the in-subtree
  *strategy* adapts because the opt-out is part of the subgame tree — the
  player responds to the opponent's ability to opt out. Option B (our wrapper)
  applies the clamp *after* the inner evaluator runs, so any inner strategy is
  frozen and gadget-unaware. This is intentional for cfvnet (where there is
  no "inner strategy" — cfvnet is a function) and a known approximation for
  the exact subtree evaluator.

## Implementation — Option B (per-call CFV adjustment)

See the commit history on `feat/subgame-exact-parity` for Option A (tree-extension)
tradeoff analysis. Option B chosen for: zero changes to range-solver, trivial
composition with any inner evaluator, ~150 lines of code, already partially shipped.

## Interface surface

### `OptOutProvider` trait (already shipped in `crates/tauri-app/src/gadget.rs`)

```rust
pub trait OptOutProvider: Send + Sync {
    /// Per-hand opt-out CFVs for the OPPONENT at this boundary, in
    /// pot-normalised bcfv units (1.0 = one half-pot won).
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

### `ConstantOptOut(f32)` (already shipped)

For testing / A-B diagnostic via `--gadget-provider=constant --gadget-constant=V`.

### `BlueprintCbvOptOut` (to build)

```rust
pub struct BlueprintCbvOptOut {
    cbv_table: Arc<CbvTable>,
    all_buckets: Arc<AllBuckets>,
    abstract_node_idx: usize,
    per_hand_cbv: [Vec<f32>; 2],  // pre-computed at construction, per player
    half_pot_chips: f32,
}

impl BlueprintCbvOptOut {
    pub fn new(
        cbv_ctx: &CbvContext,
        board: &[u8],
        abstract_node_idx: usize,
        private_cards: &[Vec<(u8, u8)>; 2],
        half_pot_chips: f32,
    ) -> Self { /* ... */ }
}
```

**Construction panics if:**
- `cbv_table` is missing from the bundle.
- `abstract_node_idx` doesn't resolve to a valid blueprint node.

**Runtime panics if:**
- `all_buckets.lookup(board, c1, c2)` returns `None` for a hand in `private_cards`
  (should be impossible — the parent `PostFlopGame::assign_zero_weights` filters
  blocker-conflicting hands out of `private_cards` before we see them).

### `GadgetEvaluator` (already shipped, may need generalisation)

Verify the shipped wrapper correctly clamps both `compute_cfvs` and
`compute_cfvs_both` on the opponent side.

## Wiring

### `compare-solve` CLI

Current: `--gadget` (bool), only wraps exact_subtree.

New:
- `--gadget` stays.
- `--gadget-provider <blueprint-cbv|constant>` (default `blueprint-cbv` when `--gadget` is set).
- `--gadget-constant <f32>` (required when `--gadget-provider=constant`).
- `setup_neural_boundaries` accepts `Option<Arc<dyn OptOutProvider>>` and wraps each `NeuralBoundaryEvaluator` when present. Mirror `setup_exact_subtree_boundaries`.
- Provider is constructed once in `run()` before either `setup_*` fn, fed to both.

### Tauri `game_solve_core`

- Add `enable_gadget: bool` to `game_solve` Tauri command params, thread through `game_solve_core`.
- When true + a boundary-cut mode is active, construct `BlueprintCbvOptOut` from `PostflopState::cbv_context`, wrap neural evaluators in `GadgetEvaluator`.
- When true + top-level "exact" mode (no cut), log that gadget is unused and continue.

### Settings UI

- `frontend/src/Settings.tsx` — add checkbox "Safe re-solving (Libratus gadget)".
- `frontend/src/types.ts` — `GlobalConfig.enable_safe_resolving: boolean`.
- `frontend/src/useGlobalConfig.ts` — default false.
- `frontend/src/strategy-tabs.ts::buildSolveParams` — pass `enableGadget: bool`.
- Greyed out when no per-street cut mode is `cfvnet` or `exact_subtree`.

## Validation

1. **Iter 9 — cfvnet baseline (no gadget):** record subgame_exp + worst_delta vs exact.
2. **Iter 10 — cfvnet + BlueprintCbvOptOut:** compare to iter 9.
3. **Iter 11 — cfvnet + ConstantOptOut(-999.0):** dominated opt-out. Must match iter 9 exactly (sanity).

Hypothesis: iter 10's strategy delta is smaller than iter 9's. If not, the
Libratus-style static CBV is not strong enough and we fall back to the bean-`akg3`
retrain path.

## Testing

### Rust unit tests (in `gadget.rs`)

- `blueprint_cbv_construct_panics_on_missing_table`
- `blueprint_cbv_lookup_panics_on_missing_bucket`
- `blueprint_cbv_returns_nonzero_for_known_spot` — integration with a real bundle fixture
- `unit_conversion_chip_to_bcfv` — standalone numeric test
- `gadget_wraps_neural_evaluator_and_clamps` — mocked `NeuralBoundaryEvaluator`

### Harness tests (runtime, not CI)

- Non-gadget cfvnet baseline
- Gadget + BlueprintCbvOptOut cfvnet
- Gadget + ConstantOptOut(-999.0) — should be no-op

## Out of scope

- Tree-extension implementation of the gadget (Option A) — deferred indefinitely.
- DeepStack-proper retrain of cfvnet with opt-out input channel — bean `akg3`.
- Unabstracted CBV via exact BR — easy to add later as a second `OptOutProvider` impl; deferred.
- Automated UI click-through test.
- Performance optimisation beyond pre-computing per-hand CBVs at construction.

## Files

To create or modify:
- `crates/tauri-app/src/gadget.rs` — add `BlueprintCbvOptOut` impl + panics.
- `crates/tauri-app/src/game_session.rs` — thread `enable_gadget` through `game_solve_core` + `setup_neural_boundaries`.
- `crates/trainer/src/main.rs` — add `--gadget-provider`, `--gadget-constant` flags.
- `crates/trainer/src/compare_solve.rs` — construct provider, pass to both setup_* fns, wire into neural path.
- `frontend/src/Settings.tsx` — checkbox.
- `frontend/src/types.ts` — config type field.
- `frontend/src/useGlobalConfig.ts` — default.
- `frontend/src/strategy-tabs.ts` — build params.
- `docs/progress/2026-04-22-subgame-exact-parity.md` — iterations 9–11.
