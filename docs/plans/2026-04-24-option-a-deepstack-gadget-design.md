# Option A: DeepStack-style gadget as CFR tree modification — Design

**Status:** Brainstorm complete 2026-04-24. Ready for `hex:writing-plans`.
**Predecessor:** `docs/plans/2026-04-24-option-a-deepstack-gadget-tree.md` (pre-brainstorm substrate).
**Brainstorm bean:** `poker_solver_rust-u3rf`.
**Parent bean (re-scoped):** `poker_solver_rust-akg3`.
**Iteration history:** `docs/progress/2026-04-22-subgame-exact-parity.md` (iters 1–14).

---

## 1. Architecture overview

Safe re-solving gadget reformulated from a post-clamp boundary wrapper into a structural CFR-D construction per [Burch/Johanson/Bowling 2014 §3](https://webdocs.cs.ualberta.ca/~bowling/papers/14aaai-cfrd.pdf). Two nested `Decision(player, [Terminate, Follow])` nodes are prepended at the top of every `PostFlopGame::node_arena`; each `Terminate` child is a depth-boundary-style terminal whose per-hand CFVs for the gadget player are pre-set to bucketed blueprint opt-outs and whose non-gadget-player CFVs are pinned to zero (neutralized). Regret matching at each gadget decision node forces every solved strategy to yield realized CFVs at-or-above that player's opt-out per the Burch §3 sufficiency condition — that is the formal safety claim we ship.

The existing `BlueprintCbvOptOut` supplies opt-out magnitudes (extended with a new `opt_out_at_subgame_root` method). The existing depth-boundary machinery supplies per-hand terminal CFV storage. A new `DispatchingBoundaryEvaluator` wrapper composes the static gadget ordinals (0/1) with dynamic cfvnet ordinals (2+) at solve time. The existing `ActionTree` and the entire blueprint-training pipeline are untouched — gadgets exist only in the solve-time `PostFlopGame`.

### Load-bearing claims

1. **Safety is a theorem, not a measurement.** MVP shipment is gated on test (4) proving `realized_CFV ≥ opt_out` post-solve. Exploitability on `JhTh9h|…|7d` may or may not improve. The safety claim is local, per-gadget-node, and independent of opt-out tightness.
2. **Zero blueprint-pipeline changes.** The gadget is solver-internal. `ActionTree`, `CbvTable`, `compute_cbvs`, `BlueprintV2Strategy` — all untouched.
3. **Additive on existing infrastructure.** `OptOutProvider` trait, `BlueprintCbvOptOut`, `ConstantOptOut`, `set_boundary_cfvs`, `boundary_ordinal`, `is_depth_boundary`, `SubtreeExactEvaluator`, `NeuralBoundaryEvaluator` — all reused unchanged. Net new: one range-solver module, one `PostFlopGame` constructor, one static-method on `BlueprintCbvOptOut`, one new `BoundaryEvaluator` impl.

### Not this design

- Not DeepStack-Leduc's algorithmic gadget (Pattern 2); we chose structural (Pattern 1).
- Not a cfvnet retrain — bean `akg3`'s original framing. This design supplants `akg3` for Tauri use cases.
- Not continual re-solving (one re-solve per Tauri spot).
- Not per-combo / un-abstracted opt-outs (Libratus strategy 4b); we accept bucketed looseness per goal (a).

---

## 2. Locked-in decisions from the brainstorm

| # | Question | Decision | Rationale |
|-|-|-|-|
| 1 | Acceptance bar | **(a) Safety only.** Formal guarantee holds; exploitability may not improve. | User goal: architectural transition from post-clamp to tree-extension; exploitability is a secondary metric. |
| 2 | Implementation pattern | **(1) Structural tree modification.** Insert Decision/Terminal pair above subgame root. | User pick over Pattern 2 algorithmic (DeepStack-Leduc style). Leverages existing depth-boundary mechanism which already supports per-hand externally-supplied CFVs. |
| 3 | Gadget count | **(C) Two nested gadgets.** `G_IP → G_OOP → R0`. | Exploitability is `BR(OOP) + BR(IP)`; bounding one side is half the safety. Nesting is extensive-form standard; each T/F node's regret-matching independently satisfies Burch §3. |
| 4 | Terminal semantics (non-gadget player) | **(ii) Neutralized = 0.** | Safety is local to the gadget player's own regret matching; breaking zero-sum at the terminal doesn't compromise it. Simplest to wire (constant, no per-iter recomputation). Mirrors DeepStack-Leduc's sidestep of this question by handling it outside the tree. |
| 5 | Tree injection point | **(a) `PostFlopGame` level.** New constructor `with_config_and_gadget`. | `ActionTree` stays a pure representation of poker betting structure. Strictly additive; no ripple through existing action-walk code. |
| 6 | Evaluator dispatch | **(x) `DispatchingBoundaryEvaluator` wrapper.** | Zero solver / `PostFlopGame` changes. Mirrors existing `GadgetEvaluator` composition pattern. Trivial to A/B vs non-gadget. |
| 7 | Tests gating MVP | **(1) + (2) + (3) + (4)** as hard gates; **(5)** informational. | (4) is the formal safety check — if it fails, the gadget isn't what we said it is. (5) reports exploitability but can't gate under goal (a). |

---

## 3. Components

### New

**`range-solver::game::gadget`** (new module, `crates/range-solver/src/game/gadget.rs`):

```rust
pub struct GadgetConfig {
    /// Per-hand OOP opt-out in bcfv units; len = num_private_hands(OOP).
    pub opt_out_oop: Vec<f32>,
    /// Per-hand IP opt-out in bcfv units; len = num_private_hands(IP).
    pub opt_out_ip: Vec<f32>,
    /// Nesting order: which player's Decision is outer. Default 1 (IP).
    pub outer_player: usize,
}

/// Prepend the 4-node gadget layer onto an already-built PostFlopGame's
/// node_arena. Updates children_offset on every pre-existing node, re-assigns
/// node_to_boundary ordinals (shifting existing by +2, reserving 0/1), and
/// sets the two static boundary_cfvs vectors.
pub(crate) fn inject_gadget_layer(game: &mut PostFlopGame, config: &GadgetConfig);
```

**`PostFlopGame::with_config_and_gadget`** (new public constructor in `crates/range-solver/src/game/mod.rs`):

```rust
pub fn with_config_and_gadget(
    card_config: CardConfig,
    action_tree: ActionTree,
    gadget_config: GadgetConfig,
) -> Result<Self, String>
```

Delegates to `with_config`, then calls `gadget::inject_gadget_layer`.

**`BlueprintCbvOptOut::opt_out_at_subgame_root`** (new associated function, `crates/tauri-app/src/gadget.rs`):

```rust
/// Compute per-hand opt-out bcfv values at the subgame root (decision node),
/// not at chance descendants. Uses blueprint's strategy to backward-induce
/// decision-node CBVs from stored chance-node CBVs. Returns
/// (opt_out_oop, opt_out_ip) in bcfv units normalised by subgame's starting
/// half-pot.
pub fn opt_out_at_subgame_root(
    cbv_context: &CbvContext,
    abstract_root: u32,
    subgame_pot_chips: f32,
    board: &[u8],
    private_cards: &[Vec<(u8, u8)>; 2],
) -> [Vec<f32>; 2];
```

**`DispatchingBoundaryEvaluator`** (new struct, `crates/tauri-app/src/gadget.rs`):

```rust
pub struct DispatchingBoundaryEvaluator {
    /// Pre-computed static CFVs for gadget ordinals 0/1.
    /// Indexed [ordinal][player] -> Vec<f32>.
    static_cfvs: [[Vec<f32>; 2]; 2],
    /// Delegate for ordinals 2+.
    inner: Arc<dyn BoundaryEvaluator>,
}
```

### Modified

- **`BoundaryEvaluator` trait** (potentially): may need `boundary_ordinal: usize` param on `compute_cfvs_both`. Plan-stage decision. Alternative: per-ordinal evaluator instantiation (matches existing `GadgetEvaluator` pattern).
- **CLI surface:** `--gadget-mode {clamp, tree}` (plan-stage; may repurpose existing `--gadget` flag).

### Reused unchanged

`OptOutProvider` trait, `ConstantOptOut`, `BlueprintCbvOptOut`, `chip_cfv_to_bcfv`, `set_boundary_cfvs`, `boundary_ordinal`, `is_depth_boundary`, `boundary_cfvs`, `SubtreeExactEvaluator`, `NeuralBoundaryEvaluator`, DCFR solver loop, `ActionTree`, `CbvTable`, `compute_cbvs`, `BlueprintV2Strategy`, entire blueprint pipeline.

### Retained for diagnostic / deleted later

`GadgetEvaluator` (post-clamp wrapper). Kept for A/B comparison during Option A rollout. Delete-bean opens once Option A ships with test (4) passing.

---

## 4. Data flow

### Init phase

```rust
// 1. Compute per-hand opt-outs at subgame root (bcfv units).
let [opt_out_oop, opt_out_ip] =
    BlueprintCbvOptOut::opt_out_at_subgame_root(
        &cbv_context, abstract_root,
        subgame_starting_pot as f32, &board, &private_cards,
    );

// 2. Build PostFlopGame with gadget layer.
let mut game = PostFlopGame::with_config_and_gadget(
    card_config, action_tree,
    GadgetConfig {
        opt_out_oop: opt_out_oop.clone(),
        opt_out_ip:  opt_out_ip.clone(),
        outer_player: 1, // IP outer
    },
)?;
game.allocate_memory(false);

// 3. Set static CFVs for gadget boundaries (ordinal 0 = G_IP.T, ordinal 1 = G_OOP.T).
let zero_oop = vec![0.0f32; game.num_private_hands(0)];
let zero_ip  = vec![0.0f32; game.num_private_hands(1)];
game.set_boundary_cfvs(0, 1, opt_out_ip);        // IP at G_IP.T
game.set_boundary_cfvs(0, 0, zero_oop.clone());  // OOP neutralized
game.set_boundary_cfvs(1, 0, opt_out_oop);       // OOP at G_OOP.T
game.set_boundary_cfvs(1, 1, zero_ip);           // IP neutralized

// 4. Compose boundary evaluator.
let eval = Arc::new(DispatchingBoundaryEvaluator::new(
    [[opt_out_oop, opt_out_ip], [zero_oop, zero_ip]],
    inner_cfvnet_evaluator,
));

// 5. Solve as usual.
solver::solve(&mut game, eval, max_iters, tolerance);
```

### Per-iter flow

The gadget layer is invisible to the solver loop — it looks like 2 normal `Decision` nodes with boundary-style terminals below. DCFR iterates over them like any other nodes.

**Forward pass reach propagation:**

| Node | Acting | Reach effect |
|-|-|-|
| `G_IP` (arena 0) | IP | σ_IP(T\|h), σ_IP(F\|h) via regret matching |
| → `G_IP.Terminate` (arena 1, boundary ord 0) | — | IP reach × σ_IP(T); OOP reach unchanged |
| → `G_OOP` (arena 2) | OOP | σ_OOP(T\|h'), σ_OOP(F\|h') via regret matching |
| → `G_OOP.Terminate` (arena 3, boundary ord 1) | — | OOP reach × σ_OOP(T); IP reach = init × σ_IP(F) carried |
| → `R0` (arena 4, former subgame root) | — | IP reach = init × σ_IP(F); OOP reach = init × σ_OOP(F) |

In-subgame reaches entering R0 are `init × σ_player(F)` — precisely the CFR-D reach-scaling effect.

**Backward pass at gadget terminals:**

| Node | Gadget player CFV | Non-gadget player CFV |
|-|-|-|
| `G_OOP.Terminate` | OOP CFV[h'] = opt_out_OOP[h'] | IP CFV[h] = 0 |
| `G_IP.Terminate` | IP CFV[h] = opt_out_IP[h] | OOP CFV[h'] = 0 |

**Regret matching at each gadget decision:**

```
V_T[hand] = gadget-player's boundary CFV (opt_out or 0 for non-gadget)
V_F[hand] = CFV from subtree below
V  [hand] = σ(T|hand) · V_T[hand] + σ(F|hand) · V_F[hand]
r_T[hand] += V_T[hand] - V[hand]
r_F[hand] += V_F[hand] - V[hand]
```

By Burch 2014 §3 sufficiency, over iterations: `avg_σ realized CFV[hand] ≥ opt_out[hand] − ε(iter)` with ε → 0.

### Error handling

**Panics (internal invariants, caller bugs):**
- `GadgetConfig.opt_out_oop.len() != num_private_hands(0)` (or IP analogue) → panic in `inject_gadget_layer`.
- Invalid `abstract_root` arena index → panic in `opt_out_at_subgame_root`.
- Empty `CbvTable` → panic (existing `BlueprintCbvOptOut` behavior).

**Graceful validation (user-actionable):**
- `subgame_starting_pot <= 0` → `Err` from `opt_out_at_subgame_root`.

**Invariants:**
- Boundary ordinals 0 and 1 are reserved for gadget terminals post-`inject_gadget_layer`. Caller must not overwrite them via `set_boundary_cfvs`. `DispatchingBoundaryEvaluator` returns static values from its own storage regardless — double-protection.
- `DispatchingBoundaryEvaluator::new` asserts inner evaluator's boundary count matches `num_boundaries - 2`.

### Unchanged

DCFR regret-discounting (α, β, γ); continuation-strategy handling; isomorphism / card-config paths; memory allocation phase.

---

## 5. Acceptance criteria

### Hard gates

1. **Tree construction.** `with_config_and_gadget` produces the expected layout; `allocate_memory(false)` succeeds; `node_to_boundary` assigns ordinals 0/1 to gadget terminals; existing subgame-internal arena indices resolve correctly post-injection.
2. **Opt-out placement.** On a hand-built fixture, `opt_out_at_subgame_root` returns expected per-hand bcfv values. `set_boundary_cfvs` at ordinals 0/1 stores those values correctly.
3. **Gadget-off parity.** With `ConstantOptOut(-∞)` for both players, final average strategies at all non-gadget nodes are bit-identical (or `1e-6` relative tolerance) to `with_config` (no-gadget).
4. **Safety invariant.** After N iterations with warmup = N/2: for every gadget player's hand `h` at their decision node, `avg_realized_CFV[h] ≥ opt_out[h] - 0.01`. This is the formal Burch 2014 §3 sufficiency check. **Ship-gating.**

### Soft gate (informational)

5. **`JhTh9h|…|7d` end-to-end.** `compare-solve` with Option A reports `subgame_exp`. Must not crash; must converge (no NaN, no divergence). Number recorded in a new `docs/progress/` entry; does not gate.

### Other ship conditions

- Full workspace `cargo build` + `cargo clippy` clean.
- Full test suite runs in < 1 minute (per CLAUDE.md rule).
- `docs/architecture.md` updated with gadget-at-root construction.
- `docs/training.md` updated if CLI changes.

---

## 6. Files touched

### New files

- `crates/range-solver/src/game/gadget.rs`
- `docs/plans/2026-04-24-option-a-deepstack-gadget-design.md` (this doc)

### Modified files

- `crates/range-solver/src/game/mod.rs` — export `with_config_and_gadget`, new `gadget` submodule
- `crates/range-solver/src/lib.rs` — re-export as needed
- `crates/range-solver/src/game/interpreter.rs` — possibly (if gadget nodes need interpreter-cache updates)
- `crates/tauri-app/src/gadget.rs` — add `BlueprintCbvOptOut::opt_out_at_subgame_root`, add `DispatchingBoundaryEvaluator`
- `crates/tauri-app/src/exploration.rs` or `crates/tauri-app/src/postflop.rs` — caller wiring
- `crates/poker-solver-trainer/src/bin/*` — `compare-solve` CLI routing

### Potentially touched

- `BoundaryEvaluator` trait (if ordinal param added)
- Explorer walk code (if existing walks need a gadget-layer filter)

---

## 7. Out of scope for MVP

**Explicitly out:**

- Beating the iter-10 no-gadget baseline (21k mbb). Goal (a) is safety-only.
- Continual re-solving (DeepStack's per-decision re-solve, [Moravčík 2017 Supplement §Continual Re-solving](https://static1.squarespace.com/static/58a75073e6f2e1c1d5b36630/t/58bed28de3df287015e43277/1488900766618/DeepStackSupplement.pdf)).
- Pattern 2 algorithmic gadget ([DeepStack-Leduc `cfrd_gadget.lua`](https://github.com/lifrordi/DeepStack-Leduc/blob/master/Source/Lookahead/cfrd_gadget.lua)).
- Per-combo / un-abstracted opt-outs (Libratus strategy 4b, [Brown & Sandholm 2017 §6](https://arxiv.org/abs/1705.02955)).
- cfvnet retrain with opt-out input channel (old `akg3` framing).
- Explorer UI changes.
- Best-response / exploitability reporting integration (the `compare-solve` harness may need a gadget-layer filter — plan-stage).

### Deferred engineering details (plan-stage decisions)

- **CBV lookup mechanism at decision-node subgame root.** (i) Parent turn-chance CBV approximation, or (ii) backward-induce turn-decision CBV via blueprint strategy. Per goal (a) either works for safety. Plan-stage.
- **`BoundaryEvaluator` trait signature.** Add ordinal param (ripples to 3 impls) vs per-ordinal evaluator instantiation. Plan-stage.
- **Arena-index strategy.** Prepend (shift +4) vs append (`root()` returns N). Append is safer; plan-stage call.
- **Nesting order** (IP outer vs OOP outer). Safety holds regardless; default IP outer per iter-14 data; make `outer_player` config-configurable for A/B.
- **CLI flag strategy.** Repurpose existing `--gadget` vs add `--gadget-mode {clamp, tree}`. Plan-stage.

### Follow-up beans (post-MVP)

- **Re-scope `akg3`** from "cfvnet retrain with opt-out input channel" to "improve opt-out tightness for Option A" (likely via strategy 4b un-abstracted CBVs). Or scrap `akg3` if Option A's safety is sufficient.
- **Retire or keep `GadgetEvaluator` post-clamp.** Delete once Option A ships and passes (4).
- **Un-abstracted per-combo opt-outs** (strategy 4b, Libratus approach).
- **Continual re-solving** for multi-spot / full-session flows.

---

## 8. Off-ramps

- **Test (4) safety invariant fails:** CFV propagation at gadget decision nodes is wrong. Debug via per-iter CFV logging at G_IP / G_OOP.
- **Trait change too invasive:** fall back to per-ordinal evaluator instantiation (existing `GadgetEvaluator` pattern).
- **Arena shifts break external callers:** switch to append strategy (gadget at N..N+3).
- **Decision-node CBV lookup complex:** use parent turn-chance CBV as MVP approximation; safety still holds.

---

## 9. Citations

- [Burch, Johanson, Bowling 2014. Solving Imperfect-Information Games Using Decomposition. AAAI-14.](https://webdocs.cs.ualberta.ca/~bowling/papers/14aaai-cfrd.pdf) — canonical CFR-D gadget construction; §3 sufficiency proof is the load-bearing safety theorem for this design.
- [Moravčík et al. 2017. DeepStack. Science 356(6337).](https://arxiv.org/abs/1701.01724) — [DeepStack Supplement](https://static1.squarespace.com/static/58a75073e6f2e1c1d5b36630/t/58bed28de3df287015e43277/1488900766618/DeepStackSupplement.pdf) §Continual Re-solving frames the gadget as a lookahead modification.
- [DeepStack-Leduc reference implementation](https://github.com/lifrordi/DeepStack-Leduc) — [`cfrd_gadget.lua`](https://github.com/lifrordi/DeepStack-Leduc/blob/master/Source/Lookahead/cfrd_gadget.lua) implements the gadget algorithmically rather than structurally; we chose structural but the DeepStack reference informed several decisions (terminal neutralization; composition over tree-level invasion).
- [Brown & Sandholm 2017. Safe and Nested Subgame Solving. NeurIPS 2017.](https://arxiv.org/abs/1705.02955) — Theorem 2 (`2·Δ` exploitability bound) is the failure mode that motivates Option A: bucketed CBV estimates make Δ large on narrow ranges.
- [Schmid et al. 2023. Student of Games. Science Advances.](https://arxiv.org/abs/2112.03178) — confirms modern practice: CVPN input is PBS = (public state, ranges); safe re-solving via auxiliary game constructed *inside* the search.

---

## 10. Predecessors

- `docs/plans/2026-04-24-option-a-deepstack-gadget-tree.md` — pre-brainstorm substrate; superseded by this doc.
- `docs/plans/2026-04-23-deepstack-gadget.md` — original MVP design (post-clamp path); superseded.
- `docs/progress/2026-04-22-subgame-exact-parity.md` — iteration history 1–14 proving the post-clamp failure mode.
- Beans: `poker_solver_rust-lay5` (completed post-clamp MVP), `poker_solver_rust-akg3` (to re-scope), `poker_solver_rust-u3rf` (this brainstorm).
