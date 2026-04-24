# Option A: DeepStack-style gadget as CFR tree modification

**Status:** Pre-brainstorm summary. Next session should start with `hex:brainstorming` on this doc.

**Parent bean:** `poker_solver_rust-akg3` (to be refined). Replaces the old "retrain cfvnet with opt-out input channel" framing, which had no published precedent per the Apr 24 literature review.

---

## Problem being solved

We built a Libratus-style **post-clamp** safe-resolving gadget (bean `lay5`, completed Apr 23). Infrastructure is correct: `OptOutProvider` trait, `BlueprintCbvOptOut`, `GadgetEvaluator` wrapper, per-boundary pot conversion, CLI + Tauri + UI wiring.

On the `JhTh9h|…|7d` 4-bet test spot, the gadget reduces exploitability from broken (218k mbb pre-fixes) to 40k mbb, but still **2× worse than no-gadget baseline (21k mbb)**. DCFR converges by iter ~20 to a degenerate fixed point — all strategies collapse because 98% of hands on the narrow check-check boundary get clamped to blueprint's opt-out floor.

The clamp is architecturally wrong because blueprint CBVs are computed over the **blueprint's range-reach distribution** but applied to the **subgame's narrowed range**. Brown & Sandholm 2017 Theorem 2 proves exploitability scales with `Δ = |estimated_CBV − true_CBV|` — exactly what our iter-14 data shows.

## Option A in one paragraph

**Move the gadget from a post-clamp wrapper on the boundary evaluator to a tree-structure modification inside the CFR lookahead.** At the root of each subgame, inject a terminal "opt-out" branch for the opponent: before entering the normal action tree, opponent has a per-infoset choice between `TERMINATE` (take the pre-computed opt-out value and collect pot share) or `FOLLOW` (enter the real subgame). Regret matching at the T/F node forces any strategy the solver produces to be at least as good for the opponent as the opt-out — that's the actual safety proof. cfvnet stays unchanged and is still queried at round boundaries as before.

## Citations (per new "recommendations require citations" rule)

- **Moravčík et al. 2017. DeepStack.** *Science* 356(6337), 508–513. [arXiv:1701.01724](https://arxiv.org/abs/1701.01724). Network input is `(pot, cards, range1, range2)` with 1000-bucket range encoding — **no opt-out input**. Gadget implemented as T/F node inside the CFR lookahead tree per the [Supplement](https://static1.squarespace.com/static/58a75073e6f2e1c1d5b36630/t/58bed28de3df287015e43277/1488900766618/DeepStackSupplement.pdf), "Continual Re-solving" section.
- **Burch, Johanson, Bowling 2014. Solving Imperfect-Information Games Using Decomposition.** AAAI-14. [arXiv:1303.4441](https://arxiv.org/abs/1303.4441). The canonical CFR-D gadget construction. Opponent gets T (terminate with opt-out bound) vs F (follow into subgame). Proof of sufficiency: opt-out bound must be an upper bound on what the opponent could have achieved had they played optimally to reach the subgame.
- **Brown & Sandholm 2017. Safe and Nested Subgame Solving.** NeurIPS 2017 Best Paper. [arXiv:1705.02955](https://arxiv.org/abs/1705.02955). Section 6 + Theorem 2: exploitability gap is `2·Δ` where Δ is per-hand CBV estimation error. **Direct warning about our failure mode:** "Unlike previous sections, exploitability might be higher than the blueprint when using this method; the solution quality ultimately depends on the accuracy of the estimates used." Libratus uses per-combo CBVs (fine-grained abstraction), not per-bucket, which is why their post-clamp works.
- **Schmid et al. 2023. Student of Games.** *Science Advances*. [arXiv:2112.03178](https://arxiv.org/abs/2112.03178). Confirms modern practice: CVPN input is PBS = (public state, ranges); safe re-solving via auxiliary game constructed *inside* the CFR search, not as NN wrapper.

## What exists now (reusable for Option A)

Branch `main` contains:

- `crates/tauri-app/src/gadget.rs` — `OptOutProvider` trait, `BlueprintCbvOptOut`, `ConstantOptOut`. **Keep** — per-hand opt-out computation is the same regardless of how the gadget is integrated. Just don't call `GadgetEvaluator` on the result; feed opt-outs into the tree construction instead.
- `crates/core/src/blueprint_v2/cbv.rs` — `CbvTable::build_node_to_ordinal_map` + `GameTree::chance_descendants` + `GameTree::pot_at_node`. All still useful.
- Per-boundary pot conversion (bug fix from Apr 24). Still correct.
- CLI flags, Tauri command, Settings UI checkbox (`--gadget`, `enable_gadget`, etc.). Can be repointed to Option A instead of the post-clamp once A exists.
- `docs/progress/2026-04-22-subgame-exact-parity.md` — full iteration history (iters 1–14) including diagnostic outputs proving the post-clamp failure mode.

## What needs to change for Option A

This is the brainstorming substrate — exact surfaces to be designed in the next session:

1. **Tree extension in `range-solver`.** At the root of a subgame (the current decision node), insert a new "gadget node" of type Player 1 (opponent) with two actions per opponent infoset: `OptOut(opponent_ordinal)` → terminal with per-combo payoff from the `OptOutProvider`, and `Enter` → the normal subgame root. This is invasive — touches `ActionTree`, `PostFlopNode`, terminal evaluation. Alternative: implement the gadget as a special `InterpreterState` at the root without changing node types, by locking strategies selectively per-iter.
2. **Terminal payoff wiring.** Gadget `OptOut` terminals need per-hand payoffs, not the single pot-share value that regular `TerminalKind::Fold` supports. Either extend `TerminalKind` with a new variant (e.g. `PerHandPayoff(ordinal)`) that pulls from a table, or compute it via a boundary-evaluator-like trait call at terminal-evaluation time.
3. **Per-hand opt-out magnitudes.** Currently `BlueprintCbvOptOut` produces per-hand floors in bcfv units via `from_cbv_context`. For Option A we need them in **chip payoff units** at the subgame's current pot, matching how regular terminals report payoffs. Conversion is: `chip_payoff[h] = opt_out_bcfv[h] * current_half_pot_chips + current_half_pot_chips` (inverse of `chip_cfv_to_bcfv`).
4. **Accuracy of opt-out bound.** Per Theorem 2, the bound has to actually bound. Current `BlueprintCbvOptOut` uses bucketed CBVs which *over*state the value on narrow ranges (this is the iter-14 failure mode). Options:
   - (a) Accept the looseness; correctness holds even with slack, just costs exploitability.
   - (b) Use un-abstracted blueprint CBVs (Libratus approach) — requires new compute pipeline.
   - (c) Use cfvnet's own output as the opt-out, computed once at the subgame root before solving, then hold fixed during solving. Circular but self-consistent.
5. **Integration with `SubtreeExactEvaluator` and `NeuralBoundaryEvaluator`.** Both currently plug in via `BoundaryEvaluator` trait at depth-limit boundaries. The gadget tree modification is at the **subgame root**, not at boundaries — so these evaluators don't change. Verify they still work under the new tree shape.
6. **Testing strategy.** Unit test: gadget with `ConstantOptOut(very_negative)` should be a no-op (T branch dominated). Gadget with `ConstantOptOut(+infinity)` should force opponent to always terminate — easy to verify per-hand. Integration: iter 15 harness run on the same `JhTh9h|…|7d` spot, compare to iter 10 (21k mbb baseline).

## Open questions to brainstorm

- **How invasive is the tree extension?** Minimal-change path vs clean extension of `TerminalKind`. The less we touch range-solver internals, the lower the risk; but doing it cleanly makes future work easier.
- **Which opt-out accuracy strategy (4a/b/c)?** (a) is cheapest, works, but exploitability gap persists. (b) aligns with Libratus but is a separate compute project. (c) is novel — needs thinking.
- **Continual re-solving semantics.** DeepStack re-solves every decision node. Our Tauri solver is per-spot. Do we need to re-solve at every decision (expensive) or only at subgame entry (cheap but not continual)?
- **How does the gadget interact with the existing depth-limit / cfvnet boundary?** The gadget is at the subgame root; cfvnet is at depth-limit boundaries. Two separate tree modifications. Are there interactions?
- **Per-combo vs per-bucket opt-outs.** DeepStack uses per-bucket (1000-bucket CBV input/output on the NN). Libratus uses per-combo. Our existing `BlueprintCbvOptOut` produces per-combo values by mapping each combo to its bucket then reading that bucket's CBV. This is a hybrid that inherits the accuracy problems of both.

## Commands to run for baseline context (next session)

```bash
# Resume on main, which has all infrastructure + all documented failure-mode evidence
git checkout main
git log --oneline -15   # review recent commits

# Read the validated artifacts in this order:
open docs/plans/2026-04-24-option-a-deepstack-gadget-tree.md    # this doc
open docs/plans/2026-04-23-deepstack-gadget.md                   # original design
open docs/progress/2026-04-22-subgame-exact-parity.md            # iter history
cat .beans/poker_solver_rust-lay5--*.md                          # MVP summary
cat .beans/poker_solver_rust-akg3--*.md                          # needs refinement

# Reproduce the failure mode if desired (~5 min wall time):
./target/release/poker-solver-trainer compare-solve \
    --bundle ./local_data/blueprints/1k_100bb_brdcfr_v2 \
    --snapshot snapshot_0013 \
    --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
    --river-boundary cfvnet \
    --river-model ./local_data/models/cfvnet_river_py_v2/checkpoint_epoch675.onnx \
    --gadget --iters 40 --tolerance 0.001 2>&1 | grep -A 4 "gadget.*call="
```

Expected output: reaches change per iter; clamp rate on boundary 0 stays at 98% throughout.
