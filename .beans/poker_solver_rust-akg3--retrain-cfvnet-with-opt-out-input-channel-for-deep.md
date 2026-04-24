---
# poker_solver_rust-akg3
title: Retrain cfvnet with opt-out input channel for DeepStack-proper gadget
status: todo
type: task
priority: normal
created_at: 2026-04-23T19:17:15Z
updated_at: 2026-04-24T02:17:19Z
blocked_by:
    - poker_solver_rust-lay5
---

## Context

The subgame-exact-parity investigation (see `docs/progress/2026-04-22-subgame-exact-parity.md`) converged on a Libratus-style safe re-solving gadget as MVP: clamp cfvnet output to `max(cfvnet_cfv, blueprint_cbv)` using pre-computed bucketed CBVs as opt-out values. See `docs/plans/2026-04-23-deepstack-gadget.md` (to be renamed "libratus-safe-subgame").

This is NOT the DeepStack-proper approach. DeepStack's gadget uses opt-out bounds as a runtime parameter propagated from the outer re-solve; their CFV network is TRAINED to consume these bounds as input and output CFVs consistent with the constraint. Our current cfvnet doesn't have an opt-out input channel.

## Goal

Extend cfvnet architecture + training pipeline to accept opt-out bounds as input, enabling proper DeepStack-style continual re-solving.

## Open questions to research

- [ ] What input encoding for opt-out bounds? Per-hand float channel? Bucketed? Flat vector?
- [ ] Training data: how do we generate subgame solves with varied opt-out bounds? Random sampling? Pluribus-style self-play with outer re-solve propagating values?
- [ ] Does retraining require changing the datagen pipeline in `crates/cfvnet/src/datagen/`?
- [ ] How much does loss increase with the extra input? (capacity concern)
- [ ] Do we need to change the inference call signature `NeuralBoundaryEvaluator::compute_cfvs_both` to accept opt-out bounds?

## Prerequisite work

- Libratus-style MVP (blueprint CBV clamp) must be implemented and validated first to establish a baseline.
- Need the current cfvnet training pipeline to be well-documented and reproducible.

## References

- Moravčík et al. 2017 (DeepStack). Science. Section on continual re-solving with CFV networks.
- Brown et al. 2020 (ReBeL). Formalises safe subgame solving for continuous-action games.
- `docs/plans/2026-04-23-deepstack-gadget.md` — MVP design doc.
- `docs/progress/2026-04-22-subgame-exact-parity.md` — iteration history.

## Related beans

- Blocked-by: The Libratus-style MVP (whichever bean tracks that work).



## Empirical Motivation (Post lay5)

Bean `lay5` (Libratus-style MVP) shipped and was exhaustively tested across 14 harness iterations on a 4-bet turn spot (docs/progress/2026-04-22-subgame-exact-parity.md). Final result: gadget reduces exploitability gap but does NOT close it — still 2× worse than no-gadget baseline (40k vs 21k mbb) after all correctness bugs were fixed.

## Why post-clamp fails for cfvnet on this spot

Per-boundary diagnostic (from the logging added in `diag/gadget-boundary-logging`) revealed the structural pattern:

- **Boundary 0 (pot=146, turn check-check → river chance):** 98% of hands clamped. Blueprint CBVs legitimately higher than cfvnet output for this narrow range.
- **Boundary 1-10 (higher pots, post-turn-betting):** 54-62% selective clamping.

**Root cause of persistent gap:** blueprint CBVs are computed via backward induction over blueprint's **range-reach distribution at each bucket**. cfvnet output reflects the **actual narrowed range at a specific boundary**. These diverge especially on filtered paths (like check-check, which selects weak hands out of the pot).

Static post-clamp can't know which opt-out to apply for which narrowed range — it just uses the blueprint-over-blueprint-range value, which overstates opt-out and drags cfvnet output up. DCFR over-commits chips defending the inflated opponent CFV → degenerate strategies.

## What akg3 needs to do

Train cfvnet to consume opt-out bounds as an **input channel** (per-hand or per-bucket), so its output is self-consistent with the given opt-out constraint. Continual re-solving (DeepStack-proper) then propagates constraint-consistent values through boundary-evaluator chains.

**Existing infrastructure that akg3 can reuse:**
- `OptOutProvider` trait (tauri-app/src/gadget.rs)
- `BlueprintCbvOptOut` with per-boundary pot normalisation (correct, tested)
- `GadgetEvaluator` wrapper (correct, tested — issue is cfvnet unaware of constraint, not wrapper)
- Full wiring: CLI flags, Tauri command, Settings UI
- Diagnostic logging + compare-solve harness + `scripts/trace_diff.py`

The retrain is the only remaining piece. Once cfvnet learns to respect opt-out bounds, the existing gadget stack will produce correct safe-subgame-solving output without further code changes.

## References

- Progress log: docs/progress/2026-04-22-subgame-exact-parity.md (iterations 1-14)
- Design doc: docs/plans/2026-04-23-deepstack-gadget.md
- Impl plan: docs/plans/2026-04-23-libratus-gadget-impl-plan.md
- Parent bean: poker_solver_rust-lay5 (completed)
