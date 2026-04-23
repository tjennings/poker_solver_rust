---
# poker_solver_rust-akg3
title: Retrain cfvnet with opt-out input channel for DeepStack-proper gadget
status: todo
type: task
priority: normal
created_at: 2026-04-23T19:17:15Z
updated_at: 2026-04-23T19:33:45Z
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
