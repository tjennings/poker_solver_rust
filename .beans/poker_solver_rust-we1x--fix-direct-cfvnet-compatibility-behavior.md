---
# poker_solver_rust-we1x
title: Fix Direct CFVNet compatibility behavior
status: in-progress
type: bug
priority: critical
created_at: 2026-05-08T20:47:13Z
updated_at: 2026-05-09T02:36:25Z
---

Direct CFVNet compatibility conversion makes subgame behavior worse: subgame bets/shoves 100% while exact checks 100%. Audit model target units, bcfv conversion, sign/player orientation, and boundary evaluator handoff; patch only once the solver convention is verified.

- [x] Verify working tree clean and test baseline
- [x] Trace solver bcfv convention and current BoundaryNet target
- [x] Identify why legacy conversion causes all-bet/all-shove policy
- [x] Patch compatibility mode or roll back unsafe mapping
- [x] Run focused and full verification
- [x] Commit corrected unit-conversion patch
- [x] Compare exact vs Direct boundary CFVs for reported policy-diff spots
- [ ] Decide whether to gate current checkpoint, retrain, or add a stronger runtime diagnostic

## Notes

The current Python BoundaryNet trainer encodes targets as `bcfv * pot / (pot + effective_stack)`. The previous compatibility patch treated outputs as `chip_ev / (pot + effective_stack)` and applied `2 * output * (pot + effective_stack) / pot - 1`, which injects a false `-1` offset. The correct inverse for this checkpoint is `output * (pot + effective_stack) / pot`.

Verification: focused `cfvnet` conversion test passed, focused Tauri solver-log test passed, `git diff --check` passed, `beans check` passed, and warm full `cargo test --quiet` passed in 51.96s.

Residual: after the corrected conversion, the reported behavior regressed from 100% bet/shove back to the earlier 50/50 check/bet split while exact remains 100% check. That remaining mismatch is a boundary-value audit/model-quality problem, not explained by the removed affine offset.

## Diagnostic Spot

Reported residual policy mismatch: subgame too passive while exact is more aggressive on `sb:2bb,bb:10bb,sb:22bb,bb:call|9s8h7d|bb:check,sb:check|6d`. Use this as a value-level audit case after the corrected scaled-bcfv conversion. Since the solve root is turn, the turn-boundary Direct CFVNet checkpoint should be configured as a river boundary (`--river-boundary cfvnet`) so boundary boards are 4-card turn boards.

## Audit Findings

Local checkpoint hashes match the handoff model:

- `best.onnx`: `afd43193ae0048aea682ead4095f52d34facada2e15dd45773076ce00491441b`
- `best.onnx.data`: `9ad3c1c595e6623c5aeda36e5f9a78bf85f9760c181604d7990eb7a9e47557f7`

Reproduced the reported turn spot with `local_data/blueprints/1k_100bb_brdcfr_v2`, `--river-boundary cfvnet`, `--river-model local_data/models/turn_boundary_v2/best.onnx`, and both `direct_normalized_legacy` and native `direct`.

The boundary handoff/integration path is exonerated by controls:

- `exact_oracle` boundary source matches full exact at root with mean mass moved `0.000`.
- `exact_subtree` raw-control boundary CFVs track oracle with aggregate correlations about `0.9965` OOP and `0.9946` IP.

The Direct checkpoint itself is value-wrong on this spot:

- `direct_normalized_legacy`: aggregate candidate-vs-oracle mean_abs about `0.480` OOP and `1.076` IP, correlations about `0.123` OOP and `0.183` IP.
- native `direct`: lower magnitude but still bad; aggregate mean_abs about `0.337` OOP and `0.733` IP, same low correlations.
- worst boundaries predict strongly positive CFVs for combos where exact oracle is negative, especially `3x7x` style board-interaction combos.

Conclusion: this residual mismatch is not a scalar unit conversion issue. Current checkpoint should not be trusted for subgame policy decisions on this turn class without retraining or a broader training/distribution audit.
