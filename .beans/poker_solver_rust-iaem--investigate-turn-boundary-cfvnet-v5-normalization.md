---
# poker_solver_rust-iaem
title: Investigate turn_boundary_cfvnet_v5 normalization mismatch
status: completed
type: bug
priority: high
created_at: 2026-05-14T04:08:04Z
updated_at: 2026-05-14T04:10:14Z
parent: poker_solver_rust-8e9f
---

Investigate why local_data/models/turn_boundary_cfvnet_v5, trained after CFVNet boundary normalization work, produces boundary CFV magnitudes incompatible with runtime contracts.

- [x] Pin v5 best.pt checkpoint metadata and config
- [x] Identify training data path and dataset manifest/target units
- [x] Inspect Rust dataset writer target units for turn_boundary river_net source
- [x] Inspect Python loader target conversion and loss target units
- [x] Compare raw model outputs against expected normalized ranges on records or diagnostic spot
- [x] Report likely root cause and next fix

## Investigation Findings

High-confidence root cause: the neural `compute_raw_cfvs_both` path returns conditional chip EV scale, but the range-solver raw boundary contract expects per-hand chip CFVs already integrated over the opponent's unnormalized reach. The neural evaluator normalizes OOP/IP inference ranges before ONNX input, then maps model output to chip EV and returns it directly. Range-solver then writes raw values directly with no `bcfv * half_pot / num_combinations * cfreach_adj` multiplier. This explains the 50x-400x magnitude inflation in compare-solve while correlations remain plausible.

Secondary contract inconsistency: Rust turn-boundary writers still store half-pot bcfv-style targets `(ev_chips - pot / 2) / (pot / 2)`. Rust `boundary_dataset.rs` treats `rec.cfvs` as direct targets. The Python loader, however, converts `cfvs * pot / (pot + stack)`, and Python manifest constants call the record normalization `chip_cfv_over_pot_plus_stack`. Those statements cannot all be true. v5's low validation loss is therefore not evidence that it matches the runtime direct normalized-EV contract.

Pinned v5 metadata: current `best.pt` is epoch 267 with val_huber=1.452532060284284e-05. The previous ONNX conversion used an earlier epoch 225 best snapshot because training continued after conversion.

Next fix: patch the neural raw-CFV adapter to multiply model conditional chip EVs by the same blocker-adjusted opponent reach factor divided by `num_combinations` that the legacy path applies, or temporarily disable the raw path for neural evaluators and use the legacy bcfv path until the raw adapter has parity tests. Then align the dataset manifest/Python loader/Rust docs around one explicit stored-target unit.
