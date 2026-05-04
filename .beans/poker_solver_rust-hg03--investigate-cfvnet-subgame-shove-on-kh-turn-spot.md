---
# poker_solver_rust-hg03
title: Investigate CFVNet subgame shove on Kh turn spot
status: in-progress
type: bug
priority: high
created_at: 2026-05-04T13:38:41Z
updated_at: 2026-05-04T14:34:11Z
parent: poker_solver_rust-e90m
---

Tauri subgame with cfvnet boundary reports an always-shove strategy while exact reports always-call on spot: sb:2bb,bb:10bb,sb:22bb,bb:call|Ks8d3c|bb:check,sb:15bb,bb:call|Kh.\n\n## Tasks\n\n- [x] Reproduce the spot in compare-solve with the local cfvnet model.\n- [x] Compare against exact and exact_oracle/exact_subtree controls where possible.\n- [x] Determine whether this is Tauri wiring, compare-solve parity, CFVNet value calibration, or model/domain mismatch.\n- [x] Fix or document the next concrete repair step.

## 2026-05-04 Cross-Check Results

`compare-solve` reproduces the Kh turn failure outside Tauri with `--river-boundary cfvnet` and `local_data/models/cfvnet_river_py_v2/checkpoint_epoch675.onnx`:

- Exact: final_exp 22.18 mbb/hand.
- CFVNet subgame: final_exp 20040.22 mbb/hand, mean mass moved 0.505, max mass moved 1.000, action-class bias Check -0.501 / Bet-Raise +0.264 / AllIn +0.237.

Controls on the same spot:

- `exact_oracle`: final_exp 4.94 mbb/hand, mean mass moved 0.004. The shared depth-boundary injection path can recover the exact root policy when supplied exact boundary values.
- `exact_subtree`: final_exp 243.43 mbb/hand, mean mass moved 0.032. This is still worse than oracle but does not reproduce the always-shove/all-bet collapse.

Conclusion so far: this isolates the reported Tauri behavior to CFVNet boundary values/model-domain behavior rather than Tauri wiring or the generic boundary injection path.

## 2026-05-04 CFVNet Boundary Diagnostic

Added `compare-solve --dump-boundary-cfvs` diagnostics that compare the injected boundary contribution against `exact_oracle` raw CFVs and print an `exact_subtree` raw-control comparison before the subgame solve.

Kh spot result with `checkpoint_epoch675.onnx`:

- CFVNet is not sign-flipped: aggregate candidate-vs-oracle correlation is positive (OOP 0.8448, IP 0.8739).
- CFVNet is not grossly unit-scaled: aggregate magnitude ratio is OOP 0.719, IP 0.683.
- CFVNet is materially noisy/damped versus oracle: aggregate mean_abs is OOP 0.397778, IP 0.346796, with boundary-level max deltas up to roughly 2.48 raw CFV.
- Exact-subtree raw control is closer overall (OOP mean_abs 0.267354/corr 0.9727, IP mean_abs 0.146042/corr 0.9859) but still has some noisy OOP high-pot boundaries.

Conclusion: this is not Tauri wiring, compare-solve parity, player orientation, or a simple scalar unit bug. The next concrete repair path is to quantify which boundary/value errors flip the root regrets, then decide between model retraining/domain coverage for paired-turn Kx spots and a safer boundary handoff/dampening strategy.


## 2026-05-04 Root Attribution Diagnostic

Added compare-solve root boundary attribution under --dump-boundary-cfvs. The diagnostic swaps the same seeded subgame between the candidate per-boundary evaluator and an exact_oracle evaluator, preserving the boundary reach snapshot, then prints root action CFVs and immediate regret-input pressure.

Kh spot result with checkpoint_epoch675.onnx:

• Max root action CFV gap: 2.472203 chips at 55bb 8c9c.
• Max root regret-input gap: 1.778852 chips at 55bb 7c7d.
• Reach-weighted root regret-input deltas versus exact oracle: Check -0.041571, 24bb +0.188941, 55bb +0.571200, All-in +0.490248.

Conclusion: CFVNet boundary errors are not merely noisy in aggregate; at the root they create positive regret pressure for the large bet and all-in branches where exact_oracle says those actions are negative or barely positive. The next repair should target CFVNet calibration/domain coverage for paired-turn Kx states, or gate/dampen CFVNet boundaries when root attribution shows large positive bet-pressure deltas.
