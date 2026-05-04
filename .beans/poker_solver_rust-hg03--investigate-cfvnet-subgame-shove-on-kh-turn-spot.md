---
# poker_solver_rust-hg03
title: Investigate CFVNet subgame shove on Kh turn spot
status: in-progress
type: bug
priority: high
created_at: 2026-05-04T13:38:41Z
updated_at: 2026-05-04T13:43:18Z
parent: poker_solver_rust-e90m
---

Tauri subgame with cfvnet boundary reports an always-shove strategy while exact reports always-call on spot: sb:2bb,bb:10bb,sb:22bb,bb:call|Ks8d3c|bb:check,sb:15bb,bb:call|Kh.\n\n## Tasks\n\n- [x] Reproduce the spot in compare-solve with the local cfvnet model.\n- [x] Compare against exact and exact_oracle/exact_subtree controls where possible.\n- [ ] Determine whether this is Tauri wiring, compare-solve parity, CFVNet value calibration, or model/domain mismatch.\n- [ ] Fix or document the next concrete repair step.

## 2026-05-04 Cross-Check Results

`compare-solve` reproduces the Kh turn failure outside Tauri with `--river-boundary cfvnet` and `local_data/models/cfvnet_river_py_v2/checkpoint_epoch675.onnx`:

- Exact: final_exp 22.18 mbb/hand.
- CFVNet subgame: final_exp 20040.22 mbb/hand, mean mass moved 0.505, max mass moved 1.000, action-class bias Check -0.501 / Bet-Raise +0.264 / AllIn +0.237.

Controls on the same spot:

- `exact_oracle`: final_exp 4.94 mbb/hand, mean mass moved 0.004. The shared depth-boundary injection path can recover the exact root policy when supplied exact boundary values.
- `exact_subtree`: final_exp 243.43 mbb/hand, mean mass moved 0.032. This is still worse than oracle but does not reproduce the always-shove/all-bet collapse.

Conclusion so far: this isolates the reported Tauri behavior to CFVNet boundary values/model-domain behavior rather than Tauri wiring or the generic boundary injection path.
