---
# poker_solver_rust-ruma
title: Run CFVNet paired-turn root-attribution sweep
status: completed
type: task
priority: high
created_at: 2026-05-04T14:40:42Z
updated_at: 2026-05-04T14:46:05Z
parent: poker_solver_rust-e90m
---

Use compare-solve --dump-boundary-cfvs to measure whether CFVNet root bet-pressure errors repeat across paired-turn Kx spots.\n\n## Tasks\n\n- [x] Define a small paired-turn validation grid around the known Kh failure.\n  Grid: same preflop/flop/action line with turns Kh, Kd, Kc as paired-top-card cases, plus Js as a non-paired control.\n- [ ] Run compare-solve CFVNet diagnostics on each spot.\n- [ ] Summarize root regret-input deltas and exact-vs-subgame strategy divergence.\n- [x] Decide whether the next repair is model/data calibration, runtime guard/fallback, or feature/input audit.


## 2026-05-04 Sweep Results

Ran compare-solve --dump-boundary-cfvs with checkpoint_epoch675.onnx on the same preflop/flop/action line, varying only the turn card.

CFVNet results:

| Turn | Root regret-input delta: 24bb | 55bb | All-in | Mean mass moved | Subgame exp |
| - | -: | -: | -: | -: | -: |
| Kh | +0.188941 | +0.571200 | +0.490248 | 0.505 | 20040.45 mbb/hand |
| Kd | +0.481991 | +0.899599 | +0.870718 | 0.597 | 17598.80 mbb/hand |
| Kc | +0.470183 | +0.879207 | +0.796605 | 0.532 | 18867.63 mbb/hand |
| Js | +0.512624 | +1.084511 | +1.011500 | 0.595 | 17981.31 mbb/hand |

The non-paired Js control also fails with the same positive large-bet/all-in root pressure, so this is broader than paired-top-card boards. Exact_subtree control on Js: final_exp 65.52 mbb/hand and mean mass moved 0.070, much smaller than CFVNet despite a few localized per-hand flips.

## Summary of Changes

Completed the first calibration sweep and narrowed the next repair away from paired-board-only retraining. The next concrete step is to audit CFVNet training-target/input encoding against runtime boundary evaluation, especially action-history/pot/remaining-stack normalization and whether the ONNX model checkpoint was trained for the same boundary target consumed by range-solver.
