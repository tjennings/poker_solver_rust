---
# poker_solver_rust-2dsn
title: Audit raw Blueprint MP strategy versus TUI display
status: in-progress
type: bug
priority: high
created_at: 2026-05-20T13:27:27Z
updated_at: 2026-05-20T15:42:14Z
parent: poker_solver_rust-kiqt
---

Determine whether suspect preflop folds are present in raw sparse storage or introduced by TUI scenario resolution/rendering.

## Subtasks

- [x] Pick suspect hands: A2s-A5s, ATs-AQs, K9s, 22, 72o
- [x] Pick suspect spots: UTG root, BTN unopened, BTN versus CO open, BB versus SB open
- [ ] Dump average strategy, current regret-matched strategy, regrets, and strategy sums for each hand/spot
- [ ] Compare raw action probabilities to TUI matrix colors and labels
- [ ] Classify failure as storage/training bug, path-resolution bug, or rendering/display bug
- [x] Add a regression test or diagnostic command if raw/TUI disagree

## Implementation Notes

Added a shared lazy strategy row query for TUI cells and diagnostics so lazy MP grids consume the same raw sparse-storage lookup path that audits can call. The row includes action labels, bucket, sparse key, regrets, strategy sums, current strategy, average strategy, and whether the average strategy came from a present row, missing-row uniform fallback, or present zero-sum uniform fallback.

Added live lazy-sparse TUI probe lines for the configured scenario set. The probes are driven directly by the shared LazyStrategyRow sparse-storage lookup, so the displayed probe state now reports the same row source as the hand grid: present (P), missing uniform fallback (M), or zero-sum uniform fallback (Z). The default probe hands cover the selected suited Ax/K9s/22/72o set, and the list is configurable with tui.strategy_probe_hands.

Updated sample_configurations/blueprint_mp_6max_250f_100t_20r.yaml to make the live probe setup explicit: configured the suspect hand list and added BTN vs CO plus BB vs SB response scenarios alongside the existing opener scenarios.

## Probe Observation

Live TUI probes reported all sampled UTG/HJ/CO opener hands as present sparse rows (P), including suspicious folds such as UTG A4s:P:F91, UTG K9s:P:F98, HJ A3s:P:F99, HJ A5s:P:F81, HJ AJs:P:F53, and CO A5s:P:F81. This rules out missing-row uniform fallback for those spots and makes the current failure class more likely to be stored strategy/training/keying/action-legality than pure TUI rendering drift.

## Effective Sample Correction

The 250f/100t/20r run uses sampled flop with exact turn/river continuation, giving roughly 300 * 1337 full-board deals per meta-iteration. That makes the suspect preflop probe output much less dismissible as merely early-training noise: exact continuation substantially lowers downstream value variance for each evaluated preflop branch. Remaining uncertainty should focus on whether suspect preflop actions are being evaluated with meaningful strategy-sum mass/regret evidence or starved/skipped by traversal pruning/keying/action-legality behavior.

## Implementation Notes

Extended live lazy-sparse TUI probe cells to show dominant average action, dominant current regret-matched action, and total strategy-sum mass from the same LazyStrategyRow. Probe cells now use the format hand:state:a<avg>/c<current>/s<mass>, which should distinguish normal DCFR average-lag progression from current-strategy fold collapse or low-mass starvation.

## Probe Observation

Current-vs-average probe output shows a mixed picture with multi-million strategy-sum mass per opener row. Some suspicious hands look like plausible DCFR average lag, e.g. HJ A3s:P:aF94/cB81/s3m. Others are current-strategy folds with substantial mass, e.g. UTG A2s/A3s/A4s current F100 with s3m, HJ A5s current F100 with s2m, and CO A2s/A4s/A5s current F100 with s2m, while nearby hands like CO A3s, CO K9s, and CO 22 are current raises. This narrows the next audit to raw row internals: action labels, bucket/key, regrets, and per-action strategy sums for adjacent suited Ax hands and comparison hands.
