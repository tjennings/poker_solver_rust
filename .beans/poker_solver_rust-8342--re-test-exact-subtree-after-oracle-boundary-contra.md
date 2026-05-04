---
# poker_solver_rust-8342
title: Re-test exact_subtree after oracle-boundary contract fixes
status: completed
type: task
priority: high
created_at: 2026-05-04T01:09:39Z
updated_at: 2026-05-04T05:29:54Z
parent: poker_solver_rust-e90m
---

After the oracle boundary matches exact on toy and canonical diagnostics, rerun exact_subtree on the canonical spot. Decide whether remaining divergence belongs to the exact_subtree evaluator or the shared depth-boundary contract.

## Current Repro\n\nTauri exact_subtree solve can produce a root strategy betting 24bb at ~99.9%, while the exact solve at the same spot checks the range 100%. Investigate boundary value injection, root strategy extraction, and compare-solve/Tauri parity.

## Repro Spot\n\nsb:2bb,bb:10bb,sb:22bb,bb:call|Ks8d3c|bb:check,sb:15bb,bb:call|Js\n

## Summary of Changes

- Reproduced the Ks8d3c Js exact_subtree divergence reported from Tauri via compare-solve.
- Fixed SubtreeExactEvaluator to provide raw per-combination CFVs through compute_raw_cfvs_both, avoiding the legacy bcfv normalization path for range-solver boundary injection.
- Added a regression test proving exact_subtree exposes raw finite per-combo boundary values distinct from legacy bcfv values.
- Validation on the repro spot improved exact_subtree subgame exploitability from 388.36 mbb/hand to 65.52 mbb/hand, while exact_oracle remained close at 3.68 mbb/hand.
- Conclusion: the shared boundary injection path can match exact when fed exact continuation values; the remaining exact_subtree root-policy divergence comes from independently resolving river subtrees rather than from the raw CFV contract.

## Validation

- cargo test -p poker-solver-tauri exact_subtree::tests -- --nocapture
- cargo test -p poker-solver-trainer compare_solve
- cargo test
