---
# poker_solver_rust-8342
title: Re-test exact_subtree after oracle-boundary contract fixes
status: in-progress
type: task
priority: high
created_at: 2026-05-04T01:09:39Z
updated_at: 2026-05-04T05:16:43Z
parent: poker_solver_rust-e90m
---

After the oracle boundary matches exact on toy and canonical diagnostics, rerun exact_subtree on the canonical spot. Decide whether remaining divergence belongs to the exact_subtree evaluator or the shared depth-boundary contract.

## Current Repro\n\nTauri exact_subtree solve can produce a root strategy betting 24bb at ~99.9%, while the exact solve at the same spot checks the range 100%. Investigate boundary value injection, root strategy extraction, and compare-solve/Tauri parity.

## Repro Spot\n\nsb:2bb,bb:10bb,sb:22bb,bb:call|Ks8d3c|bb:check,sb:15bb,bb:call|Js\n
