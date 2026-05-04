---
# poker_solver_rust-2yr4
title: Lock oracle-boundary repro outputs
status: completed
type: task
priority: high
created_at: 2026-05-04T01:09:15Z
updated_at: 2026-05-04T01:12:56Z
parent: poker_solver_rust-e90m
---

Re-run the canonical compare-solve spot with exact, exact_oracle, and exact_subtree. Record exact metrics, root strategy deltas, boundary count, boundary mode, and command lines so later fixes have a stable baseline.

## Summary of Changes

Locked the canonical compare-solve repro on commit b9633a81 for exact, exact_oracle, and exact_subtree using snapshot_0013 at 200 iterations. Added docs/research/oracle_boundary_contract_repro_2026-05-04.md with metrics, top root-strategy divergences, exact_oracle boundary ordinals, and trace file location. Key result: all-exact control matches exactly, while exact_oracle remains severely divergent at 2860.16 mbb/hand hybrid exploitability with 11 boundaries.
