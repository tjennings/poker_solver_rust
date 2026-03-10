---
# poker_solver_rust-58h4
title: Real-Time Solving Agent
status: completed
type: feature
priority: normal
created_at: 2026-03-10T03:20:45Z
updated_at: 2026-03-10T04:18:11Z
---

Pluribus-inspired play system: blueprint + real-time subgame solving for postflop decisions. See docs/plans/2026-03-09-real-time-solving-agent-design.md


## Summary of Changes

11 commits merged via `feat/real-time-solver`:
- RangeNarrower: 1326-combo opponent range tracking with card removal
- Bucket-to-combo expansion: 169 canonical hands ↔ 1326 combos bridge
- CbvTable: precomputed continuation values with bincode serialization
- CBV precomputation: backward induction during training snapshots
- SubgameCfrSolver CBV integration: `with_cbv_table` constructor
- Full-depth solve bridge: wraps range-solver crate for exact solving
- Hybrid solver dispatch: per-street combo thresholds (flop/turn configurable, river always full)
- RealTimeSolvingAgent: Agent trait impl with preflop blueprint + postflop solving
- CLI integration: `solver:<bundle_path>` agent source pattern
- Integration test: full pipeline verification

V1 TODOs remaining:
- Per-action Bayesian range narrowing (currently uniform + card removal)
- Depth-limited SubgameCfrSolver path (currently falls back to check/call)
- Hero range refinement
