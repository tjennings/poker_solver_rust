---
# poker_solver_rust-822y
title: 'Phase 5: Explorer Integration for HU Games'
status: completed
type: feature
priority: low
tags:
    - gpu
    - explorer
created_at: 2026-03-15T04:14:47Z
updated_at: 2026-03-15T23:48:38Z
parent: poker_solver_rust-twez
---

Wire the GPU solver into the Tauri frontend for interactive heads-up resolving.

Tasks:
- [ ] Tauri command: gpu_resolve (game state + model stack → strategy)
- [ ] Explorer UI integration: display GPU-resolved strategies
- [ ] Live resolving: re-resolve at each decision point (<1s)
- [ ] Off-tree action handling: safe resolving for non-abstraction bets
- [ ] Model management: load/unload CFVNet model stack
- [ ] Performance target: <1 second per resolve in Explorer
