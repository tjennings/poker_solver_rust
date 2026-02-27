---
# poker_solver_rust-0kdp
title: Multi-SPR postflop model selection
status: in-progress
type: feature
created_at: 2026-02-27T17:51:58Z
updated_at: 2026-02-27T17:51:58Z
---

Build a PostflopAbstraction per configured SPR and have the preflop solver select the closest SPR model at each showdown terminal.

## Tasks
- [ ] Task 1: Add select_closest_spr helper with tests
- [ ] Task 2: Change PostflopState to hold multiple abstractions
- [ ] Task 3: Update PostflopAbstraction::build to accept explicit SPR
- [ ] Task 4: Update PostflopBundle for multi-SPR persistence
- [ ] Task 5: Update trainer to build and attach multiple SPR models
- [ ] Task 6: Update exploitability module for multi-SPR
- [ ] Task 7: Update explorer bundle loading for multi-SPR
- [ ] Task 8: Full integration test
- [ ] Task 9: Update documentation
