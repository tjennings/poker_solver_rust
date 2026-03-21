---
# poker_solver_rust-vy5x
title: 'Audit: unify all solvers to use chip units (1BB = 2 chips)'
status: todo
type: task
created_at: 2026-03-21T12:48:22Z
updated_at: 2026-03-21T12:48:22Z
---

All three solvers should operate in chip units internally. Big Blinds is a display concern only.

## Problem

The blueprint solver (MCCFR) uses BB units (big_blind=1.0), while the range-solver and subgame solver use chips (1BB = 2 chips). This mismatch causes:
- CBV normalization bugs (BB values divided by chip values = 2x error)
- Action label conversion hacks (bb_scale parameter in v2_action_info)
- Pot/stack display requiring /2 conversion
- Constant risk of unit mismatch bugs at every boundary between systems

## Audit Scope

### 1. Blueprint Solver (MCCFR) — crates/core/src/blueprint_v2/
- [ ] game_tree.rs: tree builder uses big_blind=1.0 (BB units). Invested, pot, terminal values all in BB
- [ ] mccfr.rs: terminal_value uses BB units
- [ ] cbv_compute.rs: CBV backward induction produces BB values
- [ ] trainer.rs: training loop in BB units
- [ ] **Convert to chips at tree build time** (small_blind=1, big_blind=2, stack_depth stays as-is — it already represents chips, e.g. 100 chips = 50BB)

### 2. Subgame Solver (CfvSubgameSolver) — crates/core/src/blueprint_v2/cfv_subgame_solver.rs
- [ ] Currently receives chip-unit pot/stacks from PostflopConfig
- [ ] Showdown equity computed in chip units (OK)
- [ ] Depth boundary CBV values: currently requires BB→chip *2 hack
- [ ] If blueprint tree uses chips, CBV conversion becomes unnecessary

### 3. Range Solver — crates/range-solver/
- [ ] Already uses chips (SB=1, BB=2) via i32 pot/stack (reference implementation)
- [ ] PostflopConfig pot/stack are in chips
- [ ] No changes needed — this is the target unit system

### 4. Display Layer — frontend + v2_action_info
- [ ] Explorer.tsx: divides pot/stack by 2 for BB display
- [ ] PostflopExplorer.tsx: divides config.pot by 2 for BB display
- [ ] v2_action_info: bb_scale parameter (1.0 for blueprint, 0.5 for subgame)
- [ ] invested_offset subtracted from bet labels
- [ ] All hacks go away if everything uses chips — display just divides by 2 uniformly

### 5. Boundaries between systems
- [ ] get_preflop_ranges_core: returns pot/stack in BB, frontend multiplies by 2
- [ ] BlueprintCbvEvaluator: CBV in BB, half_pot in chips, requires *2 conversion
- [ ] populate_cbv_context: loads bucket files, constructs AllBuckets

## Recommendation

Convert the blueprint tree builder to use chips (small_blind=1, big_blind=2). stack_depth in config already represents chips (e.g. 100 = 50BB). This eliminates ALL conversion code at boundaries. Tree topology unchanged (uniform 2x scale). Requires retrain.

## Acceptance Criteria
- [ ] All three solvers use chip units
- [ ] No *2 or /2 conversion at any system boundary
- [ ] Display is the ONLY place that converts to BB (divide by 2)
- [ ] CBV evaluator needs no unit conversion hack
- [ ] Action labels need no bb_scale parameter
- [ ] get_preflop_ranges_core returns chips directly
