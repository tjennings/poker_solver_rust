# Blueprint Postflop Explorer — Immediate Strategy Display

**Date:** 2026-03-22

## Problem

After dealing a flop for a blueprint strategy, the user sees only a SOLVE button and must wait for the subgame solver to complete before seeing any strategy or navigating the action tree. The blueprint strategy is available but never displayed.

## Design

### Core Idea

Show the blueprint's own postflop strategy immediately after a flop is dealt, making it navigable before (and without) running the subgame solver.

### Backend Changes

**Thread `PostflopState` into `get_strategy_matrix_v2`**

The `AllBuckets` needed for postflop bucket lookup is already loaded in `PostflopState.cbv_context` (via `populate_cbv_context`). Pass a reference to `PostflopState` (or just `CbvContext`) into `get_strategy_matrix_v2` so the postflop branch can access it.

**Fix the postflop branch in `get_strategy_matrix_v2` (exploration.rs ~line 1030)**

Replace the uniform-distribution stub with real bucket lookup:

1. Get `AllBuckets` from `CbvContext`
2. Parse the board from the position
3. For each 13×13 cell, enumerate unblocked combos
4. For each combo: `AllBuckets::get_bucket(street, hole_cards, board)` → `strategy.get_action_probs(decision_idx, bucket)`
5. Average probabilities across combos within each cell
6. Return `StrategyMatrix` as today (no new types)

### Frontend Changes (PostflopExplorer.tsx)

**Flop dealt → blueprint strategy shown immediately**

- Disable cache check (for now)
- After flop is dealt, call `get_strategy_matrix` with board + empty action history
- Render blueprint action cards to the right of the SOLVE button
- Blueprint navigation: clicking an action card calls `get_strategy_matrix` with updated action history

**SOLVE clicked → transition to solver**

- Call `postflop_solve_street` as today
- Matrix data replaced by live solver output (same action cards — blueprint and solver trees are identical for now)
- Poll `postflop_get_progress` every 500ms — matrix updates live
- User can navigate the solver's action tree while it converges
- Street progression blocked until solve completes
- Solve complete: SOLVE button → green "Solved" (disabled)

**Next street after solve**

- Solved ranges propagate forward via existing `close_street` mechanism
- Next street shows blueprint strategy filtered by previous street's solved ranges
- SOLVE button resets to clickable state for the new street

### Constraints & Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Strategy source indicator | SOLVE button → green "Solved" when solved | Clean, no extra UI elements |
| Blueprint vs solver tree structure | Identical for now | Same bet sizes; different trees is future work |
| Cache | Disabled for now | Simplification |
| Blueprint replaced by solve | Yes, no toggle | Solved strategy supersedes blueprint |
| Multi-street blueprint | One street at a time | Blueprint shows current street; deeper streets require solving or progressing |
| Street progression during solve | Blocked | Must wait for solve to complete before propagating ranges |
| Blueprint without solve | Can progress to next street using blueprint ranges | Blueprint range propagation when no solve is run |

### Not In Scope

- Cache integration
- Different blueprint vs solver tree structures
- Side-by-side or toggle between blueprint/solved views
- Per-combo detail drill-down in blueprint cells
