# Per-Iteration Boundary Re-evaluation Design

**Date**: 2026-04-02
**Status**: Approved

## Problem

The current turn datagen pipeline pre-computes boundary CFVs once via the river neural net before solving. As DCFR iterates, the strategy (and thus reach probabilities at boundaries) evolves, but the boundary values remain stale. This means the solver is optimizing against boundary values computed from initial (often uniform) ranges rather than the current strategy's ranges.

ReBeL and DeepStack both re-evaluate leaf values every iteration using the current reaches. This produces more accurate solutions because boundary CFVs reflect the actual strategy at each point in convergence.

## Solution

Replace the current 4-stage pipeline (deal → gpu → solve → write) with a **lockstep ring buffer** pipeline where K games cycle through GPU evaluation and DCFR iterations together.

### Architecture

```
[Deal Buffer] → [Active Pool: K games]
                       ↓
                 ┌─────────────────────────────┐
                 │  Per-iteration lockstep loop │
                 │                              │
                 │  1. GPU batch eval boundaries │  ← 1 thread, 1 forward pass
                 │  2. Set boundary CFVs on all K│
                 │  3. solve_step all K parallel │  ← N solver threads
                 │  4. Extract boundary_reach    │
                 │  5. Check: iteration == N?    │
                 │     yes → graduate to write   │
                 │           inject from deal    │
                 │     no  → loop back to 1      │
                 └─────────────────────────────┘
                       ↓
                 [Write Queue] → Writer thread
```

### Lockstep Loop Detail

One orchestrator thread manages the loop:

1. **GPU batch eval** — Read `boundary_reach` from all K games (opponent reaches from last iteration; `initial_weights` for iteration 0). Build inputs for all games' boundaries into one tensor via existing `build_game_inputs`. Single `model.forward()`. Scatter results back via `set_boundary_cfvs`.

2. **Parallel solve** — N solver threads each grab games from the pool. Each calls `solve_step(game, iteration)` with `force_sequential = true`. Boundary reaches get updated during traversal (overwrite every visit, not cache-once). Increment iteration counter. Uses `std::thread::scope` — no rayon, no pool.

3. **Graduate + inject** — Any game at `solver_iterations` gets exploitability computed, results extracted, sent to write queue. New game from deal buffer replaces it. Pool shrinks when deal buffer empties.

4. **Termination** — When all games have graduated and deal buffer is empty, close write queue.

### Key Design Decisions

- **Lockstep, not async**: All K games advance together. Maximizes GPU batch size (all K games every iteration). Solver times are uniform with `force_sequential` and similar tree sizes.
- **Fixed iteration count**: Games graduate after `solver_iterations` iterations. Exploitability computed once at graduation, not during solving. Simpler and avoids expensive per-iteration exploitability checks.
- **No rayon**: Solver threads use `std::thread::scope` for the parallel `solve_step` phase. No rayon pools, no spin-wait, no nested parallelism.
- **K sizing**: Default 64. Each depth-limited turn game is ~1-5 MB, so 64 games ≈ 320 MB. GPU batch size: 64 games × ~10 boundaries × 2 players × 48 rivers ≈ 61K rows — good GPU utilization.

### Convergence

DCFR converges with per-iteration boundary re-evaluation (ReBeL Theorem 2). Leaf value error contributes linearly to exploitability; convergence rate O(1/sqrt(T)) is unaffected. The neural net is fixed during solving — only the reach inputs change per iteration.

## Changes Required

### range-solver crate

1. **`evaluation.rs`**: Change `boundary_reach` from cache-once to overwrite-every-visit. Remove the `if guard.is_empty()` check at line ~97 so reaches are updated every iteration.

2. **`PostFlopGame`**: Add `clear_boundary_cfvs()` method to reset cached boundary CFVs between iterations, forcing them to be re-read from the values set by the GPU phase.

### cfvnet crate

3. **`turn_generate.rs`**: New pipeline function (or replace existing wgpu path) implementing the lockstep ring buffer:
   - Deal gen thread fills a channel (reuse existing Stage 1)
   - Orchestrator thread runs the lockstep loop:
     - Maintains `Vec<ActiveGame>` with game + iteration counter
     - GPU phase: `build_game_inputs` for all active games → `model.forward()` → `set_boundary_cfvs`
     - Solve phase: `std::thread::scope` with N threads, each solving a slice of active games
     - Graduate phase: extract results, send to write queue, inject new games
   - Writer thread (reuse existing Stage 4)

4. **Config**: Add `active_pool_size` option (default 64) to `DatagenConfig`.

### No changes to

- `solve_step`, `solve_recursive` — they already read boundary CFVs from the game struct
- `LeafEvaluator` trait
- `build_game_inputs`, `decode_boundary_cfvs` — reused as-is for the GPU phase
- Model/training code
- CLI (pool size via config)

## Files Changed

- `crates/range-solver/src/game/evaluation.rs` — boundary_reach overwrite + clear_boundary_cfvs
- `crates/range-solver/src/game/mod.rs` — clear_boundary_cfvs method
- `crates/cfvnet/src/datagen/turn_generate.rs` — lockstep ring buffer pipeline
- `crates/cfvnet/src/config.rs` — active_pool_size config option
