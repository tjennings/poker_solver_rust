# Range-Solver Subgame Solving for GameSession — Design

**Date:** 2026-03-25

## Problem

`GameSession::solve()` is a stub. The old `CfvSubgameSolver` (now deprecated) uses bucket abstraction, limiting hand resolution. The range-solver `PostFlopGame` with `depth_limit` solves at exact 1326-combo granularity — the correct approach for real-time subgame solving (Libratus/Pluribus).

## What We're Building

Wire `PostFlopGame` with depth-limited boundary evaluation into `GameSession::solve()`, preserving all existing rollout settings and the iterative eval-interval pattern from the old solver.

## Solve Loop

```
game_solve(params) called from frontend
  → spawn background thread
  → build PostFlopGame with depth_limit from config
  → set initial ranges from session weights
  → loop {
      1. Compute boundary CFVs (RolloutLeafEvaluator with current strategy)
      2. Inject via game.set_boundary_cfvs() for each boundary
      3. Run leaf_eval_interval solve_steps
      4. Build matrix snapshot from game.strategy() at root
      5. Store snapshot in session (for game_get_state to read)
      6. Update iteration/exploitability counters
      7. Check cancel flag → break if cancelled
    }
  → finalize(game) — normalize strategy, compute EVs
  → Store final matrix with EVs
  → Mark solve complete
```

## Parameters (from frontend settings)

All four rollout settings pass through `game_solve`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 200 | Total solve iterations |
| `target_exploitability` | 3.0 | Stop early if exploitability drops below |
| `leaf_eval_interval` | 10 | Re-evaluate boundary CFVs every N iterations |
| `rollout_bias_factor` | 10.0 | Rollout evaluation bias strength |
| `rollout_num_samples` | 3 | Monte Carlo rollout samples |
| `rollout_opponent_samples` | 8 | Opponent hands sampled per rollout |
| `range_clamp_threshold` | 0.05 | Zero out combos with weight below this |

## Key Components

### 1. GameSession solve state

Add to `GameSession`:

```rust
pub struct SolveState {
    solving: AtomicBool,
    cancel: AtomicBool,
    iteration: AtomicU32,
    max_iterations: AtomicU32,
    exploitability: AtomicU32,  // f32 bits
    solve_start: Instant,
    /// Matrix snapshot updated during solve (read by get_state)
    matrix_snapshot: RwLock<Option<GameMatrix>>,
    /// Final per-hand EVs (set after finalize)
    hand_evs: RwLock<Option<Vec<f32>>>,
}
```

`get_state()` checks if solve is active and returns:
- `solve: Some(SolveStatus { ... })` with progress
- `matrix` from the solve snapshot instead of the blueprint matrix
- `actions` from the solve game tree root

### 2. Building the PostFlopGame

From the session's current state:

- **Board**: `session.board` (3-5 cards)
- **Ranges**: `session.weights[0]` (OOP/BB), `session.weights[1]` (IP/SB) — after range clamping
- **Pot/stacks**: computed from V2 tree at `session.node_idx`
- **Bet sizes**: from the blueprint config's action abstraction for the current street
- **Depth limit**: 1 (solve one street at a time, boundaries at next street transition)
- **Config**: `PostflopConfig` fields already available in the session's config

The game is built using the existing `build_game` pattern (which we removed from postflop.rs but the logic is straightforward: `CardConfig` + `TreeConfig` + `ActionTree::new` + `PostFlopGame::with_config` + `allocate_memory`).

### 3. Boundary CFV evaluation

Reuse the existing `RolloutLeafEvaluator` from `postflop.rs`. It already:
- Takes blueprint strategy, abstract tree, bucket data
- Supports configurable bias_factor, num_rollouts, opponent_samples
- Runs rollout on unit pot and scales to each boundary (batch optimization)
- Returns per-combo CFVs

The evaluator needs access to `CbvContext` (blueprint buckets + strategy) which the session already holds.

For each eval interval:
1. Get current strategy from the game (for weighting rollout opponent sampling)
2. Call evaluator for each boundary node
3. `game.set_boundary_cfvs(ordinal, player, cfvs)` for both players

### 4. Matrix snapshots during solve

After each eval interval batch:
1. Navigate game to root: `game.back_to_root()`
2. Read strategy: `game.strategy()` → per-combo probabilities
3. Read private cards: `game.private_cards(player)` → card pairs
4. Build `GameMatrix` using the same 13x13 cell construction pattern as `build_matrix`
5. Store in `solve_state.matrix_snapshot`

### 5. Cancellation

`game_cancel_solve` sets `solve_state.cancel` to true. The solve loop checks it each interval and breaks.

### 6. Frontend integration

`game_solve` command accepts the solver params:

```rust
#[tauri::command]
pub fn game_solve(
    session_state: ...,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
    leaf_eval_interval: Option<u32>,
    rollout_bias_factor: Option<f64>,
    rollout_num_samples: Option<u32>,
    rollout_opponent_samples: Option<u32>,
    range_clamp_threshold: Option<f64>,
) -> Result<(), String>
```

Frontend reads settings from `global_config` localStorage and passes them to `game_solve`. Then polls `game_get_state` every 500ms to get progress + matrix updates.

`get_state()` returns:
- `solve: Some(SolveStatus)` while solving
- `matrix` from solve snapshot (overrides blueprint matrix)
- `actions` from the solve game's root node

## What Stays From CfvSubgameSolver

- RolloutLeafEvaluator (the evaluator itself, not the solver)
- Eval interval pattern (re-evaluate boundaries periodically)
- All four rollout settings
- Blueprint warm-start concept (deferred to v2 — solve starts from uniform for now)
- Range clamping before solve

## What Changes

- Solver: PostFlopGame (exact 1326-combo) instead of CfvSubgameSolver (bucketed)
- Tree: range-solver ActionTree instead of V2 GameTree
- Strategy: per-combo instead of per-bucket
- Multi-evaluator choice node: dropped (single evaluator for v1)

## Files to Change

- Modify: `crates/tauri-app/src/game_session.rs` — SolveState, solve(), cancel, get_state integration
- Modify: `crates/tauri-app/src/game_session.rs` — game_solve command params
- Reuse: `crates/tauri-app/src/postflop.rs` — RolloutLeafEvaluator, build_subgame_solver patterns
- Reuse: `crates/range-solver/src/` — PostFlopGame, solve_step, set_boundary_cfvs, finalize
