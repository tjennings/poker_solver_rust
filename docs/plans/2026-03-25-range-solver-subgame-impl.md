# Range-Solver Subgame Solving Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace the `GameSession::solve()` stub with a real subgame solver using the range-solver `PostFlopGame` with `depth_limit` for exact 1326-combo solving, preserving all rollout settings and the iterative boundary re-evaluation pattern.

**Architecture:** `GameSession` spawns a background thread that builds a `PostFlopGame` with depth limit, runs an iterative solve loop with `RolloutLeafEvaluator` boundary re-evaluation every `leaf_eval_interval` iterations, and stores matrix snapshots in shared state for the UI to read via `game_get_state`. Cancellation via atomic flag.

**Tech Stack:** Rust, range-solver (`PostFlopGame`, `solve_step`, `finalize`), `RolloutLeafEvaluator` from postflop.rs, atomic types for thread-safe state sharing.

---

### Task 1: Add SolveState to GameSession

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

**Step 1: Define SolveState struct**

Add a shared solve state that the background thread writes and `get_state` reads:

```rust
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Shared state between solve thread and UI queries.
pub struct SolveState {
    pub solving: AtomicBool,
    pub cancel: AtomicBool,
    pub iteration: AtomicU32,
    pub max_iterations: AtomicU32,
    pub exploitability_bits: AtomicU32, // f32::to_bits / from_bits
    pub solve_start: parking_lot::RwLock<Option<Instant>>,
    /// Matrix snapshot updated during solve.
    pub matrix_snapshot: parking_lot::RwLock<Option<GameMatrix>>,
    /// Actions at the solve game's root node.
    pub solve_actions: parking_lot::RwLock<Vec<GameAction>>,
    /// Position label at the solve root.
    pub solve_position: parking_lot::RwLock<String>,
}
```

**Step 2: Add SolveState to GameSessionState**

The `SolveState` lives in `GameSessionState` (not in `GameSession` itself) because it must be `Arc`-shared with the background thread while the session is `RwLock`-protected:

```rust
pub struct GameSessionState {
    pub session: RwLock<Option<GameSession>>,
    pub solve_state: Arc<SolveState>,
}
```

Initialize `SolveState` with all defaults in the `GameSessionState` constructor.

**Step 3: Update get_state to read solve state**

In `GameSession::get_state()` (or the Tauri command wrapper), check `solve_state`:
- If `solve_state.solving` is true, populate `GameState.solve` with `SolveStatus`
- If `solve_state.matrix_snapshot` is Some, use it instead of the blueprint matrix
- If `solve_state.solve_actions` is non-empty, use them as the current actions

The key idea: `game_get_state_core` receives both `&GameSessionState` and reads from `solve_state`. If a solve is active or just completed, the solve's matrix/actions override the blueprint's.

**Step 4: Build, test, commit**

```bash
cargo build -p poker-solver-tauri
cargo test -p poker-solver-tauri -- game_session
```

```
git commit -m "feat: add SolveState to GameSession for background solve progress"
```

---

### Task 2: Update game_solve command to accept params

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs` (Tauri command + core function)
- Modify: `crates/devserver/src/main.rs` (devserver endpoint)
- Modify: `frontend/src/GameExplorer.tsx` (pass settings from global_config)

**Step 1: Update game_solve Tauri command signature**

```rust
#[tauri::command(rename_all = "snake_case")]
pub fn game_solve(
    session_state: tauri::State<'_, GameSessionState>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
    leaf_eval_interval: Option<u32>,
    rollout_bias_factor: Option<f64>,
    rollout_num_samples: Option<u32>,
    rollout_opponent_samples: Option<u32>,
    range_clamp_threshold: Option<f64>,
) -> Result<(), String> {
```

Note: use `rename_all = "snake_case"` so the frontend can use snake_case param names. If Tauri defaults to camelCase, adjust the frontend accordingly.

**Step 2: Update core function**

```rust
pub fn game_solve_core(
    session_state: &GameSessionState,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
    leaf_eval_interval: Option<u32>,
    rollout_bias_factor: Option<f64>,
    rollout_num_samples: Option<u32>,
    rollout_opponent_samples: Option<u32>,
    range_clamp_threshold: Option<f64>,
) -> Result<(), String> {
```

For now, keep the body as a stub that stores the params and returns Ok (actual solve in Task 3).

**Step 3: Update devserver endpoint**

Add a request struct with the params and pass them to `game_solve_core`.

**Step 4: Update frontend to pass settings**

In `GameExplorer.tsx`, where the solve button calls `game_solve`:

```tsx
const globalConfig = JSON.parse(localStorage.getItem('global_config') || '{}');
await invoke('game_solve', {
  maxIterations: globalConfig.solve_iterations ?? 200,
  targetExploitability: globalConfig.target_exploitability ?? 3.0,
  leafEvalInterval: globalConfig.leaf_eval_interval ?? 10,
  rolloutBiasFactor: globalConfig.rollout_bias_factor ?? 10.0,
  rolloutNumSamples: globalConfig.rollout_num_samples ?? 3,
  rolloutOpponentSamples: globalConfig.rollout_opponent_samples ?? 8,
  rangeClampThreshold: globalConfig.range_clamp_threshold ?? 0.05,
});
```

Note Tauri camelCase parameter naming.

**Step 5: Build, test, commit**

```
git commit -m "feat: game_solve accepts rollout params from frontend settings"
```

---

### Task 3: Build PostFlopGame from session state

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

**Step 1: Implement build_solve_game helper**

This builds a `PostFlopGame` from the current session state. The pattern is adapted from the old `build_game` in postflop.rs (which was removed) and `build_subgame_solver`:

```rust
use range_solver::{PostFlopGame, ActionTree, CardConfig, TreeConfig, BetSizeOptions, BoardState};
use range_solver::card::{NOT_DEALT, card_from_str, flop_from_str};
use range_solver::range::Range;

fn build_solve_game(
    board: &[String],
    oop_weights: &[f32],
    ip_weights: &[f32],
    pot: i32,
    effective_stack: i32,
    bet_sizes: &[Vec<f64>],  // from blueprint config, per raise depth
) -> Result<PostFlopGame, String> {
    // Parse board
    let (flop, turn, river, initial_state) = parse_solve_board(board)?;

    // Build ranges from 1326-element weights
    let oop_range = Range::from_raw_data(oop_weights)
        .map_err(|e| format!("Bad OOP weights: {e}"))?;
    let ip_range = Range::from_raw_data(ip_weights)
        .map_err(|e| format!("Bad IP weights: {e}"))?;

    // Parse bet sizes from blueprint config
    let bet_str = format_bet_sizes_for_range_solver(bet_sizes);
    let oop_sizes = BetSizeOptions::try_from((bet_str.as_str(), ""))
        .map_err(|e| format!("Bad bet sizes: {e}"))?;
    let ip_sizes = oop_sizes.clone();  // same for both players

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop, turn, river,
    };

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: pot,
        effective_stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        turn_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        river_bet_sizes: [oop_sizes, ip_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        depth_limit: Some(1),  // solve one street, boundaries at next
    };

    let action_tree = ActionTree::new(tree_config)
        .map_err(|e| format!("Failed to build tree: {e}"))?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;
    game.allocate_memory(false);
    Ok(game)
}
```

**Step 2: Helper to convert blueprint bet sizes to range-solver format**

The blueprint stores bet sizes as `Vec<Vec<f64>>` (pot fractions per raise depth). The range-solver `BetSizeOptions` parses from comma-separated strings like `"33%,67%,100%"`. Write a converter.

Check `crates/tauri-app/src/postflop.rs` for how this was done before (lines ~1158-1178 in `solve_depth_limited`).

**Step 3: Helper to parse board for range-solver**

```rust
fn parse_solve_board(board: &[String]) -> Result<([u8; 3], u8, u8, BoardState), String> {
    match board.len() {
        3 => {
            let flop_str = format!("{}{}{}", board[0], board[1], board[2]);
            let flop = flop_from_str(&flop_str).map_err(|e| format!("Bad flop: {e}"))?;
            Ok((flop, NOT_DEALT, NOT_DEALT, BoardState::Flop))
        }
        4 => {
            let flop_str = format!("{}{}{}", board[0], board[1], board[2]);
            let flop = flop_from_str(&flop_str).map_err(|e| format!("Bad flop: {e}"))?;
            let turn = card_from_str(&board[3]).map_err(|e| format!("Bad turn: {e}"))?;
            Ok((flop, turn, NOT_DEALT, BoardState::Turn))
        }
        5 => {
            let flop_str = format!("{}{}{}", board[0], board[1], board[2]);
            let flop = flop_from_str(&flop_str).map_err(|e| format!("Bad flop: {e}"))?;
            let turn = card_from_str(&board[3]).map_err(|e| format!("Bad turn: {e}"))?;
            let river = card_from_str(&board[4]).map_err(|e| format!("Bad river: {e}"))?;
            Ok((flop, turn, river, BoardState::River))
        }
        n => Err(format!("Board must have 3-5 cards, got {n}")),
    }
}
```

**Step 4: Build, test, commit**

```
git commit -m "feat: build_solve_game constructs PostFlopGame from session state"
```

---

### Task 4: Implement the solve loop

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

This is the core task. Replace the `solve()` stub with the real implementation.

**Step 1: Implement game_solve_core**

```rust
pub fn game_solve_core(
    session_state: &GameSessionState,
    max_iterations: Option<u32>,
    // ... other params
) -> Result<(), String> {
    // 1. Guard: reject if already solving
    if session_state.solve_state.solving.load(Ordering::Relaxed) {
        return Err("A solve is already in progress".to_string());
    }

    // 2. Read session state under lock, clone what the thread needs
    let (board, oop_w, ip_w, pot, eff_stack, bet_sizes, cbv_ctx, abstract_node_idx) = {
        let guard = session_state.session.read();
        let session = guard.as_ref().ok_or("No game session active")?;
        // ... extract board, weights, pot/stacks, bet sizes from session
    };

    // 3. Apply range clamping
    // ... clamp oop_w and ip_w

    // 4. Reset solve state atomics
    let ss = &session_state.solve_state;
    ss.iteration.store(0, Ordering::Relaxed);
    ss.max_iterations.store(max_iters, Ordering::Relaxed);
    ss.exploitability_bits.store(f32::MAX.to_bits(), Ordering::Relaxed);
    ss.cancel.store(false, Ordering::Relaxed);
    ss.solving.store(true, Ordering::Release);
    *ss.solve_start.write() = Some(Instant::now());
    *ss.matrix_snapshot.write() = None;

    // 5. Spawn background thread
    let ss_clone = Arc::clone(&session_state.solve_state);
    std::thread::spawn(move || {
        // Build game
        let mut game = match build_solve_game(&board, &oop_w, &ip_w, pot, eff_stack, &bet_sizes) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[solve] failed to build game: {e}");
                ss_clone.solving.store(false, Ordering::Release);
                return;
            }
        };

        let max_iters = max_iterations.unwrap_or(200);
        let eval_interval = leaf_eval_interval.unwrap_or(10);
        let target_exp = target_exploitability.unwrap_or(3.0);

        // Build RolloutLeafEvaluator(s) if CbvContext available
        // ... (see Task 5)

        let mut t = 0u32;
        while t < max_iters {
            if ss_clone.cancel.load(Ordering::Relaxed) { break; }

            // Re-evaluate boundary CFVs every eval_interval
            if t % eval_interval == 0 && game.num_boundary_nodes() > 0 {
                evaluate_and_inject_boundaries(&mut game, /* evaluator, weights */);
            }

            // Solve one step
            solve_step(&game, t);
            t += 1;
            ss_clone.iteration.store(t, Ordering::Relaxed);

            // Compute exploitability periodically
            if t % eval_interval == 0 {
                let exp = compute_exploitability(&game);
                ss_clone.exploitability_bits.store(exp.to_bits(), Ordering::Relaxed);

                // Snapshot matrix
                let matrix = build_solve_matrix(&game);
                *ss_clone.matrix_snapshot.write() = Some(matrix);

                if exp <= target_exp { break; }
            }
        }

        // Finalize
        finalize(&mut game);
        let final_matrix = build_solve_matrix_with_evs(&game);
        *ss_clone.matrix_snapshot.write() = Some(final_matrix);
        ss_clone.solving.store(false, Ordering::Release);
    });

    Ok(())
}
```

**Step 2: Implement game_cancel_solve_core**

```rust
pub fn game_cancel_solve_core(session_state: &GameSessionState) -> Result<(), String> {
    session_state.solve_state.cancel.store(true, Ordering::Relaxed);
    Ok(())
}
```

**Step 3: Build, test, commit**

```
git commit -m "feat: game_solve spawns background PostFlopGame solve with iterative boundary eval"
```

---

### Task 5: Wire boundary evaluation with RolloutLeafEvaluator

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`
- Reference: `crates/tauri-app/src/postflop.rs` (RolloutLeafEvaluator construction at ~lines 643-660)

**Step 1: Build RolloutLeafEvaluator inside the solve thread**

Reuse the existing `RolloutLeafEvaluator` from postflop.rs. It needs `CbvContext` (blueprint strategy + buckets + abstract tree). The session holds this via `cbv_context: Option<Arc<CbvContext>>`.

```rust
use crate::postflop::{RolloutLeafEvaluator, BiasType, CbvContext};

// Inside the solve thread, after building the game:
let evaluator = if let Some(ctx) = &cbv_ctx {
    Some(RolloutLeafEvaluator::new(
        Arc::clone(&ctx.strategy),
        Arc::new(ctx.abstract_tree.clone()),
        Arc::clone(&ctx.all_buckets),
        abstract_node_idx.unwrap_or(0),
        BiasType::Unbiased,
        rollout_bias_factor.unwrap_or(10.0),
        rollout_num_samples.unwrap_or(3),
        rollout_opponent_samples.unwrap_or(8),
        starting_stack,
        pot_f64,
    ))
} else {
    None
};
```

**Step 2: Implement evaluate_and_inject_boundaries**

```rust
fn evaluate_and_inject_boundaries(
    game: &mut PostFlopGame,
    evaluator: &RolloutLeafEvaluator,
    oop_weights: &[f32],
    ip_weights: &[f32],
    board: &[rs_poker::core::Card],
) {
    let n_boundaries = game.num_boundary_nodes();
    if n_boundaries == 0 { return; }

    // Collect boundary info: (ordinal, pot, invested)
    // ... iterate boundary nodes, get pot/invested at each

    // Build requests for batch evaluation
    // ... format as (pot, invested, player, [invested_per_player])

    // Call evaluator.evaluate_boundaries(combos, board, oop_range, ip_range, requests)
    // Returns Vec<Vec<f64>> — one CFV vector per boundary×player

    // Inject: game.set_boundary_cfvs(ordinal, player, cfvs_f32)
    for ordinal in 0..n_boundaries {
        for player in 0..2 {
            let cfvs: Vec<f32> = /* from evaluator result */;
            game.set_boundary_cfvs(ordinal, player, cfvs);
        }
    }
}
```

Note: The exact boundary evaluation API depends on how `RolloutLeafEvaluator::evaluate_boundaries` works. Check the existing `LeafEvaluator` trait signature in postflop.rs (~line 482). The evaluator returns per-combo CFVs for each boundary request.

**Step 3: Build, test, commit**

```
git commit -m "feat: boundary CFV evaluation using RolloutLeafEvaluator during solve"
```

---

### Task 6: Build solve matrix snapshots

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

**Step 1: Implement build_solve_matrix**

Build a `GameMatrix` from the current `PostFlopGame` state at the root node. This is similar to `GameSession::build_matrix` but reads from the range-solver game instead of the blueprint:

```rust
fn build_solve_matrix(game: &mut PostFlopGame) -> GameMatrix {
    game.back_to_root();
    let player = game.current_player(); // 0=OOP, 1=IP
    let strategy = game.strategy(); // per-combo strategy at root
    let private_cards = game.private_cards(player);
    let num_hands = game.num_private_hands(player);
    let initial_weights = game.initial_weights(player);

    // Build 13x13 cells from per-combo data
    // Group combos by canonical hand, average probabilities, compute weights
    // Same pattern as build_matrix_from_snapshot in postflop.rs
    // ...

    GameMatrix { cells, actions }
}
```

The key difference from blueprint matrix building: the range-solver stores strategy per-combo (not per-bucket), so each combo has its own probabilities. Group by canonical hand and average for the 13x13 cell display. Per-combo details populate the `combos` field.

**Step 2: Implement build_solve_matrix_with_evs**

After `finalize()`, also extract EVs:

```rust
fn build_solve_matrix_with_evs(game: &mut PostFlopGame) -> GameMatrix {
    game.cache_normalized_weights();
    let evs = game.expected_values(game.current_player());
    // Build matrix as above, but include EVs per cell
}
```

**Step 3: Build, test, commit**

```
git commit -m "feat: build_solve_matrix extracts 13x13 grid from PostFlopGame during solve"
```

---

### Task 7: Update get_state to show solve progress

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

**Step 1: Update game_get_state_core to read solve state**

```rust
pub fn game_get_state_core(session_state: &GameSessionState) -> Result<GameState, String> {
    let guard = session_state.session.read();
    let session = guard.as_ref().ok_or("No game session active")?;

    let mut state = session.get_state();

    // Override with solve state if active
    let ss = &session_state.solve_state;
    let is_solving = ss.solving.load(Ordering::Relaxed);
    let iteration = ss.iteration.load(Ordering::Relaxed);

    if is_solving || iteration > 0 {
        let exp = f32::from_bits(ss.exploitability_bits.load(Ordering::Relaxed));
        let max_iters = ss.max_iterations.load(Ordering::Relaxed);
        let elapsed = ss.solve_start.read()
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        state.solve = Some(SolveStatus {
            iteration,
            max_iterations: max_iters,
            exploitability: exp,
            elapsed_secs: elapsed,
            solver_name: "range".to_string(),
            is_complete: !is_solving && iteration > 0,
        });

        // Use solve matrix if available
        if let Some(matrix) = ss.matrix_snapshot.read().clone() {
            state.matrix = Some(matrix);
        }
        if !ss.solve_actions.read().is_empty() {
            state.actions = ss.solve_actions.read().clone();
        }
        if !ss.solve_position.read().is_empty() {
            state.position = ss.solve_position.read().clone();
        }
    }

    Ok(state)
}
```

**Step 2: Update frontend to poll during solve**

In `GameExplorer.tsx`, after calling `game_solve`, start polling:

```tsx
// After starting solve:
setSolving(true);
const pollId = setInterval(async () => {
    const s = await invoke<GameState>('game_get_state', {});
    setState(s);
    if (s.solve && s.solve.is_complete) {
        clearInterval(pollId);
        setSolving(false);
    }
}, 500);
```

**Step 3: Build, test, commit**

```
git commit -m "feat: game_get_state returns solve progress and matrix snapshots"
```

---

### Task 8: End-to-end test and cleanup

**Step 1: Manual test flow**

1. Load blueprint, navigate to postflop
2. Click SOLVE
3. Verify progress bar shows iterations/exploitability
4. Verify matrix updates during solve
5. Click CANCEL, verify solve stops
6. Verify final matrix has EVs after completion

**Step 2: Clean up solve state on new hand / back**

When `game_new` or `game_back` is called, reset the solve state so stale solve data doesn't leak.

**Step 3: Update bean**

```bash
beans update poker_solver_rust-lcyi --body-replace-old "- [ ] Wire PostFlopGame" --body-replace-new "- [x] Wire PostFlopGame"
```

**Step 4: Commit**

```
git commit -m "feat: range-solver subgame solving in GameSession (complete)"
```
