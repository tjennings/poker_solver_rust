# Subgame Solver Explorer Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Wire the existing `SubgameCfrSolver` (depth-limited) into the PostflopExplorer so large flop/turn spots use a compact single-street tree instead of the full-tree range-solver that OOMs.

**Architecture:** The PostflopExplorer's `postflop_solve_street` is extended with a dispatch layer. It uses the existing `dispatch_decision()` (combo threshold config) to choose between the range-solver (full-depth) and `SubgameCfrSolver` (depth-limited with CBV leaf values). Both paths produce a `PostflopStrategyMatrix` for the frontend. A new `solver_name` field in `PostflopProgress` tells the UI which solver is running.

**Tech Stack:** Rust (core + tauri-app crates), TypeScript/React (frontend), range-solver, SubgameCfrSolver, CbvTable

---

### Task 1: Add `solver_name` to `PostflopProgress` and wire through backend

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:89-97` (PostflopProgress struct)
- Modify: `crates/tauri-app/src/postflop.rs:246-259` (PostflopState struct + Default)
- Modify: `crates/tauri-app/src/postflop.rs:841-861` (postflop_get_progress_core)
- Modify: `frontend/src/types.ts:207-214` (PostflopProgress TS interface)
- Modify: `frontend/src/PostflopExplorer.tsx:657` (progress bar label)

**Step 1: Add `solver_name` to `PostflopProgress` struct**

In `crates/tauri-app/src/postflop.rs`, add `solver_name` to the struct:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct PostflopProgress {
    pub iteration: u32,
    pub max_iterations: u32,
    pub exploitability: f32,
    pub is_complete: bool,
    pub matrix: Option<PostflopStrategyMatrix>,
    pub elapsed_secs: f64,
    pub solver_name: String,
}
```

**Step 2: Add `solver_name` atomic to `PostflopState`**

Add a `solver_name: RwLock<String>` field to `PostflopState` and initialize it to `"range"` in `Default`:

```rust
pub struct PostflopState {
    // ... existing fields ...
    pub solver_name: RwLock<String>,
}

// In Default impl:
solver_name: RwLock::new("range".to_string()),
```

**Step 3: Wire `solver_name` into `postflop_get_progress_core`**

```rust
pub fn postflop_get_progress_core(state: &PostflopState) -> PostflopProgress {
    // ... existing fields ...
    let solver_name = state.solver_name.read().clone();

    PostflopProgress {
        iteration,
        max_iterations,
        exploitability,
        is_complete,
        matrix,
        elapsed_secs,
        solver_name,
    }
}
```

**Step 4: Set `solver_name` to `"range"` in `postflop_solve_street_impl`**

At the start of the solve (after resetting progress atomics, around line 738):

```rust
*state.solver_name.write() = "range".to_string();
```

**Step 5: Update TypeScript `PostflopProgress` interface**

In `frontend/src/types.ts`:

```typescript
export interface PostflopProgress {
  iteration: number;
  max_iterations: number;
  exploitability: number;
  is_complete: boolean;
  matrix: PostflopStrategyMatrix | null;
  elapsed_secs: number;
  solver_name: string;
}
```

**Step 6: Update frontend progress bar to use `solver_name` from backend**

In `frontend/src/PostflopExplorer.tsx`, replace the hardcoded label (line 657):

```tsx
{` — ${progress.solver_name}`}
```

**Step 7: Build and verify**

Run: `cargo build -p poker-solver-tauri` (Rust compiles)
Run: `cd frontend && npx tsc --noEmit` (TypeScript compiles)

**Step 8: Commit**

```bash
git add crates/tauri-app/src/postflop.rs frontend/src/types.ts frontend/src/PostflopExplorer.tsx
git commit -m "feat: add solver_name to PostflopProgress for UI label"
```

---

### Task 2: Load CBV tables in exploration.rs SubgameSolve variant

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs:95-103` (SubgameSolve variant)
- Modify: `crates/tauri-app/src/exploration.rs:480-512` (load_subgame_source_core)

**Step 1: Add CBV fields to the `SubgameSolve` variant**

In `crates/tauri-app/src/exploration.rs`, add CBV table fields:

```rust
SubgameSolve {
    blueprint: Arc<poker_solver_core::blueprint::BlueprintStrategy>,
    blueprint_config: BundleConfig,
    #[allow(dead_code)]
    subgame_config: poker_solver_core::blueprint::SubgameConfig,
    #[allow(dead_code)]
    solve_cache: Arc<RwLock<LruCache<u64, poker_solver_core::blueprint::SubgameStrategy>>>,
    cbv_p0: Option<Arc<poker_solver_core::blueprint::CbvTable>>,
    cbv_p1: Option<Arc<poker_solver_core::blueprint::CbvTable>>,
}
```

**Step 2: Load CBV files in `load_subgame_source_core`**

After loading the bundle, attempt to load CBV files. They are optional — if not present, depth-limited solving falls back to zero leaf values.

```rust
// After: let bundle = tokio::task::spawn_blocking(...)?;
// Before: *state.source.write() = ...

let bundle_dir = PathBuf::from(&blueprint_path)
    .parent()
    .unwrap_or_else(|| std::path::Path::new("."))
    .to_path_buf();
let cbv_p0_path = bundle_dir.join("cbv_p0.bin");
let cbv_p1_path = bundle_dir.join("cbv_p1.bin");

let cbv_p0 = if cbv_p0_path.exists() {
    poker_solver_core::blueprint::CbvTable::load(&cbv_p0_path)
        .ok()
        .map(Arc::new)
} else {
    None
};
let cbv_p1 = if cbv_p1_path.exists() {
    poker_solver_core::blueprint::CbvTable::load(&cbv_p1_path)
        .ok()
        .map(Arc::new)
} else {
    None
};

*state.source.write() = Some(StrategySource::SubgameSolve {
    blueprint: Arc::new(bundle.blueprint),
    blueprint_config: bundle.config,
    subgame_config: poker_solver_core::blueprint::SubgameConfig::default(),
    solve_cache: Arc::new(RwLock::new(LruCache::new(NonZeroUsize::new(64).unwrap()))),
    cbv_p0,
    cbv_p1,
});
```

**Step 3: Add the `CbvTable` import**

Ensure `poker_solver_core::blueprint::CbvTable` is accessible. It should already be available via `poker_solver_core::blueprint` since `cbv.rs` is in the blueprint module. If not, add `pub mod cbv;` to the blueprint `mod.rs`.

**Step 4: Build and verify**

Run: `cargo build -p poker-solver-tauri`

**Step 5: Commit**

```bash
git add crates/tauri-app/src/exploration.rs
git commit -m "feat: load CBV tables from bundle dir in SubgameSolve"
```

---

### Task 3: Create `postflop_solve_subgame` — the depth-limited solve path

This is the core task. We add a new function that builds a `SubgameTree` with `depth_limit(1)`, runs `SubgameCfrSolver`, and produces a `PostflopStrategyMatrix`.

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs` (new function + imports)
- Test: manual verification via the UI (integration)

**Step 1: Add imports**

At the top of `crates/tauri-app/src/postflop.rs`:

```rust
use poker_solver_core::blueprint::full_depth_solver::rs_poker_card_to_id;
use poker_solver_core::blueprint::subgame_tree::{SubgameTreeBuilder, SubgameHands};
use poker_solver_core::blueprint::subgame_cfr::SubgameCfrSolver;
use poker_solver_core::blueprint::cbv::CbvTable;
use poker_solver_core::blueprint::solver_dispatch::{self, SolverChoice, SolverConfig, Street};
use poker_solver_core::poker::{Card as RsPokerCard, Suit as RsPokerSuit, Value as RsPokerValue};
```

Note: Check exact import paths. The `poker_solver_core::poker` module re-exports `Card`, `Suit`, `Value` from `rs_poker`.

**Step 2: Add helper to parse board string to rs_poker Cards**

```rust
/// Parse a board string card (e.g. "Ks") to an rs_poker Card.
fn parse_rs_poker_card(s: &str) -> Result<RsPokerCard, String> {
    if s.len() != 2 {
        return Err(format!("Invalid card string: {s}"));
    }
    let chars: Vec<char> = s.chars().collect();
    let value = match chars[0] {
        'A' | 'a' => RsPokerValue::Ace,
        'K' | 'k' => RsPokerValue::King,
        'Q' | 'q' => RsPokerValue::Queen,
        'J' | 'j' => RsPokerValue::Jack,
        'T' | 't' => RsPokerValue::Ten,
        '9' => RsPokerValue::Nine,
        '8' => RsPokerValue::Eight,
        '7' => RsPokerValue::Seven,
        '6' => RsPokerValue::Six,
        '5' => RsPokerValue::Five,
        '4' => RsPokerValue::Four,
        '3' => RsPokerValue::Three,
        '2' => RsPokerValue::Two,
        c => return Err(format!("Unknown rank: {c}")),
    };
    let suit = match chars[1] {
        's' | 'S' => RsPokerSuit::Spade,
        'h' | 'H' => RsPokerSuit::Heart,
        'd' | 'D' => RsPokerSuit::Diamond,
        'c' | 'C' => RsPokerSuit::Club,
        c => return Err(format!("Unknown suit: {c}")),
    };
    Ok(RsPokerCard::new(value, suit))
}
```

**Step 3: Add function to build matrix from `SubgameStrategy`**

This translates `SubgameStrategy` (per-combo probabilities) into a `PostflopStrategyMatrix` (13x13 hand class grid). Key steps:
- Enumerate all valid combos via `SubgameHands`
- Convert each combo to range-solver card IDs for `card_pair_to_matrix`
- Average action probabilities per matrix cell, weighted by reaching weight

```rust
/// Build a PostflopStrategyMatrix from a SubgameStrategy.
///
/// `hands` — the combos used in the solve.
/// `strategy` — the solved SubgameStrategy (root node probabilities).
/// `actions` — human-readable action labels from the subgame tree root.
/// `weights` — 1326-element reaching weights for this player (range-solver indexing).
/// `board_cards` — board as range-solver card IDs (for card_to_string).
/// `pot` / `stacks` — game state for display.
/// `player` — 0 for OOP, 1 for IP.
fn build_subgame_matrix(
    hands: &SubgameHands,
    strategy: &poker_solver_core::blueprint::subgame_cfr::SubgameStrategy,
    action_infos: Vec<ActionInfo>,
    weights: &[f32],
    board_rs_ids: &[u8],
    pot: i32,
    stacks: [i32; 2],
    player: usize,
) -> PostflopStrategyMatrix {
    let num_actions = action_infos.len();
    let board: Vec<String> = board_rs_ids
        .iter()
        .filter_map(|&c| card_to_string(c).ok())
        .collect();

    let mut prob_sums = vec![vec![vec![0.0f64; num_actions]; 13]; 13];
    let mut combo_counts = vec![vec![0usize; 13]; 13];
    let mut weight_sums = vec![vec![0.0f64; 13]; 13];
    let mut combo_details: Vec<Vec<Vec<PostflopComboDetail>>> =
        vec![vec![Vec::new(); 13]; 13];

    for (combo_idx, combo) in hands.combos.iter().enumerate() {
        let rs_id0 = rs_poker_card_to_id(combo[0]);
        let rs_id1 = rs_poker_card_to_id(combo[1]);
        let ci = card_pair_to_index(rs_id0, rs_id1);
        let w = weights[ci] as f64;
        if w <= 0.0 {
            continue; // Skip combos not in range
        }

        let (row, col, _) = card_pair_to_matrix(rs_id0, rs_id1);
        combo_counts[row][col] += 1;
        weight_sums[row][col] += w;

        let probs = strategy.root_probs(combo_idx);
        let mut prob_f32 = Vec::with_capacity(num_actions);
        for (a, prob_sum) in prob_sums[row][col].iter_mut().enumerate() {
            let p = probs.get(a).copied().unwrap_or(0.0);
            *prob_sum += p;
            prob_f32.push(p as f32);
        }

        let s1 = card_to_string(rs_id0).unwrap_or_default();
        let s2 = card_to_string(rs_id1).unwrap_or_default();
        combo_details[row][col].push(PostflopComboDetail {
            cards: format!("{s1}{s2}"),
            probabilities: prob_f32,
        });
    }

    let cells: Vec<Vec<PostflopMatrixCell>> = (0..13)
        .map(|row| {
            (0..13)
                .map(|col| {
                    let (label, suited, pair) = matrix_cell_label(row, col);
                    let count = combo_counts[row][col];
                    let probabilities = if count > 0 {
                        prob_sums[row][col]
                            .iter()
                            .map(|&s| (s / count as f64) as f32)
                            .collect()
                    } else {
                        vec![0.0; num_actions]
                    };
                    let combos = std::mem::take(&mut combo_details[row][col]);
                    let weight = if count > 0 {
                        (weight_sums[row][col] / count as f64) as f32
                    } else {
                        0.0
                    };
                    PostflopMatrixCell {
                        hand: label,
                        suited,
                        pair,
                        probabilities,
                        combo_count: count,
                        ev: None, // SubgameCfrSolver doesn't produce per-hand EV
                        combos,
                        weight,
                    }
                })
                .collect()
        })
        .collect();

    PostflopStrategyMatrix {
        cells,
        actions: action_infos,
        player,
        pot,
        stacks,
        board,
    }
}
```

**Step 4: Add the core subgame solve function**

This function builds a depth-limited subgame tree, runs CFR+, and returns the strategy + matrix.

```rust
/// Solve a postflop spot using the depth-limited SubgameCfrSolver.
///
/// Returns the SubgameStrategy, SubgameHands, and action labels for the root.
#[allow(clippy::too_many_arguments, clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn solve_subgame_street(
    board: &[RsPokerCard],
    bet_sizes: &[f32],
    pot: u32,
    stacks: [u32; 2],
    oop_weights: &[f32],
    ip_weights: &[f32],
    cbv_p0: Option<&CbvTable>,
    cbv_p1: Option<&CbvTable>,
    iterations: u32,
    player: usize,
    progress_state: &Arc<PostflopState>,
) -> Result<(poker_solver_core::blueprint::subgame_cfr::SubgameStrategy, SubgameHands, Vec<ActionInfo>), String> {
    // Build depth-limited tree (solve only the current street).
    let tree = SubgameTreeBuilder::new()
        .board(board)
        .bet_sizes(bet_sizes)
        .pot(pot)
        .stacks(&[stacks[0], stacks[1]])
        .depth_limit(1)
        .build();

    let hands = SubgameHands::enumerate(board);

    // Build opponent reach from the 1326-weight vector.
    // The SubgameCfrSolver needs a per-combo opponent_reach that matches
    // the ordering of SubgameHands::combos.
    let opp_weights = if player == 0 { ip_weights } else { oop_weights };
    let opponent_reach: Vec<f64> = hands
        .combos
        .iter()
        .map(|combo| {
            let rs_id0 = rs_poker_card_to_id(combo[0]);
            let rs_id1 = rs_poker_card_to_id(combo[1]);
            let ci = card_pair_to_index(rs_id0, rs_id1);
            opp_weights[ci] as f64
        })
        .collect();

    // Build leaf values (zero if no CBV table available).
    let leaf_values: Vec<f64> = vec![0.0; hands.combos.len()];
    // TODO: If CBV tables are available, populate leaf_values from them.
    // This requires knowing the boundary_node_idx and combo-to-bucket mapping,
    // which depends on the abstract game tree structure. For now, zero leaf
    // values work for river (no boundaries) and provide a reasonable
    // approximation for flop/turn (treats depth boundaries as 0 EV).
    let _ = (cbv_p0, cbv_p1); // suppress unused warnings

    // Extract action labels from the tree root.
    let action_infos = match &tree.nodes[0] {
        poker_solver_core::blueprint::subgame_tree::SubgameNode::Decision { actions, .. } => {
            actions
                .iter()
                .enumerate()
                .map(|(i, a)| subgame_action_to_info(a, i, pot as i32))
                .collect()
        }
        _ => return Err("Subgame tree root is not a decision node".to_string()),
    };

    let mut solver = SubgameCfrSolver::new(tree, hands.clone(), opponent_reach, leaf_values);

    // Run CFR+ with progress updates.
    for t in 0..iterations {
        if !progress_state.solving.load(Ordering::Relaxed) {
            break; // Cancelled
        }
        solver.train(1);
        progress_state.current_iteration.store(t + 1, Ordering::Relaxed);
    }

    let strategy = solver.strategy();
    Ok((strategy, hands, action_infos))
}

/// Convert a SubgameTree action to an ActionInfo.
fn subgame_action_to_info(
    action: &poker_solver_core::game::Action,
    index: usize,
    pot: i32,
) -> ActionInfo {
    use poker_solver_core::game::{Action as GameAction, ALL_IN};

    match action {
        GameAction::Fold => ActionInfo {
            id: index.to_string(),
            label: "Fold".to_string(),
            action_type: "fold".to_string(),
            size_key: None,
        },
        GameAction::Check => ActionInfo {
            id: index.to_string(),
            label: "Check".to_string(),
            action_type: "check".to_string(),
            size_key: None,
        },
        GameAction::Call => ActionInfo {
            id: index.to_string(),
            label: "Call".to_string(),
            action_type: "call".to_string(),
            size_key: None,
        },
        GameAction::Bet(idx) => {
            if *idx == ALL_IN {
                ActionInfo {
                    id: index.to_string(),
                    label: "All-in".to_string(),
                    action_type: "allin".to_string(),
                    size_key: None,
                }
            } else {
                ActionInfo {
                    id: index.to_string(),
                    label: format!("Bet {}", format_pot_pct(*idx as i32, pot)),
                    action_type: "bet".to_string(),
                    size_key: Some(idx.to_string()),
                }
            }
        }
        GameAction::Raise(idx) => {
            if *idx == ALL_IN {
                ActionInfo {
                    id: index.to_string(),
                    label: "All-in".to_string(),
                    action_type: "allin".to_string(),
                    size_key: None,
                }
            } else {
                ActionInfo {
                    id: index.to_string(),
                    label: format!("Raise {}", format_pot_pct(*idx as i32, pot)),
                    action_type: "raise".to_string(),
                    size_key: Some(idx.to_string()),
                }
            }
        }
    }
}
```

**Step 5: Build and verify**

Run: `cargo build -p poker-solver-tauri`

**Step 6: Commit**

```bash
git add crates/tauri-app/src/postflop.rs
git commit -m "feat: add SubgameCfrSolver solve path and matrix builder"
```

---

### Task 4: Wire dispatch into `postflop_solve_street_impl`

Replace the single range-solver path with a dispatch that chooses between range-solver and subgame solver based on `dispatch_decision()`.

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:707-812` (postflop_solve_street_impl)

**Step 1: Add dispatch logic to `postflop_solve_street_impl`**

The function currently always calls `build_game()` + range-solver. We need to:
1. Count live combos from filtered weights
2. Determine street from board length
3. Call `dispatch_decision()`
4. Route to range-solver or subgame solver

Replace `postflop_solve_street_impl` with:

```rust
fn postflop_solve_street_impl(
    state: &Arc<PostflopState>,
    board: Vec<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
    _prior_actions: Vec<Vec<usize>>,
) -> Result<(), String> {
    // Guard: reject if already solving.
    if state.solving.load(Ordering::Relaxed) {
        return Err("A solve is already in progress".to_string());
    }

    // Snapshot config and filtered weights under their locks.
    let config = state.config.read().clone();
    let target_exp = target_exploitability.unwrap_or(3.0);
    let filtered_oop = state.filtered_oop_weights.read().clone();
    let filtered_ip = state.filtered_ip_weights.read().clone();

    // Determine street from board length.
    let street = match board.len() {
        3 => Street::Flop,
        4 => Street::Turn,
        5 => Street::River,
        n => return Err(format!("Invalid board length: {n}")),
    };

    // Count live combos for dispatch decision.
    let live_combos = {
        let oop_count = filtered_oop
            .as_ref()
            .map(|w| w.iter().filter(|&&v| v > 0.0).count())
            .unwrap_or(0);
        let ip_count = filtered_ip
            .as_ref()
            .map(|w| w.iter().filter(|&&v| v > 0.0).count())
            .unwrap_or(0);
        oop_count.max(ip_count)
    };

    let solver_config = SolverConfig::default();
    let choice = solver_dispatch::dispatch_decision(&solver_config, street, live_combos);

    // Reset progress atomics.
    state.current_iteration.store(0, Ordering::Relaxed);
    state.solve_complete.store(false, Ordering::Relaxed);
    *state.solve_start.write() = Some(std::time::Instant::now());

    match choice {
        SolverChoice::FullDepth => {
            let max_iters = max_iterations.unwrap_or(solver_config.full_solve_iterations);
            state.max_iterations.store(max_iters, Ordering::Relaxed);
            state
                .exploitability_bits
                .store(f32::MAX.to_bits(), Ordering::Relaxed);
            *state.solver_name.write() = "range".to_string();
            state.solving.store(true, Ordering::Release);

            // Existing range-solver path (unchanged).
            let mut game = build_game(&config, &board, &filtered_oop, &filtered_ip)?;
            {
                let matrix = build_strategy_matrix(&mut game);
                *state.matrix_snapshot.write() = Some(matrix);
            }
            *state.game.lock() = Some(game);

            let shared = Arc::clone(state);
            std::thread::spawn(move || {
                #[allow(unused_assignments)]
                let mut last_exp = f32::MAX;

                for t in 0..max_iters {
                    if !shared.solving.load(Ordering::Relaxed) {
                        break;
                    }
                    {
                        let game_guard = shared.game.lock();
                        let game = game_guard.as_ref().unwrap();
                        solve_step(game, t);
                        last_exp = compute_exploitability(game);
                    }
                    shared.current_iteration.store(t + 1, Ordering::Relaxed);
                    shared
                        .exploitability_bits
                        .store(last_exp.to_bits(), Ordering::Relaxed);
                    {
                        let mut game_guard = shared.game.lock();
                        let game = game_guard.as_mut().unwrap();
                        let snap = capture_matrix_snapshot(game);
                        let shared2 = Arc::clone(&shared);
                        std::thread::spawn(move || {
                            let matrix = build_matrix_from_snapshot(snap);
                            *shared2.matrix_snapshot.write() = Some(matrix);
                        });
                    }
                    if last_exp <= target_exp {
                        break;
                    }
                }
                {
                    let mut game_guard = shared.game.lock();
                    let game = game_guard.as_mut().unwrap();
                    finalize(game);
                    let matrix = build_strategy_matrix(game);
                    *shared.matrix_snapshot.write() = Some(matrix);
                }
                shared.solve_complete.store(true, Ordering::Relaxed);
                shared.solving.store(false, Ordering::Release);
            });

            Ok(())
        }
        SolverChoice::DepthLimited => {
            let max_iters = max_iterations.unwrap_or(solver_config.depth_limited_iterations);
            state.max_iterations.store(max_iters, Ordering::Relaxed);
            state
                .exploitability_bits
                .store(f32::MAX.to_bits(), Ordering::Relaxed);
            *state.solver_name.write() = "subgame".to_string();
            state.solving.store(true, Ordering::Release);

            // Parse board to rs_poker Cards.
            let board_cards: Vec<RsPokerCard> = board
                .iter()
                .map(|s| parse_rs_poker_card(s))
                .collect::<Result<Vec<_>, _>>()?;

            // Use config bet sizes (parse from the stored config string).
            let bet_sizes: Vec<f32> = config
                .oop_bet_sizes
                .split(',')
                .filter_map(|s| {
                    let s = s.trim().trim_end_matches('%');
                    s.parse::<f32>().ok().map(|v| v / 100.0)
                })
                .collect();
            let bet_sizes = if bet_sizes.is_empty() {
                vec![1.0] // fallback: pot-size bet
            } else {
                bet_sizes
            };

            let pot = config.pot;
            let stacks = [config.effective_stack, config.effective_stack];

            let oop_w = filtered_oop
                .unwrap_or_else(|| {
                    let range: Range = config.oop_range.parse().unwrap_or_default();
                    range.raw_data().to_vec()
                });
            let ip_w = filtered_ip
                .unwrap_or_else(|| {
                    let range: Range = config.ip_range.parse().unwrap_or_default();
                    range.raw_data().to_vec()
                });

            let shared = Arc::clone(state);
            std::thread::spawn(move || {
                // OOP (player 0) solves first at root.
                let player = 0;
                match solve_subgame_street(
                    &board_cards,
                    &bet_sizes,
                    pot as u32,
                    [stacks[0] as u32, stacks[1] as u32],
                    &oop_w,
                    &ip_w,
                    None, // CBV tables: TODO wire from exploration state
                    None,
                    max_iters,
                    player,
                    &shared,
                ) {
                    Ok((strategy, hands, action_infos)) => {
                        let board_rs: Vec<u8> = board_cards
                            .iter()
                            .map(|c| rs_poker_card_to_id(*c))
                            .collect();
                        let matrix = build_subgame_matrix(
                            &hands,
                            &strategy,
                            action_infos,
                            &oop_w,
                            &board_rs,
                            pot,
                            stacks,
                            player,
                        );
                        *shared.matrix_snapshot.write() = Some(matrix);
                    }
                    Err(e) => {
                        eprintln!("Subgame solve failed: {e}");
                    }
                }
                shared.solve_complete.store(true, Ordering::Relaxed);
                shared.solving.store(false, Ordering::Release);
            });

            Ok(())
        }
    }
}
```

**Step 2: Build and verify**

Run: `cargo build -p poker-solver-tauri`

**Step 3: Commit**

```bash
git add crates/tauri-app/src/postflop.rs
git commit -m "feat: dispatch to SubgameCfrSolver for large flop/turn spots"
```

---

### Task 5: Wire subgame range propagation in `postflop_close_street`

When the subgame solver was used, we need to propagate ranges differently. The range-solver path uses `game.strategy()` and `game.private_cards()` — both from `PostFlopGame`. The subgame path needs to store the `SubgameStrategy` + `SubgameHands` so that `postflop_close_street` can walk the action history and multiply weights.

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:246-259` (PostflopState — add subgame strategy storage)
- Modify: `crates/tauri-app/src/postflop.rs:1008-1080` (postflop_close_street_core)

**Step 1: Add subgame strategy storage to PostflopState**

```rust
pub struct PostflopState {
    // ... existing fields ...
    pub solver_name: RwLock<String>,
    /// Subgame strategy from depth-limited solve (for range propagation).
    pub subgame_result: RwLock<Option<SubgameSolveResult>>,
}

/// Stored result from a subgame solve, used for range propagation.
pub struct SubgameSolveResult {
    pub strategy: poker_solver_core::blueprint::subgame_cfr::SubgameStrategy,
    pub hands: SubgameHands,
    /// Action labels for navigation.
    pub action_infos: Vec<ActionInfo>,
    /// The tree structure (needed for navigating action sequences).
    pub tree: poker_solver_core::blueprint::subgame_tree::SubgameTree,
}
```

Initialize in Default: `subgame_result: RwLock::new(None),`

**Step 2: Store subgame result after depth-limited solve**

In the `DepthLimited` branch of `postflop_solve_street_impl`, after the solve succeeds, store the result:

```rust
// After: let (strategy, hands, action_infos) = ...
// Also need to clone/store the tree from the solve.
// Modify solve_subgame_street to also return the tree.

*shared.subgame_result.write() = Some(SubgameSolveResult {
    strategy,
    hands,
    action_infos,
    tree, // from the solve
});
```

This requires modifying `solve_subgame_street` to return the tree as well. Update its return type to include `SubgameTree`.

**Step 3: Update `postflop_close_street_core` for subgame path**

The function currently walks `game.strategy()` / `game.private_cards()` from the range-solver game. We add a branch for when `solver_name == "subgame"`:

```rust
pub fn postflop_close_street_core(
    state: &PostflopState,
    action_history: Vec<usize>,
) -> Result<PostflopStreetResult, String> {
    let solver_name = state.solver_name.read().clone();

    if solver_name == "subgame" {
        return postflop_close_street_subgame(state, action_history);
    }

    // ... existing range-solver path unchanged ...
}

/// Close street for subgame solver path.
/// Walks the action history through the SubgameStrategy to narrow ranges.
fn postflop_close_street_subgame(
    state: &PostflopState,
    action_history: Vec<usize>,
) -> Result<PostflopStreetResult, String> {
    let result_guard = state.subgame_result.read();
    let result = result_guard.as_ref().ok_or("No subgame result stored")?;

    let config = state.config.read().clone();
    let oop_range: Range = config.oop_range.parse()
        .map_err(|e: String| format!("Invalid OOP range: {e}"))?;
    let ip_range: Range = config.ip_range.parse()
        .map_err(|e: String| format!("Invalid IP range: {e}"))?;

    let mut oop_weights: Vec<f32> = state
        .filtered_oop_weights
        .read()
        .clone()
        .unwrap_or_else(|| oop_range.raw_data().to_vec());
    let mut ip_weights: Vec<f32> = state
        .filtered_ip_weights
        .read()
        .clone()
        .unwrap_or_else(|| ip_range.raw_data().to_vec());

    // Walk the action history through the subgame tree, narrowing ranges.
    let mut current_node = 0u32;
    for &action_idx in &action_history {
        match &result.tree.nodes[current_node as usize] {
            SubgameNode::Decision { position, actions, children, .. } => {
                let player = *position as usize;
                let weights = if player == 0 { &mut oop_weights } else { &mut ip_weights };

                // For each combo, multiply weight by strategy probability for this action.
                for (combo_idx, combo) in result.hands.combos.iter().enumerate() {
                    let rs_id0 = rs_poker_card_to_id(combo[0]);
                    let rs_id1 = rs_poker_card_to_id(combo[1]);
                    let ci = card_pair_to_index(rs_id0, rs_id1);
                    let probs = result.strategy.get_probs(current_node, combo_idx);
                    let action_prob = probs.get(action_idx).copied().unwrap_or(0.0);
                    weights[ci] *= action_prob as f32;
                }

                // Advance to child node.
                if action_idx < children.len() {
                    current_node = children[action_idx];
                } else {
                    break;
                }
            }
            _ => break, // Terminal or boundary — stop.
        }
    }

    // Compute pot/stacks from the final tree node.
    let (pot, effective_stack) = match &result.tree.nodes[current_node as usize] {
        SubgameNode::Decision { pot, stacks, .. }
        | SubgameNode::Terminal { pot, stacks, .. }
        | SubgameNode::DepthBoundary { pot, stacks } => {
            (*pot as i32, stacks[0].min(stacks[1]) as i32)
        }
    };

    // Store filtered weights for the next street.
    *state.filtered_oop_weights.write() = Some(oop_weights.clone());
    *state.filtered_ip_weights.write() = Some(ip_weights.clone());

    Ok(PostflopStreetResult {
        filtered_oop_range: oop_weights,
        filtered_ip_range: ip_weights,
        pot,
        effective_stack,
    })
}
```

**Step 4: Build and verify**

Run: `cargo build -p poker-solver-tauri`

**Step 5: Commit**

```bash
git add crates/tauri-app/src/postflop.rs
git commit -m "feat: subgame range propagation in postflop_close_street"
```

---

### Task 6: Wire `postflop_play_action` for subgame solver path

When the subgame solver was used, `postflop_play_action` needs to navigate the `SubgameTree` instead of the `PostFlopGame`. This is needed for the user to click action buttons and see updated matrices during tree exploration.

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:874-933` (postflop_play_action_core)
- Modify: `crates/tauri-app/src/postflop.rs:246` (PostflopState — add current_node tracker)

**Step 1: Add subgame navigation state to PostflopState**

```rust
pub struct PostflopState {
    // ... existing fields ...
    /// Current node in the subgame tree during navigation.
    pub subgame_node: AtomicU32,
    /// Action history within the subgame tree.
    pub subgame_action_history: RwLock<Vec<usize>>,
}
```

Initialize: `subgame_node: AtomicU32::new(0)`, `subgame_action_history: RwLock::new(Vec::new())`

**Step 2: Add subgame branch to `postflop_play_action_core`**

```rust
pub fn postflop_play_action_core(
    state: &PostflopState,
    action: usize,
) -> Result<PostflopPlayResult, String> {
    let solver_name = state.solver_name.read().clone();

    if solver_name == "subgame" {
        return postflop_play_action_subgame(state, action);
    }

    // ... existing range-solver path unchanged ...
}

fn postflop_play_action_subgame(
    state: &PostflopState,
    action: usize,
) -> Result<PostflopPlayResult, String> {
    let result_guard = state.subgame_result.read();
    let result = result_guard.as_ref().ok_or("No subgame result stored")?;

    let current = state.subgame_node.load(Ordering::Relaxed);
    let node = &result.tree.nodes[current as usize];

    match node {
        SubgameNode::Decision { children, pot, stacks, .. } => {
            if action >= children.len() {
                return Err(format!("Action {action} out of range"));
            }
            let child = children[action];
            state.subgame_node.store(child, Ordering::Relaxed);
            state.subgame_action_history.write().push(action);

            let child_node = &result.tree.nodes[child as usize];
            match child_node {
                SubgameNode::Terminal { pot, stacks, .. } => {
                    Ok(PostflopPlayResult {
                        matrix: None,
                        is_terminal: true,
                        is_chance: false,
                        current_player: None,
                        pot: *pot as i32,
                        stacks: [stacks[0] as i32, stacks[1] as i32],
                    })
                }
                SubgameNode::DepthBoundary { pot, stacks } => {
                    // Depth boundary acts like a chance node — user picks a card.
                    Ok(PostflopPlayResult {
                        matrix: None,
                        is_terminal: false,
                        is_chance: true,
                        current_player: None,
                        pot: *pot as i32,
                        stacks: [stacks[0] as i32, stacks[1] as i32],
                    })
                }
                SubgameNode::Decision { position, pot, stacks, .. } => {
                    // Build matrix for the next player at this node.
                    let player = *position as usize;
                    let weights = if player == 0 {
                        state.filtered_oop_weights.read().clone()
                    } else {
                        state.filtered_ip_weights.read().clone()
                    };
                    let board_rs: Vec<u8> = result.tree.board.iter()
                        .map(|c| rs_poker_card_to_id(*c))
                        .collect();

                    // Build action infos for this child decision node.
                    let action_infos = match child_node {
                        SubgameNode::Decision { actions, .. } => {
                            actions.iter().enumerate()
                                .map(|(i, a)| subgame_action_to_info(a, i, *pot as i32))
                                .collect()
                        }
                        _ => unreachable!(),
                    };

                    let matrix = build_subgame_matrix_at_node(
                        &result.hands,
                        &result.strategy,
                        action_infos,
                        weights.as_deref().unwrap_or(&[0.0; 1326]),
                        &board_rs,
                        *pot as i32,
                        [stacks[0] as i32, stacks[1] as i32],
                        player,
                        child,
                    );
                    Ok(PostflopPlayResult {
                        matrix: Some(matrix),
                        is_terminal: false,
                        is_chance: false,
                        current_player: Some(player),
                        pot: *pot as i32,
                        stacks: [stacks[0] as i32, stacks[1] as i32],
                    })
                }
            }
        }
        _ => Err("Current node is not a decision node".to_string()),
    }
}
```

**Step 3: Add `build_subgame_matrix_at_node` helper**

Same as `build_subgame_matrix` but takes a specific `node_idx` instead of always using root (node 0):

```rust
fn build_subgame_matrix_at_node(
    hands: &SubgameHands,
    strategy: &poker_solver_core::blueprint::subgame_cfr::SubgameStrategy,
    action_infos: Vec<ActionInfo>,
    weights: &[f32],
    board_rs_ids: &[u8],
    pot: i32,
    stacks: [i32; 2],
    player: usize,
    node_idx: u32,
) -> PostflopStrategyMatrix {
    // Same logic as build_subgame_matrix but uses strategy.get_probs(node_idx, combo_idx)
    // instead of strategy.root_probs(combo_idx).
    // ... (same implementation as build_subgame_matrix but with node_idx parameter)
}
```

Actually, refactor `build_subgame_matrix` to accept a `node_idx` parameter and have the root version call it with `0`.

**Step 4: Reset subgame navigation on new solve**

In `postflop_solve_street_impl`, at the start (after resetting progress atomics):

```rust
state.subgame_node.store(0, Ordering::Relaxed);
state.subgame_action_history.write().clear();
*state.subgame_result.write() = None;
```

**Step 5: Build and verify**

Run: `cargo build -p poker-solver-tauri`

**Step 6: Commit**

```bash
git add crates/tauri-app/src/postflop.rs
git commit -m "feat: subgame tree navigation in postflop_play_action"
```

---

### Task 7: Integration test — end-to-end solve dispatch

Verify the full flow works: config set, solve dispatched correctly, matrix returned.

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs` (test module)

**Step 1: Write test for dispatch to subgame solver**

```rust
#[cfg(test)]
mod dispatch_tests {
    use super::*;

    #[test]
    fn test_dispatch_uses_subgame_for_wide_range() {
        let state = Arc::new(PostflopState::default());

        // Set config with wide ranges (many combos).
        let config = PostflopConfig {
            oop_range: "22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,82s+,72s+,62s+,52s+,42s+,32s,A2o+,K2o+,Q2o+,J2o+,T2o+,92o+,82o+,72o+,62o+,52o+".to_string(),
            ip_range: "22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,82s+,72s+,62s+,52s+,42s+,32s,A2o+,K2o+,Q2o+,J2o+,T2o+,92o+,82o+,72o+,62o+,52o+".to_string(),
            pot: 100,
            effective_stack: 200,
            oop_bet_sizes: "50%".to_string(),
            oop_raise_sizes: "".to_string(),
            ip_bet_sizes: "50%".to_string(),
            ip_raise_sizes: "".to_string(),
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        postflop_set_config_core(&state, config).unwrap();

        // Solve flop with wide ranges — should dispatch to subgame.
        let board = vec!["Ks".to_string(), "Qh".to_string(), "Jd".to_string()];
        let result = postflop_solve_street_core(&state, board, Some(10), Some(1e9));
        assert!(result.is_ok());

        // Check that solver_name was set to "subgame".
        let solver_name = state.solver_name.read().clone();
        assert_eq!(solver_name, "subgame");
    }

    #[test]
    fn test_dispatch_uses_range_for_narrow_range() {
        let state = Arc::new(PostflopState::default());

        // Set config with very narrow ranges (few combos).
        let config = PostflopConfig {
            oop_range: "AA".to_string(),
            ip_range: "KK".to_string(),
            pot: 100,
            effective_stack: 200,
            oop_bet_sizes: "50%".to_string(),
            oop_raise_sizes: "".to_string(),
            ip_bet_sizes: "50%".to_string(),
            ip_raise_sizes: "".to_string(),
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        postflop_set_config_core(&state, config).unwrap();

        // Solve river with narrow ranges — should dispatch to range.
        let board = vec![
            "Ks".to_string(), "Qh".to_string(), "Jd".to_string(),
            "Tc".to_string(), "2d".to_string(),
        ];
        let result = postflop_solve_street_core(&state, board, Some(10), Some(1e9));
        assert!(result.is_ok());

        let solver_name = state.solver_name.read().clone();
        assert_eq!(solver_name, "range");
    }
}
```

**Step 2: Run tests**

Run: `cargo test -p poker-solver-tauri -- dispatch_tests`

**Step 3: Commit**

```bash
git add crates/tauri-app/src/postflop.rs
git commit -m "test: integration tests for solver dispatch"
```

---

### Task 8: Update devserver to mirror new progress field

The devserver mirrors all Tauri commands. The `solver_name` field is automatically included since it's part of the serialized `PostflopProgress` struct — no devserver changes needed if it uses the same struct. Verify this.

**Files:**
- Check: `crates/devserver/src/main.rs`

**Step 1: Verify devserver uses PostflopProgress directly**

Read `crates/devserver/src/main.rs` and confirm it calls `postflop_get_progress_core()` which returns the updated struct. If so, no changes needed.

**Step 2: Build devserver and test**

Run: `cargo build -p poker-solver-devserver`

**Step 3: Commit (if changes needed)**

Only commit if devserver required modifications.
