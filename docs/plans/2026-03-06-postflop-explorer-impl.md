# Postflop Explorer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add a "Postflop Solver" mode to the Tauri explorer that lets users configure ranges/pot/stacks/bets, solve flop strategy live with a progress bar, navigate the solved tree, and re-solve filtered ranges on turn/river.

**Architecture:** New `PostflopState` in the Tauri backend wraps the `range-solver` crate's `PostFlopGame`. Five new Tauri commands handle config, solving (async with progress polling), tree navigation, and range filtering between streets. A new `PostflopExplorer.tsx` component drives the UI, sharing matrix rendering utilities extracted from `Explorer.tsx`.

**Tech Stack:** Rust (range-solver, tauri, serde), TypeScript/React (frontend), Axum (devserver mirror)

---

## Dependency Graph

```
Task 1 (types)
  ├→ Task 2 (PostflopState + set_config)
  │    ├→ Task 3 (solve_street async)
  │    │    └→ Task 4 (get_progress polling)
  │    ├→ Task 5 (play_action navigation)
  │    └→ Task 6 (close_street range filtering)
  ├→ Task 7 (devserver endpoints)
  └→ Task 8 (extract matrix-utils.ts)
       └→ Task 9 (PostflopExplorer.tsx - entry point + config)
            └→ Task 10 (PostflopExplorer.tsx - solve + live matrix)
                 └→ Task 11 (PostflopExplorer.tsx - navigation + multi-street)
                      └→ Task 12 (integration test + polish)
```

---

### Task 1: Shared Types — Rust serde structs and TypeScript interfaces

**Files:**
- Create: `crates/tauri-app/src/postflop.rs` (just the types section)
- Modify: `crates/tauri-app/src/lib.rs` (add `pub mod postflop;`)
- Modify: `frontend/src/types.ts` (add new interfaces)

**Context:** We need serde-serializable Rust types that map 1:1 to TypeScript interfaces. The range-solver works with concrete card pairs (`(Card, Card)` where `Card = u8`, encoding `4*rank + suit`), not the 169 canonical hands the existing explorer uses. However, we still display a 13x13 matrix — we aggregate concrete combos into their canonical hand (e.g., all 6 combos of `AhKs`, `AhKd`, ... map to "AKo"). The strategy vec from `game.strategy()` has layout `action_idx * num_hands + hand_idx` where `hand_idx` maps to `game.private_cards(player)[hand_idx]`.

**Step 1: Add Rust types**

Create `crates/tauri-app/src/postflop.rs`:

```rust
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostflopConfig {
    pub oop_range: String,
    pub ip_range: String,
    pub pot: i32,
    pub effective_stack: i32,
    pub oop_bet_sizes: String,
    pub oop_raise_sizes: String,
    pub ip_bet_sizes: String,
    pub ip_raise_sizes: String,
}

impl Default for PostflopConfig {
    fn default() -> Self {
        Self {
            oop_range: "QQ+,AKs,AKo".to_string(),
            ip_range: "JJ-66,AQs-ATs,AQo,KQs".to_string(),
            pot: 30,
            effective_stack: 170,
            oop_bet_sizes: "25%,33%,75%".to_string(),
            oop_raise_sizes: "a".to_string(),
            ip_bet_sizes: "25%,33%,75%".to_string(),
            ip_raise_sizes: "a".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct PostflopConfigSummary {
    pub config: PostflopConfig,
    pub oop_combos: usize,
    pub ip_combos: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PostflopActionInfo {
    pub index: usize,
    pub label: String,
    pub action_type: String, // "fold", "check", "call", "bet", "raise", "allin"
    pub amount: Option<i32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PostflopMatrixCell {
    pub hand: String,
    pub suited: bool,
    pub pair: bool,
    pub probabilities: Vec<f32>,   // one per action, aggregated across combos
    pub combo_count: usize,        // number of concrete combos in this cell
    pub ev: Option<f32>,           // per-hand EV if available
}

#[derive(Debug, Clone, Serialize)]
pub struct PostflopStrategyMatrix {
    pub cells: Vec<Vec<PostflopMatrixCell>>,  // 13x13
    pub actions: Vec<PostflopActionInfo>,
    pub player: usize,                        // 0=OOP, 1=IP
    pub pot: i32,
    pub stacks: [i32; 2],
    pub board: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PostflopProgress {
    pub iteration: u32,
    pub max_iterations: u32,
    pub exploitability: f32,
    pub is_complete: bool,
    pub matrix: Option<PostflopStrategyMatrix>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PostflopStreetResult {
    pub filtered_oop_range: Vec<f32>,   // 1326 weights
    pub filtered_ip_range: Vec<f32>,    // 1326 weights
    pub pot: i32,
    pub effective_stack: i32,
}
```

**Step 2: Register the module**

In `crates/tauri-app/src/lib.rs`, add:
```rust
pub mod postflop;
```

**Step 3: Add TypeScript types**

Append to `frontend/src/types.ts`:

```typescript
// ---------------------------------------------------------------------------
// Postflop Solver types
// ---------------------------------------------------------------------------

export interface PostflopConfig {
  oop_range: string;
  ip_range: string;
  pot: number;
  effective_stack: number;
  oop_bet_sizes: string;
  oop_raise_sizes: string;
  ip_bet_sizes: string;
  ip_raise_sizes: string;
}

export interface PostflopConfigSummary {
  config: PostflopConfig;
  oop_combos: number;
  ip_combos: number;
}

export interface PostflopActionInfo {
  index: number;
  label: string;
  action_type: string;
  amount: number | null;
}

export interface PostflopMatrixCell {
  hand: string;
  suited: boolean;
  pair: boolean;
  probabilities: number[];
  combo_count: number;
  ev: number | null;
}

export interface PostflopStrategyMatrix {
  cells: PostflopMatrixCell[][];
  actions: PostflopActionInfo[];
  player: number;
  pot: number;
  stacks: [number, number];
  board: string[];
}

export interface PostflopProgress {
  iteration: number;
  max_iterations: number;
  exploitability: number;
  is_complete: boolean;
  matrix: PostflopStrategyMatrix | null;
}

export interface PostflopStreetResult {
  filtered_oop_range: number[];
  filtered_ip_range: number[];
  pot: number;
  effective_stack: number;
}
```

**Step 4: Verify compilation**

Run: `cargo check -p poker-solver-tauri`
Expected: PASS (types only, no logic yet)

**Step 5: Commit**

```bash
git add crates/tauri-app/src/postflop.rs crates/tauri-app/src/lib.rs frontend/src/types.ts
git commit -m "feat(postflop-explorer): add shared Rust/TS types for postflop solver mode"
```

---

### Task 2: PostflopState + `postflop_set_config` command

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs` (add state and set_config logic)
- Modify: `crates/tauri-app/src/lib.rs` (re-export state and command)
- Modify: `crates/tauri-app/src/main.rs` (register state and command)
- Modify: `crates/tauri-app/Cargo.toml` (add range-solver dependency)

**Context:** The `PostflopState` manages the game lifecycle. It must be `Send + Sync` because Tauri shares it across async command handlers. We use `parking_lot::RwLock` (already a dependency) for interior mutability. The range-solver's `PostFlopGame` is `Send` but not `Sync`, so we wrap it in a `Mutex` rather than `RwLock`.

**Step 1: Add range-solver dependency**

In `crates/tauri-app/Cargo.toml`, add under `[dependencies]`:
```toml
range-solver = { path = "../range-solver" }
```

**Step 2: Add PostflopState and set_config**

Append to `crates/tauri-app/src/postflop.rs`:

```rust
use parking_lot::{Mutex, RwLock};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{CardConfig, NOT_DEALT};
use range_solver::range::Range;
use range_solver::action_tree::{Action, ActionTree, BoardState, TreeConfig};
use range_solver::PostFlopGame;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

// ---------------------------------------------------------------------------
// Card helpers
// ---------------------------------------------------------------------------

/// Rank names for display (index 0 = Ace, 12 = Deuce).
const RANK_NAMES: [char; 13] = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];

/// Map a range-solver card (4*rank + suit, rank 0=deuce..12=ace) to a
/// canonical hand matrix index. Returns (row, col, suited) where row/col
/// are indices into the 13x13 grid (0=A, 12=2).
fn card_pair_to_matrix(c1: u8, c2: u8) -> (usize, usize, bool) {
    let rank1 = (c1 / 4) as usize;  // 0=deuce .. 12=ace
    let rank2 = (c2 / 4) as usize;
    let suit1 = c1 % 4;
    let suit2 = c2 % 4;
    let suited = suit1 == suit2;

    // Matrix uses descending rank: row/col 0 = Ace (rank 12), row/col 12 = Deuce (rank 0)
    let r1 = 12 - rank1;
    let r2 = 12 - rank2;

    if r1 == r2 {
        // Pair: on diagonal
        (r1, r2, false)
    } else if suited {
        // Suited: above diagonal (smaller row index first)
        (r1.min(r2), r1.max(r2), true)
    } else {
        // Offsuit: below diagonal (larger row index first)
        (r1.max(r2), r1.min(r2), false)
    }
}

/// Build hand label for matrix cell (e.g. "AKs", "QQ", "T9o").
fn matrix_cell_label(row: usize, col: usize) -> (String, bool, bool) {
    let r1 = RANK_NAMES[row];
    let r2 = RANK_NAMES[col];
    if row == col {
        (format!("{r1}{r2}"), false, true)
    } else if col > row {
        (format!("{r1}{r2}s"), true, false)
    } else {
        (format!("{r1}{r2}o"), false, false)
    }
}

/// Convert a range-solver `Action` to a frontend-friendly `PostflopActionInfo`.
fn action_to_info(action: &Action, index: usize) -> PostflopActionInfo {
    match action {
        Action::Fold => PostflopActionInfo {
            index, label: "Fold".to_string(), action_type: "fold".to_string(), amount: None,
        },
        Action::Check => PostflopActionInfo {
            index, label: "Check".to_string(), action_type: "check".to_string(), amount: None,
        },
        Action::Call => PostflopActionInfo {
            index, label: "Call".to_string(), action_type: "call".to_string(), amount: None,
        },
        Action::Bet(amt) => PostflopActionInfo {
            index, label: format!("Bet {amt}"), action_type: "bet".to_string(), amount: Some(*amt),
        },
        Action::Raise(amt) => PostflopActionInfo {
            index, label: format!("Raise {amt}"), action_type: "raise".to_string(), amount: Some(*amt),
        },
        Action::AllIn(amt) => PostflopActionInfo {
            index, label: format!("All-in {amt}"), action_type: "allin".to_string(), amount: Some(*amt),
        },
        _ => PostflopActionInfo {
            index, label: format!("{action}"), action_type: "unknown".to_string(), amount: None,
        },
    }
}

// ---------------------------------------------------------------------------
// PostflopState
// ---------------------------------------------------------------------------

pub struct PostflopState {
    pub config: RwLock<PostflopConfig>,
    pub game: Mutex<Option<PostFlopGame>>,
    /// Current iteration (updated by solve thread).
    pub current_iteration: AtomicU32,
    pub max_iterations: AtomicU32,
    /// Latest exploitability (stored as f32 bits).
    pub exploitability_bits: AtomicU32,
    pub solving: AtomicBool,
    pub solve_complete: AtomicBool,
    /// Snapshot of the latest strategy matrix for progress polling.
    pub matrix_snapshot: RwLock<Option<PostflopStrategyMatrix>>,
    /// Filtered ranges for multi-street solving (1326 weights each).
    pub filtered_oop_weights: RwLock<Option<Vec<f32>>>,
    pub filtered_ip_weights: RwLock<Option<Vec<f32>>>,
}

impl Default for PostflopState {
    fn default() -> Self {
        Self {
            config: RwLock::new(PostflopConfig::default()),
            game: Mutex::new(None),
            current_iteration: AtomicU32::new(0),
            max_iterations: AtomicU32::new(200),
            exploitability_bits: AtomicU32::new(f32::INFINITY.to_bits()),
            solving: AtomicBool::new(false),
            solve_complete: AtomicBool::new(false),
            matrix_snapshot: RwLock::new(None),
            filtered_oop_weights: RwLock::new(None),
            filtered_ip_weights: RwLock::new(None),
        }
    }
}

// ---------------------------------------------------------------------------
// set_config
// ---------------------------------------------------------------------------

/// Build a strategy matrix from the current game state.
/// Aggregates per-combo strategy into the 13x13 canonical hand matrix.
pub fn build_strategy_matrix(game: &PostFlopGame) -> PostflopStrategyMatrix {
    let player = game.current_player();
    let actions = game.available_actions();
    let num_actions = actions.len();
    let num_hands = game.num_private_hands(player);
    let strategy = game.strategy();
    let private_cards = game.private_cards(player);

    let action_infos: Vec<PostflopActionInfo> = actions
        .iter()
        .enumerate()
        .map(|(i, a)| action_to_info(a, i))
        .collect();

    // Aggregate into 13x13 matrix
    let mut cells = Vec::with_capacity(13);
    for row in 0..13 {
        let mut row_cells = Vec::with_capacity(13);
        for col in 0..13 {
            let (label, suited, pair) = matrix_cell_label(row, col);
            row_cells.push(PostflopMatrixCell {
                hand: label,
                suited,
                pair,
                probabilities: vec![0.0; num_actions],
                combo_count: 0,
                ev: None,
            });
        }
        cells.push(row_cells);
    }

    // Map each concrete hand to its matrix cell and accumulate
    for hand_idx in 0..num_hands {
        let (c1, c2) = private_cards[hand_idx];
        let (row, col, _) = card_pair_to_matrix(c1, c2);
        let cell = &mut cells[row][col];
        cell.combo_count += 1;
        for action_idx in 0..num_actions {
            cell.probabilities[action_idx] += strategy[action_idx * num_hands + hand_idx];
        }
    }

    // Normalize: divide accumulated probs by combo count
    for row in &mut cells {
        for cell in row.iter_mut() {
            if cell.combo_count > 0 {
                let n = cell.combo_count as f32;
                for p in &mut cell.probabilities {
                    *p /= n;
                }
            }
        }
    }

    let board: Vec<String> = game.current_board().iter().map(|&c| {
        let rank = c / 4;
        let suit = c % 4;
        let r = RANK_NAMES[12 - rank as usize];
        let s = ['c', 'd', 'h', 's'][suit as usize];
        format!("{r}{s}")
    }).collect();

    let bet_amounts = game.total_bet_amount();

    PostflopStrategyMatrix {
        cells,
        actions: action_infos,
        player,
        pot: game.tree_config().starting_pot + bet_amounts[0] + bet_amounts[1],
        stacks: [
            game.tree_config().effective_stack - bet_amounts[0],
            game.tree_config().effective_stack - bet_amounts[1],
        ],
        board,
    }
}

pub fn postflop_set_config_core(
    state: &PostflopState,
    config: PostflopConfig,
) -> Result<PostflopConfigSummary, String> {
    // Validate ranges parse
    let oop_range: Range = config.oop_range.parse()
        .map_err(|e: String| format!("Invalid OOP range: {e}"))?;
    let ip_range: Range = config.ip_range.parse()
        .map_err(|e: String| format!("Invalid IP range: {e}"))?;

    // Validate bet sizes parse
    BetSizeOptions::try_from((config.oop_bet_sizes.as_str(), config.oop_raise_sizes.as_str()))
        .map_err(|e| format!("Invalid OOP bet sizes: {e}"))?;
    BetSizeOptions::try_from((config.ip_bet_sizes.as_str(), config.ip_raise_sizes.as_str()))
        .map_err(|e| format!("Invalid IP bet sizes: {e}"))?;

    // Count combos (using a dummy flop to build card config — just for counting)
    // We count non-zero weight entries in the range data
    let oop_combos = (0..1326).filter(|&i| {
        let (c1, c2) = index_to_card_pair(i);
        oop_range.get_weight_by_cards(c1, c2) > 0.0
    }).count();
    let ip_combos = (0..1326).filter(|&i| {
        let (c1, c2) = index_to_card_pair(i);
        ip_range.get_weight_by_cards(c1, c2) > 0.0
    }).count();

    let summary = PostflopConfigSummary {
        config: config.clone(),
        oop_combos,
        ip_combos,
    };

    *state.config.write() = config;
    // Clear filtered ranges when config changes
    *state.filtered_oop_weights.write() = None;
    *state.filtered_ip_weights.write() = None;

    Ok(summary)
}

/// Map a combo index (0..1326) to a card pair.
/// Index follows the triangular pattern: for c1 < c2 in 0..52,
/// index = c1 * (101 - c1) / 2 + c2 - c1 - 1.
fn index_to_card_pair(index: usize) -> (u8, u8) {
    let mut c1 = 0u8;
    let mut remaining = index;
    loop {
        let row_size = (51 - c1) as usize;
        if remaining < row_size {
            return (c1, c1 + 1 + remaining as u8);
        }
        remaining -= row_size;
        c1 += 1;
    }
}
```

**Step 3: Check if `Range` has `get_weight_by_cards`**

The range-solver's `Range` struct stores weights in a `[f32; 1326]` array indexed by card pairs. Check the exact accessor method name. If it's not `get_weight_by_cards`, the implementer should find the equivalent method (likely something using two `u8` card IDs to look up the weight). The range-solver's `range.rs` file has accessor functions — use `data[pair_index(c1, c2)]` where `pair_index` computes the triangular index.

**Step 4: Wire up Tauri command**

In `crates/tauri-app/src/lib.rs`, add the re-export:
```rust
pub use postflop::{PostflopState, postflop_set_config_core};
```

In `crates/tauri-app/src/main.rs`, add:
```rust
// In the manage() calls:
.manage(poker_solver_tauri::PostflopState::default())

// As a new Tauri command function (can live in postflop.rs):
#[tauri::command]
pub async fn postflop_set_config(
    state: tauri::State<'_, PostflopState>,
    config: PostflopConfig,
) -> Result<PostflopConfigSummary, String> {
    postflop_set_config_core(&state, config)
}
```

Register in `invoke_handler`:
```rust
poker_solver_tauri::postflop_set_config,
```

**Step 5: Verify compilation**

Run: `cargo check -p poker-solver-tauri`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/tauri-app/src/postflop.rs crates/tauri-app/src/lib.rs crates/tauri-app/src/main.rs crates/tauri-app/Cargo.toml
git commit -m "feat(postflop-explorer): PostflopState and set_config command"
```

---

### Task 3: `postflop_solve_street` — async background solve

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs`
- Modify: `crates/tauri-app/src/lib.rs` (re-export)
- Modify: `crates/tauri-app/src/main.rs` (register command)

**Context:** The solve must run on a background thread so the UI stays responsive. We use `std::thread::spawn` (not tokio — the solve is CPU-bound). The solve thread calls `solve_step()` in a loop, updating atomic counters after each iteration. Every N iterations it also snapshots the strategy matrix into shared state so `get_progress` can return it.

**Step 1: Implement solve_street**

Add to `crates/tauri-app/src/postflop.rs`:

```rust
use range_solver::{solve_step, compute_exploitability};
use range_solver::card::flop_from_str;
use range_solver::interface::Game;

pub fn postflop_solve_street_core(
    state: &PostflopState,
    board: Vec<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
) -> Result<(), String> {
    if state.solving.load(Ordering::Relaxed) {
        return Err("Solve already in progress".to_string());
    }

    let config = state.config.read().clone();
    let iterations = max_iterations.unwrap_or(200);
    let target = target_exploitability.unwrap_or(0.01);

    // Parse board
    if board.len() < 3 || board.len() > 5 {
        return Err("Board must have 3-5 cards".to_string());
    }
    let flop_str = format!("{}{}{}", board[0], board[1], board[2]);
    let flop = flop_from_str(&flop_str).map_err(|e| format!("Invalid flop: {e}"))?;

    let turn = if board.len() >= 4 {
        range_solver::card::card_from_str(&board[3]).map_err(|e| format!("Invalid turn: {e}"))?
    } else {
        NOT_DEALT
    };
    let river = if board.len() >= 5 {
        range_solver::card::card_from_str(&board[4]).map_err(|e| format!("Invalid river: {e}"))?
    } else {
        NOT_DEALT
    };

    let initial_state = if river != NOT_DEALT {
        BoardState::River
    } else if turn != NOT_DEALT {
        BoardState::Turn
    } else {
        BoardState::Flop
    };

    // Parse ranges — use filtered ranges if available (multi-street)
    let mut oop_range: Range = config.oop_range.parse()
        .map_err(|e: String| format!("Invalid OOP range: {e}"))?;
    let mut ip_range: Range = config.ip_range.parse()
        .map_err(|e: String| format!("Invalid IP range: {e}"))?;

    // Apply filtered weights if this is a continuation street
    if let Some(weights) = state.filtered_oop_weights.read().as_ref() {
        oop_range.set_raw_weights(weights);
    }
    if let Some(weights) = state.filtered_ip_weights.read().as_ref() {
        ip_range.set_raw_weights(weights);
    }

    // Parse bet sizes
    let oop_sizes = BetSizeOptions::try_from((config.oop_bet_sizes.as_str(), config.oop_raise_sizes.as_str()))
        .map_err(|e| format!("Invalid OOP bet sizes: {e}"))?;
    let ip_sizes = BetSizeOptions::try_from((config.ip_bet_sizes.as_str(), config.ip_raise_sizes.as_str()))
        .map_err(|e| format!("Invalid IP bet sizes: {e}"))?;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop,
        turn,
        river,
    };

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: config.pot,
        effective_stack: config.effective_stack,
        flop_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        turn_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        river_bet_sizes: [oop_sizes, ip_sizes],
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config)
        .map_err(|e| format!("Failed to build action tree: {e}"))?;

    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;

    game.allocate_memory(false);

    // Reset progress counters
    state.current_iteration.store(0, Ordering::Relaxed);
    state.max_iterations.store(iterations, Ordering::Relaxed);
    state.exploitability_bits.store(f32::INFINITY.to_bits(), Ordering::Relaxed);
    state.solve_complete.store(false, Ordering::Relaxed);
    state.solving.store(true, Ordering::Relaxed);
    *state.matrix_snapshot.write() = None;

    // Take an initial snapshot before solving
    let initial_matrix = build_strategy_matrix(&game);
    *state.matrix_snapshot.write() = Some(initial_matrix);

    // Store game temporarily so we can move it to the thread
    // We need to move the game into the thread, solve, then put it back
    *state.game.lock() = None;

    // Use raw pointer dance to share state with thread
    // Safety: PostflopState is managed by Tauri and lives for the app lifetime
    let state_ptr = state as *const PostflopState as usize;

    std::thread::spawn(move || {
        // Safety: state lives for app lifetime, we only access atomics + RwLock
        let state = unsafe { &*(state_ptr as *const PostflopState) };

        let exploitability = compute_exploitability(&game);
        state.exploitability_bits.store(exploitability.to_bits(), Ordering::Relaxed);

        for t in 0..iterations {
            if !state.solving.load(Ordering::Relaxed) {
                break; // cancelled
            }

            solve_step(&game, t);

            let exploit = compute_exploitability(&game);
            state.current_iteration.store(t + 1, Ordering::Relaxed);
            state.exploitability_bits.store(exploit.to_bits(), Ordering::Relaxed);

            // Snapshot matrix every 10 iterations for live updates
            if (t + 1) % 10 == 0 || t + 1 == iterations || exploit <= target {
                let matrix = build_strategy_matrix(&game);
                *state.matrix_snapshot.write() = Some(matrix);
            }

            if exploit <= target {
                break;
            }
        }

        state.solve_complete.store(true, Ordering::Relaxed);
        state.solving.store(false, Ordering::Relaxed);

        // Put the solved game back into state
        *state.game.lock() = Some(game);
    });

    Ok(())
}
```

**Important implementation note:** The `state_ptr` approach above is a sketch. The implementer should use `Arc`-based sharing instead. Since `PostflopState` is managed by Tauri as `State<'_, PostflopState>`, the recommended pattern is:
1. Make `PostflopState` wrapped in `Arc` at registration time
2. Clone the `Arc` into the spawned thread
3. This avoids the unsafe raw pointer cast

Alternatively, move all solve-related atomics and the matrix snapshot into a separate `Arc<SolveProgress>` struct that can be safely cloned into the thread.

**Step 2: Register command**

Add Tauri command wrapper and register in `main.rs` (same pattern as Task 2).

```rust
#[tauri::command]
pub async fn postflop_solve_street(
    state: tauri::State<'_, PostflopState>,
    board: Vec<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
) -> Result<(), String> {
    postflop_solve_street_core(&state, board, max_iterations, target_exploitability)
}
```

**Step 3: Check that `Range` has `set_raw_weights`**

The implementer needs to check `range-solver/src/range.rs` for a method to set the internal `[f32; 1326]` weights directly. If no such method exists, add a `pub fn set_raw_weights(&mut self, weights: &[f32])` that copies weights into `self.data`. This is needed for filtered range propagation between streets.

**Step 4: Verify compilation**

Run: `cargo check -p poker-solver-tauri`

**Step 5: Commit**

```bash
git commit -m "feat(postflop-explorer): async background solve with progress snapshots"
```

---

### Task 4: `postflop_get_progress` — polling endpoint

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs`
- Modify: `crates/tauri-app/src/lib.rs`
- Modify: `crates/tauri-app/src/main.rs`

**Context:** The frontend polls this every ~2 seconds to get the latest matrix snapshot and progress counters. It reads from atomics and the `RwLock<Option<PostflopStrategyMatrix>>` — no blocking the solve thread.

**Step 1: Implement get_progress**

```rust
pub fn postflop_get_progress_core(state: &PostflopState) -> PostflopProgress {
    let iteration = state.current_iteration.load(Ordering::Relaxed);
    let max_iterations = state.max_iterations.load(Ordering::Relaxed);
    let exploitability = f32::from_bits(state.exploitability_bits.load(Ordering::Relaxed));
    let is_complete = state.solve_complete.load(Ordering::Relaxed);
    let matrix = state.matrix_snapshot.read().clone();

    PostflopProgress {
        iteration,
        max_iterations,
        exploitability,
        is_complete,
        matrix,
    }
}
```

**Step 2: Register Tauri command**

```rust
#[tauri::command]
pub fn postflop_get_progress(
    state: tauri::State<'_, PostflopState>,
) -> PostflopProgress {
    postflop_get_progress_core(&state)
}
```

**Step 3: Verify and commit**

Run: `cargo check -p poker-solver-tauri`

```bash
git commit -m "feat(postflop-explorer): add get_progress polling endpoint"
```

---

### Task 5: `postflop_play_action` — navigate solved tree

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs`
- Modify: `crates/tauri-app/src/lib.rs`
- Modify: `crates/tauri-app/src/main.rs`

**Context:** After the solve completes, the frontend can navigate the solved tree by playing actions. `game.play(action_index)` moves to a child node. If the child is a player node, we return the strategy matrix there. If it's terminal, we return terminal info. If it's a chance node, the frontend needs to provide a card to deal.

**Step 1: Implement play_action**

```rust
#[derive(Debug, Clone, Serialize)]
pub struct PostflopPlayResult {
    pub matrix: Option<PostflopStrategyMatrix>,
    pub is_terminal: bool,
    pub is_chance: bool,
    pub current_player: Option<usize>,
    pub pot: i32,
    pub stacks: [i32; 2],
}

pub fn postflop_play_action_core(
    state: &PostflopState,
    action: usize,
) -> Result<PostflopPlayResult, String> {
    let mut game = state.game.lock();
    let game = game.as_mut().ok_or("No game loaded")?;

    game.play(action);

    let bet_amounts = game.total_bet_amount();
    let tree_config = game.tree_config();
    let pot = tree_config.starting_pot + bet_amounts[0] + bet_amounts[1];
    let stacks = [
        tree_config.effective_stack - bet_amounts[0],
        tree_config.effective_stack - bet_amounts[1],
    ];

    if game.is_terminal_node() {
        return Ok(PostflopPlayResult {
            matrix: None,
            is_terminal: true,
            is_chance: false,
            current_player: None,
            pot,
            stacks,
        });
    }

    if game.is_chance_node() {
        return Ok(PostflopPlayResult {
            matrix: None,
            is_terminal: false,
            is_chance: true,
            current_player: None,
            pot,
            stacks,
        });
    }

    let matrix = build_strategy_matrix(game);
    Ok(PostflopPlayResult {
        matrix: Some(matrix),
        is_terminal: false,
        is_chance: false,
        current_player: Some(game.current_player()),
        pot,
        stacks,
    })
}
```

**Step 2: Register Tauri command, verify, commit**

```bash
git commit -m "feat(postflop-explorer): play_action tree navigation command"
```

---

### Task 6: `postflop_close_street` — range filtering and street transition

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs`
- Modify: `crates/range-solver/src/range.rs` (add `set_raw_weights` if needed)
- Modify: `crates/tauri-app/src/lib.rs`
- Modify: `crates/tauri-app/src/main.rs`

**Context:** When a street closes (e.g., check → bet → call), we need to:
1. Walk the solved tree through the action history
2. At each decision node, multiply the acting player's range weights by their strategy frequency for that action
3. Store the filtered ranges for the next street's solve
4. Compute the new pot and effective stack

The range-solver stores hands as `private_cards[player]: Vec<(Card, Card)>`, and strategy is indexed by hand position in that vec. We need to map back to the 1326-combo weight array.

**Step 1: Add `get_raw_weights` and `set_raw_weights` to Range**

Check if `range-solver/src/range.rs` already has these. If not, add:

```rust
impl Range {
    /// Get the raw 1326-element weight array.
    pub fn raw_weights(&self) -> &[f32; 1326] {
        &self.data
    }

    /// Set the raw 1326-element weight array.
    pub fn set_raw_weights(&mut self, weights: &[f32]) {
        assert!(weights.len() == 1326, "Expected 1326 weights, got {}", weights.len());
        self.data.copy_from_slice(weights);
    }
}
```

Also need a helper to convert a `(Card, Card)` pair to a combo index (0..1326):

```rust
/// Convert two card IDs (0..51) to a combo index in the 1326-element array.
/// Cards must satisfy c1 < c2.
pub fn pair_index(c1: u8, c2: u8) -> usize {
    let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
    (lo as usize) * (101 - lo as usize) / 2 + (hi as usize) - (lo as usize) - 1
}
```

**Step 2: Implement close_street**

```rust
pub fn postflop_close_street_core(
    state: &PostflopState,
    action_history: Vec<usize>,
) -> Result<PostflopStreetResult, String> {
    let mut game_guard = state.game.lock();
    let game = game_guard.as_mut().ok_or("No game loaded")?;

    // Reset game to root
    game.back_to_root();

    // Walk through the action history, collecting strategy weights
    let config = state.config.read().clone();
    let mut oop_range: Range = config.oop_range.parse()
        .map_err(|e: String| format!("Invalid OOP range: {e}"))?;
    let mut ip_range: Range = config.ip_range.parse()
        .map_err(|e: String| format!("Invalid IP range: {e}"))?;

    // Start with existing filtered weights if present
    let mut oop_weights: Vec<f32> = state.filtered_oop_weights.read()
        .as_ref()
        .map(|w| w.clone())
        .unwrap_or_else(|| oop_range.raw_weights().to_vec());
    let mut ip_weights: Vec<f32> = state.filtered_ip_weights.read()
        .as_ref()
        .map(|w| w.clone())
        .unwrap_or_else(|| ip_range.raw_weights().to_vec());

    for &action_idx in &action_history {
        if game.is_terminal_node() || game.is_chance_node() {
            break;
        }

        let player = game.current_player();
        let num_hands = game.num_private_hands(player);
        let strategy = game.strategy();
        let private_cards = game.private_cards(player);

        // Multiply the acting player's weights by strategy[action_idx]
        let weights = if player == 0 { &mut oop_weights } else { &mut ip_weights };
        for hand_idx in 0..num_hands {
            let (c1, c2) = private_cards[hand_idx];
            let combo_idx = range_solver::range::pair_index(c1, c2);
            let action_prob = strategy[action_idx * num_hands + hand_idx];
            weights[combo_idx] *= action_prob;
        }

        game.play(action_idx);
    }

    let bet_amounts = game.total_bet_amount();
    let tree_config = game.tree_config();
    let pot = tree_config.starting_pot + bet_amounts[0] + bet_amounts[1];
    let effective_stack = tree_config.effective_stack - bet_amounts[0].max(bet_amounts[1]);

    // Store filtered weights
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

**Step 3: Register command, verify, commit**

```bash
git commit -m "feat(postflop-explorer): close_street range filtering between streets"
```

---

### Task 7: Dev server endpoints

**Files:**
- Modify: `crates/devserver/src/main.rs`
- Modify: `crates/devserver/Cargo.toml` (add range-solver dependency if needed)

**Context:** Mirror all 5 postflop commands as POST endpoints, following the exact pattern of existing handlers. The devserver needs its own `PostflopState` instance in the Axum app state.

**Step 1: Add PostflopState to devserver**

The devserver currently uses `Arc<ExplorationState>` as `AppState`. We need to add `PostflopState` alongside it. Create a combined state struct:

```rust
use poker_solver_tauri::PostflopState;

struct CombinedState {
    exploration: ExplorationState,
    postflop: PostflopState,
}

type AppState = Arc<CombinedState>;
```

Or simpler: add a second `with_state` layer. Axum supports multiple state extractors. The simplest approach is to put `PostflopState` behind its own `Arc` and use `Extension`:

```rust
let exploration_state: Arc<ExplorationState> = Arc::new(ExplorationState::default());
let postflop_state: Arc<PostflopState> = Arc::new(PostflopState::default());
```

Then use Axum's `.layer(Extension(postflop_state.clone()))` and extract with `Extension(state): Extension<Arc<PostflopState>>`.

**Step 2: Add request param structs**

```rust
#[derive(Deserialize)]
struct PostflopConfigParams {
    config: poker_solver_tauri::postflop::PostflopConfig,
}

#[derive(Deserialize)]
struct PostflopSolveParams {
    board: Vec<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
}

#[derive(Deserialize)]
struct PostflopActionParams {
    action: usize,
}

#[derive(Deserialize)]
struct PostflopCloseStreetParams {
    action_history: Vec<usize>,
}
```

**Step 3: Add handlers**

```rust
async fn handle_postflop_set_config(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopConfigParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::postflop_set_config_core(&state, params.config))
}

async fn handle_postflop_solve_street(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopSolveParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::postflop_solve_street_core(
        &state, params.board, params.max_iterations, params.target_exploitability,
    ))
}

async fn handle_postflop_get_progress(
    Extension(state): Extension<Arc<PostflopState>>,
) -> Json<serde_json::Value> {
    to_json_value(poker_solver_tauri::postflop_get_progress_core(&state))
}

async fn handle_postflop_play_action(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopActionParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::postflop_play_action_core(&state, params.action))
}

async fn handle_postflop_close_street(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopCloseStreetParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::postflop_close_street_core(&state, params.action_history))
}
```

**Step 4: Register routes**

```rust
.route("/api/postflop_set_config", post(handle_postflop_set_config))
.route("/api/postflop_solve_street", post(handle_postflop_solve_street))
.route("/api/postflop_get_progress", post(handle_postflop_get_progress))
.route("/api/postflop_play_action", post(handle_postflop_play_action))
.route("/api/postflop_close_street", post(handle_postflop_close_street))
```

**Step 5: Test with curl**

```bash
cargo run -p poker-solver-devserver &

# Set config
curl -s -X POST http://localhost:3001/api/postflop_set_config \
  -H 'Content-Type: application/json' \
  -d '{"config":{"oop_range":"QQ+,AKs","ip_range":"JJ-66,AQs","pot":30,"effective_stack":170,"oop_bet_sizes":"33%,75%","oop_raise_sizes":"a","ip_bet_sizes":"33%,75%","ip_raise_sizes":"a"}}' | jq .

# Start solve
curl -s -X POST http://localhost:3001/api/postflop_solve_street \
  -H 'Content-Type: application/json' \
  -d '{"board":["Ah","Kd","7c"]}' | jq .

# Poll progress
sleep 2
curl -s -X POST http://localhost:3001/api/postflop_get_progress \
  -H 'Content-Type: application/json' -d '{}' | jq '.iteration, .exploitability, .is_complete'
```

**Step 6: Commit**

```bash
git commit -m "feat(postflop-explorer): devserver mirror endpoints for postflop commands"
```

---

### Task 8: Extract shared matrix utilities from Explorer.tsx

**Files:**
- Create: `frontend/src/matrix-utils.ts`
- Modify: `frontend/src/Explorer.tsx` (import from matrix-utils instead of local definitions)

**Context:** Both `Explorer.tsx` and the new `PostflopExplorer.tsx` need the same color/formatting functions. Extract these into a shared module. Do NOT change any behavior — pure refactor.

**Step 1: Create `matrix-utils.ts`**

Extract these functions from `Explorer.tsx`:
- `formatEV(potFraction: number): string`
- `sortedBetActions(actions: ActionInfo[]): ActionInfo[]`
- `displayOrderIndices(actions: ActionInfo[]): number[]`
- `displayOrderActions(actions: ActionInfo[]): ActionInfo[]`
- `formatActionLabel(action: ActionInfo): string`
- `getActionColor(action: ActionInfo, actions: ActionInfo[]): string`
- `SUIT_COLORS` and `SUIT_SYMBOLS` constants
- `matrixToHandIndex(row: number, col: number): number`

These functions should work with both `ActionInfo` (existing explorer) and `PostflopActionInfo` (new postflop explorer). Use a minimal interface:

```typescript
export interface ActionLike {
  action_type: string;
  label: string;
  id?: string;
  index?: number;
}
```

And make the utility functions accept `ActionLike[]` instead of `ActionInfo[]`.

**Step 2: Update Explorer.tsx imports**

Replace local definitions with imports from `matrix-utils.ts`. Run the frontend build to verify nothing breaks:

```bash
cd frontend && npm run build
```

**Step 3: Commit**

```bash
git commit -m "refactor(frontend): extract shared matrix utilities from Explorer.tsx"
```

---

### Task 9: PostflopExplorer.tsx — entry point split and config modal

**Files:**
- Create: `frontend/src/PostflopExplorer.tsx`
- Modify: `frontend/src/Explorer.tsx` (split entry point)
- Modify: `frontend/src/App.tsx` (optionally — or handle within Explorer)

**Context:** The existing Explorer shows a single "Load Dataset" card when no bundle is loaded (lines 1368-1381). We need to split this into two stacked cards. The cleanest approach: keep the split inside Explorer.tsx itself, using a state flag to switch between the existing explorer flow and the new PostflopExplorer component.

**Step 1: Create PostflopExplorer.tsx skeleton**

```tsx
import { useState, useCallback, useEffect, useRef } from 'react';
import { invoke } from './invoke';
import {
  PostflopConfig,
  PostflopConfigSummary,
  PostflopActionInfo,
  PostflopMatrixCell,
  PostflopStrategyMatrix,
  PostflopProgress,
  PostflopStreetResult,
} from './types';
import {
  formatEV,
  getActionColor,
  displayOrderIndices,
  formatActionLabel,
  SUIT_COLORS,
  SUIT_SYMBOLS,
} from './matrix-utils';

interface PostflopExplorerProps {
  onBack: () => void;  // return to mode selection
}

export default function PostflopExplorer({ onBack }: PostflopExplorerProps) {
  // Config state
  const [config, setConfig] = useState<PostflopConfig>({
    oop_range: 'QQ+,AKs,AKo',
    ip_range: 'JJ-66,AQs-ATs,AQo,KQs',
    pot: 30,
    effective_stack: 170,
    oop_bet_sizes: '25%,33%,75%',
    oop_raise_sizes: 'a',
    ip_bet_sizes: '25%,33%,75%',
    ip_raise_sizes: 'a',
  });
  const [configSummary, setConfigSummary] = useState<PostflopConfigSummary | null>(null);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);

  // Board input
  const [boardInput, setBoardInput] = useState('');

  // Action history for current street
  const [actionHistory, setActionHistory] = useState<{index: number; info: PostflopActionInfo}[]>([]);

  // Solve state
  const [matrix, setMatrix] = useState<PostflopStrategyMatrix | null>(null);
  const [solving, setSolving] = useState(false);
  const [progress, setProgress] = useState<PostflopProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Initialize config on mount
  useEffect(() => {
    invoke<PostflopConfigSummary>('postflop_set_config', { config })
      .then(setConfigSummary)
      .catch((e) => setError(String(e)));
  }, []);

  const handleConfigSubmit = useCallback(async (newConfig: PostflopConfig) => {
    setConfigError(null);
    try {
      const summary = await invoke<PostflopConfigSummary>('postflop_set_config', { config: newConfig });
      setConfig(newConfig);
      setConfigSummary(summary);
      setShowConfigModal(false);
    } catch (e) {
      setConfigError(String(e));
    }
  }, []);

  return (
    <div className="explorer-root">
      {error && <div className="error">{error}</div>}

      {/* Action strip */}
      <div className="action-strip">
        {/* Back button */}
        <div className="action-block" onClick={onBack} style={{ cursor: 'pointer', minWidth: 40, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </div>

        {/* Config card */}
        <div className="action-block" onClick={() => setShowConfigModal(true)} style={{ cursor: 'pointer' }}>
          <div style={{ fontSize: '0.75em', opacity: 0.7 }}>Config</div>
          <div style={{ fontSize: '0.85em' }}>
            {config.pot} pot / {config.effective_stack} eff
          </div>
        </div>

        {/* Flop input */}
        <div className="action-block">
          <input
            type="text"
            placeholder="Ah Kd 7c"
            value={boardInput}
            onChange={(e) => setBoardInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSolve();
            }}
            style={{
              background: 'transparent', border: 'none', color: 'inherit',
              fontSize: '0.85em', width: '100px', outline: 'none',
            }}
          />
        </div>

        {/* Action history blocks */}
        {actionHistory.map((item, i) => (
          <div key={i} className="action-block" style={{
            borderLeft: `3px solid ${getActionColor(item.info, matrix?.actions ?? [])}`,
          }}>
            <span style={{ fontSize: '0.85em' }}>{formatActionLabel(item.info)}</span>
          </div>
        ))}
      </div>

      {/* Progress bar */}
      {solving && progress && (
        <div className="progress-bar-container">
          <div className="progress-bar" style={{
            width: `${(progress.iteration / progress.max_iterations) * 100}%`,
          }} />
          <span className="progress-text">
            {progress.iteration}/{progress.max_iterations} — exploit: {progress.exploitability.toExponential(2)}
          </span>
        </div>
      )}

      {/* Matrix */}
      {matrix && (
        <div className="matrix-container">
          <div className="hand-matrix">
            {matrix.cells.map((row, rowIdx) => (
              <div key={rowIdx} className="matrix-row">
                {row.map((cell, colIdx) => (
                  <PostflopHandCell
                    key={colIdx}
                    cell={cell}
                    actions={matrix.actions}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Config modal */}
      {showConfigModal && (
        <ConfigModal
          config={config}
          error={configError}
          onSubmit={handleConfigSubmit}
          onClose={() => setShowConfigModal(false)}
        />
      )}
    </div>
  );
}

// Minimal hand cell component for postflop matrix
function PostflopHandCell({ cell, actions }: { cell: PostflopMatrixCell; actions: PostflopActionInfo[] }) {
  const unreachable = cell.combo_count === 0;
  return (
    <div className={`matrix-cell ${unreachable ? 'unreachable' : ''}`}>
      <span className="cell-label">{cell.hand}</span>
      {!unreachable && (
        <div className="cell-bars">
          {cell.probabilities.map((prob, i) => (
            prob > 0.005 && (
              <div
                key={i}
                className="cell-bar"
                style={{
                  width: `${prob * 100}%`,
                  backgroundColor: getActionColor(actions[i], actions),
                }}
              />
            )
          ))}
        </div>
      )}
    </div>
  );
}

// Config editing modal
function ConfigModal({ config, error, onSubmit, onClose }: {
  config: PostflopConfig;
  error: string | null;
  onSubmit: (config: PostflopConfig) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState<PostflopConfig>({ ...config });

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h3>Game Configuration</h3>

        <label>OOP Range</label>
        <textarea
          value={draft.oop_range}
          onChange={(e) => setDraft({ ...draft, oop_range: e.target.value })}
          rows={2}
        />

        <label>IP Range</label>
        <textarea
          value={draft.ip_range}
          onChange={(e) => setDraft({ ...draft, ip_range: e.target.value })}
          rows={2}
        />

        <div className="modal-row">
          <div>
            <label>Pot</label>
            <input type="number" value={draft.pot}
              onChange={(e) => setDraft({ ...draft, pot: parseInt(e.target.value) || 0 })} />
          </div>
          <div>
            <label>Effective Stack</label>
            <input type="number" value={draft.effective_stack}
              onChange={(e) => setDraft({ ...draft, effective_stack: parseInt(e.target.value) || 0 })} />
          </div>
        </div>

        <h4>Bet Sizes</h4>
        <div className="modal-row">
          <div>
            <label>OOP Bet</label>
            <input value={draft.oop_bet_sizes}
              onChange={(e) => setDraft({ ...draft, oop_bet_sizes: e.target.value })} />
          </div>
          <div>
            <label>OOP Raise</label>
            <input value={draft.oop_raise_sizes}
              onChange={(e) => setDraft({ ...draft, oop_raise_sizes: e.target.value })} />
          </div>
        </div>
        <div className="modal-row">
          <div>
            <label>IP Bet</label>
            <input value={draft.ip_bet_sizes}
              onChange={(e) => setDraft({ ...draft, ip_bet_sizes: e.target.value })} />
          </div>
          <div>
            <label>IP Raise</label>
            <input value={draft.ip_raise_sizes}
              onChange={(e) => setDraft({ ...draft, ip_raise_sizes: e.target.value })} />
          </div>
        </div>

        {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}

        <div className="modal-buttons">
          <button onClick={onClose}>Cancel</button>
          <button onClick={() => onSubmit(draft)}>Apply</button>
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Modify Explorer.tsx entry point**

Replace the "no data loaded" block (around line 1368-1381) with a vertical split:

```tsx
{!bundleInfo && !loading && !showDatasetPicker && !showPostflop && (
  <div className="mode-selection">
    <div className="mode-card" onClick={handleLoadDataset}>
      <div className="mode-icon">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
          <line x1="12" y1="11" x2="12" y2="17" />
          <line x1="9" y1="14" x2="15" y2="14" />
        </svg>
      </div>
      <span className="mode-label">Load Dataset</span>
      <span className="mode-desc">Open a solved strategy bundle</span>
    </div>
    <div className="mode-card" onClick={() => setShowPostflop(true)}>
      <div className="mode-icon">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M19.439 7.85c-.049.322.059.648.289.878l1.568 1.568c.47.47.706 1.087.706 1.704s-.235 1.233-.706 1.704l-1.611 1.611a.98.98 0 0 1-.837.276c-.47-.07-.802-.48-.968-.925a2.501 2.501 0 1 0-3.214 3.214c.446.166.855.497.925.968a.979.979 0 0 1-.276.837l-1.61 1.61a2.404 2.404 0 0 1-1.705.707 2.402 2.402 0 0 1-1.704-.706l-1.568-1.568a1.026 1.026 0 0 0-.877-.29c-.493.074-.84.504-1.02.968a2.5 2.5 0 1 1-3.237-3.237c.464-.18.894-.527.967-1.02a1.026 1.026 0 0 0-.289-.877l-1.568-1.568A2.402 2.402 0 0 1 1.998 12c0-.617.236-1.234.706-1.704L4.315 8.685a.98.98 0 0 1 .837-.276c.47.07.802.48.968.925a2.501 2.501 0 1 0 3.214-3.214c-.446-.166-.855-.497-.925-.968a.979.979 0 0 1 .276-.837l1.61-1.61a2.404 2.404 0 0 1 1.705-.707c.617 0 1.234.236 1.704.706l1.568 1.568c.23.23.556.338.877.29.493-.074.84-.504 1.02-.968a2.5 2.5 0 1 1 3.237 3.237c-.464.18-.894.527-.967 1.02Z" />
        </svg>
      </div>
      <span className="mode-label">Postflop Solver</span>
      <span className="mode-desc">Solve ranges on a specific board</span>
    </div>
  </div>
)}

{showPostflop && (
  <PostflopExplorer onBack={() => setShowPostflop(false)} />
)}
```

Add `showPostflop` state and the import:

```tsx
import PostflopExplorer from './PostflopExplorer';
// ...
const [showPostflop, setShowPostflop] = useState(false);
```

**Step 3: Add CSS for mode selection and modal**

Add to `frontend/src/App.css`:

```css
/* Mode selection (vertical split) */
.mode-selection {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 24px;
  height: 100%;
  justify-content: center;
  align-items: center;
}
.mode-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 32px 48px;
  border-radius: 12px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  cursor: pointer;
  transition: all 0.15s;
  width: 280px;
}
.mode-card:hover {
  background: rgba(255,255,255,0.08);
  border-color: #00d9ff44;
}
.mode-icon { opacity: 0.6; }
.mode-card:hover .mode-icon { opacity: 1; color: #00d9ff; }
.mode-label { font-size: 1.1em; font-weight: 600; }
.mode-desc { font-size: 0.8em; opacity: 0.5; }

/* Config modal */
.modal-overlay {
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.6);
  display: flex; align-items: center; justify-content: center;
  z-index: 100;
}
.modal-content {
  background: #1a1a2e;
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 12px;
  padding: 24px;
  min-width: 400px;
  max-width: 500px;
}
.modal-content h3 { margin: 0 0 16px; }
.modal-content h4 { margin: 16px 0 8px; }
.modal-content label {
  display: block; font-size: 0.8em;
  opacity: 0.6; margin: 8px 0 4px;
}
.modal-content input, .modal-content textarea {
  width: 100%; box-sizing: border-box;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 6px; padding: 6px 8px;
  color: inherit; font-family: inherit; font-size: 0.9em;
}
.modal-row { display: flex; gap: 12px; }
.modal-row > div { flex: 1; }
.modal-buttons {
  display: flex; gap: 8px;
  justify-content: flex-end; margin-top: 16px;
}
.modal-buttons button {
  padding: 6px 16px; border-radius: 6px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.06);
  color: inherit; cursor: pointer;
}
.modal-buttons button:last-child {
  background: #00d9ff22; border-color: #00d9ff44;
}

/* Progress bar */
.progress-bar-container {
  position: relative;
  height: 24px;
  background: rgba(255,255,255,0.04);
  border-radius: 4px;
  margin: 8px 0;
  overflow: hidden;
}
.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, #00d9ff33, #00d9ff66);
  transition: width 0.3s;
}
.progress-text {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.75em; opacity: 0.7;
}
```

**Step 4: Build and verify**

```bash
cd frontend && npm run build
```

**Step 5: Commit**

```bash
git commit -m "feat(postflop-explorer): entry point split, config modal, and PostflopExplorer skeleton"
```

---

### Task 10: PostflopExplorer.tsx — solve + live matrix updates

**Files:**
- Modify: `frontend/src/PostflopExplorer.tsx`

**Context:** Wire up the board input → `postflop_solve_street` → poll `postflop_get_progress` loop. The matrix updates live every ~2 seconds during the solve.

**Step 1: Implement solve trigger and polling**

Add to `PostflopExplorer.tsx`:

```tsx
const pollRef = useRef<number | null>(null);

const handleSolve = useCallback(async () => {
  const cards = boardInput.trim().split(/\s+/);
  if (cards.length < 3 || cards.length > 5) {
    setError('Enter 3-5 cards (e.g. Ah Kd 7c)');
    return;
  }

  setError(null);
  setSolving(true);
  setActionHistory([]);

  try {
    await invoke('postflop_solve_street', { board: cards });
  } catch (e) {
    setError(String(e));
    setSolving(false);
    return;
  }

  // Start polling
  const poll = async () => {
    try {
      const p = await invoke<PostflopProgress>('postflop_get_progress', {});
      setProgress(p);
      if (p.matrix) {
        setMatrix(p.matrix);
      }
      if (p.is_complete) {
        setSolving(false);
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = null;
      }
    } catch (e) {
      setError(String(e));
      setSolving(false);
      if (pollRef.current) clearInterval(pollRef.current);
    }
  };

  // Initial poll immediately, then every 2s
  poll();
  pollRef.current = window.setInterval(poll, 2000);
}, [boardInput]);

// Cleanup on unmount
useEffect(() => {
  return () => {
    if (pollRef.current) clearInterval(pollRef.current);
  };
}, []);
```

**Step 2: Wire the solve button into the flop input block**

Update the flop input action block to trigger solve on Enter:

Already handled in the JSX from Task 9: `onKeyDown={(e) => { if (e.key === 'Enter') handleSolve(); }}`

**Step 3: Build and test with devserver**

```bash
cd frontend && npm run build
cargo run -p poker-solver-devserver &
cd frontend && npm run dev
# Open http://localhost:5173, click Postflop Solver, enter "Ah Kd 7c", press Enter
```

**Step 4: Commit**

```bash
git commit -m "feat(postflop-explorer): live-updating matrix with solve progress polling"
```

---

### Task 11: PostflopExplorer.tsx — action navigation and multi-street

**Files:**
- Modify: `frontend/src/PostflopExplorer.tsx`

**Context:** After the solve completes, clicking an action in the matrix legend or an action button should call `postflop_play_action`. When the result is a chance node (street transition), show a new card input in the action strip. When the user enters the turn/river card, call `postflop_close_street` with the accumulated action history, then `postflop_solve_street` with the new board.

**Step 1: Add action click handler**

```tsx
const handleAction = useCallback(async (actionIndex: number) => {
  if (solving) return;

  try {
    const result = await invoke<PostflopPlayResult>('postflop_play_action', { action: actionIndex });

    const actionInfo = matrix?.actions[actionIndex];
    if (actionInfo) {
      setActionHistory(prev => [...prev, { index: actionIndex, info: actionInfo }]);
    }

    if (result.is_terminal) {
      // Hand is over (fold or showdown)
      setMatrix(null);
      // Could show terminal result here
      return;
    }

    if (result.is_chance) {
      // Street transition — need next card
      // The action strip will show a new card input
      setAwaitingCard(true);
      setMatrix(null);
      return;
    }

    if (result.matrix) {
      setMatrix(result.matrix);
    }
  } catch (e) {
    setError(String(e));
  }
}, [solving, matrix]);
```

**Step 2: Add state for multi-street**

```tsx
const [awaitingCard, setAwaitingCard] = useState(false);
const [nextCardInput, setNextCardInput] = useState('');
const [streetActionHistories, setStreetActionHistories] = useState<number[][]>([]);
```

**Step 3: Handle next street card entry**

```tsx
const handleNextCard = useCallback(async () => {
  const card = nextCardInput.trim();
  if (!card) return;

  setError(null);
  setAwaitingCard(false);

  try {
    // Close the current street — filter ranges
    const currentStreetActions = actionHistory
      .slice(streetActionHistories.flat().length)
      .map(a => a.index);

    const streetResult = await invoke<PostflopStreetResult>('postflop_close_street', {
      action_history: currentStreetActions,
    });

    // Save this street's action history
    setStreetActionHistories(prev => [...prev, currentStreetActions]);

    // Update board and solve next street
    const currentBoard = boardInput.trim().split(/\s+/);
    const newBoard = [...currentBoard, card];
    setBoardInput(newBoard.join(' '));

    // Update config pot/stack for next street
    // (The filtered ranges are already stored in PostflopState on the backend)

    setSolving(true);
    await invoke('postflop_solve_street', {
      board: newBoard,
      max_iterations: 200,
    });

    // Start polling again
    const poll = async () => {
      try {
        const p = await invoke<PostflopProgress>('postflop_get_progress', {});
        setProgress(p);
        if (p.matrix) setMatrix(p.matrix);
        if (p.is_complete) {
          setSolving(false);
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch (e) {
        setError(String(e));
        setSolving(false);
        if (pollRef.current) clearInterval(pollRef.current);
      }
    };
    poll();
    pollRef.current = window.setInterval(poll, 2000);

    setNextCardInput('');
  } catch (e) {
    setError(String(e));
  }
}, [nextCardInput, boardInput, actionHistory, streetActionHistories]);
```

**Step 4: Add action buttons below matrix**

Add clickable action buttons that display below or beside the matrix:

```tsx
{matrix && !solving && (
  <div className="action-buttons">
    {displayOrderIndices(matrix.actions).map((idx) => {
      const action = matrix.actions[idx];
      return (
        <button
          key={idx}
          className="action-button"
          style={{ borderColor: getActionColor(action, matrix.actions) }}
          onClick={() => handleAction(idx)}
        >
          {formatActionLabel(action)}
        </button>
      );
    })}
  </div>
)}
```

**Step 5: Add turn/river card input in action strip**

```tsx
{awaitingCard && (
  <div className="action-block">
    <input
      type="text"
      placeholder="Turn card"
      value={nextCardInput}
      onChange={(e) => setNextCardInput(e.target.value)}
      onKeyDown={(e) => { if (e.key === 'Enter') handleNextCard(); }}
      style={{
        background: 'transparent', border: 'none', color: 'inherit',
        fontSize: '0.85em', width: '60px', outline: 'none',
      }}
      autoFocus
    />
  </div>
)}
```

**Step 6: Build and test**

```bash
cd frontend && npm run build
```

Test the full flow with devserver: set config → enter flop → solve → click action → enter turn → re-solve.

**Step 7: Commit**

```bash
git commit -m "feat(postflop-explorer): action navigation and multi-street solving"
```

---

### Task 12: Integration test and polish

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs` (add test module)
- Modify: `frontend/src/PostflopExplorer.tsx` (CSS polish)

**Context:** Write a Rust integration test that exercises the full flow: set_config → solve_street → get_progress → play_action → close_street. Also polish any CSS issues found during manual testing.

**Step 1: Add Rust integration test**

Add to `crates/tauri-app/src/postflop.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_config_default() {
        let state = PostflopState::default();
        let config = PostflopConfig::default();
        let result = postflop_set_config_core(&state, config);
        assert!(result.is_ok());
        let summary = result.unwrap();
        assert!(summary.oop_combos > 0);
        assert!(summary.ip_combos > 0);
    }

    #[test]
    fn test_set_config_invalid_range() {
        let state = PostflopState::default();
        let mut config = PostflopConfig::default();
        config.oop_range = "INVALID".to_string();
        let result = postflop_set_config_core(&state, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_solve_and_navigate() {
        let state = PostflopState::default();
        let config = PostflopConfig::default();
        postflop_set_config_core(&state, config).unwrap();

        // Start solve (synchronous for test — we call solve_step directly)
        let board = vec!["Ah".to_string(), "Kd".to_string(), "7c".to_string()];
        postflop_solve_street_core(&state, board, Some(50), None).unwrap();

        // Wait for solve to complete
        while state.solving.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        assert!(state.solve_complete.load(Ordering::Relaxed));

        // Check progress
        let progress = postflop_get_progress_core(&state);
        assert!(progress.is_complete);
        assert!(progress.matrix.is_some());
        let matrix = progress.matrix.unwrap();
        assert_eq!(matrix.cells.len(), 13);
        assert_eq!(matrix.cells[0].len(), 13);
        assert!(!matrix.actions.is_empty());

        // Navigate: play the first action
        let result = postflop_play_action_core(&state, 0).unwrap();
        // First action could lead to terminal (fold) or another player node
        assert!(result.is_terminal || result.matrix.is_some() || result.is_chance);
    }

    #[test]
    fn test_close_street_filters_ranges() {
        let state = PostflopState::default();
        let config = PostflopConfig::default();
        postflop_set_config_core(&state, config).unwrap();

        let board = vec!["Ah".to_string(), "Kd".to_string(), "7c".to_string()];
        postflop_solve_street_core(&state, board, Some(50), None).unwrap();

        while state.solving.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // Get available actions and navigate to a non-fold action
        let progress = postflop_get_progress_core(&state);
        let matrix = progress.matrix.unwrap();
        let non_fold_idx = matrix.actions.iter()
            .find(|a| a.action_type != "fold")
            .map(|a| a.index)
            .unwrap();

        // Close street with a simple action sequence
        let result = postflop_close_street_core(&state, vec![non_fold_idx]);
        assert!(result.is_ok());
        let street_result = result.unwrap();
        assert_eq!(street_result.filtered_oop_range.len(), 1326);
        assert_eq!(street_result.filtered_ip_range.len(), 1326);
    }
}
```

**Step 2: Run tests**

```bash
cargo test -p poker-solver-tauri -- postflop --nocapture
```

**Step 3: Manual testing with devserver**

```bash
cargo run -p poker-solver-devserver &
cd frontend && npm run dev
```

Test the full flow in browser:
1. Click "Postflop Solver"
2. Click config card → modify ranges → Apply
3. Type `Ah Kd 7c` in flop input → Enter
4. Watch matrix update live
5. Click an action button
6. Enter turn card
7. Watch re-solve

**Step 4: Fix any CSS or UX issues found during testing**

**Step 5: Commit**

```bash
git commit -m "test(postflop-explorer): integration tests and UI polish"
```

---

## Post-Implementation Checklist

- [ ] All Rust tests pass: `cargo test -p poker-solver-tauri`
- [ ] Frontend builds: `cd frontend && npm run build`
- [ ] Clippy clean: `cargo clippy -p poker-solver-tauri`
- [ ] Full test suite under 1 minute: `cargo test`
- [ ] Manual test via devserver: full flow works
- [ ] Update `docs/training.md` if needed (CLI unchanged, skip)
- [ ] Update `docs/explorer.md` with postflop solver mode documentation
