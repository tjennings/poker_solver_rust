# Unified Game State Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace the split Explorer/PostflopExplorer architecture with a single backend-driven game state that flows from preflop through river, with the frontend as pure display.

**Architecture:** New `GameSession` struct in a new `crates/tauri-app/src/game_session.rs` module owns all game state. Six Tauri commands expose it. One frontend component renders it. The V2 game tree, blueprint strategy, and subgame solver are the only data sources. The range-solver full-depth path is removed.

**Tech Stack:** Rust (Tauri backend), TypeScript/React (frontend), V2 game tree, CfvSubgameSolver

---

## Phase 1: Backend Game Session (can ship independently, old frontend still works)

### Task 1: Define types in new module

**Files:**
- Create: `crates/tauri-app/src/game_session.rs`
- Modify: `crates/tauri-app/src/lib.rs` (add `mod game_session;`)

**Step 1: Create the module with all public types**

```rust
// crates/tauri-app/src/game_session.rs

use serde::Serialize;
use std::sync::Arc;
use parking_lot::RwLock;

use poker_solver_core::blueprint_v2::game_tree::{
    GameTree as V2GameTree, GameNode as V2GameNode, TreeAction,
};
use poker_solver_core::blueprint_v2::strategy::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::street::Street;

/// A single action taken in the game, for breadcrumb display.
#[derive(Debug, Clone, Serialize)]
pub struct ActionRecord {
    pub action_id: String,
    pub label: String,
    pub position: String,  // "BB" or "SB"
    pub street: String,
    pub pot: i32,
    pub stack: i32,
}

/// Solve progress info (when a subgame solve is running).
#[derive(Debug, Clone, Serialize)]
pub struct SolveStatus {
    pub iteration: u32,
    pub max_iterations: u32,
    pub exploitability: f32,
    pub elapsed_secs: f64,
    pub solver_name: String,
    pub is_complete: bool,
}

/// One action available at the current decision point.
#[derive(Debug, Clone, Serialize)]
pub struct GameAction {
    pub id: String,
    pub label: String,
    pub action_type: String,
}

/// A single cell in the 13x13 strategy matrix.
#[derive(Debug, Clone, Serialize)]
pub struct GameMatrixCell {
    pub hand: String,
    pub suited: bool,
    pub pair: bool,
    pub probabilities: Vec<f32>,      // one per action
    pub combo_count: usize,
    pub weight: f32,                   // reaching probability
    pub ev: Option<f32>,
}

/// The 13x13 strategy matrix with action labels.
#[derive(Debug, Clone, Serialize)]
pub struct GameMatrix {
    pub cells: Vec<Vec<GameMatrixCell>>,
    pub actions: Vec<GameAction>,
}

/// Complete game state returned to the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct GameState {
    pub street: String,
    pub position: String,             // "BB" or "SB" — who acts next
    pub board: Vec<String>,
    pub pot: i32,
    pub stacks: [i32; 2],            // [BB stack, SB stack]
    pub matrix: Option<GameMatrix>,
    pub actions: Vec<GameAction>,
    pub action_history: Vec<ActionRecord>,
    pub is_terminal: bool,
    pub is_chance: bool,
    pub solve: Option<SolveStatus>,
}
```

**Step 2: Add module to lib.rs**

In `crates/tauri-app/src/lib.rs`, add `mod game_session;` alongside the existing modules.

**Step 3: Build**

Run: `cargo build -p poker-solver-tauri`
Expected: Compiles (unused warnings OK).

**Step 4: Commit**

```
git commit -m "feat: add game_session module with unified GameState types"
```

---

### Task 2: GameSession struct with position helper

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

**Step 1: Add GameSession struct and position_label helper**

This is the ONE place where V2 player convention is mapped to "BB"/"SB":

```rust
use poker_solver_core::blueprint_v2::game_tree::GameTree as V2GameTree;

pub struct GameSession {
    // Blueprint data (shared across sessions via Arc)
    tree: Arc<V2GameTree>,
    strategy: Arc<BlueprintV2Strategy>,
    decision_map: Arc<Vec<u32>>,

    // Current position in the tree
    node_idx: u32,
    board: Vec<String>,
    action_history: Vec<ActionRecord>,

    // Reaching weights — always: index 0 = BB/OOP, index 1 = SB/IP
    weights: [Vec<f32>; 2],  // [oop_1326, ip_1326]
}

impl GameSession {
    /// The ONE canonical mapping from V2 tree player index to position label.
    /// V2 convention: tree.dealer = SB seat. The other seat is BB.
    fn position_label(&self, v2_player: u8) -> &'static str {
        if v2_player == self.tree.dealer { "SB" } else { "BB" }
    }

    /// Map V2 player to weight index: BB/OOP = 0, SB/IP = 1.
    fn weight_index(&self, v2_player: u8) -> usize {
        if v2_player == self.tree.dealer { 1 } else { 0 }
    }

    /// Get the V2 player at the current node.
    fn current_v2_player(&self) -> Option<u8> {
        match &self.tree.nodes[self.node_idx as usize] {
            V2GameNode::Decision { player, .. } => Some(*player),
            _ => None,
        }
    }

    /// Get the street at the current node.
    fn current_street(&self) -> Street {
        match &self.tree.nodes[self.node_idx as usize] {
            V2GameNode::Decision { street, .. } => *street,
            V2GameNode::Chance { .. } => Street::Preflop, // will advance
            V2GameNode::Terminal { .. } => Street::River,   // ended
        }
    }
}
```

**Step 2: Write test for position_label**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_label_v2_convention() {
        // V2 tree: dealer=0 means seat 0 is SB.
        // So player 0 → "SB", player 1 → "BB".
        let tree = V2GameTree::new(/* minimal test config */);
        // For a tree with dealer=0:
        // position_label(0) should return "SB"
        // position_label(1) should return "BB"
        // We'll test this via GameSession once we can construct one.
    }
}
```

Note: A proper unit test requires constructing a minimal V2GameTree. Since `V2GameTree::new` requires a full config, use a lightweight test helper or test via integration with `game_new`. Mark this as a TODO test initially, and validate via integration in Task 5.

**Step 3: Build and commit**

```
git commit -m "feat: GameSession struct with canonical position_label mapping"
```

---

### Task 3: game_new — Initialize session from loaded blueprint

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`
- Modify: `crates/tauri-app/src/lib.rs` (register command)

**Step 1: Implement `GameSession::new` and `game_new` Tauri command**

`game_new` doesn't load a bundle (that's already done via `load_bundle`). It creates a `GameSession` from the already-loaded `ExplorationState`. The session starts at the preflop root with full ranges.

```rust
use crate::exploration::ExplorationState;
use std::sync::Arc;
use parking_lot::RwLock;

/// Shared session state, accessible by Tauri commands.
pub struct GameSessionState {
    pub session: RwLock<Option<GameSession>>,
}

impl GameSession {
    pub fn from_exploration_state(state: &ExplorationState) -> Result<Self, String> {
        let source = state.source.read();
        let source = source.as_ref().ok_or("No bundle loaded")?;

        // Extract BlueprintV2 fields
        let (tree, strategy, decision_map) = match source {
            crate::exploration::StrategySource::BlueprintV2 {
                tree, strategy, decision_map, ..
            } => (Arc::clone(tree), Arc::clone(strategy), Arc::clone(decision_map)),
            _ => return Err("game_new requires a BlueprintV2 source".to_string()),
        };

        let root = tree.root;

        Ok(GameSession {
            tree,
            strategy,
            decision_map,
            node_idx: root,
            board: vec![],
            action_history: vec![],
            weights: [vec![1.0f32; 1326], vec![1.0f32; 1326]],
        })
    }
}

#[tauri::command]
pub fn game_new(
    exploration: tauri::State<'_, ExplorationState>,
    session_state: tauri::State<'_, GameSessionState>,
) -> Result<(), String> {
    let session = GameSession::from_exploration_state(&exploration)?;
    *session_state.session.write() = Some(session);
    Ok(())
}
```

**Step 2: Register in lib.rs**

Add `GameSessionState` to Tauri managed state and `game_new` to the command list.

Look at existing pattern in `crates/tauri-app/src/lib.rs` for how `ExplorationState` and `PostflopState` are registered, and follow the same pattern.

**Step 3: Build, test, commit**

```
cargo build -p poker-solver-tauri
git commit -m "feat: game_new command initializes GameSession from loaded blueprint"
```

---

### Task 4: game_get_state — Build GameState from current position

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

This is the core task. Reuse the matrix-building logic from `get_strategy_matrix_v2` (exploration.rs:1134-1206) but adapt it to work within GameSession.

**Step 1: Implement `GameSession::get_state()`**

The method must:
1. Determine current street, position, pot, stacks from the tree node
2. Build the 13x13 matrix using blueprint strategy at the current decision node
3. List available actions
4. Return the full GameState

Key references to reuse:
- Matrix cell building: `exploration.rs:1134-1206` (the RANKS loop, hand_label_from_matrix, strategy lookup)
- Preflop strategy: `strategy.get_action_probs(decision_idx, bucket)` where bucket = canonical hand index
- Postflop strategy: requires CbvContext (bucket lookup). For this task, start with preflop only. Postflop blueprint matrix will be added in Task 6.
- Pot/stacks: approximate from tree structure using `pot_at_v2_node`

```rust
impl GameSession {
    pub fn get_state(&self) -> GameState {
        let node = &self.tree.nodes[self.node_idx as usize];

        match node {
            V2GameNode::Terminal { .. } => GameState {
                street: self.street_name(),
                position: String::new(),
                board: self.board.clone(),
                pot: 0,  // TODO: compute from tree
                stacks: [0, 0],
                matrix: None,
                actions: vec![],
                action_history: self.action_history.clone(),
                is_terminal: true,
                is_chance: false,
                solve: None,
            },
            V2GameNode::Chance { .. } => GameState {
                street: self.street_name(),
                position: String::new(),
                board: self.board.clone(),
                pot: 0,
                stacks: [0, 0],
                matrix: None,
                actions: vec![],
                action_history: self.action_history.clone(),
                is_terminal: false,
                is_chance: true,
                solve: None,
            },
            V2GameNode::Decision { player, actions, street, .. } => {
                let position = self.position_label(*player).to_string();
                let decision_idx = self.decision_map
                    .get(self.node_idx as usize)
                    .copied()
                    .unwrap_or(u32::MAX);

                let game_actions = self.build_actions(actions);
                let matrix = if decision_idx != u32::MAX {
                    Some(self.build_matrix(decision_idx as usize, *player, *street, &game_actions))
                } else {
                    None
                };

                GameState {
                    street: self.street_name(),
                    position,
                    board: self.board.clone(),
                    pot: self.compute_pot(),
                    stacks: self.compute_stacks(),
                    actions: game_actions.clone(),
                    matrix: matrix.map(|cells| GameMatrix { cells, actions: game_actions }),
                    action_history: self.action_history.clone(),
                    is_terminal: false,
                    is_chance: false,
                    solve: None,
                }
            }
        }
    }
}
```

**Step 2: Implement `build_matrix` for preflop**

Port the matrix-building loop from `get_strategy_matrix_v2` (exploration.rs:1134-1206). For preflop:

```rust
fn build_matrix(&self, decision_idx: usize, player: u8, street: Street,
                actions: &[GameAction]) -> Vec<Vec<GameMatrixCell>> {
    let num_buckets = self.strategy.bucket_counts[
        self.strategy.node_street_indices[decision_idx] as usize
    ] as usize;

    let weight_idx = self.weight_index(player);
    let ranks = ['A','K','Q','J','T','9','8','7','6','5','4','3','2'];

    let mut cells = Vec::with_capacity(13);
    for (row, &rank1) in ranks.iter().enumerate() {
        let mut row_cells = Vec::with_capacity(13);
        for (col, &rank2) in ranks.iter().enumerate() {
            let (label, suited, pair) = hand_label_from_matrix(row, col, rank1, rank2);
            let hand_idx = canonical_hand_index_from_ranks(rank1, rank2, suited);

            // Strategy probs from blueprint
            let probabilities = if street == Street::Preflop {
                let bucket = if num_buckets == 169 { hand_idx as u16 }
                             else { (hand_idx % num_buckets) as u16 };
                let probs = self.strategy.get_action_probs(decision_idx, bucket);
                actions.iter().enumerate()
                    .map(|(i, _)| probs.get(i).copied().unwrap_or(0.0))
                    .collect()
            } else {
                // Postflop: requires CbvContext — implemented in Task 6
                vec![0.0; actions.len()]
            };

            // Reaching weight for acting player
            // Use 169-level canonical weight (average across combos)
            let weight = self.weights[weight_idx]
                .get(hand_idx)  // This needs 169-level indexing; see note below
                .copied()
                .unwrap_or(0.0);

            row_cells.push(GameMatrixCell {
                hand: label,
                suited,
                pair,
                probabilities,
                combo_count: if weight > 0.0 { 1 } else { 0 },
                weight,
                ev: None,
            });
        }
        cells.push(row_cells);
    }
    cells
}
```

**Important note on weights:** The session stores 1326-element weight arrays (per-combo). The matrix needs 169-level weights (per canonical hand). You need a helper that averages the 1326 weights into 169 canonical weights. Reuse the `expand_169_to_1326` / inverse mapping from `exploration.rs`. Or compute the weight per-cell by averaging combos that map to that cell, similar to what `get_strategy_matrix_v2` does at line 1182-1188.

**Step 3: Implement helper methods**

```rust
fn build_actions(&self, tree_actions: &[TreeAction]) -> Vec<GameAction> {
    tree_actions.iter().enumerate().map(|(i, a)| {
        GameAction {
            id: i.to_string(),
            label: format_tree_action(a), // reuse v2_action_info logic
            action_type: action_type_string(a),
        }
    }).collect()
}

fn street_name(&self) -> String {
    match self.current_street() {
        Street::Preflop => "Preflop",
        Street::Flop => "Flop",
        Street::Turn => "Turn",
        Street::River => "River",
    }.to_string()
}

fn compute_pot(&self) -> i32 {
    // Reuse pot_at_v2_node from exploration.rs
    (pot_at_v2_node(&self.tree, self.node_idx) * 2.0) as i32
}
```

**Step 4: Implement Tauri command**

```rust
#[tauri::command]
pub fn game_get_state(
    session_state: tauri::State<'_, GameSessionState>,
) -> Result<GameState, String> {
    let guard = session_state.session.read();
    let session = guard.as_ref().ok_or("No game session active")?;
    Ok(session.get_state())
}
```

**Step 5: Build, test, commit**

```
cargo build -p poker-solver-tauri
git commit -m "feat: game_get_state returns unified GameState with preflop matrix"
```

---

### Task 5: game_play_action — Navigate tree and update ranges

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

**Step 1: Implement play_action**

This is the core navigation method. When an action is taken:
1. Record the action in history
2. Update the acting player's range weights (multiply by strategy probs)
3. Advance `node_idx` to the child
4. If child is a chance node, stop (frontend will call `game_deal_card`)

```rust
impl GameSession {
    pub fn play_action(&mut self, action_id: &str) -> Result<(), String> {
        let action_idx: usize = action_id.parse()
            .map_err(|_| format!("Invalid action_id: {action_id}"))?;

        let (player, street, actions, children) = match &self.tree.nodes[self.node_idx as usize] {
            V2GameNode::Decision { player, street, actions, children, .. } => {
                (*player, *street, actions.clone(), children.clone())
            }
            _ => return Err("Not at a decision node".to_string()),
        };

        if action_idx >= children.len() {
            return Err(format!("Action {action_idx} out of range"));
        }

        // Record breadcrumb
        let position = self.position_label(player).to_string();
        self.action_history.push(ActionRecord {
            action_id: action_id.to_string(),
            label: format_tree_action(&actions[action_idx]),
            position,
            street: street_to_string(street),
            pot: self.compute_pot(),
            stack: 0, // TODO
        });

        // Update acting player's range weights
        let weight_idx = self.weight_index(player);
        let decision_idx = self.decision_map
            .get(self.node_idx as usize)
            .copied()
            .unwrap_or(u32::MAX);

        if decision_idx != u32::MAX {
            self.propagate_weights(decision_idx as usize, player, street, action_idx);
        }

        // Advance to child node
        self.node_idx = children[action_idx];

        // Auto-advance through chance nodes if board card already known
        // (for preflop → flop transitions where board is already set)
        if let V2GameNode::Chance { child, .. } = &self.tree.nodes[self.node_idx as usize] {
            if !self.board.is_empty() {
                self.node_idx = *child;
            }
        }

        Ok(())
    }

    /// Multiply acting player's weights by action probability at each hand.
    fn propagate_weights(&mut self, decision_idx: usize, player: u8,
                         street: Street, action_idx: usize) {
        let weight_idx = self.weight_index(player);
        let num_buckets = self.strategy.bucket_counts[
            self.strategy.node_street_indices[decision_idx] as usize
        ] as usize;

        if street == Street::Preflop {
            // Preflop: 169 canonical hands, each maps to multiple 1326 combos.
            // For each canonical hand, get the action prob and multiply ALL
            // combos belonging to that hand.
            for hand_idx in 0..169 {
                let bucket = if num_buckets == 169 { hand_idx as u16 }
                             else { (hand_idx % num_buckets) as u16 };
                let probs = self.strategy.get_action_probs(decision_idx, bucket);
                let p = probs.get(action_idx).copied().unwrap_or(0.0);
                // Multiply all 1326 combos that belong to this canonical hand
                for ci in combos_for_canonical_hand(hand_idx) {
                    self.weights[weight_idx][ci] *= p;
                }
            }
        } else {
            // Postflop: requires CbvContext bucket lookup per combo.
            // Implemented in Task 6.
        }
    }
}
```

**Note:** `combos_for_canonical_hand(hand_idx) -> Vec<usize>` maps a 169 canonical index to all 1326 combo indices. This mapping exists in `exploration.rs` as `expand_169_to_1326`. Extract or replicate the mapping.

**Step 2: Implement Tauri command**

```rust
#[tauri::command]
pub fn game_play_action(
    session_state: tauri::State<'_, GameSessionState>,
    action_id: String,
) -> Result<GameState, String> {
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.play_action(&action_id)?;
    Ok(session.get_state())
}
```

**Step 3: Build, test, commit**

```
cargo build -p poker-solver-tauri
git commit -m "feat: game_play_action navigates tree and propagates ranges"
```

---

### Task 6: Postflop matrix building with CbvContext

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

**Step 1: Add CbvContext and hand_evs to GameSession**

```rust
pub struct GameSession {
    // ... existing fields ...
    cbv_context: Option<Arc<CbvContext>>,
    hand_evs: Option<Arc<Vec<[[f64; 169]; 2]>>>,
}
```

Update `from_exploration_state` to extract these from `StrategySource::BlueprintV2`.

**Step 2: Implement postflop matrix building**

Fill in the postflop branch of `build_matrix`. Port the `postflop_cell_probs` logic from `exploration.rs:1468-1502`:

```rust
// In build_matrix, the postflop branch:
} else if let Some(ctx) = &self.cbv_context {
    let board_cards = parse_board(&self.board).ok();
    if let Some(board) = &board_cards {
        let board_slice = board_for_street_slice(board, street);
        let lookup = BucketLookup {
            all_buckets: &ctx.all_buckets,
            strategy: &ctx.strategy,
            decision_idx,
        };
        postflop_cell_probs(rank1, rank2, suited, board_slice, street, &lookup, actions)
    } else {
        vec![0.0; actions.len()]
    }
}
```

Reuse `BucketLookup`, `postflop_cell_probs`, and `bucket_probs_for_hand` from exploration.rs. These can be made `pub(crate)` if not already.

**Step 3: Add EV lookup**

For preflop and postflop, if `hand_evs` is available:

```rust
let ev = self.hand_evs
    .as_ref()
    .and_then(|evs| evs.get(decision_idx))
    .map(|node_evs| {
        let player_idx = self.weight_index(player);
        // hand_evs is indexed by V2 player convention, but we need
        // to map through: hand_evs stores [player0_evs, player1_evs]
        // where player0 = V2 seat 0. We need the acting player's EVs.
        node_evs[player as usize][hand_idx] as f32
    });
```

**Step 4: Implement postflop weight propagation**

Fill in the postflop branch of `propagate_weights` using CbvContext:

```rust
} else if let Some(ctx) = &self.cbv_context {
    let board_cards = parse_board(&self.board).unwrap();
    let board_slice = board_for_street_slice(&board_cards, street);
    // For each combo in the weight array, look up its bucket
    // and multiply by action probability.
    for hand in all_hands() {
        for (c0, c1) in hand.combos() {
            if board_slice.iter().any(|b| *b == c0 || *b == c1) {
                continue;
            }
            let id0 = rs_poker_card_to_id(c0);
            let id1 = rs_poker_card_to_id(c1);
            let ci = card_pair_to_index(id0, id1);
            let bucket = ctx.all_buckets.get_bucket(street, [c0, c1], board_slice);
            let probs = ctx.strategy.get_action_probs(decision_idx, bucket);
            let p = probs.get(action_idx).copied().unwrap_or(0.0);
            self.weights[weight_idx][ci] *= p;
        }
    }
}
```

**Step 5: Build, test, commit**

```
cargo build -p poker-solver-tauri
git commit -m "feat: postflop matrix building and range propagation with CbvContext"
```

---

### Task 7: game_deal_card and game_back

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

**Step 1: Implement deal_card**

```rust
impl GameSession {
    pub fn deal_card(&mut self, card: &str) -> Result<(), String> {
        // Validate we're at a chance node
        match &self.tree.nodes[self.node_idx as usize] {
            V2GameNode::Chance { child, .. } => {
                self.board.push(card.to_string());
                self.node_idx = *child;
                // Skip any additional chance nodes (flop deals 3 cards at once in V2)
                while let V2GameNode::Chance { child, .. } = &self.tree.nodes[self.node_idx as usize] {
                    self.node_idx = *child;
                }
                Ok(())
            }
            _ => Err("Not at a chance node — cannot deal card".to_string()),
        }
    }
}
```

**Step 2: Implement back (undo)**

For undo, we need to be able to replay from root. Store snapshots or replay from scratch:

```rust
impl GameSession {
    /// Undo the last action by replaying from root up to history.len() - 1.
    pub fn back(&mut self) -> Result<(), String> {
        if self.action_history.is_empty() {
            return Err("No actions to undo".to_string());
        }

        // Save the action IDs we want to replay (all but the last)
        let replay_ids: Vec<String> = self.action_history[..self.action_history.len() - 1]
            .iter()
            .map(|a| a.action_id.clone())
            .collect();
        let saved_board = self.board.clone();

        // Reset to root
        self.node_idx = self.tree.root;
        self.action_history.clear();
        self.weights = [vec![1.0f32; 1326], vec![1.0f32; 1326]];
        self.board.clear();

        // Replay actions
        for action_id in &replay_ids {
            // If we hit a chance node, re-deal the board cards
            if let V2GameNode::Chance { .. } = &self.tree.nodes[self.node_idx as usize] {
                // Determine how many board cards to deal based on street
                let cards_needed = match self.current_street() {
                    Street::Preflop => 3, // dealing flop
                    Street::Flop => 1,    // dealing turn
                    Street::Turn => 1,    // dealing river
                    _ => 0,
                };
                for card in saved_board.iter().take(self.board.len() + cards_needed) {
                    if self.board.len() < saved_board.len() {
                        self.deal_card(card)?;
                    }
                }
            }
            self.play_action(action_id)?;
        }

        Ok(())
    }
}
```

**Step 3: Tauri commands**

```rust
#[tauri::command]
pub fn game_deal_card(
    session_state: tauri::State<'_, GameSessionState>,
    card: String,
) -> Result<GameState, String> {
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.deal_card(&card)?;
    Ok(session.get_state())
}

#[tauri::command]
pub fn game_back(
    session_state: tauri::State<'_, GameSessionState>,
) -> Result<GameState, String> {
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.back()?;
    Ok(session.get_state())
}
```

**Step 4: Build, test, commit**

```
cargo build -p poker-solver-tauri
git commit -m "feat: game_deal_card and game_back commands"
```

---

### Task 8: game_solve — Integrate subgame solver

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

**Step 1: Add solve state to GameSession**

```rust
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use crate::postflop::{SubgameSolveResult, build_subgame_solver, snapshot_from_subgame,
                       build_matrix_from_snapshot, CbvContext};

pub struct SubgameSolveState {
    pub solving: AtomicBool,
    pub solve_complete: AtomicBool,
    pub current_iteration: AtomicU32,
    pub max_iterations: AtomicU32,
    pub exploitability_bits: AtomicU32,
    pub solve_start: Option<std::time::Instant>,
    pub result: RwLock<Option<SubgameSolveResult>>,
    pub matrix_snapshot: RwLock<Option<GameMatrix>>,
}
```

**Step 2: Implement game_solve**

Port the subgame solver invocation from `solve_depth_limited` in `postflop.rs:1503-1807`. The key changes:
- Extract OOP/IP weights from `self.weights` (already correctly mapped)
- Use the same `build_subgame_solver` function
- Store results in the session's `SubgameSolveState`
- `get_state()` checks if a solve is active and includes progress/matrix from it

**Step 3: Update get_state to include solve info**

When `self.subgame` is Some and solving is active, `get_state()` should:
- Set `solve` field with current iteration/exploitability
- Use the solver's matrix snapshot instead of the blueprint matrix
- Set `is_solving: true` (or include in SolveStatus)

**Step 4: Tauri command**

```rust
#[tauri::command]
pub fn game_solve(
    session_state: tauri::State<'_, GameSessionState>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
    // ... other solver params
) -> Result<(), String> {
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.start_solve(max_iterations, target_exploitability)?;
    Ok(())
}
```

**Step 5: Build, test, commit**

```
cargo build -p poker-solver-tauri
git commit -m "feat: game_solve integrates subgame solver into GameSession"
```

---

## Phase 2: Unified Frontend

### Task 9: GameState TypeScript types

**Files:**
- Create: `frontend/src/game-types.ts`

**Step 1: Define types matching the Rust structs**

```typescript
export interface ActionRecord {
  action_id: string;
  label: string;
  position: string;  // "BB" or "SB"
  street: string;
  pot: number;
  stack: number;
}

export interface SolveStatus {
  iteration: number;
  max_iterations: number;
  exploitability: number;
  elapsed_secs: number;
  solver_name: string;
  is_complete: boolean;
}

export interface GameAction {
  id: string;
  label: string;
  action_type: string;
}

export interface GameMatrixCell {
  hand: string;
  suited: boolean;
  pair: boolean;
  probabilities: number[];
  combo_count: number;
  weight: number;
  ev: number | null;
}

export interface GameMatrix {
  cells: GameMatrixCell[][];
  actions: GameAction[];
}

export interface GameState {
  street: string;
  position: string;
  board: string[];
  pot: number;
  stacks: [number, number];
  matrix: GameMatrix | null;
  actions: GameAction[];
  action_history: ActionRecord[];
  is_terminal: boolean;
  is_chance: boolean;
  solve: SolveStatus | null;
}
```

**Step 2: Commit**

```
git commit -m "feat: GameState TypeScript types for unified frontend"
```

---

### Task 10: GameExplorer component

**Files:**
- Create: `frontend/src/GameExplorer.tsx`

**Step 1: Build the unified component**

This component:
- Calls `game_get_state` on mount and after every action
- Renders the 13x13 matrix from `state.matrix`
- Renders action buttons from `state.actions`
- Renders breadcrumbs from `state.action_history`
- Shows board cards from `state.board`
- Shows solve progress from `state.solve`
- When `state.is_chance`, shows card picker

Reuse the existing `HandCell` rendering logic from Explorer.tsx. The matrix grid loop is identical — 13 rows × 13 cols, each cell gets probabilities and weight.

Action clicks call `game_play_action(action_id)` and re-fetch state.
Back button calls `game_back()`.
Card picker calls `game_deal_card(card)`.
Solve button calls `game_solve(params)` then polls `game_get_state` every 500ms.

**No game logic.** No `blueprintMode`. No player convention. No range management. Just render what the backend says.

**Step 2: Wire into App.tsx**

Replace the Explorer/PostflopExplorer routing with a single GameExplorer component.

**Step 3: Commit**

```
git commit -m "feat: GameExplorer unified frontend component"
```

---

## Phase 3: Cleanup

### Task 11: Remove old code

**Files to delete or gut:**
- `frontend/src/PostflopExplorer.tsx` — delete entirely
- `frontend/src/blueprint-utils.ts` — delete (blueprintToPostflopMatrix no longer needed)
- `frontend/src/blueprint-utils.test.ts` — delete

**Files to simplify:**
- `frontend/src/Explorer.tsx` — remove matrix rendering, action handling, range tracking. Keep only bundle loading UI if needed, or delete if GameExplorer handles everything.
- `frontend/src/types.ts` — remove `StrategyMatrix`, `PostflopStrategyMatrix`, `PostflopMatrixCell`, `PostflopConfig`, etc. Keep only types still used elsewhere.

**Backend cleanup:**
- `crates/tauri-app/src/postflop.rs` — remove `solve_full_depth`, `build_game`, `capture_matrix_snapshot`, `build_strategy_matrix`, the `game: Mutex<Option<PostFlopGame>>` field, and all full-depth solver paths. Keep subgame solver functions used by `game_solve`.
- `crates/tauri-app/src/exploration.rs` — remove `get_strategy_matrix` command if no longer called. Keep `load_bundle` and internal helpers used by GameSession.

**Step 1: Delete frontend files**

```bash
rm frontend/src/PostflopExplorer.tsx
rm frontend/src/blueprint-utils.ts
rm frontend/src/blueprint-utils.test.ts
```

**Step 2: Remove range-solver full-depth path from postflop.rs**

Remove:
- `use range_solver::{compute_exploitability, finalize, solve_step, PostFlopGame};`
- `game: Mutex<Option<PostFlopGame>>` from PostflopState
- `solve_full_depth` function
- `build_game` function
- `capture_matrix_snapshot` function
- `build_strategy_matrix` function
- `SolverChoice::FullDepth` dispatch branch in `postflop_solve_street_impl`

**Step 3: Build and test**

```bash
cargo build -p poker-solver-tauri
cd frontend && npm run build
cargo test
```

**Step 4: Commit**

```
git commit -m "chore: remove old Explorer/PostflopExplorer split and range-solver full-depth path"
```

---

### Task 12: Final verification

**Step 1: Test preflop blueprint navigation**
- Load a blueprint
- Navigate preflop actions (SB raise, BB call, etc.)
- Verify matrix shows correct position label and strategy

**Step 2: Test postflop blueprint navigation**
- Deal flop cards
- Navigate postflop actions
- Verify matrix updates correctly

**Step 3: Test subgame solve**
- At a postflop decision point, click solve
- Verify solve progress shows
- Verify final matrix displays with EVs

**Step 4: Test back/undo**
- Navigate several actions, then click back
- Verify state rewinds correctly

**Step 5: Test street transitions**
- Play through flop actions to a chance node
- Deal turn card
- Verify new street matrix appears

```
git commit -m "test: manual verification of unified game state flow"
```
