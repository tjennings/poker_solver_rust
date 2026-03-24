//! Unified game session: tracks state from preflop through river.
//!
//! A `GameSession` owns the V2 game tree, blueprint strategy, and range weights.
//! Six Tauri commands expose it: `game_new`, `game_get_state`, `game_play_action`,
//! `game_deal_card`, `game_back`, `game_solve`.

use std::sync::Arc;

use parking_lot::RwLock;
use serde::Serialize;

use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
use poker_solver_core::blueprint_v2::game_tree::{
    GameNode as V2GameNode, GameTree as V2GameTree, TreeAction,
};
use poker_solver_core::blueprint_v2::Street;

use crate::exploration::{
    board_for_street_slice, build_canonical_to_combo_map, canonical_hand_index_from_ranks,
    hand_label_from_matrix, parse_board, pot_at_v2_node, ActionInfo, BucketLookup, RANKS,
};
use crate::postflop::CbvContext;

// ---------------------------------------------------------------------------
// Types returned to the frontend
// ---------------------------------------------------------------------------

/// A single action taken in the game, for breadcrumb display.
#[derive(Debug, Clone, Serialize)]
pub struct ActionRecord {
    pub action_id: String,
    pub label: String,
    pub position: String, // "BB" or "SB"
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
    pub probabilities: Vec<f32>, // one per action
    pub combo_count: usize,
    pub weight: f32, // reaching probability
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
    pub position: String, // "BB" or "SB" -- who acts next
    pub board: Vec<String>,
    pub pot: i32,
    pub stacks: [i32; 2], // [BB stack, SB stack]
    pub matrix: Option<GameMatrix>,
    pub actions: Vec<GameAction>,
    pub action_history: Vec<ActionRecord>,
    pub is_terminal: bool,
    pub is_chance: bool,
    pub solve: Option<SolveStatus>,
}

// ---------------------------------------------------------------------------
// Shared state for Tauri commands
// ---------------------------------------------------------------------------

/// Shared session state, accessible by Tauri commands.
pub struct GameSessionState {
    pub session: RwLock<Option<GameSession>>,
}

impl Default for GameSessionState {
    fn default() -> Self {
        Self {
            session: RwLock::new(None),
        }
    }
}

// ---------------------------------------------------------------------------
// GameSession
// ---------------------------------------------------------------------------

pub struct GameSession {
    // Blueprint data
    tree: Box<V2GameTree>,
    strategy: Box<BlueprintV2Strategy>,
    decision_map: Vec<u32>,
    config: Box<BlueprintV2Config>,

    // Optional postflop context
    cbv_context: Option<Arc<CbvContext>>,
    hand_evs: Option<Vec<[[f64; 169]; 2]>>,

    // Current position in the tree
    node_idx: u32,
    board: Vec<String>,
    action_history: Vec<ActionRecord>,

    // Reaching weights: index 0 = BB/OOP, index 1 = SB/IP
    weights: [Vec<f32>; 2],
}

impl GameSession {
    /// Create a session from already-loaded exploration state.
    pub fn from_exploration_state(
        exploration: &crate::exploration::ExplorationState,
        cbv_context: Option<Arc<CbvContext>>,
    ) -> Result<Self, String> {
        let data = exploration.extract_blueprint_v2_data()?;
        let root = data.tree.root;

        Ok(GameSession {
            tree: data.tree,
            strategy: data.strategy,
            decision_map: data.decision_map,
            config: data.config,
            cbv_context,
            hand_evs: data.hand_evs,
            node_idx: root,
            board: vec![],
            action_history: vec![],
            weights: [vec![1.0f32; 1326], vec![1.0f32; 1326]],
        })
    }

    /// The ONE canonical mapping from V2 tree player index to position label.
    /// V2 convention: `tree.dealer` = SB seat. The other seat is BB.
    fn position_label(&self, v2_player: u8) -> &'static str {
        if v2_player == self.tree.dealer {
            "SB"
        } else {
            "BB"
        }
    }

    /// Map V2 player to weight index: BB/OOP = 0, SB/IP = 1.
    fn weight_index(&self, v2_player: u8) -> usize {
        if v2_player == self.tree.dealer {
            1
        } else {
            0
        }
    }

    /// Get the street at the current node.
    fn current_street(&self) -> Street {
        match &self.tree.nodes[self.node_idx as usize] {
            V2GameNode::Decision { street, .. } => *street,
            V2GameNode::Chance { next_street, .. } => *next_street,
            V2GameNode::Terminal { .. } => Street::River,
        }
    }

    /// Build the full `GameState` from the current session position.
    #[allow(clippy::cast_possible_truncation)]
    pub fn get_state(&self) -> GameState {
        let node = &self.tree.nodes[self.node_idx as usize];

        match node {
            V2GameNode::Terminal { pot, stacks, .. } => {
                let bb_idx = self.weight_index(1 - self.tree.dealer);
                let sb_idx = self.weight_index(self.tree.dealer);
                GameState {
                    street: street_to_string(self.current_street()),
                    position: String::new(),
                    board: self.board.clone(),
                    pot: (*pot * 2.0) as i32,
                    stacks: [
                        (stacks[bb_idx] * 2.0) as i32,
                        (stacks[sb_idx] * 2.0) as i32,
                    ],
                    matrix: None,
                    actions: vec![],
                    action_history: self.action_history.clone(),
                    is_terminal: true,
                    is_chance: false,
                    solve: None,
                }
            }
            V2GameNode::Chance { .. } => {
                let pot = self.compute_pot();
                let stacks = self.compute_stacks();
                GameState {
                    street: street_to_string(self.current_street()),
                    position: String::new(),
                    board: self.board.clone(),
                    pot,
                    stacks,
                    matrix: None,
                    actions: vec![],
                    action_history: self.action_history.clone(),
                    is_terminal: false,
                    is_chance: true,
                    solve: None,
                }
            }
            V2GameNode::Decision {
                player,
                actions,
                street,
                ..
            } => {
                let position = self.position_label(*player).to_string();
                let decision_idx = self
                    .decision_map
                    .get(self.node_idx as usize)
                    .copied()
                    .unwrap_or(u32::MAX);

                let game_actions = build_game_actions(actions);
                let matrix = if decision_idx != u32::MAX {
                    Some(self.build_matrix(
                        decision_idx as usize,
                        *player,
                        *street,
                        &game_actions,
                    ))
                } else {
                    None
                };

                let pot = self.compute_pot();
                let stacks = self.compute_stacks();

                GameState {
                    street: street_to_string(*street),
                    position,
                    board: self.board.clone(),
                    pot,
                    stacks,
                    matrix: matrix.map(|cells| GameMatrix {
                        cells,
                        actions: game_actions.clone(),
                    }),
                    actions: game_actions,
                    action_history: self.action_history.clone(),
                    is_terminal: false,
                    is_chance: false,
                    solve: None,
                }
            }
        }
    }

    /// Build the 13x13 strategy matrix for the current decision node.
    #[allow(clippy::cast_possible_truncation)]
    fn build_matrix(
        &self,
        decision_idx: usize,
        player: u8,
        street: Street,
        actions: &[GameAction],
    ) -> Vec<Vec<GameMatrixCell>> {
        let num_buckets = self.strategy.bucket_counts
            [self.strategy.node_street_indices[decision_idx] as usize]
            as usize;

        let weight_idx = self.weight_index(player);
        let board_cards = parse_board(&self.board).ok();

        // Build the canonical-to-combo map for weight averaging.
        let combo_map = build_canonical_to_combo_map();

        let mut cells = Vec::with_capacity(13);
        for (row, &rank1) in RANKS.iter().enumerate() {
            let mut row_cells = Vec::with_capacity(13);
            for (col, &rank2) in RANKS.iter().enumerate() {
                let (label, suited, pair) = hand_label_from_matrix(row, col, rank1, rank2);
                let hand_idx = canonical_hand_index_from_ranks(rank1, rank2, suited);

                // Strategy probabilities from blueprint.
                let probabilities = if street == Street::Preflop {
                    let bucket = if num_buckets == 169 {
                        hand_idx as u16
                    } else {
                        (hand_idx % num_buckets) as u16
                    };
                    let probs = self.strategy.get_action_probs(decision_idx, bucket);
                    actions
                        .iter()
                        .enumerate()
                        .map(|(i, _)| probs.get(i).copied().unwrap_or(0.0))
                        .collect()
                } else if let (Some(ctx), Some(board)) = (&self.cbv_context, &board_cards) {
                    // Postflop with bucket data.
                    let board_slice = board_for_street_slice(board, street);
                    let lookup = BucketLookup {
                        all_buckets: &ctx.all_buckets,
                        strategy: &ctx.strategy,
                        decision_idx,
                    };
                    // Use the exploration helper to get averaged probs.
                    let action_infos: Vec<ActionInfo> = actions
                        .iter()
                        .enumerate()
                        .map(|(_i, a)| ActionInfo {
                            id: a.id.clone(),
                            label: a.label.clone(),
                            action_type: a.action_type.clone(),
                            size_key: None,
                        })
                        .collect();
                    let action_probs = crate::exploration::postflop_cell_probs(
                        rank1,
                        rank2,
                        suited,
                        board_slice,
                        street,
                        &lookup,
                        &action_infos,
                    );
                    action_probs.iter().map(|ap| ap.probability).collect()
                } else {
                    vec![0.0; actions.len()]
                };

                // Compute weight: average reaching probability across combos for this hand.
                let combo_indices = &combo_map[hand_idx];
                let (weight_sum, weight_count) = combo_indices.iter().fold(
                    (0.0f32, 0usize),
                    |(sum, count), &ci| {
                        // For postflop, filter out board-blocked combos.
                        (sum + self.weights[weight_idx][ci], count + 1)
                    },
                );
                let weight = if weight_count > 0 {
                    weight_sum / weight_count as f32
                } else {
                    0.0
                };

                let combo_count = combo_indices
                    .iter()
                    .filter(|&&ci| self.weights[weight_idx][ci] > 0.0)
                    .count();

                // EV lookup.
                let ev = self
                    .hand_evs
                    .as_ref()
                    .and_then(|evs| evs.get(decision_idx))
                    .map(|node_evs| node_evs[player as usize][hand_idx] as f32);

                row_cells.push(GameMatrixCell {
                    hand: label,
                    suited,
                    pair,
                    probabilities,
                    combo_count,
                    weight,
                    ev,
                });
            }
            cells.push(row_cells);
        }
        cells
    }

    /// Navigate to the child node for the given action.
    pub fn play_action(&mut self, action_id: &str) -> Result<(), String> {
        let action_idx: usize = action_id
            .parse()
            .map_err(|_| format!("Invalid action_id: {action_id}"))?;

        let (player, street, actions, children) = match &self.tree.nodes[self.node_idx as usize] {
            V2GameNode::Decision {
                player,
                street,
                actions,
                children,
                ..
            } => (*player, *street, actions.clone(), children.clone()),
            _ => return Err("Not at a decision node".to_string()),
        };

        if action_idx >= children.len() {
            return Err(format!(
                "Action {action_idx} out of range (max {})",
                children.len() - 1
            ));
        }

        // Record breadcrumb.
        let position = self.position_label(player).to_string();
        let pot = self.compute_pot();
        self.action_history.push(ActionRecord {
            action_id: action_id.to_string(),
            label: format_tree_action(&actions[action_idx]),
            position,
            street: street_to_string(street),
            pot,
            stack: 0,
        });

        // Update acting player's range weights.
        let decision_idx = self
            .decision_map
            .get(self.node_idx as usize)
            .copied()
            .unwrap_or(u32::MAX);

        if decision_idx != u32::MAX {
            self.propagate_weights(decision_idx as usize, player, street, action_idx);
        }

        // Advance to child node.
        self.node_idx = children[action_idx];

        Ok(())
    }

    /// Deal board card(s) at a chance node.
    ///
    /// For flop transitions, the V2 tree has a single chance node but 3 cards
    /// must be dealt. Cards are buffered in `self.board`; the chance node is
    /// only advanced once enough cards are present for the next street.
    pub fn deal_card(&mut self, card: &str) -> Result<(), String> {
        match &self.tree.nodes[self.node_idx as usize] {
            V2GameNode::Chance { .. } => {
                self.board.push(card.to_string());

                // Determine how many cards the next street needs.
                let cards_needed = match self.board.len() {
                    0..=2 => 3,  // flop needs 3
                    3 => 4,      // turn needs 4 total
                    4 => 5,      // river needs 5 total
                    _ => self.board.len(),
                };

                // Only advance past chance node(s) when we have enough cards.
                if self.board.len() >= cards_needed {
                    if let V2GameNode::Chance { child, .. } =
                        &self.tree.nodes[self.node_idx as usize]
                    {
                        self.node_idx = *child;
                    }
                    // Skip additional chance nodes.
                    while let V2GameNode::Chance { child, .. } =
                        &self.tree.nodes[self.node_idx as usize]
                    {
                        self.node_idx = *child;
                    }
                }

                Ok(())
            }
            _ => Err("Not at a chance node -- cannot deal card".to_string()),
        }
    }

    /// Undo the last action by replaying from root.
    pub fn back(&mut self) -> Result<(), String> {
        if self.action_history.is_empty() {
            return Err("No actions to undo".to_string());
        }

        // Save action IDs and board to replay (all but last action).
        let replay_ids: Vec<String> = self.action_history[..self.action_history.len() - 1]
            .iter()
            .map(|a| a.action_id.clone())
            .collect();
        let saved_board = self.board.clone();

        // Reset to root.
        self.node_idx = self.tree.root;
        self.action_history.clear();
        self.weights = [vec![1.0f32; 1326], vec![1.0f32; 1326]];
        self.board.clear();

        // Replay actions.
        for action_id in &replay_ids {
            // If we hit a chance node, re-deal board cards.
            if let V2GameNode::Chance { .. } = &self.tree.nodes[self.node_idx as usize] {
                let cards_needed = match self.current_street() {
                    Street::Preflop => 0,
                    Street::Flop => 3,
                    Street::Turn => 4,
                    Street::River => 5,
                };
                let target = cards_needed.min(saved_board.len());
                while self.board.len() < target {
                    let idx = self.board.len();
                    self.deal_card(&saved_board[idx])?;
                }
            }
            self.play_action(action_id)?;
        }

        // If we're at a chance node after replay, re-deal remaining board cards.
        if let V2GameNode::Chance { .. } = &self.tree.nodes[self.node_idx as usize] {
            let cards_needed = match self.current_street() {
                Street::Preflop => 0,
                Street::Flop => 3,
                Street::Turn => 4,
                Street::River => 5,
            };
            let target = cards_needed.min(saved_board.len());
            while self.board.len() < target {
                let idx = self.board.len();
                self.deal_card(&saved_board[idx])?;
            }
        }

        Ok(())
    }

    /// Stub: subgame solver integration (not yet implemented).
    pub fn solve(&self) -> Result<(), String> {
        Err("Solve not yet implemented".to_string())
    }

    /// Multiply the acting player's weights by action probability at each hand.
    #[allow(clippy::cast_possible_truncation)]
    fn propagate_weights(
        &mut self,
        decision_idx: usize,
        player: u8,
        street: Street,
        action_idx: usize,
    ) {
        let weight_idx = self.weight_index(player);
        let num_buckets = self.strategy.bucket_counts
            [self.strategy.node_street_indices[decision_idx] as usize]
            as usize;

        if street == Street::Preflop {
            // Preflop: 169 canonical hands, each maps to multiple 1326 combos.
            let combo_map = build_canonical_to_combo_map();
            for hand_idx in 0..169 {
                let bucket = if num_buckets == 169 {
                    hand_idx as u16
                } else {
                    (hand_idx % num_buckets) as u16
                };
                let probs = self.strategy.get_action_probs(decision_idx, bucket);
                let p = probs.get(action_idx).copied().unwrap_or(0.0);
                for &ci in &combo_map[hand_idx] {
                    self.weights[weight_idx][ci] *= p;
                }
            }
        } else if let Some(ctx) = &self.cbv_context {
            // Postflop: look up each combo's bucket and multiply.
            use poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id;
            use poker_solver_core::hands::all_hands;
            use range_solver::card::card_pair_to_index;

            let board_cards = match parse_board(&self.board) {
                Ok(b) => b,
                Err(_) => return,
            };
            let board_slice = board_for_street_slice(&board_cards, street);

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
        // If no CbvContext for postflop, weights are unchanged (blueprint-only mode).
    }

    /// Compute pot at the current node (in half-BB units for display).
    #[allow(clippy::cast_possible_truncation)]
    fn compute_pot(&self) -> i32 {
        (pot_at_v2_node(&self.tree, self.node_idx) * 2.0) as i32
    }

    /// Compute remaining stacks [BB, SB] (in half-BB units for display).
    #[allow(clippy::cast_possible_truncation)]
    fn compute_stacks(&self) -> [i32; 2] {
        let pot = pot_at_v2_node(&self.tree, self.node_idx);
        let stack_depth = self.config.game.stack_depth;
        let each_invested = pot / 2.0;
        let remaining = ((stack_depth - each_invested) * 2.0) as i32;
        [remaining, remaining]
    }

    /// For testing: create a session with a tree but no real strategy.
    #[cfg(test)]
    fn new_for_test(tree: V2GameTree) -> Self {
        let root = tree.root;
        let node_count = tree.nodes.len();
        let strategy = BlueprintV2Strategy::empty();
        let config = make_test_config();
        GameSession {
            tree: Box::new(tree),
            strategy: Box::new(strategy),
            decision_map: vec![u32::MAX; node_count],
            config: Box::new(config),
            cbv_context: None,
            hand_evs: None,
            node_idx: root,
            board: vec![],
            action_history: vec![],
            weights: [vec![1.0f32; 1326], vec![1.0f32; 1326]],
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Convert a `TreeAction` to a `GameAction`.
fn build_game_actions(tree_actions: &[TreeAction]) -> Vec<GameAction> {
    tree_actions
        .iter()
        .enumerate()
        .map(|(i, a)| GameAction {
            id: i.to_string(),
            label: format_tree_action(a),
            action_type: action_type_string(a),
        })
        .collect()
}

/// Format a tree action as a human-readable label.
fn format_tree_action(action: &TreeAction) -> String {
    match action {
        TreeAction::Fold => "Fold".to_string(),
        TreeAction::Check => "Check".to_string(),
        TreeAction::Call => "Call".to_string(),
        TreeAction::Bet(amount) => format!("{amount:.0}bb"),
        TreeAction::Raise(amount) => format!("{amount:.0}bb"),
        TreeAction::AllIn => "All-in".to_string(),
    }
}

/// Get the action type string for a tree action.
fn action_type_string(action: &TreeAction) -> String {
    match action {
        TreeAction::Fold => "fold".to_string(),
        TreeAction::Check => "check".to_string(),
        TreeAction::Call => "call".to_string(),
        TreeAction::Bet(_) => "bet".to_string(),
        TreeAction::Raise(_) => "raise".to_string(),
        TreeAction::AllIn => "allin".to_string(),
    }
}

/// Convert a `Street` to its display string.
fn street_to_string(street: Street) -> String {
    match street {
        Street::Preflop => "Preflop".to_string(),
        Street::Flop => "Flop".to_string(),
        Street::Turn => "Turn".to_string(),
        Street::River => "River".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Core functions (no Tauri dependency, usable from Axum devserver)
// ---------------------------------------------------------------------------

/// Create a new game session from the loaded exploration state.
pub fn game_new_core(
    exploration: &crate::exploration::ExplorationState,
    postflop: &crate::postflop::PostflopState,
    session_state: &GameSessionState,
) -> Result<(), String> {
    let cbv_ctx = postflop.cbv_context.read().clone();
    let session = GameSession::from_exploration_state(exploration, cbv_ctx)?;
    *session_state.session.write() = Some(session);
    Ok(())
}

/// Get the current game state.
pub fn game_get_state_core(session_state: &GameSessionState) -> Result<GameState, String> {
    let guard = session_state.session.read();
    let session = guard.as_ref().ok_or("No game session active")?;
    Ok(session.get_state())
}

/// Play an action and return the new game state.
pub fn game_play_action_core(
    session_state: &GameSessionState,
    action_id: &str,
) -> Result<GameState, String> {
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.play_action(action_id)?;
    Ok(session.get_state())
}

/// Deal a board card and return the new game state.
pub fn game_deal_card_core(
    session_state: &GameSessionState,
    card: &str,
) -> Result<GameState, String> {
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.deal_card(card)?;
    Ok(session.get_state())
}

/// Undo the last action and return the new game state.
pub fn game_back_core(session_state: &GameSessionState) -> Result<GameState, String> {
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.back()?;
    Ok(session.get_state())
}

/// Start a subgame solve (stub).
pub fn game_solve_core(session_state: &GameSessionState) -> Result<(), String> {
    let guard = session_state.session.read();
    let session = guard.as_ref().ok_or("No game session active")?;
    session.solve()
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn game_new(
    exploration: tauri::State<'_, crate::exploration::ExplorationState>,
    postflop_state: tauri::State<'_, Arc<crate::postflop::PostflopState>>,
    session_state: tauri::State<'_, GameSessionState>,
) -> Result<(), String> {
    let cbv_ctx = postflop_state.cbv_context.read().clone();
    let session = GameSession::from_exploration_state(&exploration, cbv_ctx)?;
    *session_state.session.write() = Some(session);
    Ok(())
}

#[tauri::command]
pub fn game_get_state(
    session_state: tauri::State<'_, GameSessionState>,
) -> Result<GameState, String> {
    let guard = session_state.session.read();
    let session = guard.as_ref().ok_or("No game session active")?;
    Ok(session.get_state())
}

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

#[tauri::command]
pub fn game_solve(
    session_state: tauri::State<'_, GameSessionState>,
) -> Result<(), String> {
    let guard = session_state.session.read();
    let session = guard.as_ref().ok_or("No game session active")?;
    session.solve()
}

#[cfg(test)]
fn make_test_config() -> BlueprintV2Config {
    use poker_solver_core::blueprint_v2::config::*;
    BlueprintV2Config {
        game: GameConfig {
            name: "test".to_string(),
            players: 2,
            stack_depth: 100.0,
            small_blind: 0.5,
            big_blind: 1.0,
            rake_rate: 0.0,
            rake_cap: 0.0,
        },
        clustering: ClusteringConfig {
            algorithm: ClusteringAlgorithm::PotentialAwareEmd,
            preflop: StreetClusterConfig {
                buckets: 169,
                delta_bins: None,
                expected_delta: false,
                sample_boards: None,
            },
            flop: StreetClusterConfig {
                buckets: 10,
                delta_bins: None,
                expected_delta: false,
                sample_boards: None,
            },
            turn: StreetClusterConfig {
                buckets: 10,
                delta_bins: None,
                expected_delta: false,
                sample_boards: None,
            },
            river: StreetClusterConfig {
                buckets: 10,
                delta_bins: None,
                expected_delta: false,
                sample_boards: None,
            },
            seed: 42,
            kmeans_iterations: 100,
            cfvnet_river_data: None,
            per_flop: None,
        },
        action_abstraction: ActionAbstractionConfig {
            preflop: vec![],
            flop: vec![],
            turn: vec![],
            river: vec![],
        },
        training: TrainingConfig {
            cluster_path: None,
            iterations: None,
            time_limit_minutes: None,
            lcfr_warmup_iterations: 0,
            lcfr_discount_interval: 1,
            prune_after_iterations: 0,
            prune_threshold: 0,
            prune_explore_pct: 0.0,
            print_every_minutes: 1,
            batch_size: 1,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.0,
            dcfr_gamma: 2.0,
            dcfr_epoch_cap: None,
            target_strategy_delta: None,
            purify_threshold: 0.0,
            equity_cache_path: None,
        },
        snapshots: SnapshotConfig {
            warmup_minutes: 0,
            snapshot_every_minutes: 1,
            output_dir: "/tmp/test".to_string(),
            resume: false,
            max_snapshots: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // position_label tests
    // -------------------------------------------------------------------

    #[test]
    fn position_label_dealer_is_sb() {
        // V2 convention: tree.dealer = 0, so player 0 = SB.
        let session = make_test_session(0);
        assert_eq!(session.position_label(0), "SB");
    }

    #[test]
    fn position_label_non_dealer_is_bb() {
        let session = make_test_session(0);
        assert_eq!(session.position_label(1), "BB");
    }

    #[test]
    fn position_label_dealer_1() {
        // If dealer = 1, then player 1 = SB, player 0 = BB.
        let session = make_test_session(1);
        assert_eq!(session.position_label(1), "SB");
        assert_eq!(session.position_label(0), "BB");
    }

    // -------------------------------------------------------------------
    // weight_index tests
    // -------------------------------------------------------------------

    #[test]
    fn weight_index_dealer_is_ip_slot_1() {
        let session = make_test_session(0);
        // Dealer (SB/IP) maps to weight index 1.
        assert_eq!(session.weight_index(0), 1);
    }

    #[test]
    fn weight_index_non_dealer_is_oop_slot_0() {
        let session = make_test_session(0);
        // Non-dealer (BB/OOP) maps to weight index 0.
        assert_eq!(session.weight_index(1), 0);
    }

    #[test]
    fn weight_index_dealer_1() {
        let session = make_test_session(1);
        assert_eq!(session.weight_index(1), 1); // player 1 is dealer = IP = slot 1
        assert_eq!(session.weight_index(0), 0); // player 0 is BB = OOP = slot 0
    }

    // -------------------------------------------------------------------
    // build_actions tests
    // -------------------------------------------------------------------

    #[test]
    fn build_actions_fold_check_call() {
        let actions = vec![TreeAction::Fold, TreeAction::Check, TreeAction::Call];
        let result = build_game_actions(&actions);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, "0");
        assert_eq!(result[0].label, "Fold");
        assert_eq!(result[0].action_type, "fold");
        assert_eq!(result[1].label, "Check");
        assert_eq!(result[1].action_type, "check");
        assert_eq!(result[2].label, "Call");
        assert_eq!(result[2].action_type, "call");
    }

    #[test]
    fn build_actions_bet_raise_allin() {
        let actions = vec![
            TreeAction::Bet(2.5),
            TreeAction::Raise(6.0),
            TreeAction::AllIn,
        ];
        let result = build_game_actions(&actions);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].action_type, "bet");
        assert_eq!(result[1].action_type, "raise");
        assert_eq!(result[2].label, "All-in");
        assert_eq!(result[2].action_type, "allin");
    }

    #[test]
    fn build_actions_empty() {
        let result = build_game_actions(&[]);
        assert!(result.is_empty());
    }

    // -------------------------------------------------------------------
    // street_name tests
    // -------------------------------------------------------------------

    #[test]
    fn street_name_strings() {
        assert_eq!(street_to_string(Street::Preflop), "Preflop");
        assert_eq!(street_to_string(Street::Flop), "Flop");
        assert_eq!(street_to_string(Street::Turn), "Turn");
        assert_eq!(street_to_string(Street::River), "River");
    }

    // -------------------------------------------------------------------
    // GameState serialization (type existence check)
    // -------------------------------------------------------------------

    #[test]
    fn game_state_is_serializable() {
        let state = GameState {
            street: "Preflop".to_string(),
            position: "SB".to_string(),
            board: vec![],
            pot: 3,
            stacks: [100, 100],
            matrix: None,
            actions: vec![],
            action_history: vec![],
            is_terminal: false,
            is_chance: false,
            solve: None,
        };
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("Preflop"));
        assert!(json.contains("\"pot\":3"));
    }

    #[test]
    fn game_matrix_cell_is_serializable() {
        let cell = GameMatrixCell {
            hand: "AKs".to_string(),
            suited: true,
            pair: false,
            probabilities: vec![0.5, 0.3, 0.2],
            combo_count: 4,
            weight: 0.85,
            ev: Some(1.5),
        };
        let json = serde_json::to_string(&cell).unwrap();
        assert!(json.contains("AKs"));
        assert!(json.contains("0.85"));
    }

    #[test]
    fn action_record_is_serializable() {
        let record = ActionRecord {
            action_id: "0".to_string(),
            label: "Fold".to_string(),
            position: "BB".to_string(),
            street: "Preflop".to_string(),
            pot: 3,
            stack: 100,
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("Fold"));
    }

    #[test]
    fn solve_status_is_serializable() {
        let status = SolveStatus {
            iteration: 100,
            max_iterations: 1000,
            exploitability: 0.5,
            elapsed_secs: 2.3,
            solver_name: "CfvSubgame".to_string(),
            is_complete: false,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("CfvSubgame"));
    }

    // -------------------------------------------------------------------
    // get_state on a minimal tree
    // -------------------------------------------------------------------

    #[test]
    fn get_state_terminal_node() {
        let session = make_terminal_session();
        let state = session.get_state();
        assert!(state.is_terminal);
        assert!(!state.is_chance);
        assert!(state.actions.is_empty());
        assert!(state.matrix.is_none());
    }

    #[test]
    fn get_state_chance_node() {
        let session = make_chance_session();
        let state = session.get_state();
        assert!(state.is_chance);
        assert!(!state.is_terminal);
    }

    #[test]
    fn get_state_decision_node_has_actions() {
        let session = make_decision_session();
        let state = session.get_state();
        assert!(!state.is_terminal);
        assert!(!state.is_chance);
        assert!(!state.actions.is_empty());
        assert_eq!(state.position, "SB"); // player 0 = dealer = SB
    }

    // -------------------------------------------------------------------
    // play_action tests
    // -------------------------------------------------------------------

    #[test]
    fn play_action_advances_node() {
        let mut session = make_decision_session();
        let initial_node = session.node_idx;
        session.play_action("1").unwrap(); // Call action
        assert_ne!(session.node_idx, initial_node);
    }

    #[test]
    fn play_action_records_history() {
        let mut session = make_decision_session();
        assert!(session.action_history.is_empty());
        session.play_action("0").unwrap();
        assert_eq!(session.action_history.len(), 1);
        assert_eq!(session.action_history[0].action_id, "0");
        assert_eq!(session.action_history[0].position, "SB");
    }

    #[test]
    fn play_action_invalid_id_errors() {
        let mut session = make_decision_session();
        let result = session.play_action("abc");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid action_id"));
    }

    #[test]
    fn play_action_out_of_range_errors() {
        let mut session = make_decision_session();
        let result = session.play_action("99");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of range"));
    }

    #[test]
    fn play_action_on_terminal_errors() {
        let mut session = make_terminal_session();
        let result = session.play_action("0");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Not at a decision node"));
    }

    // -------------------------------------------------------------------
    // deal_card tests
    // -------------------------------------------------------------------

    #[test]
    fn deal_card_at_chance_node_advances() {
        let mut session = make_chance_session();
        assert!(session.board.is_empty());
        session.deal_card("Ah").unwrap();
        assert_eq!(session.board, vec!["Ah".to_string()]);
    }

    #[test]
    fn deal_card_at_decision_errors() {
        let mut session = make_decision_session();
        let result = session.deal_card("Ah");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Not at a chance node"));
    }

    // -------------------------------------------------------------------
    // back tests
    // -------------------------------------------------------------------

    #[test]
    fn back_with_no_history_errors() {
        let mut session = make_decision_session();
        let result = session.back();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No actions to undo"));
    }

    #[test]
    fn back_restores_to_root() {
        let mut session = make_two_level_session();
        let root = session.node_idx;
        session.play_action("1").unwrap(); // go to child
        assert_ne!(session.node_idx, root);
        session.back().unwrap();
        assert_eq!(session.node_idx, root);
        assert!(session.action_history.is_empty());
    }

    // -------------------------------------------------------------------
    // game_solve stub test
    // -------------------------------------------------------------------

    #[test]
    fn solve_returns_not_implemented() {
        let session = make_decision_session();
        let result = session.solve();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not yet implemented"));
    }

    // -------------------------------------------------------------------
    // Test helpers — minimal trees without real strategies
    // -------------------------------------------------------------------

    /// Create a GameSession with a minimal tree pointing at the root.
    /// `dealer` controls which seat is SB.
    fn make_test_session(dealer: u8) -> GameSession {
        let tree = make_minimal_decision_tree(dealer);
        GameSession::new_for_test(tree)
    }

    fn make_terminal_session() -> GameSession {
        use poker_solver_core::blueprint_v2::game_tree::TerminalKind;
        let mut tree = V2GameTree {
            nodes: vec![V2GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 10.0,
                stacks: [95.0, 95.0],
            }],
            root: 0,
            dealer: 0,
            starting_stack: 100.0,
        };
        GameSession::new_for_test(tree)
    }

    fn make_chance_session() -> GameSession {
        use poker_solver_core::blueprint_v2::game_tree::TerminalKind;
        let tree = V2GameTree {
            nodes: vec![
                V2GameNode::Chance {
                    next_street: Street::Flop,
                    child: 1,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Flop,
                    actions: vec![TreeAction::Check, TreeAction::Fold],
                    children: vec![2, 3],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 10.0,
                    stacks: [95.0, 95.0],
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 0 },
                    pot: 10.0,
                    stacks: [95.0, 95.0],
                },
            ],
            root: 0,
            dealer: 0,
            starting_stack: 100.0,
        };
        GameSession::new_for_test(tree)
    }

    fn make_decision_session() -> GameSession {
        let tree = make_minimal_decision_tree(0);
        GameSession::new_for_test(tree)
    }

    /// A two-level tree: root decision -> child decision, for testing back().
    fn make_two_level_session() -> GameSession {
        use poker_solver_core::blueprint_v2::game_tree::TerminalKind;
        let tree = V2GameTree {
            nodes: vec![
                // 0: root decision (player 0)
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Fold, TreeAction::Call],
                    children: vec![1, 2],
                    blueprint_decision_idx: None,
                },
                // 1: fold terminal
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 1 },
                    pot: 3.0,
                    stacks: [99.5, 100.5],
                },
                // 2: child decision (player 1)
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Check, TreeAction::Fold],
                    children: vec![3, 4],
                    blueprint_decision_idx: None,
                },
                // 3: showdown
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 4.0,
                    stacks: [98.0, 98.0],
                },
                // 4: fold terminal
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 0 },
                    pot: 4.0,
                    stacks: [98.0, 98.0],
                },
            ],
            root: 0,
            dealer: 0,
            starting_stack: 100.0,
        };
        GameSession::new_for_test(tree)
    }

    fn make_minimal_decision_tree(dealer: u8) -> V2GameTree {
        use poker_solver_core::blueprint_v2::game_tree::TerminalKind;
        V2GameTree {
            nodes: vec![
                // 0: decision node
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Fold, TreeAction::Call, TreeAction::AllIn],
                    children: vec![1, 2, 3],
                    blueprint_decision_idx: None,
                },
                // 1: fold terminal
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 1 },
                    pot: 3.0,
                    stacks: [99.5, 100.5],
                },
                // 2: call terminal (showdown)
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 4.0,
                    stacks: [98.0, 98.0],
                },
                // 3: all-in terminal
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 200.0,
                    stacks: [0.0, 0.0],
                },
            ],
            root: 0,
            dealer,
            starting_stack: 100.0,
        }
    }
}
