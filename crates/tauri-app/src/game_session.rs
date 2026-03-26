//! Unified game session: tracks state from preflop through river.
//!
//! A `GameSession` owns the V2 game tree, blueprint strategy, and range weights.
//! Six Tauri commands expose it: `game_new`, `game_get_state`, `game_play_action`,
//! `game_deal_card`, `game_back`, `game_solve`.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use serde::Serialize;

use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
use poker_solver_core::blueprint_v2::continuation::BiasType;
use poker_solver_core::blueprint_v2::game_tree::{
    GameNode as V2GameNode, GameTree as V2GameTree, TreeAction,
};
use poker_solver_core::blueprint_v2::LeafEvaluator;
use poker_solver_core::blueprint_v2::Street;

use range_solver::card::{card_to_string, NOT_DEALT};
use range_solver::{PostFlopGame, solve_step, finalize, compute_exploitability};

use crate::exploration::{
    board_for_street_slice, blueprint_sizes_to_range_solver, build_canonical_to_combo_map,
    canonical_hand_index_from_ranks, hand_label_from_matrix, parse_board, pot_at_v2_node,
    ActionInfo, BucketLookup, RANKS,
};
use crate::postflop::{CbvContext, RolloutLeafEvaluator, parse_rs_poker_card};

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
    /// All actions that were available at this decision point.
    pub actions: Vec<GameAction>,
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

/// Per-combo strategy detail (e.g., "AhKh" with its own action probabilities).
#[derive(Debug, Clone, Serialize)]
pub struct ComboDetail {
    pub cards: String,           // e.g. "AhKh"
    pub probabilities: Vec<f32>, // one per action
    pub weight: f32,             // reaching probability for this combo
}

/// A single cell in the 13x13 strategy matrix.
#[derive(Debug, Clone, Serialize)]
pub struct GameMatrixCell {
    pub hand: String,
    pub suited: bool,
    pub pair: bool,
    pub probabilities: Vec<f32>, // one per action (averaged across combos)
    pub combo_count: usize,
    pub weight: f32, // reaching probability
    pub ev: Option<f32>,
    pub combos: Vec<ComboDetail>,
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
// Shared solve state (background thread <-> UI queries)
// ---------------------------------------------------------------------------

/// Shared state between the background solve thread and UI queries.
///
/// Lives in `GameSessionState` (not `GameSession`) because it must be
/// `Arc`-shared with the background thread while the session is behind
/// a `RwLock`.
pub struct SolveState {
    pub solving: AtomicBool,
    pub cancel: AtomicBool,
    pub iteration: AtomicU32,
    pub max_iterations: AtomicU32,
    /// Exploitability stored as f32 bits (use `f32::to_bits` / `f32::from_bits`).
    pub exploitability_bits: AtomicU32,
    pub solve_start: RwLock<Option<Instant>>,
    /// Matrix snapshot updated during solve.
    pub matrix_snapshot: RwLock<Option<GameMatrix>>,
    /// Actions at the solve game's root node.
    pub solve_actions: RwLock<Vec<GameAction>>,
    /// Position label at the solve root.
    pub solve_position: RwLock<String>,
}

impl Default for SolveState {
    fn default() -> Self {
        Self {
            solving: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
            iteration: AtomicU32::new(0),
            max_iterations: AtomicU32::new(0),
            exploitability_bits: AtomicU32::new(0),
            solve_start: RwLock::new(None),
            matrix_snapshot: RwLock::new(None),
            solve_actions: RwLock::new(vec![]),
            solve_position: RwLock::new(String::new()),
        }
    }
}

impl SolveState {
    /// Reset all fields to defaults. Called when starting a new hand or going back.
    pub fn reset(&self) {
        self.solving.store(false, Ordering::Relaxed);
        self.cancel.store(false, Ordering::Relaxed);
        self.iteration.store(0, Ordering::Relaxed);
        self.max_iterations.store(0, Ordering::Relaxed);
        self.exploitability_bits.store(0, Ordering::Relaxed);
        *self.solve_start.write() = None;
        *self.matrix_snapshot.write() = None;
        *self.solve_actions.write() = vec![];
        *self.solve_position.write() = String::new();
    }
}

// ---------------------------------------------------------------------------
// Shared state for Tauri commands
// ---------------------------------------------------------------------------

/// Shared session state, accessible by Tauri commands.
pub struct GameSessionState {
    pub session: RwLock<Option<GameSession>>,
    pub solve_state: Arc<SolveState>,
}

impl Default for GameSessionState {
    fn default() -> Self {
        Self {
            session: RwLock::new(None),
            solve_state: Arc::new(SolveState::default()),
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

                // Build per-combo details.
                let combos = self.build_combo_details(
                    combo_indices, weight_idx, decision_idx, street, actions.len(),
                    &board_cards,
                );

                row_cells.push(GameMatrixCell {
                    hand: label,
                    suited,
                    pair,
                    probabilities,
                    combo_count,
                    weight,
                    ev,
                    combos,
                });
            }
            cells.push(row_cells);
        }
        cells
    }

    /// Build per-combo strategy details for a canonical hand's combos.
    fn build_combo_details(
        &self,
        combo_indices: &[usize],
        weight_idx: usize,
        decision_idx: usize,
        street: Street,
        num_actions: usize,
        board_cards: &Option<Vec<rs_poker::core::Card>>,
    ) -> Vec<ComboDetail> {
        use range_solver::card::{card_to_string, index_to_card_pair};

        combo_indices
            .iter()
            .filter_map(|&ci| {
                let w = self.weights[weight_idx][ci];
                if w <= 0.0 {
                    return None;
                }

                let (c1_raw, c2_raw) = index_to_card_pair(ci);
                // Show high card first: rank = id / 4, higher rank = higher id.
                let (c1, c2) = if c1_raw / 4 >= c2_raw / 4 { (c1_raw, c2_raw) } else { (c2_raw, c1_raw) };
                let s1 = card_to_string(c1).unwrap_or_default();
                let s2 = card_to_string(c2).unwrap_or_default();

                // Check board blockers for postflop.
                if let Some(board) = board_cards {
                    // Convert range-solver card IDs to rs_poker for comparison.
                    let rs_c1 = crate::exploration::range_solver_to_rs_card(c1);
                    let rs_c2 = crate::exploration::range_solver_to_rs_card(c2);
                    if board.iter().any(|b| *b == rs_c1 || *b == rs_c2) {
                        return None;
                    }
                }

                let probs = if street == Street::Preflop {
                    // Preflop: all combos of a canonical hand share the same strategy.
                    // Return empty — the cell's aggregated probs are sufficient.
                    vec![]
                } else if let (Some(ctx), Some(board)) = (&self.cbv_context, board_cards) {
                    // Postflop: per-combo bucket lookup.
                    let board_slice = board_for_street_slice(board, street);
                    let rs_c1 = crate::exploration::range_solver_to_rs_card(c1);
                    let rs_c2 = crate::exploration::range_solver_to_rs_card(c2);
                    let bucket = ctx.all_buckets.get_bucket(street, [rs_c1, rs_c2], board_slice);
                    let strategy_probs = ctx.strategy.get_action_probs(decision_idx, bucket);
                    (0..num_actions)
                        .map(|i| strategy_probs.get(i).copied().unwrap_or(0.0))
                        .collect()
                } else {
                    vec![0.0; num_actions]
                };

                Some(ComboDetail {
                    cards: format!("{s1}{s2}"),
                    probabilities: probs,
                    weight: w,
                })
            })
            .collect()
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

        // Record breadcrumb with all available actions.
        let position = self.position_label(player).to_string();
        let pot = self.compute_pot();
        let all_actions = build_game_actions(&actions);
        let wi = self.weight_index(player);
        let stack = self.compute_stacks()[wi];
        self.action_history.push(ActionRecord {
            action_id: action_id.to_string(),
            label: format_tree_action(&actions[action_idx]),
            position,
            street: street_to_string(street),
            pot,
            stack,
            actions: all_actions,
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
                // Determine target card count BEFORE pushing (based on current board).
                let cards_needed = match self.board.len() {
                    0..=2 => 3,  // flop needs 3 total
                    3 => 4,      // turn needs 4 total
                    4 => 5,      // river needs 5 total
                    _ => self.board.len() + 1,
                };

                self.board.push(card.to_string());

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
// Range-solver helpers for subgame solving
// ---------------------------------------------------------------------------

/// Parse board card strings into range-solver card IDs.
fn parse_solve_board(
    board: &[String],
) -> Result<([u8; 3], u8, u8, range_solver::BoardState), String> {
    use range_solver::card::{card_from_str, flop_from_str};

    match board.len() {
        3 => {
            let flop_str = format!("{}{}{}", board[0], board[1], board[2]);
            let flop = flop_from_str(&flop_str).map_err(|e| format!("Bad flop: {e}"))?;
            Ok((flop, NOT_DEALT, NOT_DEALT, range_solver::BoardState::Flop))
        }
        4 => {
            let flop_str = format!("{}{}{}", board[0], board[1], board[2]);
            let flop = flop_from_str(&flop_str).map_err(|e| format!("Bad flop: {e}"))?;
            let turn = card_from_str(&board[3]).map_err(|e| format!("Bad turn: {e}"))?;
            Ok((flop, turn, NOT_DEALT, range_solver::BoardState::Turn))
        }
        5 => {
            let flop_str = format!("{}{}{}", board[0], board[1], board[2]);
            let flop = flop_from_str(&flop_str).map_err(|e| format!("Bad flop: {e}"))?;
            let turn = card_from_str(&board[3]).map_err(|e| format!("Bad turn: {e}"))?;
            let river = card_from_str(&board[4]).map_err(|e| format!("Bad river: {e}"))?;
            Ok((flop, turn, river, range_solver::BoardState::River))
        }
        n => Err(format!("Board must have 3-5 cards, got {n}")),
    }
}

/// Convert blueprint bet sizes (`Vec<Vec<f64>>` pot fractions per raise depth)
/// into range-solver format strings: `(bet_str, raise_str)`.
///
/// Reuses `blueprint_sizes_to_range_solver` from exploration.rs.
fn format_bet_sizes_for_solve(sizes: &[Vec<f64>]) -> (String, String) {
    blueprint_sizes_to_range_solver(sizes)
}

/// Build a `PostFlopGame` from session state, ready for solving.
#[allow(clippy::too_many_arguments)]
fn build_solve_game(
    board: &[String],
    oop_weights: &[f32],
    ip_weights: &[f32],
    pot: i32,
    effective_stack: i32,
    bet_sizes: &[Vec<f64>],
) -> Result<PostFlopGame, String> {
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::CardConfig;
    use range_solver::range::Range;
    use range_solver::{ActionTree, TreeConfig};

    let (flop, turn, river, initial_state) = parse_solve_board(board)?;

    let oop_range =
        Range::from_raw_data(oop_weights).map_err(|e| format!("Bad OOP weights: {e}"))?;
    let ip_range =
        Range::from_raw_data(ip_weights).map_err(|e| format!("Bad IP weights: {e}"))?;

    let (bet_str, raise_str) = format_bet_sizes_for_solve(bet_sizes);
    let oop_sizes = BetSizeOptions::try_from((bet_str.as_str(), raise_str.as_str()))
        .map_err(|e| format!("Bad bet sizes: {e}"))?;
    let ip_sizes = oop_sizes.clone();

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop,
        turn,
        river,
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
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        merging_threshold: 0.0,
        // River: solve to showdown (no boundaries).
        // Flop/Turn: solve current street only, boundaries at next street.
        depth_limit: if initial_state == range_solver::BoardState::River { None } else { Some(0) },
    };

    let action_tree =
        ActionTree::new(tree_config).map_err(|e| format!("Failed to build tree: {e}"))?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;
    game.allocate_memory(false);
    Ok(game)
}

/// Convert a range-solver `Action` to a `GameAction`.
fn range_solver_action_to_game_action(
    action: &range_solver::Action,
    idx: usize,
) -> GameAction {
    let (label, action_type) = match action {
        range_solver::Action::Fold => ("Fold".to_string(), "fold"),
        range_solver::Action::Check => ("Check".to_string(), "check"),
        range_solver::Action::Call => ("Call".to_string(), "call"),
        range_solver::Action::Bet(amt) => {
            let bb = *amt as f64 / 2.0;
            (format!("{bb:.0}bb"), "bet")
        }
        range_solver::Action::Raise(amt) => {
            let bb = *amt as f64 / 2.0;
            (format!("{bb:.0}bb"), "raise")
        }
        range_solver::Action::AllIn(_) => ("All-in".to_string(), "allin"),
        _ => ("?".to_string(), "unknown"),
    };
    GameAction {
        id: idx.to_string(),
        label,
        action_type: action_type.to_string(),
    }
}

/// Build a `GameMatrix` from the current `PostFlopGame` state at the root node.
///
/// Reuses the `card_pair_to_matrix` and `matrix_cell_label` helpers from postflop.rs.
#[allow(clippy::cast_possible_truncation)]
fn build_solve_matrix(game: &mut PostFlopGame, hand_evs: Option<&[f32]>) -> GameMatrix {
    use crate::postflop::{card_pair_to_matrix, matrix_cell_label};
    use range_solver::interface::Game;

    game.back_to_root();
    let player = game.current_player();
    let strategy = game.strategy();
    let private_cards = game.private_cards(player);
    let num_hands = game.num_private_hands(player);
    let initial_weights = game.initial_weights(player);
    let available_actions = game.available_actions();

    let game_actions: Vec<GameAction> = available_actions
        .iter()
        .enumerate()
        .map(|(i, a)| range_solver_action_to_game_action(a, i))
        .collect();
    let num_actions = game_actions.len();

    // Accumulators for 13x13 grid
    let mut prob_sums = vec![vec![vec![0.0f64; num_actions]; 13]; 13];
    let mut combo_counts = vec![vec![0usize; 13]; 13];
    let mut weight_sums = vec![vec![0.0f64; 13]; 13];
    let mut ev_sums = vec![vec![0.0f64; 13]; 13];
    let mut combo_details: Vec<Vec<Vec<ComboDetail>>> = vec![vec![Vec::new(); 13]; 13];

    for (hand_idx, &(c1, c2)) in private_cards.iter().enumerate() {
        let (row, col, _) = card_pair_to_matrix(c1, c2);
        combo_counts[row][col] += 1;
        weight_sums[row][col] += initial_weights[hand_idx] as f64;
        if let Some(evs) = hand_evs {
            if hand_idx < evs.len() {
                ev_sums[row][col] += evs[hand_idx] as f64;
            }
        }

        let mut probs = Vec::with_capacity(num_actions);
        for (action_idx, prob_sum) in prob_sums[row][col].iter_mut().enumerate() {
            let prob = strategy[action_idx * num_hands + hand_idx];
            *prob_sum += prob as f64;
            probs.push(prob);
        }

        let s1 = card_to_string(c1).unwrap_or_default();
        let s2 = card_to_string(c2).unwrap_or_default();
        combo_details[row][col].push(ComboDetail {
            cards: format!("{s1}{s2}"),
            probabilities: probs,
            weight: initial_weights[hand_idx],
        });
    }

    let cells: Vec<Vec<GameMatrixCell>> = (0..13)
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
                    let ev = if count > 0 && hand_evs.is_some() {
                        Some((ev_sums[row][col] / count as f64) as f32)
                    } else {
                        None
                    };
                    let combos = std::mem::take(&mut combo_details[row][col]);
                    let weight = if count > 0 {
                        (weight_sums[row][col] / count as f64) as f32
                    } else {
                        0.0
                    };
                    let combo_count = combos.iter().filter(|c| c.weight > 0.0).count();
                    GameMatrixCell {
                        hand: label,
                        suited,
                        pair,
                        probabilities,
                        combo_count,
                        weight,
                        ev,
                        combos,
                    }
                })
                .collect()
        })
        .collect();

    GameMatrix {
        cells,
        actions: game_actions,
    }
}

/// Evaluate boundary CFVs using a `RolloutLeafEvaluator` and inject them
/// into the `PostFlopGame`.
#[allow(clippy::too_many_arguments)]
fn evaluate_and_inject_boundaries(
    game: &mut PostFlopGame,
    evaluator: &RolloutLeafEvaluator,
    board_cards: &[rs_poker::core::Card],
    oop_weights: &[f64],
    ip_weights: &[f64],
    combos: &[[rs_poker::core::Card; 2]],
) {
    let n_boundaries = game.num_boundary_nodes();
    if n_boundaries == 0 {
        return;
    }

    // Build requests: (pot, effective_stack, traverser) per boundary.
    // We need both players, so we evaluate twice (once per traverser).
    for traverser in 0..2u8 {
        let requests: Vec<(f64, f64, u8)> = (0..n_boundaries)
            .map(|ordinal| {
                let pot = game.boundary_pot(ordinal) as f64;
                (pot, 0.0, traverser)
            })
            .collect();

        let cfv_results = evaluator.evaluate_boundaries(
            combos,
            board_cards,
            oop_weights,
            ip_weights,
            &requests,
        );

        // Inject CFVs for each boundary
        for (ordinal, cfvs_f64) in cfv_results.into_iter().enumerate() {
            // Convert from SubgameHands combo ordering to PostFlopGame private_cards ordering
            let private_cards = game.private_cards(traverser as usize);
            let mut mapped_cfvs = vec![0.0f32; private_cards.len()];

            // Build lookup: (rs_poker card pair) -> combo_idx in evaluator combos
            for (hand_idx, &(c1, c2)) in private_cards.iter().enumerate() {
                // Find matching combo in evaluator combos
                let rs_c1 = crate::exploration::range_solver_to_rs_card(c1);
                let rs_c2 = crate::exploration::range_solver_to_rs_card(c2);
                for (ci, combo) in combos.iter().enumerate() {
                    if (combo[0] == rs_c1 && combo[1] == rs_c2)
                        || (combo[0] == rs_c2 && combo[1] == rs_c1)
                    {
                        mapped_cfvs[hand_idx] = cfvs_f64[ci] as f32;
                        break;
                    }
                }
            }

            // Diagnostic: dump boundary CFVs for first eval
            if traverser == 0 && ordinal < 3 {
                let pot = game.boundary_pot(ordinal);
                let nonzero: Vec<(usize, f32)> = mapped_cfvs.iter().enumerate()
                    .filter(|(_, &v)| v.abs() > 0.001)
                    .take(5)
                    .map(|(i, &v)| (i, v))
                    .collect();
                let min = mapped_cfvs.iter().cloned().fold(f32::MAX, f32::min);
                let max = mapped_cfvs.iter().cloned().fold(f32::MIN, f32::max);
                let nonzero_count = mapped_cfvs.iter().filter(|v| v.abs() > 0.001).count();
                eprintln!(
                    "[boundary inject] ordinal={ordinal} traverser={traverser} pot={pot} \
                     nonzero={nonzero_count}/{} min={min:.2} max={max:.2} samples={nonzero:?}",
                    mapped_cfvs.len()
                );
            }

            game.set_boundary_cfvs(ordinal, traverser as usize, mapped_cfvs);
        }
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
    session_state.solve_state.reset();
    Ok(())
}

/// Get the current game state, including solve progress if active.
pub fn game_get_state_core(session_state: &GameSessionState) -> Result<GameState, String> {
    let guard = session_state.session.read();
    let session = guard.as_ref().ok_or("No game session active")?;
    let mut state = session.get_state();

    // Override with solve state if active or just completed
    let ss = &session_state.solve_state;
    let is_solving = ss.solving.load(Ordering::Relaxed);
    let iteration = ss.iteration.load(Ordering::Relaxed);

    if is_solving || iteration > 0 {
        let exp = f32::from_bits(ss.exploitability_bits.load(Ordering::Relaxed));
        let max_iters = ss.max_iterations.load(Ordering::Relaxed);
        let elapsed = ss
            .solve_start
            .read()
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
        let actions = ss.solve_actions.read();
        if !actions.is_empty() {
            state.actions = actions.clone();
        }
        let position = ss.solve_position.read();
        if !position.is_empty() {
            state.position = position.clone();
        }
    }

    Ok(state)
}

/// Play an action and return the new game state.
pub fn game_play_action_core(
    session_state: &GameSessionState,
    action_id: &str,
) -> Result<GameState, String> {
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.play_action(action_id)?;
    let state = session.get_state();
    drop(guard);
    // Reset solve state so stale data doesn't leak into the next position.
    session_state.solve_state.reset();
    Ok(state)
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
    // Reset solve state so stale data doesn't leak.
    drop(guard);
    session_state.solve_state.reset();
    let guard = session_state.session.read();
    let session = guard.as_ref().ok_or("No game session active")?;
    Ok(session.get_state())
}

/// Start a subgame solve using range-solver `PostFlopGame` with depth limit.
///
/// Spawns a background thread that builds a `PostFlopGame`, runs an iterative
/// solve loop with optional `RolloutLeafEvaluator` boundary re-evaluation,
/// and stores matrix snapshots in `SolveState` for the UI to read.
#[allow(clippy::too_many_arguments)]
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
    use poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id;
    use range_solver::card::card_pair_to_index;

    // Guard: reject if already solving
    if session_state.solve_state.solving.load(Ordering::Relaxed) {
        return Err("A solve is already in progress".to_string());
    }

    // Read session state under lock, clone what the thread needs
    let (board, oop_w, ip_w, pot, eff_stack, bet_sizes, cbv_ctx, abstract_node_idx, position_label) = {
        let guard = session_state.session.read();
        let session = guard.as_ref().ok_or("No game session active")?;

        // Must be at a postflop decision node
        if session.board.len() < 3 {
            return Err("Solve requires a postflop position (deal board cards first)".to_string());
        }
        let node = &session.tree.nodes[session.node_idx as usize];
        let player = match node {
            V2GameNode::Decision { player, .. } => *player,
            _ => return Err("Not at a decision node".to_string()),
        };

        let board = session.board.clone();
        let oop_w = session.weights[0].clone();
        let ip_w = session.weights[1].clone();
        let pot = session.compute_pot();
        let stacks = session.compute_stacks();
        let eff_stack = stacks[0].min(stacks[1]);

        // Bet sizes from blueprint config for the current street
        let street = session.current_street();
        let sizes = match street {
            Street::Flop => &session.config.action_abstraction.flop,
            Street::Turn => &session.config.action_abstraction.turn,
            Street::River => &session.config.action_abstraction.river,
            Street::Preflop => return Err("Cannot solve preflop".to_string()),
        };

        let cbv_ctx = session.cbv_context.clone();
        let abs_node_idx = Some(session.node_idx);

        let position = session.position_label(player).to_string();

        (board, oop_w, ip_w, pot, eff_stack, sizes.clone(), cbv_ctx, abs_node_idx, position)
    };

    // Apply range clamping
    let clamp = range_clamp_threshold.unwrap_or(0.0) as f32;
    let mut oop_w = oop_w;
    let mut ip_w = ip_w;
    if clamp > 0.0 {
        for w in oop_w.iter_mut() {
            if *w > 0.0 && *w < clamp {
                *w = 0.0;
            }
        }
        for w in ip_w.iter_mut() {
            if *w > 0.0 && *w < clamp {
                *w = 0.0;
            }
        }
    }

    let max_iters = max_iterations.unwrap_or(200);
    let eval_interval = leaf_eval_interval.unwrap_or(10);
    let target_exp = target_exploitability.unwrap_or(3.0);

    // Reset solve state atomics
    let ss = &session_state.solve_state;
    ss.iteration.store(0, Ordering::Relaxed);
    ss.max_iterations.store(max_iters, Ordering::Relaxed);
    ss.exploitability_bits
        .store(f32::MAX.to_bits(), Ordering::Relaxed);
    ss.cancel.store(false, Ordering::Relaxed);
    ss.solving.store(true, Ordering::Release);
    *ss.solve_start.write() = Some(Instant::now());
    *ss.matrix_snapshot.write() = None;
    *ss.solve_position.write() = position_label;

    // Spawn background thread
    let ss_clone = Arc::clone(&session_state.solve_state);
    let board_clone = board.clone();
    std::thread::spawn(move || {
        // Build game
        let mut game = match build_solve_game(&board_clone, &oop_w, &ip_w, pot, eff_stack, &bet_sizes) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[solve] failed to build game: {e}");
                ss_clone.solving.store(false, Ordering::Release);
                return;
            }
        };

        // Store available actions at root
        {
            game.back_to_root();
            let actions: Vec<GameAction> = game
                .available_actions()
                .iter()
                .enumerate()
                .map(|(i, a)| range_solver_action_to_game_action(a, i))
                .collect();
            *ss_clone.solve_actions.write() = actions;
        }

        // Build RolloutLeafEvaluator if CbvContext available and boundaries exist
        let evaluator_data = if game.num_boundary_nodes() > 0 {
            if let Some(ctx) = &cbv_ctx {
                // Parse board for rs_poker cards
                let board_cards: Vec<rs_poker::core::Card> = board_clone
                    .iter()
                    .filter_map(|s| parse_rs_poker_card(s).ok())
                    .collect();

                // Enumerate combos (all non-blocked 2-card combos)
                let mut combos: Vec<[rs_poker::core::Card; 2]> = Vec::new();
                let mut oop_reach_f64 = Vec::new();
                let mut ip_reach_f64 = Vec::new();
                use poker_solver_core::hands::all_hands;
                for hand in all_hands() {
                    for (c0, c1) in hand.combos() {
                        if board_cards.iter().any(|b| *b == c0 || *b == c1) {
                            continue;
                        }
                        let id0 = rs_poker_card_to_id(c0);
                        let id1 = rs_poker_card_to_id(c1);
                        let ci = card_pair_to_index(id0, id1);
                        combos.push([c0, c1]);
                        oop_reach_f64.push(f64::from(oop_w[ci]));
                        ip_reach_f64.push(f64::from(ip_w[ci]));
                    }
                }

                let bias_factor = rollout_bias_factor.unwrap_or(10.0);
                let num_rollouts = rollout_num_samples.unwrap_or(3);
                let opp_samples = rollout_opponent_samples.unwrap_or(8);
                let starting_stack = f64::from(eff_stack) + f64::from(pot) / 2.0;

                let evaluator = RolloutLeafEvaluator::new(
                    Arc::clone(&ctx.strategy),
                    Arc::new(ctx.abstract_tree.clone()),
                    Arc::clone(&ctx.all_buckets),
                    abstract_node_idx.unwrap_or(0),
                    BiasType::Unbiased,
                    bias_factor,
                    num_rollouts,
                    opp_samples,
                    starting_stack,
                    f64::from(pot),
                );

                Some((evaluator, board_cards, combos, oop_reach_f64, ip_reach_f64))
            } else {
                None
            }
        } else {
            None
        };

        let n_boundaries = game.num_boundary_nodes();
        let (mem_est, _) = game.memory_usage();
        eprintln!("[solve] boundary nodes: {n_boundaries}, evaluator: {}", evaluator_data.is_some());
        eprintln!("[solve] abstract_node_idx: {:?}", abstract_node_idx);
        eprintln!("[solve] pot={pot}, eff_stack={eff_stack}, board={board:?}");
        eprintln!("[solve] OOP hands: {}, IP hands: {}", game.private_cards(0).len(), game.private_cards(1).len());
        eprintln!("[solve] memory: {:.1} MB", mem_est as f64 / 1_048_576.0);

        // Check if boundary CFVs are being read by the solver
        if n_boundaries > 0 {
            // Run one solve step, then check exploitability change
            if let Some((ref evaluator, ref board_cards, ref combos, ref oop_reach, ref ip_reach)) = evaluator_data {
                evaluate_and_inject_boundaries(
                    &mut game, evaluator, board_cards, oop_reach, ip_reach, combos,
                );
            }
            let exp_before = compute_exploitability(&game);
            solve_step(&game, 0);
            let exp_after = compute_exploitability(&game);
            eprintln!("[solve] after 1 step: expl {exp_before:.3} → {exp_after:.3}");
        }

        // Initial matrix snapshot
        let matrix = build_solve_matrix(&mut game, None);
        *ss_clone.matrix_snapshot.write() = Some(matrix);

        // Solve loop
        let mut t = 0u32;
        while t < max_iters {
            if ss_clone.cancel.load(Ordering::Relaxed) {
                break;
            }

            // Re-evaluate boundary CFVs every eval_interval
            if t.is_multiple_of(eval_interval) {
                if let Some((ref evaluator, ref board_cards, ref combos, ref oop_reach, ref ip_reach)) = evaluator_data {
                    evaluate_and_inject_boundaries(
                        &mut game, evaluator, board_cards, oop_reach, ip_reach, combos,
                    );
                }
            }

            solve_step(&game, t);
            t += 1;
            ss_clone.iteration.store(t, Ordering::Relaxed);

            // Compute exploitability and snapshot matrix periodically
            if t.is_multiple_of(eval_interval) {
                let exp = compute_exploitability(&game);
                ss_clone
                    .exploitability_bits
                    .store(exp.to_bits(), Ordering::Relaxed);

                let matrix = build_solve_matrix(&mut game, None);
                *ss_clone.matrix_snapshot.write() = Some(matrix);

                if exp <= target_exp {
                    break;
                }
            }
        }

        // Finalize: normalize strategy, compute EVs
        finalize(&mut game);
        game.back_to_root();
        game.cache_normalized_weights();
        let player = game.current_player();
        let evs = game.expected_values(player);
        let final_matrix = build_solve_matrix(&mut game, Some(&evs));
        *ss_clone.matrix_snapshot.write() = Some(final_matrix);

        // Compute final exploitability
        let final_exp = compute_exploitability(&game);
        ss_clone
            .exploitability_bits
            .store(final_exp.to_bits(), Ordering::Relaxed);

        ss_clone.solving.store(false, Ordering::Release);
        eprintln!(
            "[solve] complete: {} iterations, exploitability={:.4}",
            t, final_exp
        );
    });

    Ok(())
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
    session_state.solve_state.reset();
    Ok(())
}

#[tauri::command]
pub fn game_get_state(
    session_state: tauri::State<'_, GameSessionState>,
) -> Result<GameState, String> {
    game_get_state_core(&session_state)
}

#[tauri::command]
pub fn game_play_action(
    session_state: tauri::State<'_, GameSessionState>,
    action_id: String,
) -> Result<GameState, String> {
    game_play_action_core(&session_state, &action_id)
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
    game_back_core(&session_state)
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
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
    game_solve_core(
        &session_state,
        max_iterations,
        target_exploitability,
        leaf_eval_interval,
        rollout_bias_factor,
        rollout_num_samples,
        rollout_opponent_samples,
        range_clamp_threshold,
    )
}

pub fn game_cancel_solve_core(session_state: &GameSessionState) -> Result<(), String> {
    session_state
        .solve_state
        .cancel
        .store(true, Ordering::Relaxed);
    Ok(())
}

#[tauri::command]
pub fn game_cancel_solve(
    session_state: tauri::State<'_, GameSessionState>,
) -> Result<(), String> {
    game_cancel_solve_core(&session_state)
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
            optimizer: "dcfr".to_string(),
            sapcfr_eta: 0.5,
            use_baselines: false,
            baseline_alpha: 0.01,
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
            combos: vec![ComboDetail {
                cards: "AhKh".to_string(),
                probabilities: vec![0.5, 0.3, 0.2],
                weight: 1.0,
            }],
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
            actions: vec![],
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
    // SolveState tests
    // -------------------------------------------------------------------

    #[test]
    fn solve_state_defaults_to_not_solving() {
        let ss = SolveState::default();
        assert!(!ss.solving.load(std::sync::atomic::Ordering::Relaxed));
        assert!(!ss.cancel.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(ss.iteration.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(ss.max_iterations.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert!(ss.matrix_snapshot.read().is_none());
        assert!(ss.solve_actions.read().is_empty());
        assert!(ss.solve_position.read().is_empty());
    }

    #[test]
    fn game_session_state_has_solve_state() {
        let gss = GameSessionState::default();
        // solve_state should be an Arc<SolveState>
        assert!(!gss.solve_state.solving.load(std::sync::atomic::Ordering::Relaxed));
    }

    // -------------------------------------------------------------------
    // get_state reads solve progress
    // -------------------------------------------------------------------

    #[test]
    fn get_state_core_returns_solve_status_when_solving() {
        let gss = GameSessionState::default();
        // Set up a session
        let session = make_decision_session();
        *gss.session.write() = Some(session);

        // Simulate active solve
        gss.solve_state.solving.store(true, std::sync::atomic::Ordering::Relaxed);
        gss.solve_state.iteration.store(50, std::sync::atomic::Ordering::Relaxed);
        gss.solve_state.max_iterations.store(200, std::sync::atomic::Ordering::Relaxed);
        gss.solve_state.exploitability_bits.store(5.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
        *gss.solve_state.solve_start.write() = Some(std::time::Instant::now());

        let state = game_get_state_core(&gss).unwrap();
        let solve = state.solve.expect("solve should be Some");
        assert_eq!(solve.iteration, 50);
        assert_eq!(solve.max_iterations, 200);
        assert!((solve.exploitability - 5.0).abs() < 0.01);
        assert!(!solve.is_complete);
        assert_eq!(solve.solver_name, "range");
    }

    #[test]
    fn get_state_core_returns_complete_after_solve() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);

        // Simulate completed solve
        gss.solve_state.solving.store(false, std::sync::atomic::Ordering::Relaxed);
        gss.solve_state.iteration.store(200, std::sync::atomic::Ordering::Relaxed);
        gss.solve_state.max_iterations.store(200, std::sync::atomic::Ordering::Relaxed);
        gss.solve_state.exploitability_bits.store(1.5f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
        *gss.solve_state.solve_start.write() = Some(std::time::Instant::now());

        let state = game_get_state_core(&gss).unwrap();
        let solve = state.solve.expect("solve should be Some after completion");
        assert!(solve.is_complete);
        assert_eq!(solve.iteration, 200);
    }

    #[test]
    fn get_state_core_overrides_matrix_with_solve_snapshot() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);

        // Simulate solve with matrix snapshot
        gss.solve_state.solving.store(true, std::sync::atomic::Ordering::Relaxed);
        gss.solve_state.iteration.store(10, std::sync::atomic::Ordering::Relaxed);
        gss.solve_state.max_iterations.store(100, std::sync::atomic::Ordering::Relaxed);
        *gss.solve_state.solve_start.write() = Some(std::time::Instant::now());

        // Create a dummy matrix snapshot
        let dummy_matrix = GameMatrix {
            cells: vec![vec![GameMatrixCell {
                hand: "TEST".to_string(),
                suited: false,
                pair: false,
                probabilities: vec![1.0],
                combo_count: 1,
                weight: 1.0,
                ev: None,
                combos: vec![],
            }]],
            actions: vec![GameAction { id: "0".to_string(), label: "Check".to_string(), action_type: "check".to_string() }],
        };
        *gss.solve_state.matrix_snapshot.write() = Some(dummy_matrix);

        let state = game_get_state_core(&gss).unwrap();
        let matrix = state.matrix.expect("matrix should be overridden by solve snapshot");
        assert_eq!(matrix.cells[0][0].hand, "TEST");
    }

    #[test]
    fn get_state_core_no_solve_data_returns_none() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);

        // No solve has been run - iteration is 0, not solving
        let state = game_get_state_core(&gss).unwrap();
        assert!(state.solve.is_none());
    }

    // -------------------------------------------------------------------
    // game_solve_core tests
    // -------------------------------------------------------------------

    #[test]
    fn game_solve_core_rejects_no_session() {
        let gss = GameSessionState::default();
        let result = game_solve_core(&gss, None, None, None, None, None, None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No game session"));
    }

    #[test]
    fn game_solve_core_rejects_double_solve() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);
        gss.solve_state.solving.store(true, std::sync::atomic::Ordering::Relaxed);

        let result = game_solve_core(&gss, None, None, None, None, None, None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already in progress"));
    }

    // -------------------------------------------------------------------
    // game_cancel_solve tests
    // -------------------------------------------------------------------

    #[test]
    fn cancel_solve_sets_cancel_flag() {
        let gss = GameSessionState::default();
        assert!(!gss.solve_state.cancel.load(std::sync::atomic::Ordering::Relaxed));
        game_cancel_solve_core(&gss).unwrap();
        assert!(gss.solve_state.cancel.load(std::sync::atomic::Ordering::Relaxed));
    }

    // -------------------------------------------------------------------
    // parse_solve_board tests
    // -------------------------------------------------------------------

    #[test]
    fn parse_solve_board_flop() {
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()];
        let (flop, turn, river, state) = parse_solve_board(&board).unwrap();
        assert_eq!(state, range_solver::BoardState::Flop);
        assert_eq!(turn, range_solver::card::NOT_DEALT);
        assert_eq!(river, range_solver::card::NOT_DEALT);
        // Flop should be 3 valid card IDs
        assert!(flop.iter().all(|&c| c < 52));
    }

    #[test]
    fn parse_solve_board_turn() {
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string(), "Js".to_string()];
        let (_flop, turn, river, state) = parse_solve_board(&board).unwrap();
        assert_eq!(state, range_solver::BoardState::Turn);
        assert!(turn < 52);
        assert_eq!(river, range_solver::card::NOT_DEALT);
    }

    #[test]
    fn parse_solve_board_river() {
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string(), "Js".to_string(), "Ts".to_string()];
        let (_flop, _turn, river, state) = parse_solve_board(&board).unwrap();
        assert_eq!(state, range_solver::BoardState::River);
        assert!(river < 52);
    }

    #[test]
    fn parse_solve_board_invalid_length() {
        let board = vec!["Ah".to_string(), "Kd".to_string()];
        let result = parse_solve_board(&board);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("3-5 cards"));
    }

    // -------------------------------------------------------------------
    // bet size formatting tests
    // -------------------------------------------------------------------

    #[test]
    fn format_bet_sizes_single_depth() {
        let sizes = vec![vec![0.33, 0.67, 1.0]];
        let (bet_str, raise_str) = format_bet_sizes_for_solve(&sizes);
        assert!(bet_str.contains("33%"));
        assert!(bet_str.contains("67%"));
        assert!(bet_str.contains("100%"));
        // Should include allin
        assert!(bet_str.contains('a'));
        // raise_str defaults to bet_str when only one depth
        assert!(raise_str.contains('a'));
    }

    #[test]
    fn format_bet_sizes_two_depths() {
        let sizes = vec![vec![0.33, 1.0], vec![2.5, 3.0]];
        let (bet_str, raise_str) = format_bet_sizes_for_solve(&sizes);
        assert!(bet_str.contains("33%"));
        assert!(bet_str.contains("100%"));
        assert!(raise_str.contains("250%"));
        assert!(raise_str.contains("300%"));
    }

    #[test]
    fn format_bet_sizes_empty() {
        let sizes: Vec<Vec<f64>> = vec![];
        let (bet_str, raise_str) = format_bet_sizes_for_solve(&sizes);
        // Should have allin at minimum
        assert!(bet_str.contains('a'));
        assert!(raise_str.contains('a'));
    }

    // -------------------------------------------------------------------
    // solve state reset on game_new tests
    // -------------------------------------------------------------------

    #[test]
    fn game_new_resets_solve_state() {
        let gss = GameSessionState::default();
        // Simulate prior solve
        gss.solve_state.iteration.store(100, std::sync::atomic::Ordering::Relaxed);
        gss.solve_state.max_iterations.store(200, std::sync::atomic::Ordering::Relaxed);
        gss.solve_state.solving.store(false, std::sync::atomic::Ordering::Relaxed);

        // game_new_core needs ExplorationState and PostflopState, but
        // we can test the reset by calling reset_solve_state directly
        gss.solve_state.reset();

        assert_eq!(gss.solve_state.iteration.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(gss.solve_state.max_iterations.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert!(gss.solve_state.matrix_snapshot.read().is_none());
    }

    // -------------------------------------------------------------------
    // build_solve_matrix tests (basic structure)
    // -------------------------------------------------------------------

    #[test]
    fn build_solve_matrix_from_postflop_game() {
        // Build a tiny PostFlopGame and verify matrix extraction
        use range_solver::{PostFlopGame, ActionTree, CardConfig, TreeConfig, BoardState};
        use range_solver::card::{flop_from_str, NOT_DEALT};
        use range_solver::range::Range;
        use range_solver::bet_size::BetSizeOptions;

        let oop_range: Range = "AA".parse().unwrap();
        let ip_range: Range = "KK".parse().unwrap();
        let flop = flop_from_str("AhKdQc").unwrap();

        let sizes = BetSizeOptions::try_from(("50%,a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: 20,
            effective_stack: 90,
            rake_rate: 0.0,
            rake_cap: 0.0,
            flop_bet_sizes: [sizes.clone(), sizes.clone()],
            turn_bet_sizes: [sizes.clone(), sizes.clone()],
            river_bet_sizes: [sizes.clone(), sizes.clone()],
            turn_donk_sizes: None,
            river_donk_sizes: None,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.1,
            depth_limit: Some(0),
        };
        let action_tree = ActionTree::new(tree_config).unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop,
            turn: NOT_DEALT,
            river: NOT_DEALT,
        };
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);

        let matrix = build_solve_matrix(&mut game, None);
        // Should be a 13x13 grid
        assert_eq!(matrix.cells.len(), 13);
        assert_eq!(matrix.cells[0].len(), 13);
        // Should have actions
        assert!(!matrix.actions.is_empty());
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
