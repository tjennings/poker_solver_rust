//! Unified game session: tracks state from preflop through river.
//!
//! A `GameSession` owns the V2 game tree, blueprint strategy, and range weights.
//! Six Tauri commands expose it: `game_new`, `game_get_state`, `game_play_action`,
//! `game_deal_card`, `game_back`, `game_solve`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use serde::Serialize;

use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
use poker_solver_core::blueprint_v2::game_tree::{
    GameNode as V2GameNode, GameTree as V2GameTree, TreeAction,
};
use poker_solver_core::blueprint_v2::{LeafEvaluator, Street};

use range_solver::card::{card_to_string, NOT_DEALT};
use range_solver::{PostFlopGame, solve_step, finalize, compute_exploitability};
use range_solver::interface::Game;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Per-street boundary configuration
// ---------------------------------------------------------------------------

/// How to evaluate boundaries at a given street transition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum StreetBoundaryMode {
    Exact,
    Cfvnet { model_path: String },
}

impl Default for StreetBoundaryMode {
    fn default() -> Self {
        StreetBoundaryMode::Exact
    }
}

/// Per-street boundary evaluator configuration.
///
/// The solver walks streets in order from the session's root street. The first
/// non-Exact street becomes the cut point: the tree is built with a depth limit
/// that stops at that street transition, and the named cfvnet model evaluates
/// the boundary.
///
/// All-Exact (the default) means a full exact solve with no boundaries.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StreetBoundaryConfig {
    pub flop: StreetBoundaryMode,
    pub turn: StreetBoundaryMode,
    pub river: StreetBoundaryMode,
}

/// Result of resolving a `StreetBoundaryConfig` against a root street.
///
/// `None` means all-exact (no cut). `Some((depth, model_path))` means cut
/// after `depth` street transitions using the given ONNX model.
pub fn resolve_street_boundary(
    config: &StreetBoundaryConfig,
    root_street: Street,
) -> Option<(u8, String)> {
    // Streets to walk from root, in order.
    let streets: &[(Street, &StreetBoundaryMode)] = match root_street {
        Street::Flop => &[
            (Street::Flop, &config.flop),
            (Street::Turn, &config.turn),
            (Street::River, &config.river),
        ],
        Street::Turn => &[
            (Street::Turn, &config.turn),
            (Street::River, &config.river),
        ],
        Street::River => &[
            (Street::River, &config.river),
        ],
        Street::Preflop => return None, // preflop solve not supported
    };

    // Street mode `X = Cfvnet` means "cut at the card deal BEFORE street X".
    // Near tree = streets[0..index], so depth = index - 1.
    // If the root street itself is Cfvnet, no cut possible — fall through.
    for (i, (_street, mode)) in streets.iter().enumerate() {
        if let StreetBoundaryMode::Cfvnet { model_path } = mode {
            if i == 0 { continue; }
            return Some(((i - 1) as u8, model_path.clone()));
        }
    }
    None // all exact
}

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bucket: Option<u16>,     // strategy bucket ID (postflop only)
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

/// Cached strategy data for a single node in a solved subgame tree.
#[derive(Debug, Clone)]
pub struct CachedSolveNode {
    pub matrix: GameMatrix,
    pub actions: Vec<GameAction>,
    pub position: String,
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
    pub cancel: Arc<AtomicBool>,
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
    /// Cached matrices for every node in the solved subgame tree.
    /// Key: action path from solve root (e.g., `[0, 1]` = first action then second).
    pub solve_cache: RwLock<HashMap<Vec<usize>, CachedSolveNode>>,
    /// Current position within the solved tree (action path from solve root).
    pub solve_path: RwLock<Vec<usize>>,
}

impl Default for SolveState {
    fn default() -> Self {
        Self {
            solving: AtomicBool::new(false),
            cancel: Arc::new(AtomicBool::new(false)),
            iteration: AtomicU32::new(0),
            max_iterations: AtomicU32::new(0),
            exploitability_bits: AtomicU32::new(0),
            solve_start: RwLock::new(None),
            matrix_snapshot: RwLock::new(None),
            solve_actions: RwLock::new(vec![]),
            solve_position: RwLock::new(String::new()),
            solve_cache: RwLock::new(HashMap::new()),
            solve_path: RwLock::new(vec![]),
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
        *self.solve_cache.write() = HashMap::new();
        *self.solve_path.write() = vec![];
    }
}

// ---------------------------------------------------------------------------
// Shared state for Tauri commands
// ---------------------------------------------------------------------------

/// Shared session state, accessible by Tauri commands.
pub struct GameSessionState {
    pub session: RwLock<Option<GameSession>>,
    pub subgame_solve: Arc<SolveState>,
    pub exact_solve: Arc<SolveState>,
}

impl GameSessionState {
    /// Return the `SolveState` for the given mode string.
    /// `"exact"` -> `exact_solve`, anything else (including `None`, `"hybrid"`, `"subgame"`) -> `subgame_solve`.
    pub fn solve_for(&self, mode: &Option<String>) -> &Arc<SolveState> {
        match mode.as_deref() {
            Some("exact") => &self.exact_solve,
            _ => &self.subgame_solve,
        }
    }
}

impl Default for GameSessionState {
    fn default() -> Self {
        Self {
            session: RwLock::new(None),
            subgame_solve: Arc::new(SolveState::default()),
            exact_solve: Arc::new(SolveState::default()),
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
    /// Create a session from raw blueprint components.
    ///
    /// This constructor is used by the CLI (`inspect-spot`) and any other
    /// context that loads blueprint data independently of the Tauri explorer.
    pub fn new(
        config: BlueprintV2Config,
        strategy: BlueprintV2Strategy,
        tree: V2GameTree,
        decision_map: Vec<u32>,
        hand_evs: Option<Vec<[[f64; 169]; 2]>>,
    ) -> Self {
        let root = tree.root;
        GameSession {
            tree: Box::new(tree),
            strategy: Box::new(strategy),
            decision_map,
            config: Box::new(config),
            cbv_context: None,
            hand_evs,
            node_idx: root,
            board: vec![],
            action_history: vec![],
            weights: [vec![1.0f32; 1326], vec![1.0f32; 1326]],
        }
    }

    /// Set the CbvContext for postflop bucket lookups.
    pub fn set_cbv_context(&mut self, ctx: Arc<CbvContext>) {
        self.cbv_context = Some(ctx);
    }

    /// Current abstract tree node index.
    pub fn node_idx(&self) -> u32 {
        self.node_idx
    }

    /// Current OOP and IP reaching weights (BB=0, SB=1).
    pub fn weights(&self) -> (&[f32], &[f32]) {
        (&self.weights[0], &self.weights[1])
    }

    /// Create a session from already-loaded exploration state.
    pub fn from_exploration_state(
        exploration: &crate::exploration::ExplorationState,
        cbv_context: Option<Arc<CbvContext>>,
    ) -> Result<Self, String> {
        let data = exploration.extract_blueprint_v2_data()?;
        let mut session = Self::new(
            *data.config,
            *data.strategy,
            *data.tree,
            data.decision_map,
            data.hand_evs,
        );
        session.cbv_context = cbv_context;
        Ok(session)
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
                    pot: *pot as i32,
                    stacks: [
                        stacks[bb_idx] as i32,
                        stacks[sb_idx] as i32,
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
        // [matrix] log intentionally silenced — fired on every strategy-matrix
        // request (many per iter in the UI). Re-enable by uncommenting if needed.
        // eprintln!("[matrix] street={street:?}, board_len={}, cbv_context={}, board_cards={}",
        //     self.board.len(), self.cbv_context.is_some(), board_cards.as_ref().map_or(0, |b| b.len()));

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

                let (probs, bucket_id) = if street == Street::Preflop {
                    // Preflop: all combos of a canonical hand share the same strategy.
                    // Return empty — the cell's aggregated probs are sufficient.
                    (vec![], None)
                } else if let (Some(ctx), Some(board)) = (&self.cbv_context, board_cards) {
                    // Postflop: per-combo bucket lookup.
                    let board_slice = board_for_street_slice(board, street);
                    let rs_c1 = crate::exploration::range_solver_to_rs_card(c1);
                    let rs_c2 = crate::exploration::range_solver_to_rs_card(c2);
                    let bucket = ctx.all_buckets.get_bucket(street, [rs_c1, rs_c2], board_slice);
                    let strategy_probs = ctx.strategy.get_action_probs(decision_idx, bucket);
                    let probs = (0..num_actions)
                        .map(|i| strategy_probs.get(i).copied().unwrap_or(0.0))
                        .collect();
                    (probs, Some(bucket))
                } else {
                    (vec![0.0; num_actions], None)
                };

                Some(ComboDetail {
                    cards: format!("{s1}{s2}"),
                    probabilities: probs,
                    weight: w,
                    bucket: bucket_id,
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

    /// Compute pot at the current node (in chips).
    #[allow(clippy::cast_possible_truncation)]
    fn compute_pot(&self) -> i32 {
        pot_at_v2_node(&self.tree, self.node_idx) as i32
    }

    /// Compute remaining stacks [BB, SB] (in chips).
    #[allow(clippy::cast_possible_truncation)]
    fn compute_stacks(&self) -> [i32; 2] {
        let pot = pot_at_v2_node(&self.tree, self.node_idx);
        let stack_depth = self.config.game.stack_depth;
        let each_invested = pot / 2.0;
        let remaining = (stack_depth - each_invested) as i32;
        [remaining, remaining]
    }

    /// Encode the current game state as a human-readable spot string.
    ///
    /// Format: `sb:2bb,bb:call|AhKdQc|bb:check,sb:4bb`
    /// - Actions are `position:label` (lowercased), comma-separated
    /// - `|` separates street transitions (board card deals)
    /// - Board segments are card strings concatenated (e.g. "AhKdQc")
    pub fn encode_spot(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        let mut current_actions: Vec<String> = Vec::new();
        let mut prev_street = String::new();
        let mut board_idx = 0;

        for rec in &self.action_history {
            if rec.street != prev_street && !prev_street.is_empty() {
                // Flush current actions
                if !current_actions.is_empty() {
                    parts.push(current_actions.join(","));
                    current_actions.clear();
                }
                // Emit board cards for the street transition
                let new_cards = match prev_street.as_str() {
                    "Preflop" => 3,
                    _ => 1,
                };
                let end = (board_idx + new_cards).min(self.board.len());
                let board_str: String = self.board[board_idx..end].join("");
                board_idx = end;
                parts.push(board_str);
            }
            prev_street = rec.street.clone();
            current_actions.push(format!(
                "{}:{}",
                rec.position.to_lowercase(),
                rec.label.to_lowercase()
            ));
        }

        // Flush remaining actions
        if !current_actions.is_empty() {
            parts.push(current_actions.join(","));
        }

        // Emit any remaining board cards (e.g. board dealt but no actions on new street)
        if board_idx < self.board.len() {
            let remaining: String = self.board[board_idx..].join("");
            parts.push(remaining);
        }

        parts.join("|")
    }

    /// Parse a spot encoding and replay to that state.
    ///
    /// Resets to preflop root (including weights, board, action history),
    /// then replays each action and board card deal from the encoding.
    pub fn load_spot(&mut self, spot: &str) -> Result<(), String> {
        let spot = spot.trim();
        if spot.is_empty() {
            return Ok(());
        }

        // Reset to root
        self.node_idx = self.tree.root;
        self.board.clear();
        self.action_history.clear();
        self.weights = [vec![1.0f32; 1326], vec![1.0f32; 1326]];

        let segments: Vec<&str> = spot.split('|').collect();

        for segment in segments {
            let segment = segment.trim();
            if segment.is_empty() {
                continue;
            }

            if segment.contains(':') {
                // Action segment: "sb:2bb,bb:call"
                let actions: Vec<&str> = segment.split(',').collect();
                for action_str in actions {
                    let action_str = action_str.trim();
                    let (pos, label) = action_str.split_once(':').ok_or_else(|| {
                        format!(
                            "Invalid action format: '{action_str}'. Expected 'position:label'"
                        )
                    })?;

                    // Get current state to find matching action
                    let state = self.get_state();
                    let position = state.position.to_lowercase();
                    if pos.to_lowercase() != position {
                        return Err(format!(
                            "Position mismatch: '{pos}' but current position is '{}'",
                            state.position
                        ));
                    }

                    // Find matching action by label (case-insensitive)
                    let matched = state
                        .actions
                        .iter()
                        .find(|a| a.label.to_lowercase() == label.to_lowercase());

                    match matched {
                        Some(action) => {
                            let id = action.id.clone();
                            self.play_action(&id)?;
                        }
                        None => {
                            let available: Vec<String> = state
                                .actions
                                .iter()
                                .map(|a| {
                                    format!("{}:{}", position, a.label.to_lowercase())
                                })
                                .collect();
                            return Err(format!(
                                "Action '{}:{}' not found. Available: {}",
                                pos,
                                label,
                                available.join(", ")
                            ));
                        }
                    }
                }
            } else {
                // Board segment: "AhKdQc" or "7s" or "2d"
                let chars: Vec<char> = segment.chars().collect();
                if chars.len() % 2 != 0 {
                    return Err(format!(
                        "Invalid board segment: '{segment}'. Must be pairs of rank+suit."
                    ));
                }
                for chunk in chars.chunks(2) {
                    let card: String = chunk.iter().collect();
                    self.deal_card(&card)?;
                }
            }
        }

        Ok(())
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
///
/// Amounts are in chips; display converts to BB (chips / 2).
fn format_tree_action(action: &TreeAction) -> String {
    match action {
        TreeAction::Fold => "Fold".to_string(),
        TreeAction::Check => "Check".to_string(),
        TreeAction::Call => "Call".to_string(),
        TreeAction::Bet(amount) => {
            let bb = (amount / 2.0).round();
            format!("{bb:.0}bb")
        }
        TreeAction::Raise(amount) => {
            let bb = (amount / 2.0).round();
            format!("{bb:.0}bb")
        }
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
///
/// When `exact` is true, `depth_limit` is set to `None` so the game tree
/// extends through all remaining streets to showdown (no boundary nodes).
#[allow(clippy::too_many_arguments)]
pub fn build_solve_game(
    board: &[String],
    oop_weights: &[f32],
    ip_weights: &[f32],
    pot: i32,
    effective_stack: i32,
    bet_sizes: &[Vec<f64>],
    exact: bool,
    depth_limit_override: Option<u8>,
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
        // Exact mode: solve through all streets to showdown (no boundaries).
        // Subgame mode: boundaries placed after depth_limit street transitions.
        //   depth_limit=0: current street only (boundaries at next transition)
        //   depth_limit=1: current + next street (e.g., flop+turn, boundaries at river)
        //   depth_limit=2: from flop = full solve (flop+turn+river)
        // River: no boundaries needed in either mode (already at showdown).
        depth_limit: if exact || initial_state == range_solver::BoardState::River {
            None
        } else {
            Some(depth_limit_override.unwrap_or(0))
        },
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
/// Navigates to root first, then delegates to `build_solve_matrix_at_current`.
fn build_solve_matrix(game: &mut PostFlopGame, hand_evs: Option<&[f32]>) -> GameMatrix {
    game.back_to_root();
    build_solve_matrix_at_current(game, hand_evs)
}

/// Build a `GameMatrix` from the current `PostFlopGame` position (without navigating to root).
///
/// Same logic as `build_solve_matrix` but does NOT call `game.back_to_root()`.
#[allow(clippy::cast_possible_truncation)]
fn build_solve_matrix_at_current(game: &mut PostFlopGame, hand_evs: Option<&[f32]>) -> GameMatrix {
    use crate::postflop::{card_pair_to_matrix, matrix_cell_label};

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

    let mut prob_sums = vec![vec![vec![0.0f64; num_actions]; 13]; 13];
    let mut combo_counts = vec![vec![0usize; 13]; 13];
    let mut weight_sums = vec![vec![0.0f64; 13]; 13];
    let mut ev_sums = vec![vec![0.0f64; 13]; 13];
    let mut combo_details: Vec<Vec<Vec<ComboDetail>>> = vec![vec![Vec::new(); 13]; 13];

    for (hand_idx, &(c1_raw, c2_raw)) in private_cards.iter().enumerate() {
        let (row, col, _) = card_pair_to_matrix(c1_raw, c2_raw);
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

        let (c1, c2) = if c1_raw / 4 >= c2_raw / 4 { (c1_raw, c2_raw) } else { (c2_raw, c1_raw) };
        let s1 = card_to_string(c1).unwrap_or_default();
        let s2 = card_to_string(c2).unwrap_or_default();
        combo_details[row][col].push(ComboDetail {
            cards: format!("{s1}{s2}"),
            probabilities: probs,
            weight: initial_weights[hand_idx],
            bucket: None,
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

/// Walk the solved game tree and cache a `CachedSolveNode` at every decision node.
///
/// Returns a map from action path (e.g., `[0, 1]`) to cached node data.
fn build_solve_cache(game: &mut PostFlopGame) -> HashMap<Vec<usize>, CachedSolveNode> {
    let mut cache = HashMap::new();
    build_solve_cache_recursive(game, &mut vec![], &mut cache);
    cache
}

fn build_solve_cache_recursive(
    game: &mut PostFlopGame,
    path: &mut Vec<usize>,
    cache: &mut HashMap<Vec<usize>, CachedSolveNode>,
) {
    if game.is_terminal_node() || game.is_chance_node() {
        return;
    }

    let matrix = build_solve_matrix_at_current(game, None);
    let actions: Vec<GameAction> = game
        .available_actions()
        .iter()
        .enumerate()
        .map(|(i, a)| range_solver_action_to_game_action(a, i))
        .collect();
    let player = game.current_player();
    let position = if player == 0 { "OOP" } else { "IP" };

    let num_actions = actions.len();
    cache.insert(
        path.clone(),
        CachedSolveNode {
            matrix,
            actions,
            position: position.to_string(),
        },
    );

    for i in 0..num_actions {
        game.play(i);
        path.push(i);
        build_solve_cache_recursive(game, path, cache);
        path.pop();
        // Navigate back: PostFlopGame has no undo, so replay from root
        game.apply_history(path);
    }
}

/// Adapter implementing `BoundaryEvaluator` for the range-solver.
/// SPR=0 boundaries use exact matchup equity; SPR>0 uses RolloutLeafEvaluator.
pub struct SolveBoundaryEvaluator {
    /// Private cards per player, in range-solver ordering (card ID pairs).
    pub private_cards: [Vec<(u8, u8)>; 2],
    /// Board cards as rs_poker Cards (for equity computation).
    pub board_cards: Vec<rs_poker::core::Card>,
    /// Effective stack at game start.
    #[allow(dead_code)]
    pub eff_stack: f64,
    /// Rollout evaluator for SPR>0 boundaries (None if CbvContext unavailable).
    pub rollout: Option<RolloutLeafEvaluator>,
    /// Combos in rollout ordering + card mappings per player.
    pub combos: Vec<[rs_poker::core::Card; 2]>,
    /// Maps game private_cards index → combo index, per player.
    pub game_to_combo: [Vec<usize>; 2],
}

impl range_solver::game::BoundaryEvaluator for SolveBoundaryEvaluator {
    fn num_continuations(&self) -> usize { 4 }

    fn compute_cfvs(
        &self,
        player: usize,
        pot: i32,
        remaining_stack: f64,
        opponent_reach: &[f32],
        num_hands: usize,
        _continuation_index: usize,
    ) -> Vec<f32> {
        let opp = player ^ 1;

        if remaining_stack <= 0.0 {
            // SPR=0: exact equity against opponent's filtered range.
            use rayon::prelude::*;
            let hero_cards = &self.private_cards[player];
            let opp_cards = &self.private_cards[opp];

            hero_cards
                .par_iter()
                .enumerate()
                .map(|(_i, &(h1, h2))| {
                    let rs_h1 = crate::exploration::range_solver_to_rs_card(h1);
                    let rs_h2 = crate::exploration::range_solver_to_rs_card(h2);
                    let mut ev_sum = 0.0f64;
                    let mut weight_sum = 0.0f64;
                    for (j, &(o1, o2)) in opp_cards.iter().enumerate() {
                        let w = if j < opponent_reach.len() { opponent_reach[j] as f64 } else { 0.0 };
                        if w <= 0.0 { continue; }
                        let rs_o1 = crate::exploration::range_solver_to_rs_card(o1);
                        let rs_o2 = crate::exploration::range_solver_to_rs_card(o2);
                        // Skip card overlaps
                        if rs_h1 == rs_o1 || rs_h1 == rs_o2 || rs_h2 == rs_o1 || rs_h2 == rs_o2 {
                            continue;
                        }
                        if self.board_cards.iter().any(|b| *b == rs_o1 || *b == rs_o2) {
                            continue;
                        }
                        let eq = poker_solver_core::showdown_equity::compute_matchup_equity(
                            [rs_h1, rs_h2], [rs_o1, rs_o2], &self.board_cards,
                        );
                        ev_sum += eq * w;
                        weight_sum += w;
                    }
                    if weight_sum > 0.0 {
                        ((ev_sum / weight_sum) - 0.5) as f32 * 2.0
                    } else {
                        0.0 // No opponent reach — boundary unreachable, value irrelevant
                    }
                })
                .collect()
        } else if let Some(ref rollout) = self.rollout {
            // SPR > 0: rollout with boundary stack/pot.
            // Convert opponent_reach from game ordering to combo ordering.
            let opp_map = &self.game_to_combo[opp];
            let mut opp_combo_reach = vec![0.0f64; self.combos.len()];
            for (game_idx, &combo_idx) in opp_map.iter().enumerate() {
                if combo_idx < opp_combo_reach.len() && game_idx < opponent_reach.len() {
                    opp_combo_reach[combo_idx] = opponent_reach[game_idx] as f64;
                }
            }
            // If no opponent combos have reach, boundary is unreachable — value irrelevant.
            let opp_total: f64 = opp_combo_reach.iter().sum();
            if opp_total <= 0.0 {
                return vec![0.0f32; num_hands];
            }

            // Hero reach: use 1.0 for all (the solver weights externally).
            let hero_combo_reach = vec![1.0f64; self.combos.len()];

            let boundary_starting_stack = remaining_stack + pot as f64 / 2.0;
            let mut eval = RolloutLeafEvaluator::new(
                rollout.strategy.clone(),
                rollout.abstract_tree.clone(),
                rollout.all_buckets.clone(),
                rollout.abstract_start_node,
                rollout.bias,
                rollout.bias_factor,
                rollout.num_rollouts,
                rollout.num_opponent_samples,
                boundary_starting_stack,
                pot as f64,
            );
            if let Some(ref counter) = rollout.hand_counter {
                eval.hand_counter = Some(Arc::clone(counter));
            }
            eval.enumerate_decision_depth = rollout.enumerate_decision_depth;
            eval.call_counter = Arc::clone(&rollout.call_counter);
            let requests = vec![(pot as f64, 0.0, player as u8)];
            let results = eval.evaluate_boundaries(
                &self.combos, &self.board_cards, &hero_combo_reach, &opp_combo_reach, &requests,
            );
            let combo_cfvs = results.into_iter().next().unwrap_or_default();

            // Map combo ordering → game ordering
            let hero_map = &self.game_to_combo[player];
            let mut cfvs = vec![0.0f32; num_hands];
            for (game_idx, &combo_idx) in hero_map.iter().enumerate() {
                if combo_idx < combo_cfvs.len() && game_idx < cfvs.len() {
                    cfvs[game_idx] = combo_cfvs[combo_idx] as f32;
                }
            }
            cfvs
        } else {
            // No rollout available, return zero
            vec![0.0; num_hands]
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
    session_state.subgame_solve.reset();
    session_state.exact_solve.reset();
    Ok(())
}

/// Get the current game state, including solve progress if active.
///
/// `source` controls which strategy data is returned:
/// - `None` or `"blueprint"`: return blueprint data only, skip solve overlay.
/// - `"subgame"`: overlay from `subgame_solve`.
/// - `"exact"`: overlay from `exact_solve`.
pub fn game_get_state_core(session_state: &GameSessionState, source: Option<String>) -> Result<GameState, String> {
    let guard = session_state.session.read();
    let session = guard.as_ref().ok_or("No game session active")?;
    let mut state = session.get_state();

    // Blueprint source: return raw blueprint data, no solve overlay
    let ss = match source.as_deref() {
        Some("subgame") => &session_state.subgame_solve,
        Some("exact") => &session_state.exact_solve,
        _ => return Ok(state),
    };
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

        // Prefer solve cache (navigated position) over root snapshot
        let cache = ss.solve_cache.read();
        let path = ss.solve_path.read();
        if let Some(node) = cache.get(&*path) {
            state.matrix = Some(node.matrix.clone());
            state.actions = node.actions.clone();
            state.position = node.position.clone();
        } else {
            // Fall back to root snapshot (during solve or before cache is built)
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

    }

    Ok(state)
}

/// Play an action and return the new game state.
///
/// If navigating within a solved subgame tree, serves the cached matrix
/// instead of resetting the solve state.
///
/// `source` selects which solve cache to navigate within.
pub fn game_play_action_core(
    session_state: &GameSessionState,
    action_id: &str,
    source: Option<String>,
) -> Result<GameState, String> {
    let ss = match source.as_deref() {
        Some("exact") => &session_state.exact_solve,
        _ => &session_state.subgame_solve,
    };
    let cache = ss.solve_cache.read();

    if !cache.is_empty() {
        let current_path = ss.solve_path.read().clone();
        if let Some(current_node) = cache.get(&current_path) {
            if let Some(action_idx) = current_node.actions.iter().position(|a| a.id == action_id) {
                let mut new_path = current_path.clone();
                new_path.push(action_idx);

                if let Some(child_node) = cache.get(&new_path) {
                    let child_matrix = child_node.matrix.clone();
                    let child_actions = child_node.actions.clone();
                    let child_position = child_node.position.clone();
                    drop(cache);

                    // Play the action on the session for board/range tracking
                    let mut guard = session_state.session.write();
                    let session = guard.as_mut().ok_or("No game session active")?;
                    session.play_action(action_id)?;
                    let mut state = session.get_state();
                    drop(guard);

                    // Override with cached data
                    state.matrix = Some(child_matrix);
                    state.actions = child_actions;
                    state.position = child_position;
                    *ss.solve_path.write() = new_path;

                    return Ok(state);
                }
            }
        }
    }
    drop(cache);

    // Not in solved tree -- normal navigation, reset solve state
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.play_action(action_id)?;
    let state = session.get_state();
    drop(guard);
    session_state.subgame_solve.reset();
    session_state.exact_solve.reset();
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
///
/// If within a solved subgame tree, pops the last action from the solve path
/// and serves the parent's cached matrix. If at the solve root, resets the
/// solve state entirely.
///
/// `source` selects which solve cache to navigate within.
pub fn game_back_core(session_state: &GameSessionState, source: Option<String>) -> Result<GameState, String> {
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.back()?;
    let mut state = session.get_state();
    drop(guard);

    let ss = match source.as_deref() {
        Some("exact") => &session_state.exact_solve,
        _ => &session_state.subgame_solve,
    };
    let cache = ss.solve_cache.read();

    if !cache.is_empty() {
        let mut path = ss.solve_path.write();
        if !path.is_empty() {
            // Pop last action to get parent path
            path.pop();
            if let Some(node) = cache.get(&*path) {
                state.matrix = Some(node.matrix.clone());
                state.actions = node.actions.clone();
                state.position = node.position.clone();
            }
            return Ok(state);
        }
        // At solve root and going back -- navigating before solve root, reset
        drop(path);
        drop(cache);
        session_state.subgame_solve.reset();
        session_state.exact_solve.reset();
    }

    Ok(state)
}

/// Start a subgame solve using range-solver `PostFlopGame`.
///
/// Spawns a background thread that builds a `PostFlopGame`, optionally
/// configures ONNX cfvnet boundary evaluators per `StreetBoundaryConfig`,
/// runs a DCFR solve loop, and stores matrix snapshots in `SolveState`
/// for the UI to read.
#[allow(clippy::too_many_arguments)]
pub fn game_solve_core(
    session_state: &GameSessionState,
    mode: Option<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
    matrix_snapshot_interval: Option<u32>,
    range_clamp_threshold: Option<f64>,
    street_boundary_config: Option<StreetBoundaryConfig>,
    trace_boundaries: Option<String>,
    trace_iters: Option<String>,
    trace_dir: Option<String>,
) -> Result<(), String> {
    let is_exact = mode.as_deref() == Some("exact");
    let ss_ref = session_state.solve_for(&mode);

    // Guard: reject if this mode is already solving
    if ss_ref.solving.load(Ordering::Relaxed) {
        return Err("A solve is already in progress".to_string());
    }

    // Read session state under lock, clone what the thread needs
    let (board, oop_w, ip_w, pot, eff_stack, bet_sizes, cbv_ctx, current_node_idx, position_label, root_street) = {
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

        let street = session.current_street();
        let sizes = match street {
            Street::Flop => &session.config.action_abstraction.flop,
            Street::Turn => &session.config.action_abstraction.turn,
            Street::River => &session.config.action_abstraction.river,
            Street::Preflop => return Err("Cannot solve preflop".to_string()),
        };

        let cbv_ctx = session.cbv_context.clone();
        let position = session.position_label(player).to_string();
        let current_node = session.node_idx;

        (board, oop_w, ip_w, pot, eff_stack, sizes.clone(), cbv_ctx, current_node, position, street)
    };

    // Resolve StreetBoundaryConfig to (depth_limit, model_path)
    let sbc = street_boundary_config.unwrap_or_default();
    let boundary_cut = if is_exact {
        None
    } else {
        resolve_street_boundary(&sbc, root_street)
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
    let snapshot_interval = matrix_snapshot_interval.unwrap_or(10);
    let target_exp = target_exploitability.unwrap_or(3.0);

    // Build trace config (empty trace_boundaries = no tracing = zero cost)
    let trace_config = {
        let boundaries = trace_boundaries.filter(|s| !s.trim().is_empty());
        crate::boundary_trace::TraceConfig {
            boundaries,
            iters_str: trace_iters.unwrap_or_else(|| "last".to_string()),
            dir: std::path::PathBuf::from(
                trace_dir.unwrap_or_else(|| "./local_data/logs".to_string()),
            ),
        }
    };

    // Reset solve state atomics
    let ss = ss_ref;
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
    let ss_clone = Arc::clone(ss_ref);
    let board_clone = board.clone();
    std::thread::spawn(move || {
        // Build game with depth_limit from boundary config.
        // When no boundary cut is active (all-exact SBC or explicit exact mode),
        // build as exact (no boundaries).
        let depth_limit_override = boundary_cut.as_ref().map(|(depth, _)| *depth);
        let build_exact = is_exact || boundary_cut.is_none();
        let mut game = match build_solve_game(
            &board_clone, &oop_w, &ip_w, pot, eff_stack, &bet_sizes,
            build_exact, depth_limit_override,
        ) {
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

        // Set up neural boundary evaluators if a cut is active
        let n_boundaries = game.num_boundary_nodes();
        if let Some((_, ref model_path)) = boundary_cut {
            if n_boundaries > 0 {
                setup_neural_boundaries(&mut game, model_path);
            }
        }

        let (mem_est, _) = game.memory_usage();
        eprintln!(
            "[solve] depth_limit: {:?}, boundary nodes: {n_boundaries}, per_boundary: {}",
            depth_limit_override, game.per_boundary_evaluators.len(),
        );
        eprintln!("[solve] pot={pot}, eff_stack={eff_stack}, board={board_clone:?}");
        eprintln!(
            "[solve] OOP hands: {}, IP hands: {}",
            game.private_cards(0).len(), game.private_cards(1).len(),
        );
        eprintln!("[solve] memory: {:.1} MB", mem_est as f64 / 1_048_576.0);

        // Seed solver with blueprint strategy if available.
        if let Some(ref ctx) = cbv_ctx {
            let board_cards: Vec<rs_poker::core::Card> = board_clone
                .iter()
                .filter_map(|s| parse_rs_poker_card(s).ok())
                .collect();
            let seed_street = match board_cards.len() {
                3 => Street::Flop,
                4 => Street::Turn,
                _ => Street::River,
            };
            crate::postflop::seed_solver_with_blueprint(
                &game,
                &ctx.strategy,
                &ctx.all_buckets,
                &ctx.abstract_tree,
                &board_cards,
                seed_street,
                current_node_idx,
            );
        }

        // Set up boundary tracer (no-op when disabled)
        let tracer = trace_config.into_tracer(max_iters);
        let spot_paths: Option<Vec<String>> = tracer.as_ref().and_then(|_| {
            let n = game.num_boundary_nodes();
            if n > 0 {
                Some(crate::boundary_trace::build_boundary_spot_paths(&game))
            } else {
                None
            }
        });
        let preceding_map = tracer.as_ref().map(|_| {
            crate::boundary_trace::build_preceding_decision_map(&game)
        });

        // Initial matrix snapshot
        let matrix = build_solve_matrix(&mut game, None);
        *ss_clone.matrix_snapshot.write() = Some(matrix);

        // Solve loop
        let has_per_boundary = !game.per_boundary_evaluators.is_empty();
        let mut t = 0u32;
        while t < max_iters {
            if ss_clone.cancel.load(Ordering::Relaxed) {
                break;
            }

            // Neural cfvnet path: clear CFV cache every iteration so boundary
            // values are recomputed with updated opponent reaches.
            if has_per_boundary {
                game.clear_boundary_cfvs();
            }

            // Update DCFR discount params for boundary continuation regrets.
            {
                let nearest_pow4 = if t == 0 { 0 } else { 1u32 << ((t.leading_zeros() ^ 31) & !1) };
                let t_alpha = (t as i32 - 1).max(0) as f64;
                let t_gamma = (t - nearest_pow4) as f64;
                let pow_alpha = t_alpha * t_alpha.sqrt();
                let alpha = (pow_alpha / (pow_alpha + 1.0)) as f32;
                let beta = 0.5f32;
                let gamma = (t_gamma / (t_gamma + 1.0)).powi(3) as f32;
                game.set_boundary_discount(alpha, beta, gamma);
            }

            solve_step(&game, t);
            t += 1;
            ss_clone.iteration.store(t, Ordering::Relaxed);

            // Capture boundary traces after this iteration's CFVs are cached.
            if let Some(ref tr) = tracer {
                crate::boundary_trace::capture_boundary_traces(
                    &game,
                    tr,
                    spot_paths.as_deref(),
                    preceding_map.as_ref(),
                    t - 1,
                );
            }

            // Snapshot matrix and exploitability periodically
            if t.is_multiple_of(snapshot_interval) {
                let matrix = build_solve_matrix(&mut game, None);
                *ss_clone.matrix_snapshot.write() = Some(matrix);

                if is_exact {
                    let exp = compute_exploitability(&game);
                    ss_clone.exploitability_bits.store(exp.to_bits(), Ordering::Relaxed);
                    if exp.is_finite() && exp > 0.0 && exp <= target_exp {
                        eprintln!(
                            "[solve] exact converged: iter={t} exploitability={exp:.3} <= target={target_exp}"
                        );
                        break;
                    }
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

        // Compute exploitability using cached boundary CFVs
        let saved_evaluator = game.boundary_evaluator.take();
        let saved_per_boundary = std::mem::take(&mut game.per_boundary_evaluators);
        let final_exp = compute_exploitability(&game);
        game.boundary_evaluator = saved_evaluator;
        game.per_boundary_evaluators = saved_per_boundary;
        ss_clone
            .exploitability_bits
            .store(final_exp.to_bits(), Ordering::Relaxed);

        // Build solve cache for all decision nodes in the solved tree.
        let solve_cache = build_solve_cache(&mut game);
        eprintln!("[solve] cached {} decision nodes for subgame navigation", solve_cache.len());
        *ss_clone.solve_cache.write() = solve_cache;
        *ss_clone.solve_path.write() = vec![];

        ss_clone.solving.store(false, Ordering::Release);
        let reported_exp = f32::from_bits(ss_clone.exploitability_bits.load(Ordering::Relaxed));
        eprintln!(
            "[solve] complete: {} iterations, exploitability={:.4}",
            t, reported_exp
        );
    });

    Ok(())
}

/// Load an ONNX session and wire per-boundary `NeuralBoundaryEvaluator`s
/// into the game's `per_boundary_evaluators` vector.
fn setup_neural_boundaries(game: &mut PostFlopGame, model_path: &str) {
    let path = std::path::PathBuf::from(model_path);
    let boundary_boards = game.boundary_boards();
    let n_boundaries = game.num_boundary_nodes();

    if boundary_boards.is_empty() {
        eprintln!("[solve] no boundary boards found; skipping neural setup");
        return;
    }

    let session = match cfvnet::eval::boundary_evaluator::load_shared_onnx_session(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[solve] ONNX session load failed: {e}");
            return;
        }
    };

    let mut per_boundary: Vec<Arc<dyn range_solver::game::BoundaryEvaluator>> =
        Vec::with_capacity(n_boundaries);
    for board_4 in boundary_boards {
        let private_cards_pair = [
            game.private_cards(0).to_vec(),
            game.private_cards(1).to_vec(),
        ];
        let eval = cfvnet::eval::boundary_evaluator::neural_boundary_evaluator_from_shared(
            Arc::clone(&session),
            board_4,
            private_cards_pair,
        );
        per_boundary.push(Arc::new(eval));
    }
    game.per_boundary_evaluators = per_boundary;
    game.boundary_evaluator = None;

    eprintln!(
        "[solve] neural-cfvnet mode: {} boundaries (ONNX)",
        n_boundaries,
    );
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
    session_state.subgame_solve.reset();
    session_state.exact_solve.reset();
    Ok(())
}

#[tauri::command]
pub fn game_get_state(
    session_state: tauri::State<'_, GameSessionState>,
    source: Option<String>,
) -> Result<GameState, String> {
    game_get_state_core(&session_state, source)
}

#[tauri::command]
pub fn game_play_action(
    session_state: tauri::State<'_, GameSessionState>,
    action_id: String,
    source: Option<String>,
) -> Result<GameState, String> {
    game_play_action_core(&session_state, &action_id, source)
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
    source: Option<String>,
) -> Result<GameState, String> {
    game_back_core(&session_state, source)
}

#[tauri::command]
pub fn game_solve(
    session_state: tauri::State<'_, GameSessionState>,
    mode: Option<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
    matrix_snapshot_interval: Option<u32>,
    range_clamp_threshold: Option<f64>,
    street_boundary_config: Option<StreetBoundaryConfig>,
    trace_boundaries: Option<String>,
    trace_iters: Option<String>,
    trace_dir: Option<String>,
) -> Result<(), String> {
    game_solve_core(
        &session_state,
        mode,
        max_iterations,
        target_exploitability,
        matrix_snapshot_interval,
        range_clamp_threshold,
        street_boundary_config,
        trace_boundaries,
        trace_iters,
        trace_dir,
    )
}

pub fn game_cancel_solve_core(session_state: &GameSessionState, mode: Option<String>) -> Result<(), String> {
    session_state
        .solve_for(&mode)
        .cancel
        .store(true, Ordering::Relaxed);
    Ok(())
}

#[tauri::command]
pub fn game_cancel_solve(
    session_state: tauri::State<'_, GameSessionState>,
    mode: Option<String>,
) -> Result<(), String> {
    game_cancel_solve_core(&session_state, mode)
}

/// Encode the current game state as a human-readable spot string.
pub fn game_encode_spot_core(session_state: &GameSessionState) -> Result<String, String> {
    let guard = session_state.session.read();
    let session = guard.as_ref().ok_or("No game session active")?;
    Ok(session.encode_spot())
}

/// Parse a spot encoding and replay to that state, returning the new game state.
pub fn game_load_spot_core(
    session_state: &GameSessionState,
    spot: &str,
) -> Result<GameState, String> {
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.load_spot(spot)?;
    Ok(session.get_state())
}

#[tauri::command]
pub fn game_encode_spot(
    session_state: tauri::State<'_, GameSessionState>,
) -> Result<String, String> {
    game_encode_spot_core(&session_state)
}

#[tauri::command]
pub fn game_load_spot(
    session_state: tauri::State<'_, GameSessionState>,
    spot: String,
) -> Result<GameState, String> {
    game_load_spot_core(&session_state, &spot)
}

#[cfg(test)]
fn make_test_config() -> BlueprintV2Config {
    use poker_solver_core::blueprint_v2::config::*;
    BlueprintV2Config {
        game: GameConfig {
            name: "test".to_string(),
            players: 2,
            stack_depth: 200.0,
            small_blind: 1.0,
            big_blind: 2.0,
            rake_rate: 0.0,
            rake_cap: 0.0,
            allow_preflop_limp: true,
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
            brcfr_eta: 0.6,
            brcfr_warmup_iterations: 0,
            brcfr_interval: 100_000_000,
            use_baselines: false,
            baseline_alpha: 0.01,
            prune_streets: None,
            regret_floor: None,
            exploitability_interval_minutes: 0,
            exploitability_samples: 100_000,
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
                bucket: None,
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
    fn game_session_state_has_dual_solve_states() {
        let gss = GameSessionState::default();
        // Both subgame_solve and exact_solve should exist and default to not solving
        assert!(!gss.subgame_solve.solving.load(std::sync::atomic::Ordering::Relaxed));
        assert!(!gss.exact_solve.solving.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn solve_for_returns_subgame_by_default() {
        let gss = GameSessionState::default();
        gss.subgame_solve.iteration.store(42, std::sync::atomic::Ordering::Relaxed);
        let ss = gss.solve_for(&None);
        assert_eq!(ss.iteration.load(std::sync::atomic::Ordering::Relaxed), 42);
    }

    #[test]
    fn solve_for_returns_subgame_for_subgame_mode() {
        let gss = GameSessionState::default();
        gss.subgame_solve.iteration.store(77, std::sync::atomic::Ordering::Relaxed);
        let ss = gss.solve_for(&Some("subgame".to_string()));
        assert_eq!(ss.iteration.load(std::sync::atomic::Ordering::Relaxed), 77);
    }

    #[test]
    fn solve_for_returns_exact_for_exact_mode() {
        let gss = GameSessionState::default();
        gss.exact_solve.iteration.store(99, std::sync::atomic::Ordering::Relaxed);
        let ss = gss.solve_for(&Some("exact".to_string()));
        assert_eq!(ss.iteration.load(std::sync::atomic::Ordering::Relaxed), 99);
    }

    // -------------------------------------------------------------------
    // get_state reads solve progress (source-aware)
    // -------------------------------------------------------------------

    #[test]
    fn get_state_core_blueprint_source_skips_solve_overlay() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);

        // Simulate active solve on subgame
        gss.subgame_solve.solving.store(true, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve.iteration.store(50, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve.max_iterations.store(200, std::sync::atomic::Ordering::Relaxed);
        *gss.subgame_solve.solve_start.write() = Some(std::time::Instant::now());

        // source=None means blueprint, should skip solve overlay
        let state = game_get_state_core(&gss, None).unwrap();
        assert!(state.solve.is_none());

        // source="blueprint" also skips solve overlay
        let state = game_get_state_core(&gss, Some("blueprint".to_string())).unwrap();
        assert!(state.solve.is_none());
    }

    #[test]
    fn get_state_core_subgame_source_returns_solve_status() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);

        // Simulate active solve
        gss.subgame_solve.solving.store(true, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve.iteration.store(50, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve.max_iterations.store(200, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve.exploitability_bits.store(5.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
        *gss.subgame_solve.solve_start.write() = Some(std::time::Instant::now());

        let state = game_get_state_core(&gss, Some("subgame".to_string())).unwrap();
        let solve = state.solve.expect("solve should be Some");
        assert_eq!(solve.iteration, 50);
        assert_eq!(solve.max_iterations, 200);
        assert!((solve.exploitability - 5.0).abs() < 0.01);
        assert!(!solve.is_complete);
        assert_eq!(solve.solver_name, "range");
    }

    #[test]
    fn get_state_core_exact_source_returns_exact_solve_status() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);

        // Simulate active solve on exact
        gss.exact_solve.solving.store(true, std::sync::atomic::Ordering::Relaxed);
        gss.exact_solve.iteration.store(75, std::sync::atomic::Ordering::Relaxed);
        gss.exact_solve.max_iterations.store(300, std::sync::atomic::Ordering::Relaxed);
        *gss.exact_solve.solve_start.write() = Some(std::time::Instant::now());

        // source="exact" should read from exact_solve
        let state = game_get_state_core(&gss, Some("exact".to_string())).unwrap();
        let solve = state.solve.expect("solve should be Some");
        assert_eq!(solve.iteration, 75);
        assert_eq!(solve.max_iterations, 300);
    }

    #[test]
    fn get_state_core_returns_complete_after_solve() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);

        // Simulate completed solve
        gss.subgame_solve.solving.store(false, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve.iteration.store(200, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve.max_iterations.store(200, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve.exploitability_bits.store(1.5f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
        *gss.subgame_solve.solve_start.write() = Some(std::time::Instant::now());

        let state = game_get_state_core(&gss, Some("subgame".to_string())).unwrap();
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
        gss.subgame_solve.solving.store(true, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve.iteration.store(10, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve.max_iterations.store(100, std::sync::atomic::Ordering::Relaxed);
        *gss.subgame_solve.solve_start.write() = Some(std::time::Instant::now());

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
        *gss.subgame_solve.matrix_snapshot.write() = Some(dummy_matrix);

        let state = game_get_state_core(&gss, Some("subgame".to_string())).unwrap();
        let matrix = state.matrix.expect("matrix should be overridden by solve snapshot");
        assert_eq!(matrix.cells[0][0].hand, "TEST");
    }

    #[test]
    fn get_state_core_no_solve_data_returns_none() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);

        // No solve has been run - iteration is 0, not solving
        let state = game_get_state_core(&gss, Some("subgame".to_string())).unwrap();
        assert!(state.solve.is_none());
    }

    // -------------------------------------------------------------------
    // game_solve_core tests
    // -------------------------------------------------------------------

    #[test]
    fn game_solve_core_rejects_no_session() {
        let gss = GameSessionState::default();
        let result = game_solve_core(&gss, None, None, None, None, None, None, None, None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No game session"));
    }

    #[test]
    fn game_solve_core_rejects_double_solve_same_mode() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);
        gss.subgame_solve.solving.store(true, std::sync::atomic::Ordering::Relaxed);

        // Default mode (subgame) should reject when subgame is already solving
        let result = game_solve_core(&gss, None, None, None, None, None, None, None, None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already in progress"));
    }

    #[test]
    fn game_solve_core_rejects_double_solve_exact_mode() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);
        gss.exact_solve.solving.store(true, std::sync::atomic::Ordering::Relaxed);

        // Exact mode should reject when exact is already solving
        let result = game_solve_core(&gss, Some("exact".to_string()), None, None, None, None, None, None, None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already in progress"));
    }

    #[test]
    fn game_solve_core_allows_different_mode_concurrent() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);
        // Subgame is already solving
        gss.subgame_solve.solving.store(true, std::sync::atomic::Ordering::Relaxed);

        // Exact mode should NOT be rejected (different mode)
        // It will still fail because it's a preflop node, but the error
        // should NOT be "already in progress"
        let result = game_solve_core(&gss, Some("exact".to_string()), None, None, None, None, None, None, None, None);
        assert!(result.is_err());
        assert!(!result.unwrap_err().contains("already in progress"));
    }

    // -------------------------------------------------------------------
    // game_cancel_solve tests
    // -------------------------------------------------------------------

    #[test]
    fn cancel_solve_sets_cancel_flag_subgame() {
        let gss = GameSessionState::default();
        assert!(!gss.subgame_solve.cancel.load(std::sync::atomic::Ordering::Relaxed));
        game_cancel_solve_core(&gss, None).unwrap();
        assert!(gss.subgame_solve.cancel.load(std::sync::atomic::Ordering::Relaxed));
        // exact_solve should be unaffected
        assert!(!gss.exact_solve.cancel.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn cancel_solve_sets_cancel_flag_exact() {
        let gss = GameSessionState::default();
        assert!(!gss.exact_solve.cancel.load(std::sync::atomic::Ordering::Relaxed));
        game_cancel_solve_core(&gss, Some("exact".to_string())).unwrap();
        assert!(gss.exact_solve.cancel.load(std::sync::atomic::Ordering::Relaxed));
        // subgame_solve should be unaffected
        assert!(!gss.subgame_solve.cancel.load(std::sync::atomic::Ordering::Relaxed));
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
    // build_solve_game exact mode tests
    // -------------------------------------------------------------------

    #[test]
    fn build_solve_game_default_has_boundary_nodes_for_flop() {
        // Default (subgame) mode: flop solve has depth_limit=Some(0), producing boundary nodes
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5, 1.0]];
        let game = build_solve_game(&board, &weights, &weights, 20, 90, &sizes, false, None).unwrap();
        // Flop solve with depth_limit=Some(0) should have boundary nodes
        assert!(game.num_boundary_nodes() > 0);
    }

    #[test]
    fn build_solve_game_exact_has_no_boundary_nodes_for_flop() {
        // Exact mode: flop solve has depth_limit=None, no boundary nodes
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5, 1.0]];
        let game = build_solve_game(&board, &weights, &weights, 20, 90, &sizes, true, None).unwrap();
        // Exact solve with depth_limit=None should have no boundary nodes
        assert_eq!(game.num_boundary_nodes(), 0);
    }

    #[test]
    fn build_solve_game_depth_limit_1_allows_flop_to_turn() {
        // depth_limit_override=1 on flop: flop->turn allowed, turn->river blocked
        // Should still have boundary nodes (at turn->river transition)
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5, 1.0]];
        let game = build_solve_game(&board, &weights, &weights, 20, 90, &sizes, false, Some(1)).unwrap();
        // With depth_limit=1, there should still be boundary nodes (at turn->river)
        assert!(game.num_boundary_nodes() > 0);
    }

    #[test]
    fn build_solve_game_depth_limit_2_from_flop_has_no_boundaries() {
        // depth_limit_override=2 on flop: flop->turn->river both allowed = full solve
        // Should have no boundary nodes (equivalent to exact)
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5, 1.0]];
        let game = build_solve_game(&board, &weights, &weights, 20, 90, &sizes, false, Some(2)).unwrap();
        // depth_limit=2 from flop = full solve, no boundaries
        assert_eq!(game.num_boundary_nodes(), 0);
    }

    #[test]
    fn build_solve_game_depth_limit_none_defaults_to_zero() {
        // depth_limit_override=None should behave like depth_limit=0 (current behavior)
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5, 1.0]];
        let game = build_solve_game(&board, &weights, &weights, 20, 90, &sizes, false, None).unwrap();
        // Should have boundary nodes (same as depth_limit=0)
        assert!(game.num_boundary_nodes() > 0);
    }

    #[test]
    fn build_solve_game_exact_ignores_depth_limit_override() {
        // exact=true should always use depth_limit=None regardless of override
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5, 1.0]];
        let game = build_solve_game(&board, &weights, &weights, 20, 90, &sizes, true, Some(0)).unwrap();
        // Exact mode ignores depth_limit_override
        assert_eq!(game.num_boundary_nodes(), 0);
    }

    #[test]
    fn build_solve_game_river_ignores_depth_limit_override() {
        // River solve should always use depth_limit=None regardless of override
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string(), "7s".to_string(), "2h".to_string()];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5, 1.0]];
        let game = build_solve_game(&board, &weights, &weights, 20, 90, &sizes, false, Some(0)).unwrap();
        // River solve has no boundaries regardless of depth_limit_override
        assert_eq!(game.num_boundary_nodes(), 0);
    }

    // -------------------------------------------------------------------
    // solve state reset on game_new tests
    // -------------------------------------------------------------------

    #[test]
    fn game_new_resets_solve_state() {
        let gss = GameSessionState::default();
        // Simulate prior solve
        gss.subgame_solve.iteration.store(100, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve.max_iterations.store(200, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve.solving.store(false, std::sync::atomic::Ordering::Relaxed);

        // game_new_core needs ExplorationState and PostflopState, but
        // we can test the reset by calling reset_solve_state directly
        gss.subgame_solve.reset();

        assert_eq!(gss.subgame_solve.iteration.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(gss.subgame_solve.max_iterations.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert!(gss.subgame_solve.matrix_snapshot.read().is_none());
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
                pot: 20.0,
                stacks: [190.0, 190.0],
            }],
            root: 0,
            dealer: 0,
            starting_stack: 200.0,
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
                    pot: 20.0,
                    stacks: [190.0, 190.0],
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 0 },
                    pot: 20.0,
                    stacks: [190.0, 190.0],
                },
            ],
            root: 0,
            dealer: 0,
            starting_stack: 200.0,
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
                    pot: 6.0,
                    stacks: [199.0, 201.0],
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
                    pot: 8.0,
                    stacks: [196.0, 196.0],
                },
                // 4: fold terminal
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 0 },
                    pot: 8.0,
                    stacks: [196.0, 196.0],
                },
            ],
            root: 0,
            dealer: 0,
            starting_stack: 200.0,
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
                    pot: 6.0,
                    stacks: [199.0, 201.0],
                },
                // 2: call terminal (showdown)
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 8.0,
                    stacks: [196.0, 196.0],
                },
                // 3: all-in terminal
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 400.0,
                    stacks: [0.0, 0.0],
                },
            ],
            root: 0,
            dealer,
            starting_stack: 200.0,
        }
    }

    /// A multi-street tree: preflop SB raise/fold -> BB call/fold -> Chance -> Flop decisions.
    /// Dealer = 0 (SB = player 0, BB = player 1).
    /// All values in chips (1 BB = 2 chips).
    ///
    /// Nodes:
    /// 0: SB decision (Preflop) [Fold(->1), Bet 2bb(->2)]
    /// 1: Terminal (fold)
    /// 2: BB decision (Preflop) [Fold(->3), Call(->4)]
    /// 3: Terminal (fold)
    /// 4: Chance (Flop) -> 5
    /// 5: BB decision (Flop) [Check(->6), Bet 4bb(->7)]
    /// 6: SB decision (Flop) [Check(->8), Bet 4bb(->9)]
    /// 7: SB decision (Flop) [Fold(->10), Call(->11)]
    /// 8: Terminal (showdown)
    /// 9: Terminal (showdown)
    /// 10: Terminal (fold)
    /// 11: Chance (Turn) -> 12
    /// 12: BB decision (Turn) [Check(->13), Bet 10bb(->14)]
    /// 13: Terminal (showdown)
    /// 14: Terminal (showdown)
    fn make_multi_street_tree() -> V2GameTree {
        use poker_solver_core::blueprint_v2::game_tree::TerminalKind;
        V2GameTree {
            nodes: vec![
                // 0: SB decision (Preflop)
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Fold, TreeAction::Bet(4.0)],
                    children: vec![1, 2],
                    blueprint_decision_idx: None,
                },
                // 1: Terminal (SB fold)
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 1 },
                    pot: 3.0,
                    stacks: [199.0, 201.0],
                },
                // 2: BB decision (Preflop)
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Fold, TreeAction::Call],
                    children: vec![3, 4],
                    blueprint_decision_idx: None,
                },
                // 3: Terminal (BB fold)
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 0 },
                    pot: 5.0,
                    stacks: [201.0, 199.0],
                },
                // 4: Chance (Flop)
                V2GameNode::Chance {
                    next_street: Street::Flop,
                    child: 5,
                },
                // 5: BB decision (Flop)
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Flop,
                    actions: vec![TreeAction::Check, TreeAction::Bet(8.0)],
                    children: vec![6, 7],
                    blueprint_decision_idx: None,
                },
                // 6: SB decision (Flop) after BB check
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Flop,
                    actions: vec![TreeAction::Check, TreeAction::Bet(8.0)],
                    children: vec![8, 9],
                    blueprint_decision_idx: None,
                },
                // 7: SB decision (Flop) after BB bet
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Flop,
                    actions: vec![TreeAction::Fold, TreeAction::Call],
                    children: vec![10, 11],
                    blueprint_decision_idx: None,
                },
                // 8: Terminal (check-check showdown)
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 8.0,
                    stacks: [196.0, 196.0],
                },
                // 9: Terminal (check-bet showdown)
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 16.0,
                    stacks: [192.0, 192.0],
                },
                // 10: Terminal (SB fold to BB bet)
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 1 },
                    pot: 16.0,
                    stacks: [192.0, 192.0],
                },
                // 11: Chance (Turn)
                V2GameNode::Chance {
                    next_street: Street::Turn,
                    child: 12,
                },
                // 12: BB decision (Turn)
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Turn,
                    actions: vec![TreeAction::Check, TreeAction::Bet(20.0)],
                    children: vec![13, 14],
                    blueprint_decision_idx: None,
                },
                // 13: Terminal (showdown)
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 24.0,
                    stacks: [188.0, 188.0],
                },
                // 14: Terminal (showdown)
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 64.0,
                    stacks: [168.0, 168.0],
                },
            ],
            root: 0,
            dealer: 0,
            starting_stack: 200.0,
        }
    }

    fn make_multi_street_session() -> GameSession {
        let tree = make_multi_street_tree();
        GameSession::new_for_test(tree)
    }

    // -------------------------------------------------------------------
    // core function tests (encode_spot_core, load_spot_core)
    // -------------------------------------------------------------------

    #[test]
    fn encode_spot_core_no_session_errors() {
        let gss = GameSessionState::default();
        let result = game_encode_spot_core(&gss);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No game session"));
    }

    #[test]
    fn encode_spot_core_returns_encoding() {
        let gss = GameSessionState::default();
        let mut session = make_multi_street_session();
        session.play_action("0").unwrap(); // SB fold
        *gss.session.write() = Some(session);
        let result = game_encode_spot_core(&gss).unwrap();
        assert_eq!(result, "sb:fold");
    }

    #[test]
    fn load_spot_core_no_session_errors() {
        let gss = GameSessionState::default();
        let result = game_load_spot_core(&gss, "sb:fold");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No game session"));
    }

    #[test]
    fn load_spot_core_returns_game_state() {
        let gss = GameSessionState::default();
        let session = make_multi_street_session();
        *gss.session.write() = Some(session);
        let state = game_load_spot_core(&gss, "sb:2bb,bb:fold").unwrap();
        assert_eq!(state.action_history.len(), 2);
        assert!(state.is_terminal);
    }

    // -------------------------------------------------------------------
    // load_spot tests
    // -------------------------------------------------------------------

    #[test]
    fn load_spot_empty_string_is_noop() {
        let mut session = make_multi_street_session();
        let root = session.node_idx;
        session.load_spot("").unwrap();
        assert_eq!(session.node_idx, root);
        assert!(session.action_history.is_empty());
        assert!(session.board.is_empty());
    }

    #[test]
    fn load_spot_whitespace_only_is_noop() {
        let mut session = make_multi_street_session();
        session.load_spot("  \n  ").unwrap();
        assert!(session.action_history.is_empty());
    }

    #[test]
    fn load_spot_preflop_fold() {
        let mut session = make_multi_street_session();
        session.load_spot("sb:fold").unwrap();
        assert_eq!(session.action_history.len(), 1);
        assert_eq!(session.action_history[0].label, "Fold");
        assert_eq!(session.action_history[0].position, "SB");
    }

    #[test]
    fn load_spot_preflop_two_actions() {
        let mut session = make_multi_street_session();
        session.load_spot("sb:2bb,bb:fold").unwrap();
        assert_eq!(session.action_history.len(), 2);
        assert_eq!(session.action_history[0].label, "2bb");
        assert_eq!(session.action_history[1].label, "Fold");
    }

    #[test]
    fn load_spot_case_insensitive_labels() {
        let mut session = make_multi_street_session();
        session.load_spot("SB:FOLD").unwrap();
        assert_eq!(session.action_history.len(), 1);
        assert_eq!(session.action_history[0].label, "Fold");
    }

    #[test]
    fn load_spot_board_segment_parsed() {
        let mut session = make_multi_street_session();
        session.load_spot("sb:2bb,bb:call|Td9d6h").unwrap();
        assert_eq!(session.board, vec!["Td", "9d", "6h"]);
        assert_eq!(session.action_history.len(), 2);
    }

    #[test]
    fn load_spot_flop_actions_after_board() {
        let mut session = make_multi_street_session();
        session.load_spot("sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb").unwrap();
        assert_eq!(session.action_history.len(), 4);
        assert_eq!(session.board.len(), 3);
        assert_eq!(session.action_history[2].label, "Check");
        assert_eq!(session.action_history[2].street, "Flop");
        assert_eq!(session.action_history[3].label, "4bb");
    }

    #[test]
    fn load_spot_turn_deal() {
        let mut session = make_multi_street_session();
        session
            .load_spot("sb:2bb,bb:call|Td9d6h|bb:4bb,sb:call|Kh")
            .unwrap();
        assert_eq!(session.board.len(), 4);
        assert_eq!(session.board[3], "Kh");
    }

    #[test]
    fn load_spot_invalid_action_errors() {
        let mut session = make_multi_street_session();
        let result = session.load_spot("sb:invalid");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("not found"), "Error was: {err}");
        assert!(err.contains("Available"), "Error was: {err}");
    }

    #[test]
    fn load_spot_position_mismatch_errors() {
        let mut session = make_multi_street_session();
        // First action should be SB, not BB
        let result = session.load_spot("bb:fold");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Position mismatch"), "Error was: {err}");
    }

    #[test]
    fn load_spot_invalid_format_errors() {
        let mut session = make_multi_street_session();
        let result = session.load_spot("nocolon");
        // "nocolon" has no colon, so it's treated as a board segment
        // With odd length it should error
        assert!(result.is_err());
    }

    #[test]
    fn load_spot_resets_prior_state() {
        let mut session = make_multi_street_session();
        // Play some actions first
        session.play_action("0").unwrap(); // SB fold
        assert_eq!(session.action_history.len(), 1);

        // load_spot should reset and replay from scratch
        session.load_spot("sb:2bb,bb:call|Td9d6h").unwrap();
        assert_eq!(session.action_history.len(), 2);
        assert_eq!(session.board.len(), 3);
    }

    // -------------------------------------------------------------------
    // encode/load round-trip tests
    // -------------------------------------------------------------------

    #[test]
    fn round_trip_preflop_fold() {
        let mut session1 = make_multi_street_session();
        session1.play_action("0").unwrap(); // SB fold
        let encoded = session1.encode_spot();

        let mut session2 = make_multi_street_session();
        session2.load_spot(&encoded).unwrap();
        assert_eq!(session2.encode_spot(), encoded);
        assert_eq!(session2.action_history.len(), session1.action_history.len());
    }

    #[test]
    fn round_trip_preflop_to_flop() {
        let mut session1 = make_multi_street_session();
        session1.play_action("1").unwrap(); // SB 2bb
        session1.play_action("1").unwrap(); // BB call
        session1.deal_card("Ah").unwrap();
        session1.deal_card("Kd").unwrap();
        session1.deal_card("Qc").unwrap();
        let encoded = session1.encode_spot();

        let mut session2 = make_multi_street_session();
        session2.load_spot(&encoded).unwrap();
        assert_eq!(session2.encode_spot(), encoded);
        assert_eq!(session2.board, session1.board);
    }

    #[test]
    fn round_trip_flop_actions() {
        let mut session1 = make_multi_street_session();
        session1.play_action("1").unwrap(); // SB 2bb
        session1.play_action("1").unwrap(); // BB call
        session1.deal_card("Td").unwrap();
        session1.deal_card("9d").unwrap();
        session1.deal_card("6h").unwrap();
        session1.play_action("0").unwrap(); // BB check
        session1.play_action("1").unwrap(); // SB 4bb
        let encoded = session1.encode_spot();

        let mut session2 = make_multi_street_session();
        session2.load_spot(&encoded).unwrap();
        assert_eq!(session2.encode_spot(), encoded);
        assert_eq!(session2.action_history.len(), 4);
    }

    #[test]
    fn round_trip_turn_deal() {
        let mut session1 = make_multi_street_session();
        session1.play_action("1").unwrap(); // SB 2bb
        session1.play_action("1").unwrap(); // BB call
        session1.deal_card("Td").unwrap();
        session1.deal_card("9d").unwrap();
        session1.deal_card("6h").unwrap();
        session1.play_action("1").unwrap(); // BB 4bb
        session1.play_action("1").unwrap(); // SB call
        session1.deal_card("Kh").unwrap();
        let encoded = session1.encode_spot();

        let mut session2 = make_multi_street_session();
        session2.load_spot(&encoded).unwrap();
        assert_eq!(session2.encode_spot(), encoded);
        assert_eq!(session2.board, vec!["Td", "9d", "6h", "Kh"]);
    }

    // -------------------------------------------------------------------
    // encode_spot tests
    // -------------------------------------------------------------------

    #[test]
    fn encode_spot_empty_history() {
        let session = make_multi_street_session();
        assert_eq!(session.encode_spot(), "");
    }

    #[test]
    fn encode_spot_preflop_fold() {
        let mut session = make_multi_street_session();
        session.play_action("0").unwrap(); // SB fold
        assert_eq!(session.encode_spot(), "sb:fold");
    }

    #[test]
    fn encode_spot_preflop_two_actions() {
        let mut session = make_multi_street_session();
        session.play_action("1").unwrap(); // SB 2bb
        session.play_action("0").unwrap(); // BB fold
        assert_eq!(session.encode_spot(), "sb:2bb,bb:fold");
    }

    #[test]
    fn encode_spot_preflop_to_flop_deal() {
        let mut session = make_multi_street_session();
        session.play_action("1").unwrap(); // SB 2bb
        session.play_action("1").unwrap(); // BB call -> Chance node
        session.deal_card("Td").unwrap();
        session.deal_card("9d").unwrap();
        session.deal_card("6h").unwrap();
        assert_eq!(session.encode_spot(), "sb:2bb,bb:call|Td9d6h");
    }

    #[test]
    fn encode_spot_flop_action_after_board() {
        let mut session = make_multi_street_session();
        session.play_action("1").unwrap(); // SB 2bb
        session.play_action("1").unwrap(); // BB call
        session.deal_card("Td").unwrap();
        session.deal_card("9d").unwrap();
        session.deal_card("6h").unwrap();
        session.play_action("0").unwrap(); // BB check
        session.play_action("1").unwrap(); // SB 4bb
        assert_eq!(
            session.encode_spot(),
            "sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb"
        );
    }

    #[test]
    fn encode_spot_flop_to_turn_deal() {
        let mut session = make_multi_street_session();
        session.play_action("1").unwrap(); // SB 2bb
        session.play_action("1").unwrap(); // BB call
        session.deal_card("Td").unwrap();
        session.deal_card("9d").unwrap();
        session.deal_card("6h").unwrap();
        session.play_action("1").unwrap(); // BB 4bb
        session.play_action("1").unwrap(); // SB call -> Chance (Turn)
        session.deal_card("Kh").unwrap();
        assert_eq!(
            session.encode_spot(),
            "sb:2bb,bb:call|Td9d6h|bb:4bb,sb:call|Kh"
        );
    }

    #[test]
    fn encode_spot_turn_action() {
        let mut session = make_multi_street_session();
        session.play_action("1").unwrap(); // SB 2bb
        session.play_action("1").unwrap(); // BB call
        session.deal_card("Td").unwrap();
        session.deal_card("9d").unwrap();
        session.deal_card("6h").unwrap();
        session.play_action("1").unwrap(); // BB 4bb
        session.play_action("1").unwrap(); // SB call -> Turn
        session.deal_card("Kh").unwrap();
        session.play_action("1").unwrap(); // BB 10bb
        assert_eq!(
            session.encode_spot(),
            "sb:2bb,bb:call|Td9d6h|bb:4bb,sb:call|Kh|bb:10bb"
        );
    }

    #[test]
    fn round_trip_turn_action_verifies_all_fields() {
        let mut session1 = make_multi_street_session();
        session1.play_action("1").unwrap(); // SB 2bb
        session1.play_action("1").unwrap(); // BB call
        session1.deal_card("Td").unwrap();
        session1.deal_card("9d").unwrap();
        session1.deal_card("6h").unwrap();
        session1.play_action("1").unwrap(); // BB 4bb
        session1.play_action("1").unwrap(); // SB call -> Turn
        session1.deal_card("Kh").unwrap();
        session1.play_action("1").unwrap(); // BB 10bb (terminal)

        let encoded = session1.encode_spot();
        let state1 = session1.get_state();

        let mut session2 = make_multi_street_session();
        session2.load_spot(&encoded).unwrap();
        let state2 = session2.get_state();

        // Verify round-trip fidelity
        assert_eq!(session2.encode_spot(), encoded);
        assert_eq!(state2.action_history.len(), state1.action_history.len());
        assert_eq!(state2.board, state1.board);
        assert_eq!(state2.position, state1.position);
        assert_eq!(state2.street, state1.street);
        assert_eq!(state2.is_terminal, state1.is_terminal);
    }

    #[test]
    fn round_trip_flop_deal_only_verifies_position() {
        // Encode a spot at a chance node (board dealt, waiting for action)
        let mut session1 = make_multi_street_session();
        session1.play_action("1").unwrap(); // SB 2bb
        session1.play_action("1").unwrap(); // BB call
        session1.deal_card("Ah").unwrap();
        session1.deal_card("Kd").unwrap();
        session1.deal_card("Qc").unwrap();

        let encoded = session1.encode_spot();
        let state1 = session1.get_state();

        let mut session2 = make_multi_street_session();
        session2.load_spot(&encoded).unwrap();
        let state2 = session2.get_state();

        assert_eq!(session2.encode_spot(), encoded);
        assert_eq!(state2.board, state1.board);
        assert_eq!(state2.position, state1.position);
        assert_eq!(state2.street, state1.street);
        assert_eq!(state2.action_history.len(), state1.action_history.len());
    }

    // -------------------------------------------------------------------
    // Solve cache tests
    // -------------------------------------------------------------------

    /// Helper to create a dummy CachedSolveNode with a recognizable hand label.
    fn make_cached_node(hand_label: &str, action_labels: &[&str], position: &str) -> CachedSolveNode {
        let actions: Vec<GameAction> = action_labels
            .iter()
            .enumerate()
            .map(|(i, &lbl)| GameAction {
                id: i.to_string(),
                label: lbl.to_string(),
                action_type: "check".to_string(),
            })
            .collect();
        let matrix = GameMatrix {
            cells: vec![vec![GameMatrixCell {
                hand: hand_label.to_string(),
                suited: false,
                pair: false,
                probabilities: vec![1.0; action_labels.len()],
                combo_count: 1,
                weight: 1.0,
                ev: None,
                combos: vec![],
            }]],
            actions: actions.clone(),
        };
        CachedSolveNode {
            matrix,
            actions,
            position: position.to_string(),
        }
    }

    #[test]
    fn solve_state_default_has_empty_cache_and_path() {
        let ss = SolveState::default();
        assert!(ss.solve_cache.read().is_empty());
        assert!(ss.solve_path.read().is_empty());
    }

    #[test]
    fn solve_state_reset_clears_cache_and_path() {
        let ss = SolveState::default();
        // Populate cache and path
        ss.solve_cache.write().insert(vec![], make_cached_node("ROOT", &["Check", "Bet"], "OOP"));
        ss.solve_cache.write().insert(vec![0], make_cached_node("CHILD0", &["Fold", "Call"], "IP"));
        ss.solve_path.write().push(0);

        assert!(!ss.solve_cache.read().is_empty());
        assert!(!ss.solve_path.read().is_empty());

        ss.reset();

        assert!(ss.solve_cache.read().is_empty());
        assert!(ss.solve_path.read().is_empty());
    }

    #[test]
    fn play_action_serves_cached_matrix_within_solved_tree() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        *gss.session.write() = Some(session);

        // Populate solve cache: root and one child
        let root_node = make_cached_node("ROOT", &["Fold", "Call"], "BB");
        let child_node = make_cached_node("CHILD", &["Check", "Fold"], "SB");
        gss.subgame_solve.solve_cache.write().insert(vec![], root_node);
        gss.subgame_solve.solve_cache.write().insert(vec![1], child_node);
        // Mark solve as completed so iteration > 0
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);

        // Play action "1" (Call) which maps to cache path [1]
        let source = Some("subgame".to_string());
        let state = game_play_action_core(&gss, "1", source).unwrap();
        let matrix = state.matrix.expect("should have cached matrix");
        assert_eq!(matrix.cells[0][0].hand, "CHILD");
        assert_eq!(state.position, "SB");
        // Path should now be [1]
        assert_eq!(*gss.subgame_solve.solve_path.read(), vec![1]);
    }

    #[test]
    fn play_action_resets_when_navigating_outside_solved_tree() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        *gss.session.write() = Some(session);

        // Populate solve cache with only root (no children cached)
        let root_node = make_cached_node("ROOT", &["Fold", "Call"], "BB");
        gss.subgame_solve.solve_cache.write().insert(vec![], root_node);
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);

        // Play action "1" (Call) -- path [1] not in cache, should reset
        let source = Some("subgame".to_string());
        let _state = game_play_action_core(&gss, "1", source).unwrap();
        // Cache should be cleared
        assert!(gss.subgame_solve.solve_cache.read().is_empty());
        assert!(gss.subgame_solve.solve_path.read().is_empty());
    }

    #[test]
    fn back_serves_parent_cached_matrix() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        *gss.session.write() = Some(session);

        // Navigate to child first (without cache, so it resets -- we need to
        // set cache AFTER navigating to simulate post-solve navigation)
        // Instead, pre-populate cache and navigate within it:
        let root_node = make_cached_node("ROOT", &["Fold", "Call"], "BB");
        let child_node = make_cached_node("CHILD", &["Check", "Fold"], "SB");
        gss.subgame_solve.solve_cache.write().insert(vec![], root_node);
        gss.subgame_solve.solve_cache.write().insert(vec![1], child_node);
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);

        // Play action to get to child (within solved tree)
        let source = Some("subgame".to_string());
        let _ = game_play_action_core(&gss, "1", source).unwrap();
        assert_eq!(*gss.subgame_solve.solve_path.read(), vec![1]);

        // Now go back -- should serve root cached matrix
        let state = game_back_core(&gss, Some("subgame".to_string())).unwrap();
        let matrix = state.matrix.expect("should have root cached matrix");
        assert_eq!(matrix.cells[0][0].hand, "ROOT");
        assert_eq!(*gss.subgame_solve.solve_path.read(), Vec::<usize>::new());
    }

    #[test]
    fn back_at_solve_root_shows_root_matrix() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        *gss.session.write() = Some(session);

        // Play one action outside solve cache first, then set up cache at the child
        // Actually, test the case where we're at solve root (path is empty) and go back
        // We need to first navigate to a position, THEN set up solve cache

        // Navigate to child node first
        {
            let mut guard = gss.session.write();
            let s = guard.as_mut().unwrap();
            s.play_action("1").unwrap(); // now at node 2
        }

        // Set up solve cache at this position (the solve root is the current node)
        let root_node = make_cached_node("SOLVE_ROOT", &["Check", "Fold"], "SB");
        gss.subgame_solve.solve_cache.write().insert(vec![], root_node);
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);
        // Path is empty (at solve root)

        // Go back -- should clear cache (navigating before solve root)
        let _state = game_back_core(&gss, Some("subgame".to_string())).unwrap();
        // Cache should be cleared since we went before the solve root
        assert!(gss.subgame_solve.solve_cache.read().is_empty());
    }

    #[test]
    fn get_state_core_serves_cached_matrix_at_solve_path() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        *gss.session.write() = Some(session);

        // Set up completed solve with cache
        gss.subgame_solve.solving.store(false, Ordering::Relaxed);
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);
        gss.subgame_solve.max_iterations.store(100, Ordering::Relaxed);
        *gss.subgame_solve.solve_start.write() = Some(std::time::Instant::now());

        let root_node = make_cached_node("CACHE_ROOT", &["Fold", "Call"], "BB");
        let child_node = make_cached_node("CACHE_CHILD", &["Check", "Fold"], "SB");
        gss.subgame_solve.solve_cache.write().insert(vec![], root_node);
        gss.subgame_solve.solve_cache.write().insert(vec![1], child_node);

        // At solve root (empty path), should serve root cache
        let state = game_get_state_core(&gss, Some("subgame".to_string())).unwrap();
        let matrix = state.matrix.expect("should have cached matrix");
        assert_eq!(matrix.cells[0][0].hand, "CACHE_ROOT");

        // Navigate to child path
        *gss.subgame_solve.solve_path.write() = vec![1];
        let state = game_get_state_core(&gss, Some("subgame".to_string())).unwrap();
        let matrix = state.matrix.expect("should have child cached matrix");
        assert_eq!(matrix.cells[0][0].hand, "CACHE_CHILD");
    }

    #[test]
    fn build_solve_cache_contains_root_and_children() {
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

        let cache = build_solve_cache(&mut game);
        // Root should be present
        assert!(cache.contains_key(&vec![]), "cache should contain root entry");
        // Root should have actions
        assert!(!cache[&vec![]].actions.is_empty());
        // Root should have a 13x13 matrix
        assert_eq!(cache[&vec![]].matrix.cells.len(), 13);
        // Should have more than just root (children for each action at root)
        assert!(cache.len() > 1, "cache should contain child entries too, got {}", cache.len());
    }

    #[test]
    fn build_solve_matrix_at_current_works_without_back_to_root() {
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

        // Navigate to a child node first
        let num_actions = game.available_actions().len();
        assert!(num_actions > 0);
        game.play(0); // play first action

        // If we're NOT at a terminal/chance node, build matrix at current position
        if !game.is_terminal_node() && !game.is_chance_node() {
            let matrix = build_solve_matrix_at_current(&mut game, None);
            assert_eq!(matrix.cells.len(), 13);
            assert!(!matrix.actions.is_empty());
        }
    }

    // (RefreshProgress tests deleted: rollout boundary-eval path removed)

    // -------------------------------------------------------------------
    // StreetBoundaryConfig + resolve_street_boundary tests
    // -------------------------------------------------------------------

    #[test]
    fn sbc_all_exact_returns_none() {
        let config = StreetBoundaryConfig::default();
        assert!(resolve_street_boundary(&config, Street::Flop).is_none());
        assert!(resolve_street_boundary(&config, Street::Turn).is_none());
        assert!(resolve_street_boundary(&config, Street::River).is_none());
    }

    #[test]
    fn sbc_cfvnet_at_river_from_flop_root() {
        // river=Cfvnet from flop root → cut before river card → depth=1
        // (near tree = flop + turn, 1 street transition before the river cut).
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::Exact,
            river: StreetBoundaryMode::Cfvnet {
                model_path: "/models/river.onnx".to_string(),
            },
        };
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(result, Some((1, "/models/river.onnx".to_string())));
    }

    #[test]
    fn sbc_cfvnet_at_turn_from_flop_root() {
        // turn=Cfvnet from flop root → cut before turn card → depth=0
        // (near tree = flop only, 0 transitions).
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::Cfvnet {
                model_path: "/models/turn.onnx".to_string(),
            },
            river: StreetBoundaryMode::Exact,
        };
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(result, Some((0, "/models/turn.onnx".to_string())));
    }

    #[test]
    fn sbc_cfvnet_at_flop_from_flop_root_is_ignored() {
        // flop=Cfvnet on flop root is degenerate — can't cut before our
        // current position. Falls through to all-exact (None).
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Cfvnet {
                model_path: "/models/flop.onnx".to_string(),
            },
            turn: StreetBoundaryMode::Exact,
            river: StreetBoundaryMode::Exact,
        };
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(result, None);
    }

    #[test]
    fn sbc_first_cfvnet_wins_when_multiple() {
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::Cfvnet {
                model_path: "/models/turn.onnx".to_string(),
            },
            river: StreetBoundaryMode::Cfvnet {
                model_path: "/models/river.onnx".to_string(),
            },
        };
        // First non-exact wins: turn cut at depth 0 from flop root.
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(result, Some((0, "/models/turn.onnx".to_string())));
    }

    #[test]
    fn sbc_cfvnet_at_river_from_turn_root() {
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::Exact,
            river: StreetBoundaryMode::Cfvnet {
                model_path: "/models/river.onnx".to_string(),
            },
        };
        // From turn root: near tree = turn, cut before river = depth=0.
        let result = resolve_street_boundary(&config, Street::Turn);
        assert_eq!(result, Some((0, "/models/river.onnx".to_string())));
    }

    #[test]
    fn sbc_cfvnet_at_turn_from_turn_root_is_ignored() {
        // turn=Cfvnet on turn root is degenerate — same as flop case above.
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::Cfvnet {
                model_path: "/models/turn.onnx".to_string(),
            },
            river: StreetBoundaryMode::Exact,
        };
        let result = resolve_street_boundary(&config, Street::Turn);
        assert_eq!(result, None);
    }

    #[test]
    fn sbc_preflop_root_returns_none() {
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Cfvnet {
                model_path: "/models/flop.onnx".to_string(),
            },
            turn: StreetBoundaryMode::Exact,
            river: StreetBoundaryMode::Exact,
        };
        assert!(resolve_street_boundary(&config, Street::Preflop).is_none());
    }

    #[test]
    fn sbc_default_is_all_exact() {
        let config = StreetBoundaryConfig::default();
        assert!(matches!(config.flop, StreetBoundaryMode::Exact));
        assert!(matches!(config.turn, StreetBoundaryMode::Exact));
        assert!(matches!(config.river, StreetBoundaryMode::Exact));
    }

    #[test]
    fn sbc_serde_roundtrip_exact() {
        let config = StreetBoundaryConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: StreetBoundaryConfig = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed.flop, StreetBoundaryMode::Exact));
    }

    #[test]
    fn sbc_serde_roundtrip_cfvnet() {
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::Exact,
            river: StreetBoundaryMode::Cfvnet {
                model_path: "/models/river.onnx".to_string(),
            },
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: StreetBoundaryConfig = serde_json::from_str(&json).unwrap();
        if let StreetBoundaryMode::Cfvnet { model_path } = &parsed.river {
            assert_eq!(model_path, "/models/river.onnx");
        } else {
            panic!("expected Cfvnet for river");
        }
    }

    #[test]
    fn sbc_serde_from_json_tagged() {
        let json = r#"{
            "flop": {"mode": "exact"},
            "turn": {"mode": "exact"},
            "river": {"mode": "cfvnet", "model_path": "/path/to/model.onnx"}
        }"#;
        let config: StreetBoundaryConfig = serde_json::from_str(json).unwrap();
        assert!(matches!(config.flop, StreetBoundaryMode::Exact));
        assert!(matches!(config.turn, StreetBoundaryMode::Exact));
        if let StreetBoundaryMode::Cfvnet { model_path } = &config.river {
            assert_eq!(model_path, "/path/to/model.onnx");
        } else {
            panic!("expected Cfvnet");
        }
    }
}
