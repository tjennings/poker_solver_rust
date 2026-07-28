//! Unified game session: tracks state from preflop through river.
//!
//! A `GameSession` owns the V2 game tree, blueprint strategy, and range weights.
//! Six Tauri commands expose it: `game_new`, `game_get_state`, `game_play_action`,
//! `game_deal_card`, `game_back`, `game_solve`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use serde::Serialize;

use poker_solver_core::blueprint_mp::config::{BlueprintMpConfig, ForcedBetKind};
use poker_solver_core::blueprint_mp::game_tree::TreeAction as MpTreeAction;
use poker_solver_core::blueprint_mp::lazy_mccfr::{
    LazyBettingSnapshot, LazyMpGame, LazyResolvedSpot,
};
use poker_solver_core::blueprint_mp::Street as MpStreet;
use poker_solver_core::blueprint_universal::{
    ActionDescriptor, ActionKind, BundleKind, LoadedBundle, MpLazyKey,
};
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
use poker_solver_core::blueprint_v2::game_tree::{
    GameNode as V2GameNode, GameTree as V2GameTree, TreeAction,
};
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::blueprint_v2::{LeafEvaluator, Street};
use poker_solver_core::hands::CanonicalHand;
use poker_solver_core::poker::{Card as RsPokerCard, Value as RsPokerValue};

use range_solver::card::{card_pair_to_index, card_to_string, index_to_card_pair, NOT_DEALT};
use range_solver::interface::Game;
use range_solver::{compute_exploitability, finalize, solve_step, PostFlopGame};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Per-street boundary configuration
// ---------------------------------------------------------------------------

/// How to evaluate boundaries at a given street transition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum StreetBoundaryMode {
    Exact,
    Cfvnet {
        model_path: String,
        #[serde(default)]
        inference_mode: cfvnet::eval::boundary_evaluator::BoundaryInferenceMode,
    },
    /// Cut here and solve the downstream subtree exactly (full CFR).
    ExactSubtree,
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

/// What kind of boundary evaluator to use at a depth cut.
#[derive(Clone, Debug, PartialEq)]
pub enum BoundaryKind {
    /// Use an ONNX cfvnet model for boundary CFVs.
    Cfvnet {
        model_path: String,
        inference_mode: cfvnet::eval::boundary_evaluator::BoundaryInferenceMode,
    },
    /// Solve the downstream subtree exactly (full CFR).
    ExactSubtree,
}

/// Result of resolving a `StreetBoundaryConfig` against a root street.
///
/// `None` means all-exact (no cut). `Some((depth, kind))` means cut
/// after `depth` street transitions using the given boundary evaluator.
pub fn resolve_street_boundary(
    config: &StreetBoundaryConfig,
    root_street: Street,
) -> Option<(u8, BoundaryKind)> {
    // Streets to walk from root, in order.
    let streets: &[(Street, &StreetBoundaryMode)] = match root_street {
        Street::Flop => &[
            (Street::Flop, &config.flop),
            (Street::Turn, &config.turn),
            (Street::River, &config.river),
        ],
        Street::Turn => &[(Street::Turn, &config.turn), (Street::River, &config.river)],
        Street::River => &[(Street::River, &config.river)],
        Street::Preflop => return None, // preflop solve not supported
    };

    // First non-Exact street defines the cut point.
    // If the root street itself is non-Exact, no cut possible — skip it.
    for (i, (street, mode)) in streets.iter().enumerate() {
        if i == 0 && !matches!(mode, StreetBoundaryMode::Exact) {
            continue;
        }
        match mode {
            StreetBoundaryMode::Exact => {}
            StreetBoundaryMode::Cfvnet {
                model_path,
                inference_mode,
            } => {
                return Some((
                    (i - 1) as u8,
                    BoundaryKind::Cfvnet {
                        model_path: model_path.clone(),
                        inference_mode: *inference_mode,
                    },
                ));
            }
            StreetBoundaryMode::ExactSubtree => {
                // A turn-root solve is small enough to solve exactly, and the
                // river exact-subtree shortcut is not yet strategy-equivalent
                // at in-street bet/raise response nodes.
                if root_street == Street::Turn && *street == Street::River {
                    continue;
                }
                return Some(((i - 1) as u8, BoundaryKind::ExactSubtree));
            }
        }
    }
    None // all exact
}

pub fn validate_cfvnet_boundary_cut(
    boundary_cut: &Option<(u8, BoundaryKind)>,
    root_street: Street,
) -> Result<(), String> {
    if matches!(boundary_cut, Some((0, BoundaryKind::Cfvnet { .. }))) && root_street == Street::Flop
    {
        return Err(
            "CFVNet boundary from a flop solve would evaluate 3-card flop boards, \
             but the ONNX evaluator supports only 4-card and 5-card boards. \
             Use a river CFVNet boundary or Exact Subtree for earlier cuts."
                .to_string(),
        );
    }
    Ok(())
}

fn inference_mode_label(
    inference_mode: cfvnet::eval::boundary_evaluator::BoundaryInferenceMode,
) -> &'static str {
    match inference_mode {
        cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::RiverEnumeratedTurn => {
            "river_enumerated_turn"
        }
        cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::Direct => "direct",
        cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::DirectNormalizedLegacy => {
            "direct_normalized_legacy"
        }
    }
}

fn boundary_evaluator_log_line(
    solver_mode: &str,
    boundary_cut: &Option<(u8, BoundaryKind)>,
) -> String {
    match boundary_cut {
        Some((
            depth,
            BoundaryKind::Cfvnet {
                model_path,
                inference_mode,
            },
        )) => {
            let evaluator = match inference_mode {
                cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::Direct => "Direct CFVNet",
                cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::DirectNormalizedLegacy => {
                    "Direct CFVNet (legacy scaled bcfv)"
                }
                cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::RiverEnumeratedTurn => {
                    "CFVNet"
                }
            };
            format!(
                "[solve] solver: {solver_mode}; boundary evaluator: {evaluator}; depth_limit={depth}; inference_mode={}; model={model_path}",
                inference_mode_label(*inference_mode)
            )
        }
        Some((depth, BoundaryKind::ExactSubtree)) => format!(
            "[solve] solver: {solver_mode}; boundary evaluator: Exact Subtree; depth_limit={depth}"
        ),
        None => format!(
            "[solve] solver: {solver_mode}; boundary evaluator: Exact full tree; depth_limit=none; model=none"
        ),
    }
}

use crate::exploration::{
    blueprint_sizes_to_range_solver, board_for_street_slice, build_canonical_to_combo_map,
    canonical_hand_index_from_ranks, hand_label_from_matrix, parse_board, pot_at_v2_node,
    ActionInfo, BucketLookup, RANKS,
};
use crate::postflop::{parse_rs_poker_card, CbvContext, RolloutLeafEvaluator};

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
    pub bucket: Option<u16>, // strategy bucket ID (postflop only)
}

/// A single cell in the 13x13 strategy matrix.
#[derive(Debug, Clone, Serialize)]
pub struct GameMatrixCell {
    pub hand: String,
    pub suited: bool,
    pub pair: bool,
    /// One value per action. Universal lazy MP preflop and postflop matrix
    /// values are root-reach-weighted and sum to `weight`. Other sources may
    /// expose conditional strategy frequencies.
    pub probabilities: Vec<f32>,
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

/// Session position where a solve cache was rooted.
#[derive(Debug, Clone)]
pub struct SolveAnchor {
    pub node_idx: u32,
    pub board: Vec<String>,
    pub action_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct SolveGameRoot {
    pub starting_pot: i32,
    pub initial_player: u8,
    pub initial_stacks: [i32; 2],
    pub initial_prev_action: range_solver::Action,
    pub initial_prev_amount: i32,
    pub initial_amount: i32,
    pub initial_num_bets: i32,
}

impl SolveGameRoot {
    fn fresh_street(pot: i32, effective_stack: i32) -> Self {
        Self {
            starting_pot: pot,
            initial_player: 0,
            initial_stacks: [effective_stack, effective_stack],
            initial_prev_action: range_solver::Action::None,
            initial_prev_amount: 0,
            initial_amount: 0,
            initial_num_bets: 0,
        }
    }
}

pub fn effective_stack_for_solve_root(root: &SolveGameRoot) -> i32 {
    root.initial_stacks
        .iter()
        .copied()
        .filter(|stack| *stack > 0)
        .min()
        .unwrap_or(1)
}

/// Exact solve inputs captured from a Universal MP lazy session.
///
/// `raw_reaches_by_seat` stays in the bundle's actual seat order. The exact
/// solver maps those arrays to OOP/IP using `oop_seat` and `ip_seat`; keeping
/// both facts in this type prevents display-order assumptions from leaking
/// into solver construction.
#[derive(Debug, Clone)]
pub struct UniversalMpSolveSnapshot {
    pub board: Vec<String>,
    pub raw_reaches_by_seat: [Vec<f32>; 2],
    pub acting_seat: u8,
    pub pot: i32,
    pub remaining_stacks: [i32; 2],
    pub street_bets: [i32; 2],
    pub facing_bet: bool,
    pub raise_count: u8,
    pub last_raise_to: i32,
    pub last_aggressive_action: Option<MpTreeAction>,
    pub oop_seat: u8,
    pub ip_seat: u8,
    pub root: SolveGameRoot,
    pub bet_sizes: Vec<Vec<f64>>,
    pub action_history: Vec<ActionRecord>,
    pub actions: Vec<GameAction>,
}

#[derive(Debug, Clone, Copy)]
struct BettingState {
    pot: f64,
    stacks: [f64; 2],
    street_bets: [f64; 2],
    num_bets: u8,
    last_aggressive_action: Option<TreeAction>,
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
    /// Monotonically increasing identity for the solve/session state.
    pub generation: AtomicU64,
    /// Serializes worker publications with reset/start invalidation.
    publish_gate: RwLock<()>,
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
    /// Session anchor for the cached solve tree.
    pub solve_anchor: RwLock<Option<SolveAnchor>>,
}

impl Default for SolveState {
    fn default() -> Self {
        Self {
            solving: AtomicBool::new(false),
            cancel: Arc::new(AtomicBool::new(false)),
            generation: AtomicU64::new(0),
            publish_gate: RwLock::new(()),
            iteration: AtomicU32::new(0),
            max_iterations: AtomicU32::new(0),
            exploitability_bits: AtomicU32::new(0),
            solve_start: RwLock::new(None),
            matrix_snapshot: RwLock::new(None),
            solve_actions: RwLock::new(vec![]),
            solve_position: RwLock::new(String::new()),
            solve_cache: RwLock::new(HashMap::new()),
            solve_path: RwLock::new(vec![]),
            solve_anchor: RwLock::new(None),
        }
    }
}

impl SolveState {
    /// Reset all fields and invalidate every worker from the previous state.
    pub fn reset(&self) {
        let _publish_guard = self.publish_gate.write();
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.solving.store(false, Ordering::Relaxed);
        self.cancel.store(true, Ordering::Release);
        self.iteration.store(0, Ordering::Relaxed);
        self.max_iterations.store(0, Ordering::Relaxed);
        self.exploitability_bits.store(0, Ordering::Relaxed);
        *self.solve_start.write() = None;
        *self.matrix_snapshot.write() = None;
        *self.solve_actions.write() = vec![];
        *self.solve_position.write() = String::new();
        *self.solve_cache.write() = HashMap::new();
        *self.solve_path.write() = vec![];
        *self.solve_anchor.write() = None;
    }

    fn publish_if_current<F>(&self, generation: u64, publish: F) -> bool
    where
        F: FnOnce(&Self),
    {
        let _publish_guard = self.publish_gate.read();
        if self.generation.load(Ordering::Acquire) != generation {
            return false;
        }
        publish(self);
        true
    }
}

// ---------------------------------------------------------------------------
// Shared state for Tauri commands
// ---------------------------------------------------------------------------

/// Shared session state, accessible by Tauri commands.
pub struct GameSessionState {
    pub session: RwLock<Option<GameSession>>,
    /// Separate lazy MP session. The HU `session` field remains unchanged so
    /// existing GameSession and solver tests keep their original ownership and
    /// state model.
    pub mp_session: RwLock<Option<LazyMpSession>>,
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
            mp_session: RwLock::new(None),
            subgame_solve: Arc::new(SolveState::default()),
            exact_solve: Arc::new(SolveState::default()),
        }
    }
}

// ---------------------------------------------------------------------------
// Lazy MP GameSession
// ---------------------------------------------------------------------------

/// Read-only navigation over a two-player universal lazy MP bundle.
///
/// The cursor remains the semantic lazy public model. The adapter adds a typed
/// board and uses file-backed `AllBuckets` lookups for completed postflop
/// streets; it never materializes a dense MP tree or fabricates rows.
pub struct LazyMpSession {
    game: LazyMpGame,
    spot: LazyResolvedSpot,
    bundle: Arc<LoadedBundle>,
    config: BlueprintMpConfig,
    all_buckets: Option<Arc<AllBuckets>>,
    bucket_error: Option<String>,
    bucket_source: Option<crate::exploration::UniversalMpData>,
    board: Vec<RsPokerCard>,
    action_history: Vec<ActionRecord>,
    terminal: bool,
}

impl LazyMpSession {
    pub(crate) fn from_exploration_data(
        data: crate::exploration::UniversalMpData,
    ) -> Result<Self, String> {
        let bundle_kind = data.bundle.kind();
        if bundle_kind != BundleKind::UniversalMpLazy {
            return Err(format!(
                "Universal MP GameExplorer supports only universal_mp_lazy bundles; received {bundle_kind}"
            ));
        }
        let config = data.config.clone().ok_or_else(|| {
            "game_new requires a retained MP config.yaml for universal MP navigation".to_string()
        })?;
        let manifest = data
            .bundle
            .manifest()
            .ok_or_else(|| "Universal MP bundle is missing its manifest".to_string())?;
        if manifest.game.num_players != config.game.num_players {
            return Err(format!(
                "Universal MP config player count ({}) does not match bundle manifest ({})",
                config.game.num_players, manifest.game.num_players
            ));
        }
        if config.game.num_players != 2 {
            return Err(format!(
                "Universal MP GameExplorer currently supports 2-player sessions; bundle has {} players",
                config.game.num_players
            ));
        }
        mp_big_blind_amount(&config)?;

        let game = LazyMpGame::new(&config.game, &config.action_abstraction);
        let spot = LazyResolvedSpot::root(&game);
        let bucket_source = crate::exploration::UniversalMpData {
            bundle: Arc::clone(&data.bundle),
            config: data.config.clone(),
            config_dir: data.config_dir.clone(),
            bundle_dir: data.bundle_dir.clone(),
        };
        Ok(Self {
            game,
            spot,
            bundle: data.bundle,
            config: *config,
            all_buckets: None,
            bucket_error: None,
            bucket_source: Some(bucket_source),
            board: vec![],
            action_history: vec![],
            terminal: false,
        })
    }

    fn ensure_all_buckets(&mut self) -> Result<&AllBuckets, String> {
        if self.all_buckets.is_none() && self.bucket_error.is_none() {
            let source = self
                .bucket_source
                .take()
                .ok_or_else(|| "Universal MP flop bucket source is unavailable".to_string())?;
            let bucket_load_started = Instant::now();
            match crate::exploration::load_mp_all_buckets(&source) {
                Ok(all_buckets) => self.all_buckets = Some(all_buckets),
                Err(error) => self.bucket_error = Some(error),
            }
            eprintln!(
                "[game_new] universal MP first bucket load completed in {:.3}s",
                bucket_load_started.elapsed().as_secs_f64()
            );
        }

        self.all_buckets.as_deref().ok_or_else(|| {
            self.bucket_error
                .clone()
                .unwrap_or_else(|| "Universal MP flop bucket source is unavailable".to_string())
        })
    }

    fn current_actions(&self) -> Vec<MpTreeAction> {
        self.spot.actions(&self.game)
    }

    fn position_label(&self, seat: u8) -> String {
        self.config
            .game
            .blinds
            .iter()
            .find(|blind| blind.seat == seat)
            .map(|blind| match blind.kind {
                poker_solver_core::blueprint_mp::config::ForcedBetKind::SmallBlind => "SB",
                poker_solver_core::blueprint_mp::config::ForcedBetKind::BigBlind => "BB",
                _ => "P",
            })
            .map(str::to_string)
            .unwrap_or_else(|| format!("P{}", seat + 1))
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn display_stacks(&self) -> [i32; 2] {
        let stacks = self.spot.stacks();
        let sb_seat = self
            .config
            .game
            .blinds
            .iter()
            .find(|blind| {
                blind.kind == poker_solver_core::blueprint_mp::config::ForcedBetKind::SmallBlind
            })
            .map_or(0, |blind| blind.seat);
        let bb_seat = self
            .config
            .game
            .blinds
            .iter()
            .find(|blind| {
                blind.kind == poker_solver_core::blueprint_mp::config::ForcedBetKind::BigBlind
            })
            .map_or(1, |blind| blind.seat);
        [
            stacks[bb_seat as usize].0 as i32,
            stacks[sb_seat as usize].0 as i32,
        ]
    }

    fn current_actions_at(&self, spot: LazyResolvedSpot) -> Vec<MpTreeAction> {
        spot.actions(&self.game)
    }

    fn probabilities_for_bucket(
        &self,
        spot: LazyResolvedSpot,
        bucket: u16,
        mp_actions: &[MpTreeAction],
    ) -> Result<Vec<f32>, String> {
        let key = spot.key_for_bucket(bucket);
        let lazy_key = MpLazyKey {
            seat: key.seat,
            street: spot.street() as u8,
            local_bucket: key.local_bucket(),
            history_hash: key.history_hash,
            history_len: key.history_len,
        };
        let Some(view) = self.bundle.query_mp_lazy(&lazy_key) else {
            return Err(format!(
                "Universal MP sparse row is missing: seat={}, street={}, local_bucket={}, history_hash={}, history_len={}",
                lazy_key.seat,
                lazy_key.street,
                lazy_key.local_bucket,
                lazy_key.history_hash,
                lazy_key.history_len
            ));
        };

        if view.actions.len() != mp_actions.len() || view.probs.len() != view.actions.len() {
            return Err(format!(
                "Universal MP sparse row schema mismatch: seat={}, street={}, local_bucket={}, history_hash={}, history_len={}, row_actions={}, row_probs={}, legal_actions={}",
                lazy_key.seat,
                lazy_key.street,
                lazy_key.local_bucket,
                lazy_key.history_hash,
                lazy_key.history_len,
                view.actions.len(),
                view.probs.len(),
                mp_actions.len()
            ));
        }

        let mut probabilities = vec![0.0; mp_actions.len()];
        let mut seen = vec![false; mp_actions.len()];
        for (row_index, descriptor) in view.actions.iter().enumerate() {
            let action_index = usize::from(descriptor.source_action_index);
            if action_index >= mp_actions.len() || seen[action_index] {
                return Err(format!(
                    "Universal MP sparse row action schema is incompatible: seat={}, street={}, local_bucket={}, history_hash={}, history_len={}, source_action_index={action_index}, legal_actions={}",
                    lazy_key.seat,
                    lazy_key.street,
                    lazy_key.local_bucket,
                    lazy_key.history_hash,
                    lazy_key.history_len,
                    mp_actions.len()
                ));
            }
            validate_mp_action_descriptor(descriptor, &mp_actions[action_index]).map_err(|error| {
                format!(
                    "Universal MP sparse row action schema mismatch at seat={}, street={}, local_bucket={}, history_hash={}, history_len={}, source_action_index={action_index}: {error}",
                    lazy_key.seat,
                    lazy_key.street,
                    lazy_key.local_bucket,
                    lazy_key.history_hash,
                    lazy_key.history_len
                )
            })?;
            seen[action_index] = true;
            probabilities[action_index] = view.probs[row_index];
        }
        if seen.iter().any(|seen| !seen) {
            return Err(format!(
                "Universal MP sparse row action schema omits a legal action: seat={}, street={}, local_bucket={}, history_hash={}, history_len={}",
                lazy_key.seat,
                lazy_key.street,
                lazy_key.local_bucket,
                lazy_key.history_hash,
                lazy_key.history_len
            ));
        }
        Ok(probabilities)
    }

    fn preflop_bucket_for_hand(&self, hand_index: usize) -> Result<u16, String> {
        let bucket_count = usize::from(self.config.clustering.preflop.buckets);
        if bucket_count == 0 {
            return Err("Universal MP config has zero preflop buckets".to_string());
        }
        Ok(if bucket_count == 169 {
            hand_index as u16
        } else {
            hand_index.min(bucket_count - 1) as u16
        })
    }

    /// Propagate each seat's root reach through the exact public action path.
    ///
    /// This intentionally keeps per-seat reach independent: an action only
    /// changes the reach of the seat that selected it. Opponent actions do not
    /// reduce the other seat's root reach.
    fn preflop_root_reaches(
        &self,
        action_history: &[ActionRecord],
    ) -> Result<[Vec<f32>; 2], String> {
        let mut reaches = [vec![1.0; 169], vec![1.0; 169]];
        let mut replay_spot = LazyResolvedSpot::root(&self.game);

        for record in action_history {
            if replay_spot.street() != MpStreet::Preflop {
                return Err(format!(
                    "Universal MP preflop reach replay reached {} before action {}",
                    mp_street_to_string(replay_spot.street()),
                    record.action_id
                ));
            }
            let mp_actions = replay_spot.actions(&self.game);
            let action_index = record.action_id.parse::<usize>().map_err(|_| {
                format!(
                    "Invalid MP action_id during reach replay: {}",
                    record.action_id
                )
            })?;
            if action_index >= mp_actions.len() {
                return Err(format!(
                    "MP action {action_index} out of range during reach replay (max {})",
                    mp_actions.len().saturating_sub(1)
                ));
            }
            let seat = usize::from(replay_spot.to_act().index());
            if seat >= reaches.len() {
                return Err(format!(
                    "Universal MP reach replay encountered unsupported seat {}",
                    replay_spot.to_act().index()
                ));
            }

            for hand_index in 0..169 {
                let bucket = self.preflop_bucket_for_hand(hand_index)?;
                let probabilities =
                    self.probabilities_for_bucket(replay_spot, bucket, &mp_actions)?;
                reaches[seat][hand_index] *= probabilities[action_index];
            }

            replay_spot = replay_spot
                .advance(&self.game, action_index)
                .ok_or_else(|| {
                    format!(
                        "Universal MP reach replay action {} terminated before the requested state",
                        record.action_id
                    )
                })?;
        }

        Ok(reaches)
    }

    /// Propagate concrete-combo reach through the public action path for a
    /// completed postflop board. Each action changes only the reach of its
    /// acting seat, while blockers are checked against the complete board.
    fn postflop_root_reaches(
        &mut self,
        action_history: &[ActionRecord],
        board: &[RsPokerCard],
    ) -> Result<[Vec<f32>; 2], String> {
        let mut reaches = [vec![1.0; 1326], vec![1.0; 1326]];
        let mut replay_spot = LazyResolvedSpot::root(&self.game);

        for record in action_history {
            let street = replay_spot.street();
            if !matches!(
                street,
                MpStreet::Preflop | MpStreet::Flop | MpStreet::Turn | MpStreet::River
            ) {
                return Err(format!(
                    "Universal MP postflop reach replay reached {} before action {}",
                    mp_street_to_string(street),
                    record.action_id
                ));
            }
            let mp_actions = replay_spot.actions(&self.game);
            let action_index = record.action_id.parse::<usize>().map_err(|_| {
                format!(
                    "Invalid MP action_id during flop reach replay: {}",
                    record.action_id
                )
            })?;
            if action_index >= mp_actions.len() {
                return Err(format!(
                    "MP action {action_index} out of range during flop reach replay (max {})",
                    mp_actions.len().saturating_sub(1)
                ));
            }
            let seat = usize::from(replay_spot.to_act().index());
            if seat >= reaches.len() {
                return Err(format!(
                    "Universal MP flop reach replay encountered unsupported seat {}",
                    replay_spot.to_act().index()
                ));
            }

            let mut probabilities_by_bucket = HashMap::new();
            for combo_index in 0..1326 {
                let (card1, card2) = index_to_card_pair(combo_index);
                let combo = [
                    crate::exploration::range_solver_to_rs_card(card1),
                    crate::exploration::range_solver_to_rs_card(card2),
                ];
                if combo_is_blocked(combo, board) {
                    reaches[seat][combo_index] = 0.0;
                    continue;
                }

                let bucket = if street == MpStreet::Preflop {
                    let hand = CanonicalHand::from_cards(combo[0], combo[1]);
                    self.preflop_bucket_for_hand(hand.index())?
                } else {
                    let visible_board_cards = mp_required_board_cards(street);
                    self.postflop_bucket(
                        replay_spot,
                        "concrete combo",
                        combo,
                        &board[..visible_board_cards],
                    )?
                };
                let probabilities = match probabilities_by_bucket.get(&bucket) {
                    Some(probabilities) => probabilities,
                    None => {
                        let probabilities =
                            self.probabilities_for_bucket(replay_spot, bucket, &mp_actions)?;
                        probabilities_by_bucket.insert(bucket, probabilities);
                        probabilities_by_bucket
                            .get(&bucket)
                            .expect("inserted MP bucket probabilities")
                    }
                };
                reaches[seat][combo_index] *= probabilities[action_index];
            }

            replay_spot = replay_spot
                .advance(&self.game, action_index)
                .ok_or_else(|| {
                    format!(
                        "Universal MP postflop reach replay action {} terminated before the requested state",
                        record.action_id
                    )
                })?;
        }

        Ok(reaches)
    }

    /// Capture the exact supported MP solve inputs at the current flop spot.
    ///
    /// Reaches are replayed from the raw sparse rows and remain in actual seat
    /// order. Betting metadata comes from the lazy core state, including the
    /// raw action that established a live bet; no display label is parsed.
    pub fn exact_solve_snapshot(&mut self) -> Result<UniversalMpSolveSnapshot, String> {
        if self.terminal {
            return Err(
                "Exact solve requires a non-terminal Universal MP flop decision".to_string(),
            );
        }
        if self.spot.street() != MpStreet::Flop || self.board.len() != 3 {
            return Err(
                "Exact solve is supported for UniversalMpLazy flop decisions only".to_string(),
            );
        }

        let board = mp_board_strings(&self.board);
        let action_history = self.action_history.clone();
        let raw_reaches_by_seat =
            self.postflop_root_reaches(&action_history, &self.board.clone())?;
        let actions =
            build_mp_game_actions(&self.current_actions(), mp_big_blind_amount(&self.config)?);
        let betting = self.spot.betting_snapshot();
        let (sb_seat, bb_seat) = mp_sb_bb_seats(&self.config)?;
        let big_blind = mp_big_blind_amount(&self.config)?;
        let root = mp_solve_root(&betting, sb_seat, bb_seat, big_blind)?;

        Ok(UniversalMpSolveSnapshot {
            board,
            raw_reaches_by_seat,
            acting_seat: betting.to_act.index(),
            pot: chips_to_i32(betting.pot),
            remaining_stacks: chips_array_to_i32(betting.stacks),
            street_bets: chips_array_to_i32(betting.street_bets),
            facing_bet: betting.facing_bet,
            raise_count: betting.raise_count,
            last_raise_to: chips_to_i32(betting.last_raise_to),
            last_aggressive_action: betting.last_aggressive_action,
            oop_seat: bb_seat,
            ip_seat: sb_seat,
            root,
            bet_sizes: mp_flop_bet_sizes(&self.config)?,
            action_history,
            actions,
        })
    }

    fn postflop_bucket(
        &mut self,
        spot: LazyResolvedSpot,
        hand: &str,
        hole_cards: [RsPokerCard; 2],
        board: &[RsPokerCard],
    ) -> Result<u16, String> {
        let street = spot.street();
        if street == MpStreet::Preflop {
            return Err("Universal MP postflop bucket lookup received a preflop spot".to_string());
        }
        let street_name = mp_street_to_string(street).to_ascii_lowercase();
        let all_buckets = self.ensure_all_buckets()?;
        let bucket_street = match street {
            MpStreet::Flop => Street::Flop,
            MpStreet::Turn => Street::Turn,
            MpStreet::River => Street::River,
            MpStreet::Preflop => unreachable!("preflop was rejected above"),
        };
        all_buckets
            .try_get_bucket(bucket_street, hole_cards, board)
            .map_err(|error| {
                format!(
                    "Universal MP {street_name} bucket lookup failed: seat={}, street={street_name}, hand={}, board={:?}, hole_cards={:?}: {error}",
                    spot.to_act().index(),
                    hand,
                    board,
                    hole_cards
                )
            })
    }

    fn build_postflop_matrix(
        &mut self,
        spot: LazyResolvedSpot,
        board: &[RsPokerCard],
        mp_actions: &[MpTreeAction],
        action_history: &[ActionRecord],
    ) -> Result<GameMatrix, String> {
        let required_board_cards = mp_required_board_cards(spot.street());
        if !matches!(
            spot.street(),
            MpStreet::Flop | MpStreet::Turn | MpStreet::River
        ) {
            return Err(format!(
                "Universal MP postflop matrix requires a postflop decision, got {}",
                mp_street_to_string(spot.street())
            ));
        }
        if board.len() != required_board_cards {
            return Err(format!(
                "Universal MP {} matrix requires exactly {} board cards, got {}",
                mp_street_to_string(spot.street()).to_ascii_lowercase(),
                required_board_cards,
                board.len(),
            ));
        }
        let actions = build_mp_game_actions(mp_actions, mp_big_blind_amount(&self.config)?);
        let reaches = self.postflop_root_reaches(action_history, board)?;
        let acting_seat = usize::from(spot.to_act().index());
        if acting_seat >= reaches.len() {
            return Err(format!(
                "Universal MP postflop matrix encountered unsupported seat {}",
                spot.to_act().index()
            ));
        }
        let mut cells = Vec::with_capacity(13);
        for (row, &rank1) in RANKS.iter().enumerate() {
            let mut row_cells = Vec::with_capacity(13);
            for (col, &rank2) in RANKS.iter().enumerate() {
                let (label, suited, pair) = hand_label_from_matrix(row, col, rank1, rank2);
                let hand = CanonicalHand::new(mp_rank_value(rank1), mp_rank_value(rank2), suited);
                let mut sum_probabilities = vec![0.0; actions.len()];
                let mut sum_reach = 0.0;
                let mut combos = Vec::new();
                let mut probabilities_by_bucket = HashMap::new();
                for (card1, card2) in hand.combos() {
                    let combo = [card1, card2];
                    if combo_is_blocked(combo, board) {
                        continue;
                    }
                    let bucket = self.postflop_bucket(spot, &label, [card1, card2], board)?;
                    let probabilities = match probabilities_by_bucket.get(&bucket) {
                        Some(probabilities) => probabilities,
                        None => {
                            let probabilities =
                                self.probabilities_for_bucket(spot, bucket, mp_actions)?;
                            probabilities_by_bucket.insert(bucket, probabilities);
                            probabilities_by_bucket
                                .get(&bucket)
                                .expect("inserted MP bucket probabilities")
                        }
                    };
                    let combo_index = card_pair_to_index(
                        crate::exploration::rs_card_to_range_solver(card1),
                        crate::exploration::rs_card_to_range_solver(card2),
                    );
                    let combo_reach = reaches[acting_seat][combo_index];
                    sum_reach += combo_reach;
                    for (sum, probability) in sum_probabilities.iter_mut().zip(probabilities.iter())
                    {
                        *sum += combo_reach * *probability;
                    }
                    let cards = format!(
                        "{}{}",
                        card_to_string(crate::exploration::rs_card_to_range_solver(card1))
                            .unwrap_or_default(),
                        card_to_string(crate::exploration::rs_card_to_range_solver(card2))
                            .unwrap_or_default()
                    );
                    combos.push(ComboDetail {
                        cards,
                        probabilities: probabilities
                            .iter()
                            .map(|probability| combo_reach * *probability)
                            .collect(),
                        weight: combo_reach,
                        bucket: Some(bucket),
                    });
                }
                let combo_count = combos.len();
                let probabilities = if combo_count == 0 {
                    vec![0.0; actions.len()]
                } else {
                    sum_probabilities
                        .into_iter()
                        .map(|probability| probability / combo_count as f32)
                        .collect()
                };
                row_cells.push(GameMatrixCell {
                    hand: label,
                    suited,
                    pair,
                    probabilities,
                    combo_count,
                    weight: if combo_count == 0 {
                        0.0
                    } else {
                        sum_reach / combo_count as f32
                    },
                    ev: None,
                    combos,
                });
            }
            cells.push(row_cells);
        }
        Ok(GameMatrix { cells, actions })
    }

    #[allow(clippy::cast_possible_truncation)]
    fn state_at(
        &mut self,
        spot: LazyResolvedSpot,
        board: &[RsPokerCard],
        action_history: &[ActionRecord],
        terminal: bool,
    ) -> Result<GameState, String> {
        let stacks = self.display_stacks_at(spot);
        let board_strings = mp_board_strings(board);
        let street = mp_street_to_string(spot.street());
        if terminal {
            return Ok(GameState {
                street,
                position: String::new(),
                board: board_strings,
                pot: spot.pot().0 as i32,
                stacks,
                matrix: None,
                actions: vec![],
                action_history: action_history.to_vec(),
                is_terminal: true,
                is_chance: false,
                solve: None,
            });
        }

        let required_board_cards = mp_required_board_cards(spot.street());
        if required_board_cards > board.len() {
            return Ok(GameState {
                street,
                position: String::new(),
                board: board_strings,
                pot: spot.pot().0 as i32,
                stacks,
                matrix: None,
                actions: vec![],
                action_history: action_history.to_vec(),
                is_terminal: false,
                is_chance: true,
                solve: None,
            });
        }

        let mp_actions = self.current_actions_at(spot);
        let actions = build_mp_game_actions(&mp_actions, mp_big_blind_amount(&self.config)?);
        if matches!(
            spot.street(),
            MpStreet::Flop | MpStreet::Turn | MpStreet::River
        ) {
            let matrix = self.build_postflop_matrix(spot, board, &mp_actions, action_history)?;
            return Ok(GameState {
                street,
                position: self.position_label(spot.to_act().index()),
                board: board_strings,
                pot: spot.pot().0 as i32,
                stacks,
                matrix: Some(matrix),
                actions,
                action_history: action_history.to_vec(),
                is_terminal: false,
                is_chance: false,
                solve: None,
            });
        }
        debug_assert_eq!(spot.street(), MpStreet::Preflop);

        let reaches = self.preflop_root_reaches(action_history)?;
        let acting_seat = usize::from(spot.to_act().index());
        let mut cells = Vec::with_capacity(13);
        for (row, &rank1) in RANKS.iter().enumerate() {
            let mut row_cells = Vec::with_capacity(13);
            for (col, &rank2) in RANKS.iter().enumerate() {
                let (label, suited, pair) = hand_label_from_matrix(row, col, rank1, rank2);
                let hand_index = canonical_hand_index_from_ranks(rank1, rank2, suited);
                let bucket = self.preflop_bucket_for_hand(hand_index)?;
                let reach = reaches[acting_seat][hand_index];
                let conditional_probabilities =
                    self.probabilities_for_bucket(spot, bucket, &mp_actions)?;
                let probabilities = conditional_probabilities
                    .into_iter()
                    .map(|probability| reach * probability)
                    .collect();
                row_cells.push(GameMatrixCell {
                    hand: label,
                    suited,
                    pair,
                    probabilities,
                    combo_count: 0,
                    weight: reach,
                    ev: None,
                    combos: vec![],
                });
            }
            cells.push(row_cells);
        }

        Ok(GameState {
            street,
            position: self.position_label(spot.to_act().index()),
            board: board_strings,
            pot: spot.pot().0 as i32,
            stacks,
            matrix: Some(GameMatrix {
                cells,
                actions: actions.clone(),
            }),
            actions,
            action_history: action_history.to_vec(),
            is_terminal: false,
            is_chance: false,
            solve: None,
        })
    }

    fn display_stacks_at(&self, spot: LazyResolvedSpot) -> [i32; 2] {
        let stacks = spot.stacks();
        let sb_seat = self
            .config
            .game
            .blinds
            .iter()
            .find(|blind| {
                blind.kind == poker_solver_core::blueprint_mp::config::ForcedBetKind::SmallBlind
            })
            .map_or(0, |blind| blind.seat);
        let bb_seat = self
            .config
            .game
            .blinds
            .iter()
            .find(|blind| {
                blind.kind == poker_solver_core::blueprint_mp::config::ForcedBetKind::BigBlind
            })
            .map_or(1, |blind| blind.seat);
        [
            stacks[bb_seat as usize].0 as i32,
            stacks[sb_seat as usize].0 as i32,
        ]
    }

    fn get_state(&mut self) -> Result<GameState, String> {
        let board = self.board.clone();
        let action_history = self.action_history.clone();
        self.state_at(self.spot, &board, &action_history, self.terminal)
    }

    fn state_shell(&self) -> GameState {
        let is_chance =
            !self.terminal && mp_required_board_cards(self.spot.street()) > self.board.len();
        GameState {
            street: mp_street_to_string(self.spot.street()),
            position: if self.terminal || is_chance {
                String::new()
            } else {
                self.position_label(self.spot.to_act().index())
            },
            board: mp_board_strings(&self.board),
            pot: self.spot.pot().0 as i32,
            stacks: self.display_stacks(),
            matrix: None,
            actions: vec![],
            action_history: self.action_history.clone(),
            is_terminal: self.terminal,
            is_chance,
            solve: None,
        }
    }

    fn action_transition(
        &self,
        action_id: &str,
    ) -> Result<(ActionRecord, LazyResolvedSpot, bool), String> {
        if self.terminal {
            return Err("Universal MP session is already terminal".to_string());
        }
        if mp_required_board_cards(self.spot.street()) > self.board.len() {
            return Err(format!(
                "Universal MP session is at a {} chance boundary; deal the remaining board cards before playing an action",
                mp_street_to_string(self.spot.street())
            ));
        }
        let action_index = action_id
            .parse::<usize>()
            .map_err(|_| format!("Invalid MP action_id: {action_id}"))?;
        let mp_actions = self.current_actions();
        let actions = build_mp_game_actions(&mp_actions, mp_big_blind_amount(&self.config)?);
        if action_index >= mp_actions.len() {
            return Err(format!(
                "MP action {action_index} out of range (max {})",
                mp_actions.len().saturating_sub(1)
            ));
        }
        let stacks = self.display_stacks();
        let record = ActionRecord {
            action_id: action_id.to_string(),
            label: actions[action_index].label.clone(),
            position: self.position_label(self.spot.to_act().index()),
            street: mp_street_to_string(self.spot.street()),
            pot: self.spot.pot().0 as i32,
            stack: stacks[if self.spot.to_act().index() == 1 {
                0
            } else {
                1
            }],
            actions,
        };
        let next_spot = self.spot.advance(&self.game, action_index);
        let next_terminal = next_spot.is_none();
        Ok((record, next_spot.unwrap_or(self.spot), next_terminal))
    }

    fn commit_action_transition(
        &mut self,
        record: ActionRecord,
        next_spot: LazyResolvedSpot,
        next_terminal: bool,
    ) {
        self.spot = next_spot;
        self.action_history.push(record);
        self.terminal = next_terminal;
    }

    fn play_action(&mut self, action_id: &str) -> Result<GameState, String> {
        let (record, next_spot, next_terminal) = self.action_transition(action_id)?;
        let mut next_history = self.action_history.clone();
        next_history.push(record.clone());
        let board = self.board.clone();
        let state = self.state_at(next_spot, &board, &next_history, next_terminal)?;
        self.commit_action_transition(record, next_spot, next_terminal);
        Ok(state)
    }

    fn play_action_without_state(&mut self, action_id: &str) -> Result<GameState, String> {
        let (record, next_spot, next_terminal) = self.action_transition(action_id)?;
        self.commit_action_transition(record, next_spot, next_terminal);
        Ok(self.state_shell())
    }

    fn deal_card(&mut self, card: &str) -> Result<GameState, String> {
        if self.terminal {
            return Err("Universal MP session is already terminal".to_string());
        }
        let required_board_cards = mp_required_board_cards(self.spot.street());
        let valid_chance = match self.spot.street() {
            MpStreet::Flop => self.board.len() < required_board_cards,
            MpStreet::Turn | MpStreet::River => self.board.len() + 1 == required_board_cards,
            MpStreet::Preflop => false,
        };
        if !valid_chance {
            return Err(format!(
                "Universal MP session is not at a supported board chance state; street={}, board_cards={}, state unchanged",
                mp_street_to_string(self.spot.street()),
                self.board.len()
            ));
        }
        let parsed = parse_rs_poker_card(card)?;
        if self.board.contains(&parsed) {
            return Err(format!(
                "Duplicate board card {card} is illegal; board unchanged"
            ));
        }
        let mut next_board = self.board.clone();
        next_board.push(parsed);
        let action_history = self.action_history.clone();
        let state = self.state_at(self.spot, &next_board, &action_history, false)?;
        self.board = next_board;
        Ok(state)
    }

    fn back(&mut self) -> Result<GameState, String> {
        if self.action_history.is_empty() {
            return Err("No actions to undo".to_string());
        }
        let last = self.action_history.last().cloned().unwrap();
        let target_history = self.action_history[..self.action_history.len() - 1].to_vec();
        let mut target_spot = LazyResolvedSpot::root(&self.game);
        let mut target_terminal = false;
        for record in &target_history {
            let action_index = record
                .action_id
                .parse::<usize>()
                .map_err(|_| format!("Invalid MP action_id during back: {}", record.action_id))?;
            target_spot = match target_spot.advance(&self.game, action_index) {
                Some(next) => next,
                None => {
                    target_terminal = true;
                    break;
                }
            };
        }
        let undone_street = match last.street.as_str() {
            "Preflop" => MpStreet::Preflop,
            "Flop" => MpStreet::Flop,
            "Turn" => MpStreet::Turn,
            "River" => MpStreet::River,
            street => return Err(format!("Invalid MP street during back: {street}")),
        };
        let target_board_len = self.board.len().min(mp_required_board_cards(undone_street));
        let target_board = self.board[..target_board_len].to_vec();
        let state = self.state_at(target_spot, &target_board, &target_history, target_terminal)?;
        self.spot = target_spot;
        self.board = target_board;
        self.action_history = target_history;
        self.terminal = target_terminal;
        Ok(state)
    }
}

fn mp_street_to_string(street: MpStreet) -> String {
    match street {
        MpStreet::Preflop => "Preflop".to_string(),
        MpStreet::Flop => "Flop".to_string(),
        MpStreet::Turn => "Turn".to_string(),
        MpStreet::River => "River".to_string(),
    }
}

const fn mp_required_board_cards(street: MpStreet) -> usize {
    match street {
        MpStreet::Preflop => 0,
        MpStreet::Flop => 3,
        MpStreet::Turn => 4,
        MpStreet::River => 5,
    }
}

fn mp_board_strings(board: &[RsPokerCard]) -> Vec<String> {
    board
        .iter()
        .map(|card| {
            card_to_string(crate::exploration::rs_card_to_range_solver(*card))
                .unwrap_or_else(|_| "??".to_string())
        })
        .collect()
}

fn combo_is_blocked(combo: [RsPokerCard; 2], board: &[RsPokerCard]) -> bool {
    board
        .iter()
        .any(|card| *card == combo[0] || *card == combo[1])
}

fn mp_rank_value(rank: char) -> RsPokerValue {
    match rank {
        'A' => RsPokerValue::Ace,
        'K' => RsPokerValue::King,
        'Q' => RsPokerValue::Queen,
        'J' => RsPokerValue::Jack,
        'T' => RsPokerValue::Ten,
        '9' => RsPokerValue::Nine,
        '8' => RsPokerValue::Eight,
        '7' => RsPokerValue::Seven,
        '6' => RsPokerValue::Six,
        '5' => RsPokerValue::Five,
        '4' => RsPokerValue::Four,
        '3' => RsPokerValue::Three,
        '2' => RsPokerValue::Two,
        _ => unreachable!("RANKS contains only legal ranks"),
    }
}

fn expected_mp_action_kind(action: &MpTreeAction) -> ActionKind {
    match action {
        MpTreeAction::Fold => ActionKind::Fold,
        MpTreeAction::Check => ActionKind::Check,
        MpTreeAction::Call => ActionKind::Call,
        MpTreeAction::Lead(_) => ActionKind::Bet,
        MpTreeAction::Raise(_) => ActionKind::Raise,
        MpTreeAction::AllIn => ActionKind::AllInBetRaise,
    }
}

fn validate_mp_action_descriptor(
    descriptor: &ActionDescriptor,
    action: &MpTreeAction,
) -> Result<(), String> {
    if descriptor.kind == ActionKind::Opaque {
        return Ok(());
    }
    let kind_matches = match action {
        MpTreeAction::AllIn => matches!(
            descriptor.kind,
            ActionKind::AllInCall | ActionKind::AllInBetRaise
        ),
        _ => descriptor.kind == expected_mp_action_kind(action),
    };
    if !kind_matches {
        return Err(format!(
            "row kind {:?} does not match legal action {:?}",
            descriptor.kind, action
        ));
    }
    let expected_amount = match action {
        MpTreeAction::Lead(amount) | MpTreeAction::Raise(amount) => {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            {
                Some(amount.round() as u32)
            }
        }
        _ => None,
    };
    if let Some(expected_amount) = expected_amount {
        if descriptor.amount_chips != expected_amount {
            return Err(format!(
                "row amount {} does not match legal amount {}",
                descriptor.amount_chips, expected_amount
            ));
        }
    }
    Ok(())
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
                    stacks: [stacks[bb_idx] as i32, stacks[sb_idx] as i32],
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
                    Some(self.build_matrix(decision_idx as usize, *player, *street, &game_actions))
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
                let (weight_sum, weight_count) =
                    combo_indices
                        .iter()
                        .fold((0.0f32, 0usize), |(sum, count), &ci| {
                            // For postflop, filter out board-blocked combos.
                            (sum + self.weights[weight_idx][ci], count + 1)
                        });
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
                    combo_indices,
                    weight_idx,
                    decision_idx,
                    street,
                    actions.len(),
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
                let (c1, c2) = if c1_raw / 4 >= c2_raw / 4 {
                    (c1_raw, c2_raw)
                } else {
                    (c2_raw, c1_raw)
                };
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
                    let bucket = ctx
                        .all_buckets
                        .get_bucket(street, [rs_c1, rs_c2], board_slice);
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
                    0..=2 => 3, // flop needs 3 total
                    3 => 4,     // turn needs 4 total
                    4 => 5,     // river needs 5 total
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

    fn values_by_range_slot(&self, values_by_v2_player: [f64; 2]) -> [i32; 2] {
        let bb_player = (1 - self.tree.dealer) as usize;
        let sb_player = self.tree.dealer as usize;
        [
            values_by_v2_player[bb_player].round() as i32,
            values_by_v2_player[sb_player].round() as i32,
        ]
    }

    fn approximate_stacks_from_pot(&self) -> [i32; 2] {
        let pot = pot_at_v2_node(&self.tree, self.node_idx);
        let stack_depth = self.config.game.stack_depth;
        let each_invested = pot / 2.0;
        let remaining = (stack_depth - each_invested) as i32;
        [remaining, remaining]
    }

    /// Compute remaining stacks [BB, SB] (in chips).
    #[allow(clippy::cast_possible_truncation)]
    fn compute_stacks(&self) -> [i32; 2] {
        self.replay_betting_state()
            .map(|state| self.values_by_range_slot(state.stacks))
            .unwrap_or_else(|_| self.approximate_stacks_from_pot())
    }

    fn replay_betting_state(&self) -> Result<BettingState, String> {
        let sb = self.tree.dealer as usize;
        let bb = (1 - self.tree.dealer) as usize;
        let mut stacks = [self.config.game.stack_depth; 2];
        stacks[sb] -= self.config.game.small_blind;
        stacks[bb] -= self.config.game.big_blind;
        let mut street_bets = [0.0; 2];
        street_bets[sb] = self.config.game.small_blind;
        street_bets[bb] = self.config.game.big_blind;
        let mut pot = self.config.game.small_blind + self.config.game.big_blind;
        let mut node_idx = self.tree.root;
        let mut num_bets = 0u8;
        let mut last_aggressive_action = None;

        for record in &self.action_history {
            while let V2GameNode::Chance { child, .. } = &self.tree.nodes[node_idx as usize] {
                node_idx = *child;
                street_bets = [0.0; 2];
                num_bets = 0;
                last_aggressive_action = None;
            }

            let (player, actions, children) = match &self.tree.nodes[node_idx as usize] {
                V2GameNode::Decision {
                    player,
                    actions,
                    children,
                    ..
                } => (*player as usize, actions, children),
                _ => {
                    return Err(format!(
                        "Action history reached non-decision node {node_idx}"
                    ));
                }
            };

            let action_idx: usize = record
                .action_id
                .parse()
                .map_err(|_| format!("Invalid action_id in history: {}", record.action_id))?;
            let action = actions
                .get(action_idx)
                .ok_or_else(|| format!("Action {action_idx} out of range at node {node_idx}"))?;

            let opponent = player ^ 1;
            match action {
                TreeAction::Fold | TreeAction::Check => {}
                TreeAction::Call => {
                    let to_call = (street_bets[opponent] - street_bets[player]).max(0.0);
                    let actual = to_call.min(stacks[player]);
                    stacks[player] -= actual;
                    street_bets[player] += actual;
                    pot += actual;
                }
                TreeAction::Bet(amount) | TreeAction::Raise(amount) => {
                    let additional = (*amount - street_bets[player]).max(0.0).min(stacks[player]);
                    stacks[player] -= additional;
                    street_bets[player] += additional;
                    pot += additional;
                    num_bets = num_bets.saturating_add(1);
                    last_aggressive_action = Some(*action);
                }
                TreeAction::AllIn => {
                    let additional = stacks[player];
                    stacks[player] = 0.0;
                    street_bets[player] += additional;
                    pot += additional;
                    num_bets = num_bets.saturating_add(1);
                    last_aggressive_action = Some(*action);
                }
            }

            node_idx = *children
                .get(action_idx)
                .ok_or_else(|| format!("Action {action_idx} has no child at node {node_idx}"))?;
        }

        while let V2GameNode::Chance { child, .. } = &self.tree.nodes[node_idx as usize] {
            node_idx = *child;
            street_bets = [0.0; 2];
            num_bets = 0;
            last_aggressive_action = None;
        }

        Ok(BettingState {
            pot,
            stacks,
            street_bets,
            num_bets,
            last_aggressive_action,
        })
    }

    fn solve_game_root_for_player(&self, v2_player: u8) -> Result<SolveGameRoot, String> {
        let state = self.replay_betting_state()?;
        let initial_player = self.weight_index(v2_player) as u8;
        let initial_stacks = self.values_by_range_slot(state.stacks);
        let street_bets = self.values_by_range_slot(state.street_bets);
        let actor = initial_player as usize;
        let opponent = actor ^ 1;
        let to_call = (street_bets[opponent] - street_bets[actor]).max(0);
        let matched_amount = street_bets[0].min(street_bets[1]);
        let current_pot = self.compute_pot();
        debug_assert!((state.pot - f64::from(current_pot)).abs() < 1.0);
        let street_start_pot = (current_pot - street_bets[0] - street_bets[1]).max(1);

        let (initial_prev_action, initial_prev_amount, initial_num_bets) = if to_call > 0 {
            let prev_amount = street_bets[opponent];
            let prev_action = match state.last_aggressive_action {
                Some(TreeAction::AllIn) => range_solver::Action::AllIn(prev_amount),
                Some(TreeAction::Raise(_)) if state.num_bets > 1 => {
                    range_solver::Action::Raise(prev_amount)
                }
                _ => range_solver::Action::Bet(prev_amount),
            };
            (prev_action, prev_amount, i32::from(state.num_bets.max(1)))
        } else {
            (range_solver::Action::None, 0, 0)
        };

        Ok(SolveGameRoot {
            starting_pot: street_start_pot,
            initial_player,
            initial_stacks,
            initial_prev_action,
            initial_prev_amount,
            initial_amount: matched_amount,
            initial_num_bets,
        })
    }

    pub fn solve_game_root(&self) -> Result<SolveGameRoot, String> {
        let node = &self.tree.nodes[self.node_idx as usize];
        let player = match node {
            V2GameNode::Decision { player, .. } => *player,
            _ => return Err("Not at a decision node".to_string()),
        };
        self.solve_game_root_for_player(player)
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
                        format!("Invalid action format: '{action_str}'. Expected 'position:label'")
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
                                .map(|a| format!("{}:{}", position, a.label.to_lowercase()))
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
/// Resolve a relative path against the Cargo workspace root so outputs
/// land in the project's real directory regardless of the launcher's CWD
/// (Tauri desktop apps and `cargo run` from crate subdirs both differ from
/// the workspace root). Walks parent dirs looking for a `Cargo.toml` that
/// declares `[workspace]`; falls back to joining against the current CWD.
fn resolve_against_workspace_root(relative: &std::path::Path) -> std::path::PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let mut cursor: &std::path::Path = &cwd;
    loop {
        let candidate = cursor.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(content) = std::fs::read_to_string(&candidate) {
                if content.contains("[workspace]") {
                    return cursor.join(relative);
                }
            }
        }
        match cursor.parent() {
            Some(parent) => cursor = parent,
            None => break,
        }
    }
    cwd.join(relative)
}

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

fn mp_big_blind_amount(config: &BlueprintMpConfig) -> Result<f64, String> {
    config
        .game
        .blinds
        .iter()
        .find(|blind| blind.kind == ForcedBetKind::BigBlind)
        .map(|blind| blind.amount)
        .filter(|amount| *amount > 0.0)
        .ok_or_else(|| "Universal MP config must define a positive BigBlind forced bet".to_string())
}

fn mp_sb_bb_seats(config: &BlueprintMpConfig) -> Result<(u8, u8), String> {
    let sb = config
        .game
        .blinds
        .iter()
        .find(|blind| blind.kind == ForcedBetKind::SmallBlind)
        .map(|blind| blind.seat)
        .ok_or_else(|| "Universal MP config must define a SmallBlind forced bet".to_string())?;
    let bb = config
        .game
        .blinds
        .iter()
        .find(|blind| blind.kind == ForcedBetKind::BigBlind)
        .map(|blind| blind.seat)
        .ok_or_else(|| "Universal MP config must define a BigBlind forced bet".to_string())?;
    if sb == bb || sb >= 2 || bb >= 2 {
        return Err(format!(
            "Universal MP exact solve requires distinct SB/BB seats among 0..2; got SB={sb}, BB={bb}"
        ));
    }
    Ok((sb, bb))
}

fn chips_to_i32(chips: poker_solver_core::blueprint_mp::Chips) -> i32 {
    chips.0.round() as i32
}

/// Convert raw MP chip amounts to the integer units used by range-solver.
///
/// Range-solver has no blind-size field: its integer amounts are raw chip
/// units. The configured BB belongs at the display and semantic-matching
/// boundaries, not in this conversion. Keep the validated BB parameter here
/// so every MP root conversion shares the same config contract.
fn mp_chips_to_solver_units(amount: f64, big_blind: f64) -> i32 {
    debug_assert!(big_blind > 0.0);
    amount.round() as i32
}

fn chips_array_to_i32(
    chips: [poker_solver_core::blueprint_mp::Chips; poker_solver_core::blueprint_mp::MAX_PLAYERS],
) -> [i32; 2] {
    [chips_to_i32(chips[0]), chips_to_i32(chips[1])]
}

fn mp_flop_bet_sizes(config: &BlueprintMpConfig) -> Result<Vec<Vec<f64>>, String> {
    let parse_sizes = |values: &[serde_yaml::Value], label: &str| {
        values
            .iter()
            .map(|value| {
                value.as_f64().or_else(|| {
                    value
                        .as_str()
                        .and_then(|value| value.trim().parse::<f64>().ok())
                })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("Universal MP {label} size is not numeric"))
    };
    let flop = &config.action_abstraction.flop;
    let lead = parse_sizes(&flop.lead, "flop lead")?;
    let raise = flop.raise.first().map_or_else(
        || Ok(lead.clone()),
        |sizes| parse_sizes(sizes, "flop raise"),
    )?;
    Ok(vec![lead, raise])
}

fn mp_solve_root(
    betting: &LazyBettingSnapshot,
    sb_seat: u8,
    bb_seat: u8,
    big_blind: f64,
) -> Result<SolveGameRoot, String> {
    let actual_actor = betting.to_act.index();
    let initial_player = if actual_actor == bb_seat {
        0
    } else if actual_actor == sb_seat {
        1
    } else {
        return Err(format!(
            "Universal MP exact solve encountered unsupported acting seat {actual_actor}"
        ));
    };
    let initial_stacks = [
        mp_chips_to_solver_units(betting.stacks[bb_seat as usize].0, big_blind),
        mp_chips_to_solver_units(betting.stacks[sb_seat as usize].0, big_blind),
    ];
    let street_bets = [
        mp_chips_to_solver_units(betting.street_bets[bb_seat as usize].0, big_blind),
        mp_chips_to_solver_units(betting.street_bets[sb_seat as usize].0, big_blind),
    ];
    let starting_pot =
        (mp_chips_to_solver_units(betting.pot.0, big_blind) - street_bets[0] - street_bets[1])
            .max(1);
    let matched_amount = street_bets[0].min(street_bets[1]);

    let (initial_prev_action, initial_prev_amount, initial_num_bets) = if betting.facing_bet {
        let Some(action) = betting.last_aggressive_action else {
            return Err(
                "Universal MP exact solve is missing raw aggressive action metadata".to_string(),
            );
        };
        let amount = mp_chips_to_solver_units(betting.last_raise_to.0, big_blind);
        let previous = match action {
            MpTreeAction::Lead(_) => range_solver::Action::Bet(amount),
            MpTreeAction::Raise(_) => range_solver::Action::Raise(amount),
            MpTreeAction::AllIn => range_solver::Action::AllIn(amount),
            MpTreeAction::Fold | MpTreeAction::Check | MpTreeAction::Call => {
                return Err(format!(
                    "Universal MP exact solve received non-aggressive raw action {action:?} while facing a bet"
                ));
            }
        };
        (previous, amount, i32::from(betting.raise_count.max(1)))
    } else {
        (range_solver::Action::None, 0, 0)
    };

    Ok(SolveGameRoot {
        starting_pot,
        initial_player,
        initial_stacks,
        initial_prev_action,
        initial_prev_amount,
        initial_amount: matched_amount,
        initial_num_bets,
    })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn build_mp_game_actions(actions: &[MpTreeAction], big_blind: f64) -> Vec<GameAction> {
    actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            let (label, action_type) = match action {
                MpTreeAction::Fold => ("Fold".to_string(), "fold".to_string()),
                MpTreeAction::Check => ("Check".to_string(), "check".to_string()),
                MpTreeAction::Call => ("Call".to_string(), "call".to_string()),
                MpTreeAction::Lead(amount) => {
                    (format_mp_amount(*amount, big_blind), "bet".to_string())
                }
                MpTreeAction::Raise(amount) => {
                    (format_mp_amount(*amount, big_blind), "raise".to_string())
                }
                MpTreeAction::AllIn => ("All-in".to_string(), "allin".to_string()),
            };
            GameAction {
                id: index.to_string(),
                label,
                action_type,
            }
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn format_mp_amount(amount: f64, big_blind: f64) -> String {
    let bb = if big_blind > 0.0 {
        amount / big_blind
    } else {
        amount
    };
    if (bb - bb.round()).abs() < 0.05 {
        format!("{:.0}bb", bb.round())
    } else {
        format!("{bb:.1}bb")
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct SemanticAction {
    action_type: String,
    amount_bb: Option<i32>,
}

fn action_amount_bb_from_label(label: &str) -> Option<i32> {
    let normalized = label.trim().to_ascii_lowercase().replace(' ', "");
    let bb = normalized.strip_suffix("bb")?;
    bb.parse::<f64>().ok().map(|amount| amount.round() as i32)
}

fn semantic_action_from_tree_action(action: &TreeAction) -> SemanticAction {
    let amount_bb = match action {
        TreeAction::Bet(amount) | TreeAction::Raise(amount) => Some((amount / 2.0).round() as i32),
        _ => None,
    };

    SemanticAction {
        action_type: action_type_string(action),
        amount_bb,
    }
}

fn semantic_action_from_mp_tree_action(action: &MpTreeAction, big_blind: f64) -> SemanticAction {
    let (action_type, amount_bb) = match action {
        MpTreeAction::Fold => ("fold", None),
        MpTreeAction::Check => ("check", None),
        MpTreeAction::Call => ("call", None),
        MpTreeAction::Lead(amount) => ("bet", Some(mp_amount_to_bb(*amount, big_blind))),
        MpTreeAction::Raise(amount) => ("raise", Some(mp_amount_to_bb(*amount, big_blind))),
        MpTreeAction::AllIn => ("allin", None),
    };
    SemanticAction {
        action_type: action_type.to_string(),
        amount_bb,
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn mp_amount_to_bb(amount: f64, big_blind: f64) -> i32 {
    if big_blind > 0.0 {
        (amount / big_blind).round() as i32
    } else {
        amount.round() as i32
    }
}

fn semantic_action_from_game_action(action: &GameAction) -> SemanticAction {
    let action_type = action.action_type.to_ascii_lowercase();
    let amount_bb = match action_type.as_str() {
        "bet" | "raise" => action_amount_bb_from_label(&action.label),
        _ => None,
    };

    SemanticAction {
        action_type,
        amount_bb,
    }
}

fn semantic_action_from_record(record: &ActionRecord) -> Option<SemanticAction> {
    record
        .actions
        .iter()
        .find(|action| action.id == record.action_id)
        .map(semantic_action_from_game_action)
}

fn semantic_actions_match(left: &SemanticAction, right: &SemanticAction) -> bool {
    if left.action_type != right.action_type {
        return false;
    }

    match left.action_type.as_str() {
        "fold" | "check" | "call" | "allin" => true,
        "bet" | "raise" => left.amount_bb.is_some() && left.amount_bb == right.amount_bb,
        _ => false,
    }
}

fn session_action_id_matching_cached_action(
    session: &GameSession,
    cached_action: &GameAction,
) -> Result<String, String> {
    let cached_semantic = semantic_action_from_game_action(cached_action);
    let V2GameNode::Decision { actions, .. } = &session.tree.nodes[session.node_idx as usize]
    else {
        return Err("Cannot map solver action: session is not at a decision node".to_string());
    };

    actions
        .iter()
        .enumerate()
        .find_map(|(idx, action)| {
            let session_semantic = semantic_action_from_tree_action(action);
            semantic_actions_match(&cached_semantic, &session_semantic).then(|| idx.to_string())
        })
        .ok_or_else(|| {
            format!(
                "Solver action '{}' ({}) does not match any action at the current session node",
                cached_action.label, cached_action.action_type
            )
        })
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

/// Build the `CardConfig` and `ActionTree` for a postflop solve without
/// constructing the `PostFlopGame`. Useful when the caller needs to pass
/// these to `make_per_boundary_gadget_game` or other paths that consume
/// configs directly rather than building via `PostFlopGame::with_config`.
///
/// When `exact` is true, `depth_limit` is set to `None` so the game tree
/// extends through all remaining streets to showdown (no boundary nodes).
#[allow(clippy::too_many_arguments)]
pub fn build_solve_game_parts(
    board: &[String],
    oop_weights: &[f32],
    ip_weights: &[f32],
    pot: i32,
    effective_stack: i32,
    bet_sizes: &[Vec<f64>],
    exact: bool,
    depth_limit_override: Option<u8>,
) -> Result<(range_solver::card::CardConfig, range_solver::ActionTree), String> {
    build_solve_game_parts_with_root(
        board,
        oop_weights,
        ip_weights,
        pot,
        effective_stack,
        bet_sizes,
        exact,
        depth_limit_override,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_solve_game_parts_with_root(
    board: &[String],
    oop_weights: &[f32],
    ip_weights: &[f32],
    pot: i32,
    effective_stack: i32,
    bet_sizes: &[Vec<f64>],
    exact: bool,
    depth_limit_override: Option<u8>,
    root: Option<SolveGameRoot>,
) -> Result<(range_solver::card::CardConfig, range_solver::ActionTree), String> {
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::CardConfig;
    use range_solver::range::Range;
    use range_solver::{ActionTree, TreeConfig};

    let (flop, turn, river, initial_state) = parse_solve_board(board)?;

    let oop_range =
        Range::from_raw_data(oop_weights).map_err(|e| format!("Bad OOP weights: {e}"))?;
    let ip_range = Range::from_raw_data(ip_weights).map_err(|e| format!("Bad IP weights: {e}"))?;

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
    let root = root.unwrap_or_else(|| SolveGameRoot::fresh_street(pot, effective_stack));

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: root.starting_pot,
        effective_stack,
        initial_player: root.initial_player,
        initial_stacks: Some(root.initial_stacks),
        initial_prev_action: root.initial_prev_action,
        initial_prev_amount: root.initial_prev_amount,
        initial_amount: root.initial_amount,
        initial_num_bets: root.initial_num_bets,
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
        depth_limit: if exact || initial_state == range_solver::BoardState::River {
            None
        } else {
            Some(depth_limit_override.unwrap_or(0))
        },
    };

    let action_tree =
        ActionTree::new(tree_config).map_err(|e| format!("Failed to build tree: {e}"))?;
    Ok((card_config, action_tree))
}

/// Build a `PostFlopGame` from session state, ready for solving.
///
/// Delegates to [`build_solve_game_parts`] for config construction, then
/// builds and allocates the game.
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
    build_solve_game_with_root(
        board,
        oop_weights,
        ip_weights,
        pot,
        effective_stack,
        bet_sizes,
        exact,
        depth_limit_override,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_solve_game_with_root(
    board: &[String],
    oop_weights: &[f32],
    ip_weights: &[f32],
    pot: i32,
    effective_stack: i32,
    bet_sizes: &[Vec<f64>],
    exact: bool,
    depth_limit_override: Option<u8>,
    root: Option<SolveGameRoot>,
) -> Result<PostFlopGame, String> {
    let (card_config, action_tree) = build_solve_game_parts_with_root(
        board,
        oop_weights,
        ip_weights,
        pot,
        effective_stack,
        bet_sizes,
        exact,
        depth_limit_override,
        root,
    )?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;
    game.allocate_memory(false);
    Ok(game)
}

/// Full per-boundary gadget game construction for the Tauri solve path.
///
/// Uses Option A2: injects a 4-node gadget subtree at each cfvnet depth-boundary
/// terminal. Post-injection, `game.root()` is the real subgame root (no gadget
/// layer at the top). Cfvnet boundaries remain at ordinals 0..N; gadget terminals
/// occupy N..3N with pre-populated opt-out values.
#[allow(clippy::too_many_arguments)]
fn build_gadget_tree_game_for_solve(
    board: &[String],
    oop_w: &[f32],
    ip_w: &[f32],
    pot: i32,
    eff_stack: i32,
    bet_sizes: &[Vec<f64>],
    root: SolveGameRoot,
    depth_limit: Option<u8>,
    cbv_ctx: &Option<Arc<crate::postflop::CbvContext>>,
    current_node_idx: u32,
    boundary_cut: &Option<(u8, BoundaryKind)>,
    solve_iters: u32,
    target_exp: f32,
) -> Result<PostFlopGame, String> {
    let ctx = cbv_ctx
        .as_ref()
        .ok_or("enable_gadget=true but no CbvContext loaded (blueprint must include CBV tables)")?;

    let board_u8: Vec<u8> = board
        .iter()
        .map(|s| parse_rs_poker_card(s).expect("board card must parse"))
        .map(|c| crate::exploration::rs_card_to_range_solver(c))
        .collect();

    // Build inner boundary evaluator for cfvnet ordinals 0..N.
    let inner_evaluator = build_inner_evaluator_for_solve(
        board,
        oop_w,
        ip_w,
        pot,
        eff_stack,
        bet_sizes,
        root,
        depth_limit,
        &board_u8,
        boundary_cut,
        solve_iters,
        target_exp,
    )?;

    // Build gadget game via A2 per-boundary injection.
    let (card_config, action_tree) = build_solve_game_parts_with_root(
        board,
        oop_w,
        ip_w,
        pot,
        eff_stack,
        bet_sizes,
        false,
        depth_limit,
        Some(root),
    )?;
    eprintln!("[solve] gadget-tree (A2): BlueprintCbvOptOut (per-boundary pot)");
    crate::gadget::make_per_boundary_gadget_game(
        card_config,
        action_tree,
        ctx,
        current_node_idx,
        &board_u8,
        inner_evaluator,
    )
}

/// Build the inner boundary evaluator for cfvnet ordinals 0..N (Tauri solve path).
#[allow(clippy::too_many_arguments)]
fn build_inner_evaluator_for_solve(
    board: &[String],
    oop_w: &[f32],
    ip_w: &[f32],
    pot: i32,
    eff_stack: i32,
    bet_sizes: &[Vec<f64>],
    root: SolveGameRoot,
    depth_limit: Option<u8>,
    board_u8: &[u8],
    boundary_cut: &Option<(u8, BoundaryKind)>,
    solve_iters: u32,
    target_exp: f32,
) -> Result<Arc<dyn range_solver::game::BoundaryEvaluator>, String> {
    let (tmp_cc, tmp_at) = build_solve_game_parts_with_root(
        board,
        oop_w,
        ip_w,
        pot,
        eff_stack,
        bet_sizes,
        false,
        depth_limit,
        Some(root),
    )?;
    let tmp_game = PostFlopGame::with_config(tmp_cc, tmp_at)
        .map_err(|e| format!("Failed to build temp game: {e}"))?;
    let private_cards: [Vec<(u8, u8)>; 2] = [
        tmp_game.private_cards(0).to_vec(),
        tmp_game.private_cards(1).to_vec(),
    ];
    let tree_cfg = tmp_game.tree_config().clone();
    let initial_weights = [
        vec![1.0f32; private_cards[0].len()],
        vec![1.0f32; private_cards[1].len()],
    ];
    drop(tmp_game);

    let evaluator: Arc<dyn range_solver::game::BoundaryEvaluator> = match boundary_cut {
        Some((_, BoundaryKind::ExactSubtree)) | None => Arc::new(
            crate::exact_subtree::SubtreeExactEvaluator::new(
                board_u8.to_vec(),
                private_cards,
                initial_weights,
                tree_cfg,
            )
            .with_solve_iters(solve_iters)
            .with_target_exploitability(target_exp),
        ),
        Some((
            _,
            BoundaryKind::Cfvnet {
                model_path,
                inference_mode,
            },
        )) => {
            let session = cfvnet::eval::boundary_evaluator::load_shared_onnx_session(
                std::path::Path::new(model_path),
            )
            .map_err(|e| format!("ONNX session load failed: {e}"))?;
            Arc::new(
                cfvnet::eval::boundary_evaluator::neural_boundary_evaluator_from_shared_with_mode(
                    session,
                    board_u8.to_vec(),
                    private_cards,
                    *inference_mode,
                ),
            )
        }
    };
    Ok(evaluator)
}

/// Convert a range-solver `Action` to a `GameAction` using the legacy HU units.
fn range_solver_action_to_game_action(action: &range_solver::Action, idx: usize) -> GameAction {
    range_solver_action_to_game_action_with_big_blind(action, idx, 2.0)
}

/// Convert a range-solver action to a game action using the configured MP chip units.
fn range_solver_action_to_game_action_with_big_blind(
    action: &range_solver::Action,
    idx: usize,
    big_blind: f64,
) -> GameAction {
    let (label, action_type) = match action {
        range_solver::Action::Fold => ("Fold".to_string(), "fold"),
        range_solver::Action::Check => ("Check".to_string(), "check"),
        range_solver::Action::Call => ("Call".to_string(), "call"),
        range_solver::Action::Bet(amt) => (format_mp_amount(*amt as f64, big_blind), "bet"),
        range_solver::Action::Raise(amt) => (format_mp_amount(*amt as f64, big_blind), "raise"),
        range_solver::Action::AllIn(_) => ("All-in".to_string(), "allin"),
        _ => ("?".to_string(), "unknown"),
    };
    GameAction {
        id: idx.to_string(),
        label,
        action_type: action_type.to_string(),
    }
}

/// Build a `GameMatrix` from the current `PostFlopGame` state at the root.
fn build_solve_matrix(game: &mut PostFlopGame, hand_evs: Option<&[f32]>) -> GameMatrix {
    build_solve_matrix_with_big_blind(game, hand_evs, None)
}

/// Build a solve matrix using the supplied chip-to-BB conversion for actions.
fn build_solve_matrix_with_big_blind(
    game: &mut PostFlopGame,
    hand_evs: Option<&[f32]>,
    big_blind: Option<f64>,
) -> GameMatrix {
    game.back_to_root();
    build_solve_matrix_at_current_with_big_blind(game, hand_evs, big_blind)
}

/// Build a `GameMatrix` from the current `PostFlopGame` position (without navigating to root).
///
/// Same logic as `build_solve_matrix` but does NOT call `game.back_to_root()`.
#[allow(clippy::cast_possible_truncation)]
fn build_solve_matrix_at_current(game: &mut PostFlopGame, hand_evs: Option<&[f32]>) -> GameMatrix {
    build_solve_matrix_at_current_with_big_blind(game, hand_evs, None)
}

/// Build a solve matrix at the current position using the supplied action units.
fn build_solve_matrix_at_current_with_big_blind(
    game: &mut PostFlopGame,
    hand_evs: Option<&[f32]>,
    big_blind: Option<f64>,
) -> GameMatrix {
    use crate::postflop::{card_pair_to_matrix, matrix_cell_label};

    let player = game.current_player();
    let strategy = game.strategy();
    let num_hands = game.num_private_hands(player);
    game.cache_normalized_weights();
    let aggregation_weights = game.normalized_weights(player).to_vec();
    let display_weights = game.weights(player).to_vec();
    let private_cards = game.private_cards(player);
    let available_actions = game.available_actions();

    let game_actions: Vec<GameAction> = available_actions
        .iter()
        .enumerate()
        .map(|(i, a)| {
            big_blind.map_or_else(
                || range_solver_action_to_game_action(a, i),
                |big_blind| range_solver_action_to_game_action_with_big_blind(a, i, big_blind),
            )
        })
        .collect();
    let num_actions = game_actions.len();

    let mut prob_sums = vec![vec![vec![0.0f64; num_actions]; 13]; 13];
    let mut combo_counts = vec![vec![0usize; 13]; 13];
    let mut weight_sums = vec![vec![0.0f64; 13]; 13];
    let mut aggregation_weight_sums = vec![vec![0.0f64; 13]; 13];
    let mut ev_sums = vec![vec![0.0f64; 13]; 13];
    let mut combo_details: Vec<Vec<Vec<ComboDetail>>> = vec![vec![Vec::new(); 13]; 13];

    for (hand_idx, &(c1_raw, c2_raw)) in private_cards.iter().enumerate() {
        let (row, col, _) = card_pair_to_matrix(c1_raw, c2_raw);
        let strategy_w = aggregation_weights[hand_idx] as f64;
        let display_w = display_weights[hand_idx] as f64;
        combo_counts[row][col] += 1;
        weight_sums[row][col] += display_w;
        aggregation_weight_sums[row][col] += strategy_w;
        if let Some(evs) = hand_evs {
            if hand_idx < evs.len() {
                ev_sums[row][col] += evs[hand_idx] as f64 * strategy_w;
            }
        }

        let mut probs = Vec::with_capacity(num_actions);
        for (action_idx, prob_sum) in prob_sums[row][col].iter_mut().enumerate() {
            let prob = strategy[action_idx * num_hands + hand_idx];
            *prob_sum += prob as f64 * strategy_w;
            probs.push(prob);
        }

        let (c1, c2) = if c1_raw / 4 >= c2_raw / 4 {
            (c1_raw, c2_raw)
        } else {
            (c2_raw, c1_raw)
        };
        let s1 = card_to_string(c1).unwrap_or_default();
        let s2 = card_to_string(c2).unwrap_or_default();
        combo_details[row][col].push(ComboDetail {
            cards: format!("{s1}{s2}"),
            probabilities: probs,
            weight: display_weights[hand_idx],
            bucket: None,
        });
    }

    let cells: Vec<Vec<GameMatrixCell>> = (0..13)
        .map(|row| {
            (0..13)
                .map(|col| {
                    let (label, suited, pair) = matrix_cell_label(row, col);
                    let count = combo_counts[row][col];
                    let display_total_w = weight_sums[row][col];
                    let strategy_total_w = aggregation_weight_sums[row][col];
                    // Reach-weighted mean: P(action | class) = Σ P(action|combo) * w(combo) / Σ w(combo).
                    // Using simple /count here would treat blocker-adjusted combos as
                    // equally likely as full-weight ones, producing a misleading
                    // aggregate for hand classes whose combos have uneven reach.
                    let probabilities = if strategy_total_w > 0.0 {
                        prob_sums[row][col]
                            .iter()
                            .map(|&s| (s / strategy_total_w) as f32)
                            .collect()
                    } else {
                        vec![0.0; num_actions]
                    };
                    let ev = if strategy_total_w > 0.0 && hand_evs.is_some() {
                        Some((ev_sums[row][col] / strategy_total_w) as f32)
                    } else {
                        None
                    };
                    let combos = std::mem::take(&mut combo_details[row][col]);
                    let weight = if count > 0 {
                        (display_total_w / count as f64) as f32
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
fn build_solve_cache(
    game: &mut PostFlopGame,
    player_labels: &[String; 2],
) -> HashMap<Vec<usize>, CachedSolveNode> {
    build_solve_cache_with_big_blind(game, player_labels, None)
}

fn build_solve_cache_with_big_blind(
    game: &mut PostFlopGame,
    player_labels: &[String; 2],
    big_blind: Option<f64>,
) -> HashMap<Vec<usize>, CachedSolveNode> {
    let mut cache = HashMap::new();
    build_solve_cache_recursive(game, player_labels, big_blind, &mut vec![], &mut cache);
    cache
}

fn build_solve_cache_recursive(
    game: &mut PostFlopGame,
    player_labels: &[String; 2],
    big_blind: Option<f64>,
    path: &mut Vec<usize>,
    cache: &mut HashMap<Vec<usize>, CachedSolveNode>,
) {
    if game.is_terminal_node() || game.is_chance_node() {
        return;
    }

    let matrix = build_solve_matrix_at_current_with_big_blind(game, None, big_blind);
    let actions: Vec<GameAction> = game
        .available_actions()
        .iter()
        .enumerate()
        .map(|(i, a)| {
            big_blind.map_or_else(
                || range_solver_action_to_game_action(a, i),
                |big_blind| range_solver_action_to_game_action_with_big_blind(a, i, big_blind),
            )
        })
        .collect();
    let player = game.current_player();
    let position = player_labels
        .get(player)
        .cloned()
        .unwrap_or_else(|| format!("P{player}"));

    let num_actions = actions.len();
    cache.insert(
        path.clone(),
        CachedSolveNode {
            matrix,
            actions,
            position,
        },
    );

    for i in 0..num_actions {
        game.play(i);
        path.push(i);
        build_solve_cache_recursive(game, player_labels, big_blind, path, cache);
        path.pop();
        // Navigate back: PostFlopGame has no undo, so replay from root.
        game.back_to_root();
        for &action in path.iter() {
            game.play(action);
        }
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
    fn num_continuations(&self) -> usize {
        4
    }

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
                        let w = if j < opponent_reach.len() {
                            opponent_reach[j] as f64
                        } else {
                            0.0
                        };
                        if w <= 0.0 {
                            continue;
                        }
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
                            [rs_h1, rs_h2],
                            [rs_o1, rs_o2],
                            &self.board_cards,
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
                &self.combos,
                &self.board_cards,
                &hero_combo_reach,
                &opp_combo_reach,
                &requests,
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
    if let Some(mp_data) = exploration.extract_universal_mp_data() {
        let mp_session_started = Instant::now();
        let mp_session = LazyMpSession::from_exploration_data(mp_data?)?;
        *session_state.mp_session.write() = Some(mp_session);
        *session_state.session.write() = None;
        session_state.subgame_solve.reset();
        session_state.exact_solve.reset();
        eprintln!(
            "[game_new] universal MP LazyMpSession initialized in {:.3}s",
            mp_session_started.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    let session = GameSession::from_exploration_state(exploration, cbv_ctx)?;
    *session_state.session.write() = Some(session);
    *session_state.mp_session.write() = None;
    session_state.subgame_solve.reset();
    session_state.exact_solve.reset();
    Ok(())
}

fn solve_state_for_source(
    session_state: &GameSessionState,
    source: Option<&str>,
) -> Option<Arc<SolveState>> {
    match source {
        Some("subgame") => Some(Arc::clone(&session_state.subgame_solve)),
        Some("exact") => Some(Arc::clone(&session_state.exact_solve)),
        _ => None,
    }
}

fn resolve_solve_path_from_session(ss: &SolveState, session: &GameSession) -> Option<Vec<usize>> {
    let anchor = ss.solve_anchor.read().clone();
    let Some(anchor) = anchor.as_ref() else {
        return Some(ss.solve_path.read().clone());
    };

    if session.board != anchor.board {
        return None;
    }

    if session.action_history.len() < anchor.action_ids.len() {
        return None;
    }

    if session
        .action_history
        .iter()
        .take(anchor.action_ids.len())
        .map(|a| &a.action_id)
        .ne(anchor.action_ids.iter())
    {
        return None;
    }

    let mut path = Vec::new();
    let cache = ss.solve_cache.read();
    for record in session.action_history.iter().skip(anchor.action_ids.len()) {
        let node = cache.get(&path)?;
        let record_semantic = semantic_action_from_record(record)?;
        let action_idx = node.actions.iter().position(|action| {
            let cached_semantic = semantic_action_from_game_action(action);
            semantic_actions_match(&record_semantic, &cached_semantic)
        })?;
        path.push(action_idx);
        if !cache.contains_key(&path) {
            return None;
        }
    }

    if path.is_empty() && session.node_idx != anchor.node_idx {
        return None;
    }

    Some(path)
}

fn resolve_solve_path_from_mp_session(
    ss: &SolveState,
    session: &LazyMpSession,
) -> Option<Vec<usize>> {
    let anchor = ss.solve_anchor.read().clone();
    let Some(anchor) = anchor.as_ref() else {
        return Some(ss.solve_path.read().clone());
    };
    if mp_board_strings(&session.board) != anchor.board
        || session.action_history.len() < anchor.action_ids.len()
        || session
            .action_history
            .iter()
            .take(anchor.action_ids.len())
            .map(|action| &action.action_id)
            .ne(anchor.action_ids.iter())
    {
        return None;
    }

    let mut path = Vec::new();
    let cache = ss.solve_cache.read();
    for record in session.action_history.iter().skip(anchor.action_ids.len()) {
        let node = cache.get(&path)?;
        let record_semantic = semantic_action_from_record(record)?;
        let action_idx = node.actions.iter().position(|action| {
            semantic_actions_match(&record_semantic, &semantic_action_from_game_action(action))
        })?;
        path.push(action_idx);
        if !cache.contains_key(&path) {
            return None;
        }
    }
    Some(path)
}

fn apply_exact_solve_overlay(state: &mut GameState, ss: &SolveState, path: Option<Vec<usize>>) {
    let is_solving = ss.solving.load(Ordering::Relaxed);
    let iteration = ss.iteration.load(Ordering::Relaxed);
    if !is_solving && iteration == 0 {
        return;
    }
    let Some(path) = path else {
        // The MP session no longer matches the solve anchor. Keep the live
        // state untouched and do not expose status from an unrelated solve.
        state.solve = None;
        return;
    };
    let exp = f32::from_bits(ss.exploitability_bits.load(Ordering::Relaxed));
    let max_iters = ss.max_iterations.load(Ordering::Relaxed);
    let elapsed = ss
        .solve_start
        .read()
        .map(|time| time.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    state.solve = Some(SolveStatus {
        iteration,
        max_iterations: max_iters,
        exploitability: exp,
        elapsed_secs: elapsed,
        solver_name: "range".to_string(),
        is_complete: !is_solving && iteration > 0,
    });

    set_solve_path_if_changed(ss, &path);
    if let Some(node) = cached_node_for_path(ss, &path) {
        state.matrix = Some(node.matrix);
        state.actions = node.actions;
        state.position = node.position;
    } else if path.is_empty() {
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

fn set_solve_path_if_changed(ss: &SolveState, path: &[usize]) {
    if ss.solve_path.read().as_slice() != path {
        *ss.solve_path.write() = path.to_vec();
    }
}

fn cached_node_for_path(ss: &SolveState, path: &[usize]) -> Option<CachedSolveNode> {
    ss.solve_cache.read().get(path).cloned()
}

fn validate_solve_root_actor_label(
    current_player: usize,
    player_labels: &[String; 2],
    position_label: &str,
) -> Result<(), String> {
    let range_label = player_labels
        .get(current_player)
        .cloned()
        .unwrap_or_else(|| format!("P{current_player}"));
    if range_label != position_label {
        return Err(format!(
            "Solve root actor mismatch: session is {position_label} but range solver root is {range_label}"
        ));
    }
    Ok(())
}

fn reset_solve_state_for_start(
    ss: &SolveState,
    max_iters: u32,
    position_label: String,
    solve_anchor: SolveAnchor,
) -> u64 {
    let _publish_guard = ss.publish_gate.write();
    let generation = ss.generation.fetch_add(1, Ordering::AcqRel) + 1;
    ss.iteration.store(0, Ordering::Relaxed);
    ss.max_iterations.store(max_iters, Ordering::Relaxed);
    ss.exploitability_bits
        .store(f32::MAX.to_bits(), Ordering::Relaxed);
    ss.cancel.store(false, Ordering::Relaxed);
    ss.solving.store(true, Ordering::Release);
    *ss.solve_start.write() = Some(Instant::now());
    *ss.matrix_snapshot.write() = None;
    *ss.solve_actions.write() = vec![];
    *ss.solve_position.write() = position_label;
    *ss.solve_anchor.write() = Some(solve_anchor);
    *ss.solve_cache.write() = HashMap::new();
    *ss.solve_path.write() = vec![];
    generation
}

/// Get the current game state, including solve progress if active.
///
/// `source` controls which strategy data is returned:
/// - `None` or `"blueprint"`: return blueprint data only, skip solve overlay.
/// - `"subgame"`: overlay from `subgame_solve`.
/// - `"exact"`: overlay from `exact_solve`.
pub fn game_get_state_core(
    session_state: &GameSessionState,
    source: Option<String>,
) -> Result<GameState, String> {
    if session_state.mp_session.read().is_some() {
        let mut guard = session_state.mp_session.write();
        let session = guard.as_mut().ok_or("No MP game session active")?;
        let mut state = session.get_state()?;
        if source.as_deref() == Some("exact") {
            let ss = session_state.exact_solve.as_ref();
            let path = resolve_solve_path_from_mp_session(ss, session);
            apply_exact_solve_overlay(&mut state, ss, path);
        }
        return Ok(state);
    }
    let guard = session_state.session.read();
    let session = guard.as_ref().ok_or("No game session active")?;
    let mut state = session.get_state();

    // Blueprint source: return raw blueprint data, no solve overlay
    let Some(ss) = solve_state_for_source(session_state, source.as_deref()) else {
        return Ok(state);
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

        if let Some(path) = resolve_solve_path_from_session(&ss, session) {
            set_solve_path_if_changed(&ss, &path);
            // Prefer solve cache (navigated position) over root snapshot
            if let Some(node) = cached_node_for_path(&ss, &path) {
                state.matrix = Some(node.matrix.clone());
                state.actions = node.actions.clone();
                state.position = node.position.clone();
            } else if path.is_empty() {
                // Fall back to root snapshot during solve or before cache is built.
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
    if session_state.mp_session.read().is_some() {
        let mut guard = session_state.mp_session.write();
        let session = guard.as_mut().ok_or("No MP game session active")?;
        let previous_street = session.spot.street();
        let big_blind = mp_big_blind_amount(&session.config)?;
        let ss = solve_state_for_source(session_state, source.as_deref());
        let cached_navigation = ss.as_ref().and_then(|ss| {
            let current_path = resolve_solve_path_from_mp_session(ss, session)?;
            let cache = ss.solve_cache.read();
            let current_node = cache.get(&current_path)?;
            let cached_index = current_node
                .actions
                .iter()
                .position(|action| action.id == action_id)?;
            let cached_action = &current_node.actions[cached_index];
            let session_action_id = session
                .current_actions()
                .iter()
                .enumerate()
                .find(|(_, current)| {
                    semantic_actions_match(
                        &semantic_action_from_game_action(cached_action),
                        &semantic_action_from_mp_tree_action(current, big_blind),
                    )
                })
                .map(|(index, _)| index.to_string())?;
            let mut next_path = current_path;
            next_path.push(cached_index);
            cache
                .contains_key(&next_path)
                .then_some((session_action_id, next_path))
        });
        if let (Some(ss), Some((session_action_id, next_path))) = (ss.as_ref(), cached_navigation) {
            let mut state = session.play_action_without_state(&session_action_id)?;
            let street_changed = session.spot.street() != previous_street;
            let solve_in_progress = session_state.subgame_solve.solving.load(Ordering::Acquire)
                || session_state.exact_solve.solving.load(Ordering::Acquire);
            if street_changed || solve_in_progress {
                session_state.subgame_solve.reset();
                session_state.exact_solve.reset();
                state.solve = None;
                return Ok(state);
            }
            apply_exact_solve_overlay(&mut state, ss, Some(next_path));
            return Ok(state);
        }
        let session_action_id = action_id.to_string();
        let mut state = session.play_action(&session_action_id)?;
        let street_changed = session.spot.street() != previous_street;
        let solve_in_progress = session_state.subgame_solve.solving.load(Ordering::Acquire)
            || session_state.exact_solve.solving.load(Ordering::Acquire);
        if street_changed || solve_in_progress {
            session_state.subgame_solve.reset();
            session_state.exact_solve.reset();
            state.solve = None;
        }
        if let Some(ss) = ss.as_ref() {
            let path = resolve_solve_path_from_mp_session(ss, session);
            apply_exact_solve_overlay(&mut state, ss, path);
        }
        return Ok(state);
    }
    let ss = solve_state_for_source(session_state, source.as_deref());
    let source_navigation = if let Some(ss) = ss.as_ref() {
        let guard = session_state.session.read();
        let session = guard.as_ref().ok_or("No game session active")?;
        if let Some(current_path) = resolve_solve_path_from_session(ss, session) {
            set_solve_path_if_changed(ss, &current_path);

            let cache = ss.solve_cache.read();
            if let Some(current_node) = cache.get(&current_path) {
                if let Some(action_idx) = current_node
                    .actions
                    .iter()
                    .position(|action| action.id == action_id)
                {
                    let cached_action = &current_node.actions[action_idx];
                    let session_action_id =
                        session_action_id_matching_cached_action(session, cached_action)?;
                    let mut new_path = current_path;
                    new_path.push(action_idx);
                    Some((
                        session_action_id,
                        cache.get(&new_path).map(|child_node| {
                            (
                                new_path,
                                child_node.matrix.clone(),
                                child_node.actions.clone(),
                                child_node.position.clone(),
                            )
                        }),
                    ))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let session_action_id = match source_navigation {
        Some((
            session_action_id,
            Some((new_path, child_matrix, child_actions, child_position)),
        )) => {
            let ss = ss
                .as_ref()
                .expect("source navigation exists only when solve state exists");

            // Play the action on the session for board/range tracking
            let mut guard = session_state.session.write();
            let session = guard.as_mut().ok_or("No game session active")?;
            session.play_action(&session_action_id)?;
            let mut state = session.get_state();
            drop(guard);

            // Override with cached data
            state.matrix = Some(child_matrix);
            state.actions = child_actions;
            state.position = child_position;
            *ss.solve_path.write() = new_path;

            return Ok(state);
        }
        Some((session_action_id, None)) => session_action_id,
        None => action_id.to_string(),
    };

    // Normal navigation. Blueprint/None sources never consume solve caches, and
    // source-specific cache misses preserve existing caches for later reuse.
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.play_action(&session_action_id)?;
    let mut state = session.get_state();
    if let Some(ss) = ss.as_ref() {
        if let Some(path) = resolve_solve_path_from_session(ss, session) {
            if let Some(node) = cached_node_for_path(ss, &path) {
                state.matrix = Some(node.matrix.clone());
                state.actions = node.actions.clone();
                state.position = node.position.clone();
            }
            set_solve_path_if_changed(ss, &path);
        }
    }
    Ok(state)
}

/// Deal a board card and return the new game state.
pub fn game_deal_card_core(
    session_state: &GameSessionState,
    card: &str,
) -> Result<GameState, String> {
    if session_state.mp_session.read().is_some() {
        let mut guard = session_state.mp_session.write();
        let session = guard.as_mut().ok_or("No MP game session active")?;
        let mut state = session.deal_card(card)?;
        session_state.subgame_solve.reset();
        session_state.exact_solve.reset();
        state.solve = None;
        return Ok(state);
    }
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.deal_card(card)?;
    Ok(session.get_state())
}

/// Undo the last action and return the new game state.
///
/// If within a solved subgame tree, pops the last action from the solve path
/// and serves the parent's cached matrix. If the selected source is blueprint
/// or the selected solve cache cannot serve the current session path, this is
/// normal session navigation and existing solve caches are preserved.
///
/// `source` selects which solve cache to navigate within.
pub fn game_back_core(
    session_state: &GameSessionState,
    source: Option<String>,
) -> Result<GameState, String> {
    if session_state.mp_session.read().is_some() {
        let mut guard = session_state.mp_session.write();
        let session = guard.as_mut().ok_or("No MP game session active")?;
        let mut state = session.back()?;
        session_state.subgame_solve.reset();
        session_state.exact_solve.reset();
        state.solve = None;
        return Ok(state);
    }
    let ss = solve_state_for_source(session_state, source.as_deref());
    let cached_parent = ss.as_ref().and_then(|ss| {
        let guard = session_state.session.read();
        let session = guard.as_ref()?;
        let mut path = resolve_solve_path_from_session(ss, session)?;
        set_solve_path_if_changed(ss, &path);

        if path.is_empty() {
            return None;
        }
        path.pop();
        cached_node_for_path(ss, &path).map(|node| {
            (
                path,
                node.matrix.clone(),
                node.actions.clone(),
                node.position.clone(),
            )
        })
    });

    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.back()?;
    let mut state = session.get_state();
    let after_back_path = if cached_parent.is_none() {
        ss.as_ref()
            .and_then(|ss| resolve_solve_path_from_session(ss, session))
    } else {
        None
    };
    drop(guard);

    if let (Some(ss), Some((parent_path, matrix, actions, position))) = (ss.as_ref(), cached_parent)
    {
        state.matrix = Some(matrix);
        state.actions = actions;
        state.position = position;
        *ss.solve_path.write() = parent_path;
    } else if let (Some(ss), Some(path)) = (ss.as_ref(), after_back_path) {
        if let Some(node) = cached_node_for_path(ss, &path) {
            state.matrix = Some(node.matrix.clone());
            state.actions = node.actions.clone();
            state.position = node.position.clone();
        }
        set_solve_path_if_changed(ss, &path);
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
    enable_gadget: Option<bool>,
) -> Result<(), String> {
    let is_exact = mode.as_deref() == Some("exact");
    let ss_ref = session_state.solve_for(&mode);

    // Guard: reject if this mode is already solving
    if ss_ref.solving.load(Ordering::Relaxed) {
        return Err("A solve is already in progress".to_string());
    }

    // Read session state under lock, clone what the thread needs
    let (
        board,
        oop_w,
        ip_w,
        pot,
        eff_stack,
        bet_sizes,
        cbv_ctx,
        current_node_idx,
        position_label,
        solve_anchor,
        player_labels,
        solve_root,
        root_street,
        action_big_blind,
    ) = {
        if session_state.mp_session.read().is_some() {
            if !is_exact {
                return Err(
                    "UniversalMpLazy currently supports Exact solve for flop decisions only"
                        .to_string(),
                );
            }
            let mut guard = session_state.mp_session.write();
            let session = guard.as_mut().ok_or("No MP game session active")?;
            let action_big_blind = mp_big_blind_amount(&session.config)?;
            let snapshot = session.exact_solve_snapshot()?;
            let oop_w = snapshot.raw_reaches_by_seat[usize::from(snapshot.oop_seat)].clone();
            let ip_w = snapshot.raw_reaches_by_seat[usize::from(snapshot.ip_seat)].clone();
            let position = if snapshot.acting_seat == snapshot.oop_seat {
                "BB"
            } else {
                "SB"
            };
            let solve_anchor = SolveAnchor {
                node_idx: 0,
                board: snapshot.board.clone(),
                action_ids: snapshot
                    .action_history
                    .iter()
                    .map(|action| action.action_id.clone())
                    .collect(),
            };
            (
                snapshot.board,
                oop_w,
                ip_w,
                snapshot.pot,
                effective_stack_for_solve_root(&snapshot.root),
                snapshot.bet_sizes,
                None,
                0,
                position.to_string(),
                solve_anchor,
                ["BB".to_string(), "SB".to_string()],
                snapshot.root,
                Street::Flop,
                Some(action_big_blind),
            )
        } else {
            let guard = session_state.session.read();
            let session = guard.as_ref().ok_or("No game session active")?;

            // Must be at a postflop decision node
            if session.board.len() < 3 {
                return Err(
                    "Solve requires a postflop position (deal board cards first)".to_string(),
                );
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
            let solve_root = session.solve_game_root_for_player(player)?;
            let eff_stack = effective_stack_for_solve_root(&solve_root);

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
            let solve_anchor = SolveAnchor {
                node_idx: current_node,
                board: board.clone(),
                action_ids: session
                    .action_history
                    .iter()
                    .map(|a| a.action_id.clone())
                    .collect(),
            };
            let player_labels = [
                session.position_label(1 - session.tree.dealer).to_string(),
                session.position_label(session.tree.dealer).to_string(),
            ];

            (
                board,
                oop_w,
                ip_w,
                pot,
                eff_stack,
                sizes.clone(),
                cbv_ctx,
                current_node,
                position,
                solve_anchor,
                player_labels,
                solve_root,
                street,
                None,
            )
        }
    };

    // Resolve StreetBoundaryConfig to (depth_limit, model_path)
    let sbc = street_boundary_config.unwrap_or_default();
    let boundary_cut = if is_exact {
        None
    } else {
        resolve_street_boundary(&sbc, root_street)
    };
    validate_cfvnet_boundary_cut(&boundary_cut, root_street)?;
    eprintln!(
        "{}",
        boundary_evaluator_log_line(if is_exact { "exact" } else { "subgame" }, &boundary_cut)
    );

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

    let gadget_enabled = enable_gadget.unwrap_or(false);

    let max_iters = max_iterations.unwrap_or(200);
    let snapshot_interval = matrix_snapshot_interval.unwrap_or(10);
    let target_exp = target_exploitability.unwrap_or(3.0);

    // Build trace config (empty trace_boundaries = no tracing = zero cost).
    // Relative trace_dir is resolved against the workspace root (walk up
    // from CWD looking for a `Cargo.toml` with `[workspace]`) so files
    // land in the project's local_data/logs regardless of whether the
    // solver is launched from the workspace root or a crate subdir.
    //
    // The resolved dir is further suffixed with `exact/` or `subgame/` so
    // both modes can coexist without the user having to move files before
    // running the other mode.
    let trace_config = {
        let boundaries = trace_boundaries.filter(|s| !s.trim().is_empty());
        let raw_dir =
            std::path::PathBuf::from(trace_dir.unwrap_or_else(|| "./local_data/logs".to_string()));
        let base = if raw_dir.is_absolute() {
            raw_dir
        } else {
            resolve_against_workspace_root(&raw_dir)
        };
        let dir = if is_exact {
            base.join("exact")
        } else {
            let subgame = base.join("subgame");
            match &boundary_cut {
                Some((_, BoundaryKind::Cfvnet { .. })) => subgame.join("cfvnet"),
                Some((_, BoundaryKind::ExactSubtree)) => subgame.join("exact_subtree"),
                None => subgame,
            }
        };
        if boundaries.is_some() {
            eprintln!("[solve] trace output dir: {}", dir.display());
        }
        crate::boundary_trace::TraceConfig {
            boundaries,
            iters_str: trace_iters.unwrap_or_else(|| "last".to_string()),
            dir,
        }
    };

    // Reset solve state for this mode before building, so progress snapshots
    // cannot read a stale solved tree from an earlier solve.
    let ss = ss_ref;
    let solve_generation = reset_solve_state_for_start(ss, max_iters, position_label, solve_anchor);

    let depth_limit_override = boundary_cut.as_ref().map(|(depth, _)| *depth);
    let build_exact = is_exact || boundary_cut.is_none();

    // Gadget tree mode (A2): when gadget is enabled AND a boundary
    // cut is active, build via make_per_boundary_gadget_game which
    // injects per-boundary gadget subtrees. game.root() remains the
    // real subgame root.
    let gadget_tree_active = gadget_enabled && boundary_cut.is_some() && cbv_ctx.is_some();

    let game_result = if gadget_tree_active {
        build_gadget_tree_game_for_solve(
            &board,
            &oop_w,
            &ip_w,
            pot,
            eff_stack,
            &bet_sizes,
            solve_root,
            depth_limit_override,
            &cbv_ctx,
            current_node_idx,
            &boundary_cut,
            max_iters,
            target_exp,
        )
    } else {
        if gadget_enabled && boundary_cut.is_none() {
            eprintln!("[solve] enable_gadget=true but no boundary cut; gadget has no effect");
        }
        if gadget_enabled && cbv_ctx.is_none() {
            eprintln!("[solve] enable_gadget=true but no CbvContext; gadget has no effect");
        }
        build_solve_game_with_root(
            &board,
            &oop_w,
            &ip_w,
            pot,
            eff_stack,
            &bet_sizes,
            build_exact,
            depth_limit_override,
            Some(solve_root),
        )
    };
    let mut game = match game_result {
        Ok(game) => game,
        Err(e) => {
            ss.solving.store(false, Ordering::Release);
            return Err(e);
        }
    };

    let position_label_for_guard = ss.solve_position.read().clone();
    if let Err(e) = validate_solve_root_actor_label(
        game.current_player(),
        &player_labels,
        &position_label_for_guard,
    ) {
        debug_assert!(false, "{e}");
        eprintln!("[solve] {e}");
    }

    // Spawn background thread
    let ss_clone = Arc::clone(ss_ref);
    let board_clone = board.clone();
    std::thread::spawn(move || {
        // Store available actions at the explorer-visible root.
        // Under A2, game.root() IS the real subgame root.
        {
            game.back_to_root();
            let actions: Vec<GameAction> = game
                .available_actions()
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    action_big_blind.map_or_else(
                        || range_solver_action_to_game_action(a, i),
                        |big_blind| {
                            range_solver_action_to_game_action_with_big_blind(a, i, big_blind)
                        },
                    )
                })
                .collect();
            if !ss_clone.publish_if_current(solve_generation, |state| {
                *state.solve_actions.write() = actions;
            }) {
                return;
            }
        }

        // Set up boundary evaluators for non-gadget path (opt_out=None).
        // In gadget-tree (A2) mode, boundaries are already wired by
        // make_per_boundary_gadget_game.
        if !gadget_tree_active {
            let n_boundaries = game.num_boundary_nodes();
            if let Some((_, ref kind)) = boundary_cut {
                if n_boundaries > 0 {
                    match kind {
                        BoundaryKind::Cfvnet {
                            model_path,
                            inference_mode,
                        } => {
                            setup_neural_boundaries(&mut game, model_path, *inference_mode, None);
                        }
                        BoundaryKind::ExactSubtree => {
                            setup_exact_subtree_boundaries_with_gadget(
                                &mut game, None, max_iters, target_exp,
                            );
                        }
                    }
                }
            }
        }

        let n_boundaries = game.num_boundary_nodes();
        let (mem_est, _) = game.memory_usage();
        if gadget_tree_active {
            let n_original = n_boundaries / 3;
            eprintln!(
                "[solve] gadget-tree (A2): {} original boundaries, {} total (incl. {} gadget terminals)",
                n_original,
                n_boundaries,
                n_boundaries - n_original,
            );
        } else {
            eprintln!(
                "[solve] depth_limit: {:?}, boundary nodes: {n_boundaries}, per_boundary: {}",
                depth_limit_override,
                game.per_boundary_evaluators.len(),
            );
        }
        eprintln!("[solve] pot={pot}, eff_stack={eff_stack}, board={board_clone:?}");
        eprintln!(
            "[solve] OOP hands: {}, IP hands: {}",
            game.private_cards(0).len(),
            game.private_cards(1).len(),
        );
        eprintln!("[solve] memory: {:.1} MB", mem_est as f64 / 1_048_576.0);

        // Seed solver with blueprint strategy if available.
        if let Some(ref ctx) = cbv_ctx {
            let board_cards: Vec<rs_poker::core::Card> = board_clone
                .iter()
                .map(|s| parse_rs_poker_card(s).expect("board card must parse"))
                .collect();
            let seed_street = match board_cards.len() {
                3 => Street::Flop,
                4 => Street::Turn,
                _ => Street::River,
            };
            // Under A2, game.root() IS the real subgame root (no offset).
            let seed_start = 0;
            crate::postflop::seed_solver_with_blueprint(
                &game,
                &ctx.strategy,
                &ctx.all_buckets,
                &ctx.abstract_tree,
                &board_cards,
                seed_street,
                current_node_idx,
                seed_start,
            );
        }

        // Set up boundary tracer (no-op when disabled).
        // Under A2, cfvnet boundaries are at ordinals 0..N (no leading skip).
        // Gadget terminals at N..3N have static pre-populated values;
        // tracing them is harmless (shows constant opt-out values).
        let tracer = trace_config.into_tracer(max_iters);
        let spot_paths: Option<Vec<String>> = tracer.as_ref().and_then(|_| {
            let n = game.num_boundary_nodes();
            if n > 0 {
                Some(crate::boundary_trace::build_boundary_spot_paths(&game))
            } else {
                None
            }
        });
        let preceding_map = tracer
            .as_ref()
            .map(|_| crate::boundary_trace::build_preceding_decision_map(&game));

        // Initial matrix snapshot
        let matrix = build_solve_matrix_with_big_blind(&mut game, None, action_big_blind);
        if !ss_clone.publish_if_current(solve_generation, |state| {
            *state.matrix_snapshot.write() = Some(matrix);
        }) {
            return;
        }

        // Solve loop
        let has_per_boundary = !game.per_boundary_evaluators.is_empty();
        let mut t = 0u32;
        while t < max_iters {
            if ss_clone.generation.load(Ordering::Acquire) != solve_generation
                || ss_clone.cancel.load(Ordering::Acquire)
            {
                break;
            }

            // Neural cfvnet path: clear CFV cache every iteration so boundary
            // values are recomputed with updated opponent reaches.
            if has_per_boundary {
                game.clear_boundary_cfvs();
            }

            // Update DCFR discount params for boundary continuation regrets.
            {
                let nearest_pow4 = if t == 0 {
                    0
                } else {
                    1u32 << ((t.leading_zeros() ^ 31) & !1)
                };
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
            if !ss_clone.publish_if_current(solve_generation, |state| {
                state.iteration.store(t, Ordering::Relaxed);
            }) {
                return;
            }

            // Capture boundary traces after this iteration's CFVs are cached.
            if let Some(ref tr) = tracer {
                crate::boundary_trace::capture_boundary_traces(
                    &mut game,
                    tr,
                    spot_paths.as_deref(),
                    preceding_map.as_ref(),
                    t - 1,
                );
            }

            // Snapshot matrix and exploitability periodically
            if t.is_multiple_of(snapshot_interval) {
                let matrix = build_solve_matrix_with_big_blind(&mut game, None, action_big_blind);
                if !ss_clone.publish_if_current(solve_generation, |state| {
                    *state.matrix_snapshot.write() = Some(matrix);
                }) {
                    return;
                }

                if is_exact {
                    let exp = compute_exploitability(&game);
                    if !ss_clone.publish_if_current(solve_generation, |state| {
                        state
                            .exploitability_bits
                            .store(exp.to_bits(), Ordering::Relaxed);
                    }) {
                        return;
                    }
                    if exp.is_finite() && exp > 0.0 && exp <= target_exp {
                        eprintln!(
                            "[solve] exact converged: iter={t} exploitability={exp:.3} <= target={target_exp}"
                        );
                        break;
                    }
                }
            }
        }

        if ss_clone.generation.load(Ordering::Acquire) != solve_generation {
            return;
        }

        // Finalize: normalize strategy, compute EVs.
        // (This replaces the per-node strategy buffer with the CFR
        // time-averaged equilibrium — the same values the UI reads.)
        if has_per_boundary {
            game.clear_boundary_cfvs();
        }
        finalize(&mut game);

        // Flush the "last iter" trace AFTER finalize so the strategy recorded
        // matches the UI's displayed (time-averaged) strategy. Runs whether or
        // not the natural in-loop path caught iter=last — this record reflects
        // the *finalized* equilibrium strategy, which is distinct from any
        // per-iter raw strategy captured inside the loop.
        if let Some(ref tr) = tracer {
            let final_iter = t.saturating_sub(1);
            crate::boundary_trace::capture_boundary_traces_forced(
                &mut game,
                tr,
                spot_paths.as_deref(),
                preceding_map.as_ref(),
                final_iter,
            );
        }
        // Navigate to root for final matrix + EVs.
        // Under A2, game.root() IS the real subgame root.
        game.back_to_root();
        game.cache_normalized_weights();
        let player = game.current_player();
        let evs = game.expected_values(player);
        let final_matrix =
            build_solve_matrix_with_big_blind(&mut game, Some(&evs), action_big_blind);
        if !ss_clone.publish_if_current(solve_generation, |state| {
            *state.matrix_snapshot.write() = Some(final_matrix);
        }) {
            return;
        }

        // Compute exploitability using cached boundary CFVs
        let saved_evaluator = game.boundary_evaluator.take();
        let saved_per_boundary = std::mem::take(&mut game.per_boundary_evaluators);
        let final_exp = compute_exploitability(&game);
        game.boundary_evaluator = saved_evaluator;
        game.per_boundary_evaluators = saved_per_boundary;
        if !ss_clone.publish_if_current(solve_generation, |state| {
            state
                .exploitability_bits
                .store(final_exp.to_bits(), Ordering::Relaxed);
        }) {
            return;
        }

        // Build solve cache for all decision nodes in the solved tree.
        game.back_to_root();
        let solve_cache =
            build_solve_cache_with_big_blind(&mut game, &player_labels, action_big_blind);
        eprintln!(
            "[solve] cached {} decision nodes for subgame navigation",
            solve_cache.len()
        );
        if !ss_clone.publish_if_current(solve_generation, |state| {
            *state.solve_cache.write() = solve_cache;
            *state.solve_path.write() = vec![];
            state.solving.store(false, Ordering::Release);
        }) {
            return;
        }
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
///
/// When `opt_out` is `Some`, each evaluator is wrapped in a `GadgetEvaluator`
/// that clamps the opponent's CFVs upward to the opt-out values.
fn setup_neural_boundaries(
    game: &mut PostFlopGame,
    model_path: &str,
    inference_mode: cfvnet::eval::boundary_evaluator::BoundaryInferenceMode,
    opt_out: Option<Arc<dyn crate::gadget::OptOutProvider>>,
) {
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

    let gadget_label = if opt_out.is_some() { " + gadget" } else { "" };
    let mut per_boundary: Vec<Arc<dyn range_solver::game::BoundaryEvaluator>> =
        Vec::with_capacity(n_boundaries);
    for (ordinal, board_4) in boundary_boards.into_iter().enumerate() {
        let private_cards_pair = [
            game.private_cards(0).to_vec(),
            game.private_cards(1).to_vec(),
        ];
        let neural_eval =
            cfvnet::eval::boundary_evaluator::neural_boundary_evaluator_from_shared_with_mode(
                Arc::clone(&session),
                board_4.clone(),
                private_cards_pair.clone(),
                inference_mode,
            );
        let inner: Arc<dyn range_solver::game::BoundaryEvaluator> = Arc::new(neural_eval);
        let wrapped: Arc<dyn range_solver::game::BoundaryEvaluator> = match &opt_out {
            Some(provider) => Arc::new(crate::gadget::GadgetEvaluator::new(
                inner,
                Arc::clone(provider),
                ordinal,
                board_4,
                private_cards_pair,
            )),
            None => inner,
        };
        per_boundary.push(wrapped);
    }
    game.per_boundary_evaluators = per_boundary;
    game.boundary_evaluator = None;

    eprintln!(
        "[solve] neural-cfvnet mode: {n_boundaries} boundaries (ONNX, {inference_mode:?}){gadget_label}",
    );
}

/// Wire per-boundary `SubtreeExactEvaluator`s into the game's
/// `per_boundary_evaluators` vector. Each boundary gets its own
/// evaluator that solves the downstream subtree exactly via DCFR.
///
/// When `opt_out` is `Some`, each evaluator is wrapped in a `GadgetEvaluator`
/// that clamps the opponent's CFVs upward to the opt-out values.
fn setup_exact_subtree_boundaries_with_gadget(
    game: &mut PostFlopGame,
    opt_out: Option<Arc<dyn crate::gadget::OptOutProvider>>,
    solve_iters: u32,
    target_exp: f32,
) {
    let boundary_boards = game.boundary_boards();
    let n_boundaries = game.num_boundary_nodes();
    if boundary_boards.is_empty() {
        eprintln!("[solve] no boundary boards found; skipping exact subtree setup");
        return;
    }
    let bet_sizes = game.tree_config().clone();
    let gadget_label = if opt_out.is_some() { " + gadget" } else { "" };
    let mut per_boundary: Vec<Arc<dyn range_solver::game::BoundaryEvaluator>> =
        Vec::with_capacity(n_boundaries);
    let private_cards = [
        game.private_cards(0).to_vec(),
        game.private_cards(1).to_vec(),
    ];
    let initial_weights = [
        game.initial_weights(0).to_vec(),
        game.initial_weights(1).to_vec(),
    ];
    for (ordinal, board) in boundary_boards.iter().enumerate() {
        let eval: Arc<dyn range_solver::game::BoundaryEvaluator> = Arc::new(
            crate::exact_subtree::SubtreeExactEvaluator::new(
                board.clone(),
                private_cards.clone(),
                initial_weights.clone(),
                bet_sizes.clone(),
            )
            .with_solve_iters(solve_iters)
            .with_target_exploitability(target_exp),
        );
        let wrapped: Arc<dyn range_solver::game::BoundaryEvaluator> = match &opt_out {
            Some(provider) => Arc::new(crate::gadget::GadgetEvaluator::new(
                eval,
                Arc::clone(provider),
                ordinal,
                board.clone(),
                private_cards.clone(),
            )),
            None => eval,
        };
        per_boundary.push(wrapped);
    }
    game.per_boundary_evaluators = per_boundary;
    game.boundary_evaluator = None;
    eprintln!("[solve] exact-subtree mode: {n_boundaries} boundaries (full CFR){gadget_label}");
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
    game_new_core(&exploration, &postflop_state, &session_state)
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
    game_deal_card_core(&session_state, &card)
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
    enable_gadget: Option<bool>,
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
        enable_gadget,
    )
}

pub fn game_cancel_solve_core(
    session_state: &GameSessionState,
    mode: Option<String>,
) -> Result<(), String> {
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
                metric: Default::default(),
            },
            flop: StreetClusterConfig {
                buckets: 10,
                delta_bins: None,
                expected_delta: false,
                sample_boards: None,
                metric: Default::default(),
            },
            turn: StreetClusterConfig {
                buckets: 10,
                delta_bins: None,
                expected_delta: false,
                sample_boards: None,
                metric: Default::default(),
            },
            river: StreetClusterConfig {
                buckets: 10,
                delta_bins: None,
                expected_delta: false,
                sample_boards: None,
                metric: Default::default(),
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
            storage_backend: "dense".to_string(),
            sapcfr_eta: 0.5,
            brcfr_eta: 0.6,
            brcfr_warmup_iterations: 0,
            brcfr_interval: 100_000_000,
            use_baselines: false,
            baseline_alpha: 0.01,
            baseline_validation: Default::default(),
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
            format: SnapshotFormat::Legacy,
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
        assert_eq!(
            ss.max_iterations.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert!(ss.matrix_snapshot.read().is_none());
        assert!(ss.solve_actions.read().is_empty());
        assert!(ss.solve_position.read().is_empty());
    }

    #[test]
    fn game_session_state_has_dual_solve_states() {
        let gss = GameSessionState::default();
        // Both subgame_solve and exact_solve should exist and default to not solving
        assert!(!gss
            .subgame_solve
            .solving
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(!gss
            .exact_solve
            .solving
            .load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn solve_for_returns_subgame_by_default() {
        let gss = GameSessionState::default();
        gss.subgame_solve
            .iteration
            .store(42, std::sync::atomic::Ordering::Relaxed);
        let ss = gss.solve_for(&None);
        assert_eq!(ss.iteration.load(std::sync::atomic::Ordering::Relaxed), 42);
    }

    #[test]
    fn solve_for_returns_subgame_for_subgame_mode() {
        let gss = GameSessionState::default();
        gss.subgame_solve
            .iteration
            .store(77, std::sync::atomic::Ordering::Relaxed);
        let ss = gss.solve_for(&Some("subgame".to_string()));
        assert_eq!(ss.iteration.load(std::sync::atomic::Ordering::Relaxed), 77);
    }

    #[test]
    fn solve_for_returns_exact_for_exact_mode() {
        let gss = GameSessionState::default();
        gss.exact_solve
            .iteration
            .store(99, std::sync::atomic::Ordering::Relaxed);
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
        gss.subgame_solve
            .solving
            .store(true, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve
            .iteration
            .store(50, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve
            .max_iterations
            .store(200, std::sync::atomic::Ordering::Relaxed);
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
        gss.subgame_solve
            .solving
            .store(true, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve
            .iteration
            .store(50, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve
            .max_iterations
            .store(200, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve
            .exploitability_bits
            .store(5.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
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
        gss.exact_solve
            .solving
            .store(true, std::sync::atomic::Ordering::Relaxed);
        gss.exact_solve
            .iteration
            .store(75, std::sync::atomic::Ordering::Relaxed);
        gss.exact_solve
            .max_iterations
            .store(300, std::sync::atomic::Ordering::Relaxed);
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
        gss.subgame_solve
            .solving
            .store(false, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve
            .iteration
            .store(200, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve
            .max_iterations
            .store(200, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve
            .exploitability_bits
            .store(1.5f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
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
        gss.subgame_solve
            .solving
            .store(true, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve
            .iteration
            .store(10, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve
            .max_iterations
            .store(100, std::sync::atomic::Ordering::Relaxed);
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
            actions: vec![GameAction {
                id: "0".to_string(),
                label: "Check".to_string(),
                action_type: "check".to_string(),
            }],
        };
        *gss.subgame_solve.matrix_snapshot.write() = Some(dummy_matrix);

        let state = game_get_state_core(&gss, Some("subgame".to_string())).unwrap();
        let matrix = state
            .matrix
            .expect("matrix should be overridden by solve snapshot");
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
        let result = game_solve_core(
            &gss, None, None, None, None, None, None, None, None, None, None,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No game session"));
    }

    #[test]
    fn game_solve_core_rejects_double_solve_same_mode() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);
        gss.subgame_solve
            .solving
            .store(true, std::sync::atomic::Ordering::Relaxed);

        // Default mode (subgame) should reject when subgame is already solving
        let result = game_solve_core(
            &gss, None, None, None, None, None, None, None, None, None, None,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already in progress"));
    }

    #[test]
    fn game_solve_core_rejects_double_solve_exact_mode() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);
        gss.exact_solve
            .solving
            .store(true, std::sync::atomic::Ordering::Relaxed);

        // Exact mode should reject when exact is already solving
        let result = game_solve_core(
            &gss,
            Some("exact".to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already in progress"));
    }

    #[test]
    fn game_solve_core_allows_different_mode_concurrent() {
        let gss = GameSessionState::default();
        let session = make_decision_session();
        *gss.session.write() = Some(session);
        // Subgame is already solving
        gss.subgame_solve
            .solving
            .store(true, std::sync::atomic::Ordering::Relaxed);

        // Exact mode should NOT be rejected (different mode)
        // It will still fail because it's a preflop node, but the error
        // should NOT be "already in progress"
        let result = game_solve_core(
            &gss,
            Some("exact".to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err());
        assert!(!result.unwrap_err().contains("already in progress"));
    }

    #[test]
    fn game_solve_core_accepts_enable_gadget_param() {
        let gss = GameSessionState::default();
        // Should reject (no session) but must accept the enable_gadget parameter
        let result = game_solve_core(
            &gss,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(true),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No game session"));
    }

    // -------------------------------------------------------------------
    // game_cancel_solve tests
    // -------------------------------------------------------------------

    #[test]
    fn cancel_solve_sets_cancel_flag_subgame() {
        let gss = GameSessionState::default();
        assert!(!gss
            .subgame_solve
            .cancel
            .load(std::sync::atomic::Ordering::Relaxed));
        game_cancel_solve_core(&gss, None).unwrap();
        assert!(gss
            .subgame_solve
            .cancel
            .load(std::sync::atomic::Ordering::Relaxed));
        // exact_solve should be unaffected
        assert!(!gss
            .exact_solve
            .cancel
            .load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn cancel_solve_sets_cancel_flag_exact() {
        let gss = GameSessionState::default();
        assert!(!gss
            .exact_solve
            .cancel
            .load(std::sync::atomic::Ordering::Relaxed));
        game_cancel_solve_core(&gss, Some("exact".to_string())).unwrap();
        assert!(gss
            .exact_solve
            .cancel
            .load(std::sync::atomic::Ordering::Relaxed));
        // subgame_solve should be unaffected
        assert!(!gss
            .subgame_solve
            .cancel
            .load(std::sync::atomic::Ordering::Relaxed));
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
        let board = vec![
            "Ah".to_string(),
            "Kd".to_string(),
            "Qc".to_string(),
            "Js".to_string(),
        ];
        let (_flop, turn, river, state) = parse_solve_board(&board).unwrap();
        assert_eq!(state, range_solver::BoardState::Turn);
        assert!(turn < 52);
        assert_eq!(river, range_solver::card::NOT_DEALT);
    }

    #[test]
    fn parse_solve_board_river() {
        let board = vec![
            "Ah".to_string(),
            "Kd".to_string(),
            "Qc".to_string(),
            "Js".to_string(),
            "Ts".to_string(),
        ];
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
        let game =
            build_solve_game(&board, &weights, &weights, 20, 90, &sizes, false, None).unwrap();
        // Flop solve with depth_limit=Some(0) should have boundary nodes
        assert!(game.num_boundary_nodes() > 0);
    }

    #[test]
    fn build_solve_game_exact_has_no_boundary_nodes_for_flop() {
        // Exact mode: flop solve has depth_limit=None, no boundary nodes
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5, 1.0]];
        let game =
            build_solve_game(&board, &weights, &weights, 20, 90, &sizes, true, None).unwrap();
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
        let game =
            build_solve_game(&board, &weights, &weights, 20, 90, &sizes, false, Some(1)).unwrap();
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
        let game =
            build_solve_game(&board, &weights, &weights, 20, 90, &sizes, false, Some(2)).unwrap();
        // depth_limit=2 from flop = full solve, no boundaries
        assert_eq!(game.num_boundary_nodes(), 0);
    }

    #[test]
    fn build_solve_game_depth_limit_none_defaults_to_zero() {
        // depth_limit_override=None should behave like depth_limit=0 (current behavior)
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5, 1.0]];
        let game =
            build_solve_game(&board, &weights, &weights, 20, 90, &sizes, false, None).unwrap();
        // Should have boundary nodes (same as depth_limit=0)
        assert!(game.num_boundary_nodes() > 0);
    }

    #[test]
    fn build_solve_game_exact_ignores_depth_limit_override() {
        // exact=true should always use depth_limit=None regardless of override
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5, 1.0]];
        let game =
            build_solve_game(&board, &weights, &weights, 20, 90, &sizes, true, Some(0)).unwrap();
        // Exact mode ignores depth_limit_override
        assert_eq!(game.num_boundary_nodes(), 0);
    }

    #[test]
    fn build_solve_game_river_ignores_depth_limit_override() {
        // River solve should always use depth_limit=None regardless of override
        let board = vec![
            "Ah".to_string(),
            "Kd".to_string(),
            "Qc".to_string(),
            "7s".to_string(),
            "2h".to_string(),
        ];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5, 1.0]];
        let game =
            build_solve_game(&board, &weights, &weights, 20, 90, &sizes, false, Some(0)).unwrap();
        // River solve has no boundaries regardless of depth_limit_override
        assert_eq!(game.num_boundary_nodes(), 0);
    }

    // -------------------------------------------------------------------
    // build_solve_game_parts test
    // -------------------------------------------------------------------

    #[test]
    fn build_solve_game_parts_returns_card_config_and_action_tree() {
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5, 1.0]];
        let (card_config, action_tree) =
            build_solve_game_parts(&board, &weights, &weights, 20, 90, &sizes, false, None)
                .unwrap();
        // The card_config should have both ranges set
        assert!(!card_config.range[0].is_empty());
        assert!(!card_config.range[1].is_empty());
        // The action tree should have no invalid terminals
        assert!(action_tree.invalid_terminals().is_empty());
    }

    #[test]
    fn build_solve_game_with_root_can_start_sb_facing_bb_bet() {
        let board = vec![
            "Ah".to_string(),
            "Kd".to_string(),
            "Qc".to_string(),
            "Js".to_string(),
        ];
        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5], vec![1.0]];
        let root = SolveGameRoot {
            starting_pot: 40,
            initial_player: 1,
            initial_stacks: [140, 180],
            initial_prev_action: range_solver::Action::Bet(40),
            initial_prev_amount: 40,
            initial_amount: 0,
            initial_num_bets: 1,
        };

        let game = build_solve_game_with_root(
            &board,
            &weights,
            &weights,
            80,
            140,
            &sizes,
            true,
            None,
            Some(root),
        )
        .unwrap();

        assert_eq!(game.current_player(), 1);
        let actions = game.available_actions();
        assert!(actions.contains(&range_solver::Action::Fold));
        assert!(actions.contains(&range_solver::Action::Call));
        assert!(actions
            .iter()
            .any(|action| matches!(action, range_solver::Action::Raise(_))));
    }

    // -------------------------------------------------------------------
    // solve state reset on game_new tests
    // -------------------------------------------------------------------

    #[test]
    fn game_new_resets_solve_state() {
        let gss = GameSessionState::default();
        // Simulate prior solve
        gss.subgame_solve
            .iteration
            .store(100, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve
            .max_iterations
            .store(200, std::sync::atomic::Ordering::Relaxed);
        gss.subgame_solve
            .solving
            .store(false, std::sync::atomic::Ordering::Relaxed);

        // game_new_core needs ExplorationState and PostflopState, but
        // we can test the reset by calling reset_solve_state directly
        gss.subgame_solve.reset();

        assert_eq!(
            gss.subgame_solve
                .iteration
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            gss.subgame_solve
                .max_iterations
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert!(gss.subgame_solve.matrix_snapshot.read().is_none());
    }

    // -------------------------------------------------------------------
    // build_solve_matrix tests (basic structure)
    // -------------------------------------------------------------------

    #[test]
    fn build_solve_matrix_from_postflop_game() {
        // Build a tiny PostFlopGame and verify matrix extraction
        use range_solver::bet_size::BetSizeOptions;
        use range_solver::card::{flop_from_str, NOT_DEALT};
        use range_solver::range::Range;
        use range_solver::{ActionTree, BoardState, CardConfig, PostFlopGame, TreeConfig};

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
            ..Default::default()
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

    fn make_bet_amount_session() -> GameSession {
        use poker_solver_core::blueprint_v2::game_tree::TerminalKind;
        let tree = V2GameTree {
            nodes: vec![
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Fold, TreeAction::Bet(8.0)],
                    children: vec![1, 2],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 1 },
                    pot: 6.0,
                    stacks: [199.0, 201.0],
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Fold, TreeAction::Call],
                    children: vec![3, 4],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 0 },
                    pot: 16.0,
                    stacks: [192.0, 200.0],
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 16.0,
                    stacks: [192.0, 192.0],
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

    fn make_turn_root_check_to_sb_session() -> GameSession {
        use poker_solver_core::blueprint_v2::game_tree::TerminalKind;

        let tree = V2GameTree {
            nodes: vec![
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Bet(4.0)],
                    children: vec![1],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Call],
                    children: vec![2],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Chance {
                    next_street: Street::Flop,
                    child: 3,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Flop,
                    actions: vec![TreeAction::Check],
                    children: vec![4],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Flop,
                    actions: vec![TreeAction::Check],
                    children: vec![5],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Chance {
                    next_street: Street::Turn,
                    child: 6,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Turn,
                    actions: vec![TreeAction::Check, TreeAction::Bet(48.0)],
                    children: vec![7, 8],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Turn,
                    actions: vec![TreeAction::Check, TreeAction::Bet(48.0)],
                    children: vec![9, 10],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Turn,
                    actions: vec![TreeAction::Fold, TreeAction::Call],
                    children: vec![11, 12],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 8.0,
                    stacks: [196.0, 196.0],
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 56.0,
                    stacks: [172.0, 196.0],
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 1 },
                    pot: 56.0,
                    stacks: [196.0, 172.0],
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 104.0,
                    stacks: [172.0, 172.0],
                },
            ],
            root: 0,
            dealer: 0,
            starting_stack: 200.0,
        };

        let mut session = GameSession::new_for_test(tree);
        session.play_action("0").unwrap();
        session.play_action("0").unwrap();
        session.deal_card("Ks").unwrap();
        session.deal_card("8d").unwrap();
        session.deal_card("3c").unwrap();
        session.play_action("0").unwrap();
        session.play_action("0").unwrap();
        session.deal_card("Js").unwrap();
        session
    }

    fn make_turn_street_navigation_session() -> GameSession {
        use poker_solver_core::blueprint_v2::game_tree::TerminalKind;

        let terminal = |pot: f64| V2GameNode::Terminal {
            kind: TerminalKind::Showdown,
            pot,
            stacks: [100.0, 100.0],
        };

        let tree = V2GameTree {
            nodes: vec![
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Bet(4.0)],
                    children: vec![1],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Call],
                    children: vec![2],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Chance {
                    next_street: Street::Flop,
                    child: 3,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Flop,
                    actions: vec![TreeAction::Check],
                    children: vec![4],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Flop,
                    actions: vec![TreeAction::Check],
                    children: vec![5],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Chance {
                    next_street: Street::Turn,
                    child: 6,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Turn,
                    actions: vec![
                        TreeAction::Check,
                        TreeAction::Bet(48.0),
                        TreeAction::Bet(110.0),
                        TreeAction::AllIn,
                    ],
                    children: vec![7, 11, 12, 13],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Turn,
                    actions: vec![
                        TreeAction::Check,
                        TreeAction::Bet(48.0),
                        TreeAction::Bet(110.0),
                        TreeAction::AllIn,
                    ],
                    children: vec![14, 8, 9, 10],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Turn,
                    actions: vec![TreeAction::Fold, TreeAction::Call, TreeAction::AllIn],
                    children: vec![15, 16, 17],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Turn,
                    actions: vec![TreeAction::Fold, TreeAction::Call, TreeAction::AllIn],
                    children: vec![18, 19, 20],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Turn,
                    actions: vec![TreeAction::Fold, TreeAction::Call],
                    children: vec![21, 22],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Turn,
                    actions: vec![TreeAction::Fold, TreeAction::Call, TreeAction::AllIn],
                    children: vec![23, 24, 25],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Turn,
                    actions: vec![TreeAction::Fold, TreeAction::Call, TreeAction::AllIn],
                    children: vec![26, 27, 28],
                    blueprint_decision_idx: None,
                },
                terminal(400.0),
                terminal(400.0),
                terminal(104.0),
                terminal(104.0),
                terminal(104.0),
                terminal(400.0),
                terminal(166.0),
                terminal(166.0),
                terminal(400.0),
                terminal(400.0),
                terminal(400.0),
                terminal(104.0),
                terminal(104.0),
                terminal(400.0),
                terminal(166.0),
                terminal(166.0),
                terminal(400.0),
            ],
            root: 0,
            dealer: 0,
            starting_stack: 200.0,
        };

        let mut session = GameSession::new_for_test(tree);
        session.play_action("0").unwrap();
        session.play_action("0").unwrap();
        session.deal_card("Ks").unwrap();
        session.deal_card("8d").unwrap();
        session.deal_card("3c").unwrap();
        session.play_action("0").unwrap();
        session.play_action("0").unwrap();
        session.deal_card("Js").unwrap();
        session
    }

    fn make_turn_sb_facing_bb_bet_session() -> GameSession {
        use poker_solver_core::blueprint_v2::game_tree::TerminalKind;

        let tree = V2GameTree {
            nodes: vec![
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Bet(20.0)],
                    children: vec![1],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Call],
                    children: vec![2],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Chance {
                    next_street: Street::Flop,
                    child: 3,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Flop,
                    actions: vec![TreeAction::Check],
                    children: vec![4],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Flop,
                    actions: vec![TreeAction::Check],
                    children: vec![5],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Chance {
                    next_street: Street::Turn,
                    child: 6,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Turn,
                    actions: vec![TreeAction::Bet(40.0)],
                    children: vec![7],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Turn,
                    actions: vec![TreeAction::Fold, TreeAction::Call, TreeAction::Raise(120.0)],
                    children: vec![8, 9, 10],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 1 },
                    pot: 80.0,
                    stacks: [180.0, 140.0],
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 120.0,
                    stacks: [140.0, 140.0],
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 200.0,
                    stacks: [60.0, 140.0],
                },
            ],
            root: 0,
            dealer: 0,
            starting_stack: 200.0,
        };

        let mut session = GameSession::new_for_test(tree);
        session.play_action("0").unwrap();
        session.play_action("0").unwrap();
        session.deal_card("Ah").unwrap();
        session.deal_card("Kd").unwrap();
        session.deal_card("Qc").unwrap();
        session.play_action("0").unwrap();
        session.play_action("0").unwrap();
        session.deal_card("Js").unwrap();
        session.play_action("0").unwrap();
        session
    }

    #[test]
    fn solve_root_config_matches_sb_to_act_facing_bb_bet() {
        let session = make_turn_sb_facing_bb_bet_session();
        let node = &session.tree.nodes[session.node_idx as usize];
        let player = match node {
            V2GameNode::Decision { player, .. } => *player,
            _ => panic!("expected decision node"),
        };
        assert_eq!(session.position_label(player), "SB");

        let root = session.solve_game_root_for_player(player).unwrap();
        assert_eq!(root.initial_player, 1);
        assert_eq!(root.initial_stacks, [140, 180]);
        assert_eq!(root.initial_prev_action, range_solver::Action::Bet(40));
        assert_eq!(root.initial_prev_amount, 40);
        assert_eq!(root.initial_amount, 0);
        assert_eq!(root.initial_num_bets, 1);
        assert_eq!(root.starting_pot, 40);

        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5], vec![1.0]];
        let game = build_solve_game_with_root(
            &session.board,
            &weights,
            &weights,
            session.compute_pot(),
            effective_stack_for_solve_root(&root),
            &sizes,
            true,
            None,
            Some(root),
        )
        .unwrap();

        assert_eq!(game.current_player(), 1);
        assert!(validate_solve_root_actor_label(
            game.current_player(),
            &["BB".to_string(), "SB".to_string()],
            session.position_label(player),
        )
        .is_ok());
    }

    fn make_turn_sb_facing_bb_all_in_session() -> GameSession {
        use poker_solver_core::blueprint_v2::game_tree::TerminalKind;

        let tree = V2GameTree {
            nodes: vec![
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Bet(20.0)],
                    children: vec![1],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Preflop,
                    actions: vec![TreeAction::Call],
                    children: vec![2],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Chance {
                    next_street: Street::Flop,
                    child: 3,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Flop,
                    actions: vec![TreeAction::Check],
                    children: vec![4],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Flop,
                    actions: vec![TreeAction::Check],
                    children: vec![5],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Chance {
                    next_street: Street::Turn,
                    child: 6,
                },
                V2GameNode::Decision {
                    player: 1,
                    street: Street::Turn,
                    actions: vec![TreeAction::AllIn],
                    children: vec![7],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Decision {
                    player: 0,
                    street: Street::Turn,
                    actions: vec![TreeAction::Fold, TreeAction::Call],
                    children: vec![8, 9],
                    blueprint_decision_idx: None,
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Fold { winner: 1 },
                    pot: 220.0,
                    stacks: [180.0, 0.0],
                },
                V2GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 400.0,
                    stacks: [0.0, 0.0],
                },
            ],
            root: 0,
            dealer: 0,
            starting_stack: 200.0,
        };

        let mut session = GameSession::new_for_test(tree);
        session.play_action("0").unwrap();
        session.play_action("0").unwrap();
        session.deal_card("Ah").unwrap();
        session.deal_card("Kd").unwrap();
        session.deal_card("Qc").unwrap();
        session.play_action("0").unwrap();
        session.play_action("0").unwrap();
        session.deal_card("Js").unwrap();
        session.play_action("0").unwrap();
        session
    }

    #[test]
    fn build_solve_game_with_root_accepts_all_in_response_with_one_zero_stack() {
        let session = make_turn_sb_facing_bb_all_in_session();
        let node = &session.tree.nodes[session.node_idx as usize];
        let player = match node {
            V2GameNode::Decision { player, .. } => *player,
            _ => panic!("expected decision node"),
        };

        let root = session.solve_game_root_for_player(player).unwrap();
        assert_eq!(root.initial_player, 1);
        assert_eq!(root.initial_stacks, [0, 180]);
        assert_eq!(root.initial_prev_action, range_solver::Action::AllIn(180));
        assert_eq!(effective_stack_for_solve_root(&root), 180);

        let weights = vec![1.0f32; 1326];
        let sizes = vec![vec![0.5], vec![1.0]];
        let game = build_solve_game_with_root(
            &session.board,
            &weights,
            &weights,
            session.compute_pot(),
            effective_stack_for_solve_root(&root),
            &sizes,
            true,
            None,
            Some(root),
        )
        .unwrap();

        assert_eq!(game.current_player(), 1);
        let actions = game.available_actions();
        assert_eq!(actions.len(), 2);
        assert!(actions.contains(&range_solver::Action::Fold));
        assert!(actions.contains(&range_solver::Action::Call));
        assert!(!actions
            .iter()
            .any(|action| matches!(action, range_solver::Action::Raise(_))));
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
        session
            .load_spot("sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb")
            .unwrap();
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
    fn make_cached_node(
        hand_label: &str,
        action_labels: &[&str],
        position: &str,
    ) -> CachedSolveNode {
        fn action_type_for_label(label: &str) -> String {
            match label.trim().to_ascii_lowercase().as_str() {
                "fold" => "fold".to_string(),
                "check" => "check".to_string(),
                "call" => "call".to_string(),
                "all-in" | "allin" => "allin".to_string(),
                _ => "bet".to_string(),
            }
        }

        let actions: Vec<GameAction> = action_labels
            .iter()
            .enumerate()
            .map(|(i, &lbl)| GameAction {
                id: i.to_string(),
                label: lbl.to_string(),
                action_type: action_type_for_label(lbl),
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

    fn make_cached_node_with_actions(
        hand_label: &str,
        actions: Vec<GameAction>,
        position: &str,
    ) -> CachedSolveNode {
        let matrix = GameMatrix {
            cells: vec![vec![GameMatrixCell {
                hand: hand_label.to_string(),
                suited: false,
                pair: false,
                probabilities: vec![1.0; actions.len()],
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

    fn anchor_solve_to_current_session(ss: &SolveState, session: &GameSession) {
        *ss.solve_anchor.write() = Some(SolveAnchor {
            node_idx: session.node_idx,
            board: session.board.clone(),
            action_ids: session
                .action_history
                .iter()
                .map(|a| a.action_id.clone())
                .collect(),
        });
    }

    fn seed_turn_street_navigation_cache(ss: &SolveState) {
        let root = make_cached_node("ROOT_BB", &["Check", "24bb", "55bb", "All-in"], "BB");
        let sb_after_check =
            make_cached_node("SB_AFTER_CHECK", &["Check", "24bb", "55bb", "All-in"], "SB");
        let bb_vs_sb_24 = make_cached_node("BB_VS_SB_24", &["Fold", "Call", "All-in"], "BB");
        let bb_vs_sb_55 = make_cached_node("BB_VS_SB_55", &["Fold", "Call", "All-in"], "BB");
        let bb_vs_sb_allin = make_cached_node("BB_VS_SB_ALLIN", &["Fold", "Call"], "BB");
        let sb_vs_bb_24 = make_cached_node("SB_VS_BB_24", &["Fold", "Call", "All-in"], "SB");
        let sb_vs_bb_55 = make_cached_node("SB_VS_BB_55", &["Fold", "Call", "All-in"], "SB");

        let mut cache = ss.solve_cache.write();
        cache.insert(vec![], root);
        cache.insert(vec![0], sb_after_check);
        cache.insert(vec![0, 1], bb_vs_sb_24);
        cache.insert(vec![0, 2], bb_vs_sb_55);
        cache.insert(vec![0, 3], bb_vs_sb_allin);
        cache.insert(vec![1], sb_vs_bb_24);
        cache.insert(vec![2], sb_vs_bb_55);
        ss.iteration.store(100, Ordering::Relaxed);
    }

    fn matrix_label(state: &GameState) -> &str {
        state.matrix.as_ref().expect("expected solved matrix").cells[0][0]
            .hand
            .as_str()
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
        ss.solve_cache
            .write()
            .insert(vec![], make_cached_node("ROOT", &["Check", "Bet"], "BB"));
        ss.solve_cache
            .write()
            .insert(vec![0], make_cached_node("CHILD0", &["Fold", "Call"], "SB"));
        ss.solve_path.write().push(0);

        assert!(!ss.solve_cache.read().is_empty());
        assert!(!ss.solve_path.read().is_empty());

        ss.reset();

        assert!(ss.solve_cache.read().is_empty());
        assert!(ss.solve_path.read().is_empty());
    }

    #[test]
    fn stale_solve_generation_cannot_publish_after_reset() {
        let ss = SolveState::default();
        let generation = reset_solve_state_for_start(
            &ss,
            10,
            "BB".to_string(),
            SolveAnchor {
                node_idx: 0,
                board: vec!["As".to_string(), "Kd".to_string(), "Qh".to_string()],
                action_ids: vec![],
            },
        );
        assert!(ss.publish_if_current(generation, |state| {
            state.iteration.store(1, Ordering::Relaxed);
        }));

        ss.reset();

        assert_ne!(ss.generation.load(Ordering::Acquire), generation);
        assert!(!ss.publish_if_current(generation, |state| {
            state.iteration.store(99, Ordering::Relaxed);
        }));
        assert_eq!(ss.iteration.load(Ordering::Relaxed), 0);
        assert!(!ss.solving.load(Ordering::Relaxed));
        assert!(ss.cancel.load(Ordering::Acquire));
    }

    #[test]
    fn play_action_serves_cached_matrix_within_solved_tree() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        *gss.session.write() = Some(session);

        // Populate solve cache: root and one child
        let root_node = make_cached_node("ROOT", &["Fold", "Call"], "BB");
        let child_node = make_cached_node("CHILD", &["Check", "Fold"], "SB");
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![], root_node);
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![1], child_node);
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
    fn play_action_preserves_cache_when_source_path_misses() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        *gss.session.write() = Some(session);

        // Populate solve cache with only root (no children cached)
        let root_node = make_cached_node("ROOT", &["Fold", "Call"], "BB");
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![], root_node);
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);

        // Play action "1" (Call) -- path [1] not in cache, should not reset
        let source = Some("subgame".to_string());
        let _state = game_play_action_core(&gss, "1", source).unwrap();
        // Cache should be preserved for possible later reuse.
        assert!(!gss.subgame_solve.solve_cache.read().is_empty());
        assert!(gss.subgame_solve.solve_path.read().is_empty());
    }

    #[test]
    fn blueprint_play_does_not_consume_or_clear_subgame_cache() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        *gss.session.write() = Some(session);

        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![], make_cached_node("ROOT", &["Fold", "Call"], "BB"));
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![1], make_cached_node("CHILD", &["Check"], "SB"));
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);

        let _state = game_play_action_core(&gss, "1", None).unwrap();

        assert_eq!(*gss.subgame_solve.solve_path.read(), Vec::<usize>::new());
        assert!(gss.subgame_solve.solve_cache.read().contains_key(&vec![1]));
    }

    #[test]
    fn source_switch_after_blueprint_navigation_derives_cached_child_path() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        anchor_solve_to_current_session(&gss.subgame_solve, &session);
        *gss.session.write() = Some(session);

        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![], make_cached_node("ROOT", &["Fold", "Call"], "BB"));
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![1], make_cached_node("CHILD", &["Check"], "SB"));
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);

        let _blueprint_state = game_play_action_core(&gss, "1", None).unwrap();
        assert_eq!(*gss.subgame_solve.solve_path.read(), Vec::<usize>::new());

        let state = game_get_state_core(&gss, Some("subgame".to_string())).unwrap();
        let matrix = state.matrix.expect("subgame child matrix");

        assert_eq!(matrix.cells[0][0].hand, "CHILD");
        assert_eq!(state.position, "SB");
        assert_eq!(*gss.subgame_solve.solve_path.read(), vec![1]);
    }

    #[test]
    fn source_switch_matches_cached_path_by_action_semantics_not_id() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        anchor_solve_to_current_session(&gss.subgame_solve, &session);
        *gss.session.write() = Some(session);

        let root_actions = vec![
            GameAction {
                id: "0".to_string(),
                label: "Call".to_string(),
                action_type: "call".to_string(),
            },
            GameAction {
                id: "1".to_string(),
                label: "Fold".to_string(),
                action_type: "fold".to_string(),
            },
        ];
        gss.subgame_solve.solve_cache.write().insert(
            vec![],
            make_cached_node_with_actions("ROOT", root_actions, "BB"),
        );
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![0], make_cached_node("CHILD_CALL", &["Check"], "SB"));
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);

        let _blueprint_state = game_play_action_core(&gss, "1", None).unwrap();
        let state = game_get_state_core(&gss, Some("subgame".to_string())).unwrap();
        let matrix = state.matrix.expect("subgame child matrix");

        assert_eq!(matrix.cells[0][0].hand, "CHILD_CALL");
        assert_eq!(*gss.subgame_solve.solve_path.read(), vec![0]);
    }

    #[test]
    fn source_play_maps_solver_action_id_to_session_action_semantically() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        anchor_solve_to_current_session(&gss.subgame_solve, &session);
        *gss.session.write() = Some(session);

        let root_actions = vec![
            GameAction {
                id: "0".to_string(),
                label: "Call".to_string(),
                action_type: "call".to_string(),
            },
            GameAction {
                id: "1".to_string(),
                label: "Fold".to_string(),
                action_type: "fold".to_string(),
            },
        ];
        gss.subgame_solve.solve_cache.write().insert(
            vec![],
            make_cached_node_with_actions("ROOT", root_actions, "BB"),
        );
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![0], make_cached_node("CHILD_CALL", &["Check"], "SB"));
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);

        let state = game_play_action_core(&gss, "0", Some("subgame".to_string())).unwrap();
        let matrix = state.matrix.expect("subgame child matrix");

        assert_eq!(matrix.cells[0][0].hand, "CHILD_CALL");
        assert_eq!(*gss.subgame_solve.solve_path.read(), vec![0]);
        let session = gss.session.read();
        let record = &session.as_ref().unwrap().action_history[0];
        assert_eq!(record.action_id, "1");
        assert_eq!(record.label, "Call");
    }

    #[test]
    fn source_play_matches_bet_amount_when_solver_action_id_differs() {
        let gss = GameSessionState::default();
        let session = make_bet_amount_session();
        anchor_solve_to_current_session(&gss.subgame_solve, &session);
        *gss.session.write() = Some(session);

        let root_actions = vec![
            GameAction {
                id: "0".to_string(),
                label: "4bb".to_string(),
                action_type: "bet".to_string(),
            },
            GameAction {
                id: "1".to_string(),
                label: "Fold".to_string(),
                action_type: "fold".to_string(),
            },
        ];
        gss.subgame_solve.solve_cache.write().insert(
            vec![],
            make_cached_node_with_actions("ROOT", root_actions, "SB"),
        );
        gss.subgame_solve.solve_cache.write().insert(
            vec![0],
            make_cached_node("CHILD_BET", &["Fold", "Call"], "BB"),
        );
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);

        let state = game_play_action_core(&gss, "0", Some("subgame".to_string())).unwrap();
        let matrix = state.matrix.expect("subgame child matrix");

        assert_eq!(matrix.cells[0][0].hand, "CHILD_BET");
        assert_eq!(*gss.subgame_solve.solve_path.read(), vec![0]);
        let session = gss.session.read();
        let record = &session.as_ref().unwrap().action_history[0];
        assert_eq!(record.action_id, "1");
        assert_eq!(record.label, "4bb");
    }

    #[test]
    fn source_play_overlays_cached_matrix_after_session_action_id_fallback() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        anchor_solve_to_current_session(&gss.subgame_solve, &session);
        *gss.session.write() = Some(session);

        let root_actions = vec![GameAction {
            id: "solver-call".to_string(),
            label: "Call".to_string(),
            action_type: "call".to_string(),
        }];
        gss.subgame_solve.solve_cache.write().insert(
            vec![],
            make_cached_node_with_actions("ROOT", root_actions, "BB"),
        );
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![0], make_cached_node("CHILD_CALL", &["Check"], "SB"));
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);

        let state = game_play_action_core(&gss, "1", Some("subgame".to_string())).unwrap();
        let matrix = state
            .matrix
            .expect("subgame child matrix should overlay fallback navigation");

        assert_eq!(matrix.cells[0][0].hand, "CHILD_CALL");
        assert_eq!(state.position, "SB");
        assert_eq!(*gss.subgame_solve.solve_path.read(), vec![0]);
    }

    fn assert_turn_street_navigation_uses_solve_cache(source: &str) {
        let gss = GameSessionState::default();
        let session = make_turn_street_navigation_session();
        let mode = Some(source.to_string());
        let ss = gss.solve_for(&mode);
        anchor_solve_to_current_session(ss, &session);
        seed_turn_street_navigation_cache(ss);
        *gss.session.write() = Some(session);

        let state = game_get_state_core(&gss, mode.clone()).unwrap();
        assert_eq!(matrix_label(&state), "ROOT_BB");
        assert_eq!(state.position, "BB");

        let state = game_play_action_core(&gss, "0", mode.clone()).unwrap();
        assert_eq!(matrix_label(&state), "SB_AFTER_CHECK");
        assert_eq!(state.position, "SB");
        assert_eq!(*ss.solve_path.read(), vec![0]);

        let state = game_play_action_core(&gss, "1", mode.clone()).unwrap();
        assert_eq!(matrix_label(&state), "BB_VS_SB_24");
        assert_eq!(state.position, "BB");
        assert_eq!(*ss.solve_path.read(), vec![0, 1]);

        let state = game_back_core(&gss, mode.clone()).unwrap();
        assert_eq!(matrix_label(&state), "SB_AFTER_CHECK");
        assert_eq!(*ss.solve_path.read(), vec![0]);

        let state = game_play_action_core(&gss, "2", mode.clone()).unwrap();
        assert_eq!(matrix_label(&state), "BB_VS_SB_55");
        assert_eq!(*ss.solve_path.read(), vec![0, 2]);

        let state = game_back_core(&gss, mode.clone()).unwrap();
        assert_eq!(matrix_label(&state), "SB_AFTER_CHECK");
        let state = game_play_action_core(&gss, "3", mode.clone()).unwrap();
        assert_eq!(matrix_label(&state), "BB_VS_SB_ALLIN");
        assert_eq!(*ss.solve_path.read(), vec![0, 3]);

        let state = game_back_core(&gss, mode.clone()).unwrap();
        assert_eq!(matrix_label(&state), "SB_AFTER_CHECK");
        let state = game_back_core(&gss, mode.clone()).unwrap();
        assert_eq!(matrix_label(&state), "ROOT_BB");
        assert_eq!(*ss.solve_path.read(), Vec::<usize>::new());

        let state = game_play_action_core(&gss, "1", mode.clone()).unwrap();
        assert_eq!(matrix_label(&state), "SB_VS_BB_24");
        assert_eq!(state.position, "SB");
        assert_eq!(*ss.solve_path.read(), vec![1]);

        let state = game_back_core(&gss, mode.clone()).unwrap();
        assert_eq!(matrix_label(&state), "ROOT_BB");
        let state = game_play_action_core(&gss, "2", mode).unwrap();
        assert_eq!(matrix_label(&state), "SB_VS_BB_55");
        assert_eq!(*ss.solve_path.read(), vec![2]);
    }

    #[test]
    fn solved_matrix_navigation_covers_turn_street_subgame_actions() {
        assert_turn_street_navigation_uses_solve_cache("subgame");
    }

    #[test]
    fn solved_matrix_navigation_covers_turn_street_exact_actions() {
        assert_turn_street_navigation_uses_solve_cache("exact");
    }

    #[test]
    fn solve_then_bb_check_serves_solved_sb_child_matrix() {
        let gss = GameSessionState::default();
        let session = make_turn_root_check_to_sb_session();
        *gss.session.write() = Some(session);

        game_solve_core(
            &gss,
            Some("subgame".to_string()),
            Some(1),
            None,
            Some(1),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while gss.subgame_solve.solving.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "solve did not finish before timeout"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert!(
            gss.subgame_solve.solve_cache.read().contains_key(&vec![0]),
            "completed root solve should cache the SB child after BB checks"
        );

        let state = game_play_action_core(&gss, "0", Some("subgame".to_string())).unwrap();
        let matrix = state
            .matrix
            .expect("BB check should return cached solved SB matrix");

        assert_eq!(state.position, "SB");
        assert_eq!(*gss.subgame_solve.solve_path.read(), vec![0]);
        let returned_actions: Vec<_> = matrix
            .actions
            .iter()
            .map(|action| (&action.label, &action.action_type))
            .collect();
        let cache = gss.subgame_solve.solve_cache.read();
        let cached_actions: Vec<_> = cache[&vec![0]]
            .actions
            .iter()
            .map(|action| (&action.label, &action.action_type))
            .collect();
        assert_eq!(returned_actions, cached_actions);
    }

    #[test]
    fn solve_start_clears_stale_cache_and_path_for_mode() {
        let ss = SolveState::default();
        ss.solve_cache
            .write()
            .insert(vec![], make_cached_node("STALE", &["Call"], "BB"));
        ss.solve_path.write().push(0);
        *ss.solve_actions.write() = vec![GameAction {
            id: "0".to_string(),
            label: "Stale".to_string(),
            action_type: "call".to_string(),
        }];

        reset_solve_state_for_start(
            &ss,
            123,
            "BB".to_string(),
            SolveAnchor {
                node_idx: 7,
                board: vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()],
                action_ids: vec!["1".to_string()],
            },
        );

        assert!(ss.solve_cache.read().is_empty());
        assert!(ss.solve_path.read().is_empty());
        assert!(ss.solve_actions.read().is_empty());
        assert_eq!(ss.max_iterations.load(Ordering::Relaxed), 123);
        assert_eq!(ss.solve_position.read().as_str(), "BB");
        assert!(ss.solving.load(Ordering::Relaxed));
    }

    #[test]
    fn solve_root_actor_guard_reports_session_and_range_labels() {
        let labels = ["BB".to_string(), "SB".to_string()];
        let err = validate_solve_root_actor_label(0, &labels, "SB").unwrap_err();

        assert_eq!(
            err,
            "Solve root actor mismatch: session is SB but range solver root is BB"
        );
        assert!(validate_solve_root_actor_label(0, &labels, "BB").is_ok());
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
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![], root_node);
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![1], child_node);
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
        assert!(gss.subgame_solve.solve_cache.read().contains_key(&vec![]));
    }

    #[test]
    fn back_at_solve_root_preserves_cache_without_overlay() {
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
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![], root_node);
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);
        // Path is empty (at solve root)

        // Go back -- before solve root, source cache is not consumed or cleared.
        let _state = game_back_core(&gss, Some("subgame".to_string())).unwrap();
        assert!(gss.subgame_solve.solve_cache.read().contains_key(&vec![]));
        assert!(gss.subgame_solve.solve_path.read().is_empty());
    }

    #[test]
    fn exact_and_subgame_caches_are_independent() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        *gss.session.write() = Some(session);

        gss.subgame_solve.solve_cache.write().insert(
            vec![],
            make_cached_node("SUB_ROOT", &["Fold", "Call"], "BB"),
        );
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![1], make_cached_node("SUB_CHILD", &["Check"], "SB"));
        gss.exact_solve.solve_cache.write().insert(
            vec![],
            make_cached_node("EXACT_ROOT", &["Fold", "Call"], "BB"),
        );
        gss.exact_solve
            .solve_cache
            .write()
            .insert(vec![1], make_cached_node("EXACT_CHILD", &["Check"], "SB"));

        let state = game_play_action_core(&gss, "1", Some("exact".to_string())).unwrap();
        let matrix = state.matrix.expect("exact child matrix");

        assert_eq!(matrix.cells[0][0].hand, "EXACT_CHILD");
        assert_eq!(*gss.exact_solve.solve_path.read(), vec![1]);
        assert_eq!(*gss.subgame_solve.solve_path.read(), Vec::<usize>::new());
        assert!(gss.subgame_solve.solve_cache.read().contains_key(&vec![1]));
    }

    #[test]
    fn stale_solve_anchor_does_not_overlay_unrelated_session_state() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        *gss.session.write() = Some(session);

        {
            let session = gss.session.read();
            let session = session.as_ref().unwrap();
            *gss.subgame_solve.solve_anchor.write() = Some(SolveAnchor {
                node_idx: session.node_idx,
                board: vec!["Ah".to_string(), "Kd".to_string(), "Qc".to_string()],
                action_ids: session
                    .action_history
                    .iter()
                    .map(|a| a.action_id.clone())
                    .collect(),
            });
        }
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);
        gss.subgame_solve.solve_cache.write().insert(
            vec![],
            make_cached_node("CACHE_ROOT", &["Fold", "Call"], "BB"),
        );
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![1], make_cached_node("CACHE_CHILD", &["Check"], "SB"));

        let _ = game_play_action_core(&gss, "1", None).unwrap();
        let state = game_get_state_core(&gss, Some("subgame".to_string())).unwrap();

        if let Some(matrix) = state.matrix {
            assert_ne!(matrix.cells[0][0].hand, "CACHE_ROOT");
        }
        assert_eq!(*gss.subgame_solve.solve_path.read(), Vec::<usize>::new());
        assert!(gss.subgame_solve.solve_cache.read().contains_key(&vec![]));
    }

    #[test]
    fn get_state_core_serves_cached_matrix_at_solve_path() {
        let gss = GameSessionState::default();
        let session = make_two_level_session();
        *gss.session.write() = Some(session);

        // Set up completed solve with cache
        gss.subgame_solve.solving.store(false, Ordering::Relaxed);
        gss.subgame_solve.iteration.store(100, Ordering::Relaxed);
        gss.subgame_solve
            .max_iterations
            .store(100, Ordering::Relaxed);
        *gss.subgame_solve.solve_start.write() = Some(std::time::Instant::now());

        let root_node = make_cached_node("CACHE_ROOT", &["Fold", "Call"], "BB");
        let child_node = make_cached_node("CACHE_CHILD", &["Check", "Fold"], "SB");
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![], root_node);
        gss.subgame_solve
            .solve_cache
            .write()
            .insert(vec![1], child_node);

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
        use range_solver::bet_size::BetSizeOptions;
        use range_solver::card::{flop_from_str, NOT_DEALT};
        use range_solver::range::Range;
        use range_solver::{ActionTree, BoardState, CardConfig, PostFlopGame, TreeConfig};

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
            ..Default::default()
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

        let player_labels = ["BB".to_string(), "SB".to_string()];
        let cache = build_solve_cache(&mut game, &player_labels);
        // Root should be present
        assert!(
            cache.contains_key(&vec![]),
            "cache should contain root entry"
        );
        assert!(
            cache
                .values()
                .all(|node| node.position != "OOP" && node.position != "IP"),
            "cache should use seat labels, not OOP/IP"
        );
        assert!(
            cache
                .values()
                .all(|node| node.position == "BB" || node.position == "SB"),
            "cache should contain only BB/SB labels"
        );
        // Root should have actions
        assert!(!cache[&vec![]].actions.is_empty());
        // Root should have a 13x13 matrix
        assert_eq!(cache[&vec![]].matrix.cells.len(), 13);
        // Should have more than just root (children for each action at root)
        assert!(
            cache.len() > 1,
            "cache should contain child entries too, got {}",
            cache.len()
        );
    }

    #[test]
    fn build_solve_matrix_at_current_works_without_back_to_root() {
        use range_solver::bet_size::BetSizeOptions;
        use range_solver::card::{flop_from_str, NOT_DEALT};
        use range_solver::range::Range;
        use range_solver::{ActionTree, BoardState, CardConfig, PostFlopGame, TreeConfig};

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
            ..Default::default()
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

    #[test]
    fn build_solve_matrix_at_current_uses_navigated_reach_weights() {
        use range_solver::bet_size::BetSizeOptions;
        use range_solver::card::{card_from_str, flop_from_str, NOT_DEALT};
        use range_solver::range::Range;
        use range_solver::{Action, ActionTree, BoardState, CardConfig, PostFlopGame, TreeConfig};

        let oop_range: Range = "AA,KK,QQ,JJ".parse().unwrap();
        let ip_range: Range = "AA,KK,QQ,JJ".parse().unwrap();
        let flop = flop_from_str("2c3d4h").unwrap();
        let turn = card_from_str("Js").unwrap();

        let sizes = BetSizeOptions::try_from(("50%,a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
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
            ..Default::default()
        };
        let action_tree = ActionTree::new(tree_config).unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop,
            turn,
            river: NOT_DEALT,
        };
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);

        let root_actions = game.available_actions();
        let check_idx = root_actions
            .iter()
            .position(|action| *action == Action::Check)
            .expect("turn root should offer check");
        let num_actions = root_actions.len();
        let num_hands = game.num_private_hands(game.current_player());

        let mut locked_strategy = vec![0.0f32; num_actions * num_hands];
        let non_check_idx = (0..num_actions)
            .find(|idx| *idx != check_idx)
            .expect("turn root should have a non-check action");
        for hand_idx in 0..num_hands {
            locked_strategy[non_check_idx * num_hands + hand_idx] = 1.0;
        }
        game.lock_current_strategy(&locked_strategy);
        game.play(check_idx);

        assert!(!game.is_terminal_node());
        assert!(!game.is_chance_node());

        let matrix = build_solve_matrix_at_current(&mut game, None);
        let total_weight: f32 = matrix.cells.iter().flatten().map(|cell| cell.weight).sum();

        assert_eq!(
            total_weight, 0.0,
            "child matrix must use the post-action reach, not the root range"
        );
    }

    #[test]
    fn build_solve_matrix_at_current_keeps_display_weights_bounded() {
        use range_solver::bet_size::BetSizeOptions;
        use range_solver::card::{card_from_str, flop_from_str, NOT_DEALT};
        use range_solver::range::Range;
        use range_solver::{Action, ActionTree, BoardState, CardConfig, PostFlopGame, TreeConfig};

        let oop_range: Range = "AA,KK,QQ,JJ".parse().unwrap();
        let ip_range: Range = "AA,KK,QQ,JJ".parse().unwrap();
        let flop = flop_from_str("2c3d4h").unwrap();
        let turn = card_from_str("Js").unwrap();

        let sizes = BetSizeOptions::try_from(("50%,a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
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
            ..Default::default()
        };
        let action_tree = ActionTree::new(tree_config).unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop,
            turn,
            river: NOT_DEALT,
        };
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);

        let check_idx = game
            .available_actions()
            .iter()
            .position(|action| *action == Action::Check)
            .expect("turn root should offer check");
        game.play(check_idx);

        assert!(!game.is_terminal_node());
        assert!(!game.is_chance_node());

        let matrix = build_solve_matrix_at_current(&mut game, None);
        let max_weight = matrix
            .cells
            .iter()
            .flatten()
            .map(|cell| cell.weight)
            .fold(0.0f32, f32::max);

        assert!(
            max_weight <= 1.0,
            "display weights drive cell bar height and must stay in 0..1, got {max_weight}"
        );
    }

    // (RefreshProgress tests deleted: rollout boundary-eval path removed)

    // -------------------------------------------------------------------
    // StreetBoundaryConfig + resolve_street_boundary tests
    // -------------------------------------------------------------------

    fn cfvnet_mode(model_path: &str) -> StreetBoundaryMode {
        StreetBoundaryMode::Cfvnet {
            model_path: model_path.to_string(),
            inference_mode: cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::default(),
        }
    }

    fn cfvnet_kind(model_path: &str) -> BoundaryKind {
        BoundaryKind::Cfvnet {
            model_path: model_path.to_string(),
            inference_mode: cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::default(),
        }
    }

    fn direct_cfvnet_kind(model_path: &str) -> BoundaryKind {
        BoundaryKind::Cfvnet {
            model_path: model_path.to_string(),
            inference_mode: cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::Direct,
        }
    }

    fn direct_normalized_legacy_cfvnet_kind(model_path: &str) -> BoundaryKind {
        BoundaryKind::Cfvnet {
            model_path: model_path.to_string(),
            inference_mode:
                cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::DirectNormalizedLegacy,
        }
    }

    #[test]
    fn boundary_evaluator_log_line_includes_direct_model() {
        let cut = Some((0, direct_cfvnet_kind("/models/turn.onnx")));
        assert_eq!(
            boundary_evaluator_log_line("subgame", &cut),
            "[solve] solver: subgame; boundary evaluator: Direct CFVNet; depth_limit=0; inference_mode=direct; model=/models/turn.onnx"
        );
    }

    #[test]
    fn boundary_evaluator_log_line_includes_direct_legacy_model() {
        let cut = Some((0, direct_normalized_legacy_cfvnet_kind("/models/turn.onnx")));
        assert_eq!(
            boundary_evaluator_log_line("subgame", &cut),
            "[solve] solver: subgame; boundary evaluator: Direct CFVNet (legacy scaled bcfv); depth_limit=0; inference_mode=direct_normalized_legacy; model=/models/turn.onnx"
        );
    }

    #[test]
    fn boundary_evaluator_log_line_includes_legacy_model() {
        let cut = Some((1, cfvnet_kind("/models/river.onnx")));
        assert_eq!(
            boundary_evaluator_log_line("subgame", &cut),
            "[solve] solver: subgame; boundary evaluator: CFVNet; depth_limit=1; inference_mode=river_enumerated_turn; model=/models/river.onnx"
        );
    }

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
            river: cfvnet_mode("/models/river.onnx"),
        };
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(result, Some((1, cfvnet_kind("/models/river.onnx"))));
    }

    #[test]
    fn sbc_cfvnet_at_turn_from_flop_root() {
        // turn=Cfvnet from flop root → cut before turn card → depth=0
        // (near tree = flop only, 0 transitions).
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: cfvnet_mode("/models/turn.onnx"),
            river: StreetBoundaryMode::Exact,
        };
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(result, Some((0, cfvnet_kind("/models/turn.onnx"))));
    }

    #[test]
    fn sbc_rejects_cfvnet_cut_that_would_use_flop_boards() {
        let boundary_cut = Some((0, cfvnet_kind("/models/turn.onnx")));
        let err = validate_cfvnet_boundary_cut(&boundary_cut, Street::Flop).unwrap_err();
        assert!(err.contains("3-card flop boards"));
    }

    #[test]
    fn sbc_allows_cfvnet_cut_with_turn_boundary_boards() {
        let boundary_cut = Some((0, cfvnet_kind("/models/river.onnx")));
        assert!(validate_cfvnet_boundary_cut(&boundary_cut, Street::Turn).is_ok());
    }

    #[test]
    fn sbc_allows_exact_subtree_cut_from_flop_root() {
        let boundary_cut = Some((0, BoundaryKind::ExactSubtree));
        assert!(validate_cfvnet_boundary_cut(&boundary_cut, Street::Flop).is_ok());
    }

    #[test]
    fn sbc_cfvnet_at_flop_from_flop_root_is_ignored() {
        // flop=Cfvnet on flop root is degenerate — can't cut before our
        // current position. Falls through to all-exact (None).
        let config = StreetBoundaryConfig {
            flop: cfvnet_mode("/models/flop.onnx"),
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
            turn: cfvnet_mode("/models/turn.onnx"),
            river: cfvnet_mode("/models/river.onnx"),
        };
        // First non-exact wins: turn cut at depth 0 from flop root.
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(result, Some((0, cfvnet_kind("/models/turn.onnx"))));
    }

    #[test]
    fn sbc_cfvnet_at_river_from_turn_root() {
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::Exact,
            river: cfvnet_mode("/models/river.onnx"),
        };
        // From turn root: near tree = turn, cut before river = depth=0.
        let result = resolve_street_boundary(&config, Street::Turn);
        assert_eq!(result, Some((0, cfvnet_kind("/models/river.onnx"))));
    }

    #[test]
    fn sbc_cfvnet_at_turn_from_turn_root_is_ignored() {
        // turn=Cfvnet on turn root is degenerate — same as flop case above.
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: cfvnet_mode("/models/turn.onnx"),
            river: StreetBoundaryMode::Exact,
        };
        let result = resolve_street_boundary(&config, Street::Turn);
        assert_eq!(result, None);
    }

    #[test]
    fn sbc_preflop_root_returns_none() {
        let config = StreetBoundaryConfig {
            flop: cfvnet_mode("/models/flop.onnx"),
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
            river: cfvnet_mode("/models/river.onnx"),
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: StreetBoundaryConfig = serde_json::from_str(&json).unwrap();
        if let StreetBoundaryMode::Cfvnet { model_path, .. } = &parsed.river {
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
        if let StreetBoundaryMode::Cfvnet {
            model_path,
            inference_mode,
        } = &config.river
        {
            assert_eq!(model_path, "/path/to/model.onnx");
            assert_eq!(
                *inference_mode,
                cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::RiverEnumeratedTurn
            );
        } else {
            panic!("expected Cfvnet");
        }
    }

    #[test]
    fn sbc_serde_accepts_direct_cfvnet_inference_mode() {
        let json = r#"{
            "flop": {"mode": "exact"},
            "turn": {"mode": "exact"},
            "river": {
                "mode": "cfvnet",
                "model_path": "/path/to/turn_boundary.onnx",
                "inference_mode": "direct"
            }
        }"#;
        let config: StreetBoundaryConfig = serde_json::from_str(json).unwrap();
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(
            result,
            Some((1, direct_cfvnet_kind("/path/to/turn_boundary.onnx")))
        );
    }

    #[test]
    fn sbc_serde_accepts_direct_normalized_legacy_cfvnet_inference_mode() {
        let json = r#"{
            "flop": {"mode": "exact"},
            "turn": {"mode": "exact"},
            "river": {
                "mode": "cfvnet",
                "model_path": "/path/to/turn_boundary.onnx",
                "inference_mode": "direct_normalized_legacy"
            }
        }"#;
        let config: StreetBoundaryConfig = serde_json::from_str(json).unwrap();
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(
            result,
            Some((
                1,
                direct_normalized_legacy_cfvnet_kind("/path/to/turn_boundary.onnx")
            ))
        );
    }

    // -------------------------------------------------------------------
    // BoundaryKind + ExactSubtree tests
    // -------------------------------------------------------------------

    #[test]
    fn sbc_exact_subtree_at_river_from_flop_root() {
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::Exact,
            river: StreetBoundaryMode::ExactSubtree,
        };
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(result, Some((1, BoundaryKind::ExactSubtree)));
    }

    #[test]
    fn sbc_exact_subtree_at_turn_from_flop_root() {
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::ExactSubtree,
            river: StreetBoundaryMode::Exact,
        };
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(result, Some((0, BoundaryKind::ExactSubtree)));
    }

    #[test]
    fn sbc_exact_subtree_at_river_from_turn_root() {
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::Exact,
            river: StreetBoundaryMode::ExactSubtree,
        };
        let result = resolve_street_boundary(&config, Street::Turn);
        assert_eq!(result, None);
    }

    #[test]
    fn sbc_exact_subtree_at_root_street_is_ignored() {
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::ExactSubtree,
            turn: StreetBoundaryMode::Exact,
            river: StreetBoundaryMode::Exact,
        };
        assert_eq!(resolve_street_boundary(&config, Street::Flop), None);
    }

    #[test]
    fn sbc_cfvnet_returns_boundary_kind_cfvnet() {
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::Exact,
            river: cfvnet_mode("/models/river.onnx"),
        };
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(result, Some((1, cfvnet_kind("/models/river.onnx"))));
    }

    #[test]
    fn sbc_exact_subtree_wins_over_later_cfvnet() {
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::ExactSubtree,
            river: cfvnet_mode("/models/river.onnx"),
        };
        let result = resolve_street_boundary(&config, Street::Flop);
        assert_eq!(result, Some((0, BoundaryKind::ExactSubtree)));
    }

    #[test]
    fn sbc_serde_roundtrip_exact_subtree() {
        let config = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::ExactSubtree,
            river: StreetBoundaryMode::Exact,
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: StreetBoundaryConfig = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed.turn, StreetBoundaryMode::ExactSubtree));
    }

    #[test]
    fn sbc_serde_from_json_exact_subtree_tag() {
        let json = r#"{
            "flop": {"mode": "exact"},
            "turn": {"mode": "exact_subtree"},
            "river": {"mode": "exact"}
        }"#;
        let config: StreetBoundaryConfig = serde_json::from_str(json).unwrap();
        assert!(matches!(config.turn, StreetBoundaryMode::ExactSubtree));
    }

    // Option A build_gadget_tree_solve_game and advance_past_gadget tests:
    // DELETED (Phase D). The root gadget is retired. Under A2,
    // game.root() IS the real subgame root.
}
