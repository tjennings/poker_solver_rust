//! Exploration commands for navigating trained HUNL strategies.
//!
//! Allows users to load a strategy bundle or a rule-based agent config
//! and explore the game tree, viewing strategies at each decision point.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use poker_solver_core::abstraction::{CardAbstraction, Street};
use poker_solver_core::abstraction::isomorphism::{CanonicalBoard, SuitMapping};
use poker_solver_core::agent::{AgentConfig, FrequencyMap};
use poker_solver_core::blueprint::{AbstractionModeConfig, BundleConfig, StrategyBundle};
use poker_solver_core::blueprint_v2::bundle::{self as v2_bundle, BlueprintV2Strategy};
use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
use poker_solver_core::blueprint_v2::game_tree::{GameNode as V2GameNode, GameTree as V2GameTree, TreeAction};
use poker_solver_core::hand_class::{classify, intra_class_strength, HandClass};
use poker_solver_core::hands::CanonicalHand;
use poker_solver_core::info_key::{canonical_hand_index, canonical_hand_index_from_str, cards_from_rank_chars, encode_hand_v2, spr_bucket, InfoKey};
use poker_solver_core::showdown_equity;
use poker_solver_core::poker::{Card, Suit, Value};

/// Callback for reporting bucket computation progress: `(completed, total, board_key)`.
type ProgressCallback = Box<dyn Fn(usize, usize, &str) + Send + 'static>;

/// Event payload for bucket computation progress.
#[derive(Debug, Clone, Serialize)]
pub struct BucketProgressEvent {
    pub completed: usize,
    pub total: usize,
    pub board_key: String,
}

/// Event payload for subgame solving progress.
#[derive(Debug, Clone, Serialize)]
pub struct SubgameProgressEvent {
    pub iteration: u32,
    pub max_iterations: u32,
    pub board_key: String,
    pub elapsed_ms: u64,
}

/// State for the exploration view.
pub struct ExplorationState {
    /// Currently loaded strategy source (bundle or agent)
    source: RwLock<Option<StrategySource>>,
    /// Cached bucket computations: board_key -> (rank1, rank2, suited) -> bucket
    #[allow(clippy::type_complexity)]
    bucket_cache: Arc<RwLock<HashMap<String, HashMap<(char, char, bool), u16>>>>,
    /// Current computation progress
    computation_progress: Arc<AtomicUsize>,
    /// Total items to compute
    computation_total: Arc<AtomicUsize>,
    /// Whether computation is in progress
    computing: Arc<AtomicBool>,
    /// Board key currently being computed
    computing_board_key: Arc<RwLock<Option<String>>>,
    /// Abstraction for bucket computation (cloned for thread use)
    abstraction_boundaries: Arc<RwLock<Option<poker_solver_core::abstraction::BucketBoundaries>>>,
    /// Suit mapping established by flop canonicalization, applied to turn/river cards
    suit_mapping: RwLock<Option<SuitMapping>>,
}

/// A loaded strategy source — either a trained bundle, rule-based agent,
/// preflop solve, or subgame solve backed by a blueprint.
enum StrategySource {
    Bundle {
        config: BundleConfig,
        blueprint: poker_solver_core::blueprint::BlueprintStrategy,
    },
    Agent(AgentConfig),
    PreflopSolve {
        config: poker_solver_core::preflop::PreflopConfig,
        strategy: poker_solver_core::preflop::PreflopStrategy,
        /// Hand-averaged postflop EV table from the companion PostflopBundle, if present.
        /// Layout: `[pos0: 169×169, pos1: 169×169]`.
        hand_avg_values: Option<Vec<f64>>,
    },
    SubgameSolve {
        blueprint: Arc<poker_solver_core::blueprint::BlueprintStrategy>,
        blueprint_config: BundleConfig,
        #[allow(dead_code)]
        subgame_config: poker_solver_core::blueprint::SubgameConfig,
        /// Cache of solved subgames
        #[allow(dead_code)]
        solve_cache: Arc<RwLock<HashMap<u64, poker_solver_core::blueprint::SubgameStrategy>>>,
    },
    BlueprintV2 {
        config: Box<BlueprintV2Config>,
        strategy: Box<BlueprintV2Strategy>,
        tree: Box<V2GameTree>,
        /// Arena-index to decision-node-index mapping.
        decision_map: Vec<u32>,
    },
}

impl Default for ExplorationState {
    fn default() -> Self {
        Self {
            source: RwLock::new(None),
            bucket_cache: Arc::new(RwLock::new(HashMap::new())),
            computation_progress: Arc::new(AtomicUsize::new(0)),
            computation_total: Arc::new(AtomicUsize::new(0)),
            computing: Arc::new(AtomicBool::new(false)),
            computing_board_key: Arc::new(RwLock::new(None)),
            abstraction_boundaries: Arc::new(RwLock::new(None)),
            suit_mapping: RwLock::new(None),
        }
    }
}

/// Information about a loaded bundle or agent.
#[derive(Debug, Clone, Serialize)]
pub struct BundleInfo {
    pub name: Option<String>,
    pub stack_depth: u32,
    pub bet_sizes: Vec<f32>,
    pub info_sets: usize,
    pub iterations: u64,
    pub preflop_only: bool,
}

/// Information about an available agent config.
#[derive(Debug, Clone, Serialize)]
pub struct AgentInfo {
    pub name: String,
    pub path: String,
}

/// A single cell in the hand matrix.
#[derive(Debug, Clone, Serialize)]
pub struct MatrixCell {
    /// Hand label (e.g., "AKs", "QQ", "T9o")
    pub hand: String,
    /// Whether this is a suited hand
    pub suited: bool,
    /// Whether this is a pocket pair
    pub pair: bool,
    /// Action probabilities (fold, check/call, bets/raises...)
    pub probabilities: Vec<ActionProb>,
    /// Whether this hand was filtered out by the range threshold.
    pub filtered: bool,
}

/// An action with its probability.
#[derive(Debug, Clone, Serialize)]
pub struct ActionProb {
    pub action: String,
    pub probability: f32,
}

/// The full 13x13 strategy matrix.
#[derive(Debug, Clone, Serialize)]
pub struct StrategyMatrix {
    /// 13x13 grid of cells (row-major: [row][col])
    pub cells: Vec<Vec<MatrixCell>>,
    /// Available actions at this decision point
    pub actions: Vec<ActionInfo>,
    /// Current street
    pub street: String,
    /// Current pot size (after replaying history)
    pub pot: u32,
    /// Current player's stack
    pub stack: u32,
    /// Amount to call (0 if check is available)
    pub to_call: u32,
    /// Player 1 (SB) remaining stack
    pub stack_p1: u32,
    /// Player 2 (BB) remaining stack
    pub stack_p2: u32,
    /// Per-player stacks (N-player generalized)
    pub stacks: Vec<u32>,
}

/// Information about an available action.
#[derive(Debug, Clone, Serialize)]
pub struct ActionInfo {
    /// Action identifier (e.g., "fold", "call", "raise:150")
    pub id: String,
    /// Human-readable label
    pub label: String,
    /// Action type for styling
    pub action_type: String,
    /// Key for matching against `raise_sizes` config (e.g. `"0.5"`, `"1.0"`, `"allin"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_key: Option<String>,
}

/// Current game state for exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationPosition {
    /// Board cards as strings (e.g., ["As", "Kh", "7d"])
    pub board: Vec<String>,
    /// Action history as strings (e.g., ["r:100", "c"])
    pub history: Vec<String>,
    /// Current pot size
    pub pot: u32,
    /// Per-player stacks (index 0 = P1/SB, index 1 = P2/BB, etc.)
    pub stacks: Vec<u32>,
    /// Whose turn (0 = P1/SB, 1 = P2/BB)
    pub to_act: u8,
    /// Number of players in the game
    pub num_players: u8,
    /// Whether each player is still active (not folded)
    pub active_players: Vec<bool>,
}

impl Default for ExplorationPosition {
    fn default() -> Self {
        Self {
            board: vec![],
            history: vec![],
            pot: 3,        // SB + BB posted (internal units)
            stacks: vec![199, 198], // 100BB default: SB posted 1, BB posted 2
            to_act: 0,    // SB acts first preflop
            num_players: 2,
            active_players: vec![true, true],
        }
    }
}

/// Load a strategy bundle from a directory, or an agent from a `.toml` file
/// (core logic, no Tauri dependency).
///
/// Bundle loading (which can deserialize ~1 GB of blueprint data) runs on a
/// blocking thread so the async runtime stays responsive.
pub async fn load_bundle_core(
    state: &ExplorationState,
    path: String,
) -> Result<BundleInfo, String> {
    let bundle_path = PathBuf::from(&path);

    let (info, source, boundaries) = if path.ends_with(".toml") {
        let (info, source) = load_agent(&bundle_path)?;
        (info, source, None)
    } else if bundle_path.join("blueprint.bin").exists() {
        // Postflop strategy bundle
        let bundle = tokio::task::spawn_blocking(move || {
            StrategyBundle::load(&bundle_path)
                .map_err(|e| format!("Failed to load bundle: {e}"))
        })
        .await
        .map_err(|e| format!("Load task panicked: {e}"))??;

        let info = BundleInfo {
            name: Some("Trained Bundle".to_string()),
            stack_depth: bundle.config.game.stack_depth,
            bet_sizes: bundle.config.game.bet_sizes.clone(),
            info_sets: bundle.blueprint.len(),
            iterations: bundle.blueprint.iterations_trained(),
            preflop_only: false,
        };
        let boundaries = bundle.boundaries;
        let source = StrategySource::Bundle {
            config: bundle.config,
            blueprint: bundle.blueprint,
        };
        (info, source, boundaries)
    } else if bundle_path.join("strategy.bin").exists() {
        // Preflop strategy bundle
        let bundle = tokio::task::spawn_blocking(move || {
            let bp = bundle_path.clone();
            let preflop = poker_solver_core::preflop::PreflopBundle::load(&bp)
                .map_err(|e| format!("Failed to load preflop bundle: {e}"))?;

            // Hand-averaged EV table: prefer PreflopBundle's embedded copy,
            // fall back to postflop/ subdirectory, then co-located solve.bin.
            let hand_avg_values = preflop.hand_avg_values.clone().or_else(|| {
                let postflop_dir = bp.join("postflop");
                // Try multi-SPR bundle (handles both new and legacy layouts)
                if let Ok(abstractions) = poker_solver_core::preflop::PostflopBundle::load_multi(
                    &poker_solver_core::preflop::PostflopModelConfig::default(),
                    &postflop_dir,
                ) {
                    return abstractions.into_iter().next().map(|a| a.hand_avg_values);
                }
                // Legacy: co-located solve.bin at root
                let solve_bin = bp.join("solve.bin");
                if solve_bin.exists() {
                    return poker_solver_core::preflop::PostflopBundle::load_hand_avg_values(&solve_bin).ok();
                }
                None
            });

            Ok::<_, String>((preflop, hand_avg_values))
        })
        .await
        .map_err(|e| format!("Load task panicked: {e}"))??;

        let (preflop_bundle, hand_avg_values) = bundle;
        let info = BundleInfo {
            name: Some("Preflop Solve".into()),
            stack_depth: preflop_bundle.config.stacks.first().copied().unwrap_or(0) / 2,
            bet_sizes: vec![],
            info_sets: preflop_bundle.strategy.len(),
            iterations: 0,
            preflop_only: true,
        };
        let source = StrategySource::PreflopSolve {
            config: preflop_bundle.config,
            strategy: preflop_bundle.strategy,
            hand_avg_values,
        };
        (info, source, None)
    } else if bundle_path.join("config.yaml").exists() {
        // Blueprint V2 bundle — delegate to the dedicated loader.
        return load_blueprint_v2_core(state, path).await;
    } else {
        return Err(
            "Directory contains neither blueprint.bin, strategy.bin, nor config.yaml".to_string(),
        );
    };

    *state.abstraction_boundaries.write() = boundaries;
    *state.source.write() = Some(source);
    state.bucket_cache.write().clear();
    *state.suit_mapping.write() = None;

    Ok(info)
}

/// Load a strategy bundle (Tauri wrapper).
#[tauri::command]
pub async fn load_bundle(
    state: State<'_, ExplorationState>,
    path: String,
) -> Result<BundleInfo, String> {
    load_bundle_core(&state, path).await
}

fn load_agent(path: &Path) -> Result<(BundleInfo, StrategySource), String> {
    let agent = AgentConfig::load(path).map_err(|e| format!("Failed to load agent: {e}"))?;

    let info = BundleInfo {
        name: agent.game.name.clone(),
        stack_depth: agent.game.stack_depth,
        bet_sizes: agent.game.bet_sizes.clone(),
        info_sets: 0,
        iterations: 0,
        preflop_only: false,
    };

    Ok((info, StrategySource::Agent(agent)))
}

/// Load a preflop strategy bundle from a directory (core logic, no Tauri dependency).
pub async fn load_preflop_solve_core(
    state: &ExplorationState,
    path: String,
) -> Result<BundleInfo, String> {
    let bundle_path = PathBuf::from(&path);
    let (preflop_bundle, hand_avg_values) = tokio::task::spawn_blocking(move || {
        let preflop = poker_solver_core::preflop::PreflopBundle::load(&bundle_path)
            .map_err(|e| format!("Failed to load preflop bundle: {e}"))?;

        let hand_avg_values = {
            let postflop_dir = bundle_path.join("postflop");
            if let Ok(abstractions) = poker_solver_core::preflop::PostflopBundle::load_multi(
                &poker_solver_core::preflop::PostflopModelConfig::default(),
                &postflop_dir,
            ) {
                abstractions.into_iter().next().map(|a| a.hand_avg_values)
            } else {
                None
            }
        };

        Ok::<_, String>((preflop, hand_avg_values))
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))??;

    let info = BundleInfo {
        name: Some("Preflop Solve".into()),
        stack_depth: preflop_bundle.config.stacks.first().copied().unwrap_or(0) / 2,
        bet_sizes: vec![],
        info_sets: preflop_bundle.strategy.len(),
        iterations: 0,
        preflop_only: true,
    };

    *state.source.write() = Some(StrategySource::PreflopSolve {
        config: preflop_bundle.config,
        strategy: preflop_bundle.strategy,
        hand_avg_values,
    });

    Ok(info)
}

/// Load a preflop strategy bundle (Tauri wrapper).
#[tauri::command]
pub async fn load_preflop_solve(
    state: State<'_, ExplorationState>,
    path: String,
) -> Result<BundleInfo, String> {
    load_preflop_solve_core(&state, path).await
}

/// Solve a preflop game live with the given stack depth and iterations
/// (core logic, no Tauri dependency).
pub async fn solve_preflop_live_core(
    state: &ExplorationState,
    stack_depth: u32,
    iterations: u64,
) -> Result<BundleInfo, String> {
    let config = poker_solver_core::preflop::PreflopConfig::heads_up(stack_depth);
    let config_clone = config.clone();

    let (strategy, len) = tokio::task::spawn_blocking(move || {
        let mut solver = poker_solver_core::preflop::PreflopSolver::new(&config_clone);
        solver.train(iterations);
        let strat = solver.strategy();
        let len = strat.len();
        (strat, len)
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?;

    let info = BundleInfo {
        name: Some(format!("Preflop {stack_depth}BB")),
        stack_depth,
        bet_sizes: vec![],
        info_sets: len,
        iterations,
        preflop_only: true,
    };

    *state.source.write() = Some(StrategySource::PreflopSolve {
        config,
        strategy,
        hand_avg_values: None,
    });

    Ok(info)
}

/// Solve a preflop game live (Tauri wrapper).
#[tauri::command]
pub async fn solve_preflop_live(
    state: State<'_, ExplorationState>,
    stack_depth: u32,
    iterations: u64,
) -> Result<BundleInfo, String> {
    solve_preflop_live_core(&state, stack_depth, iterations).await
}

/// Load a blueprint for subgame solving (core logic, no Tauri dependency).
pub async fn load_subgame_source_core(
    state: &ExplorationState,
    blueprint_path: String,
) -> Result<BundleInfo, String> {
    let bundle_path = PathBuf::from(&blueprint_path);
    let bundle = tokio::task::spawn_blocking(move || {
        poker_solver_core::blueprint::StrategyBundle::load(&bundle_path)
            .map_err(|e| format!("Failed to load bundle: {e}"))
    })
    .await
    .map_err(|e| format!("Load task panicked: {e}"))??;

    let info = BundleInfo {
        name: Some("Subgame Solve".to_string()),
        stack_depth: bundle.config.game.stack_depth,
        bet_sizes: bundle.config.game.bet_sizes.clone(),
        info_sets: bundle.blueprint.len(),
        iterations: bundle.blueprint.iterations_trained(),
        preflop_only: false,
    };

    *state.source.write() = Some(StrategySource::SubgameSolve {
        blueprint: Arc::new(bundle.blueprint),
        blueprint_config: bundle.config,
        subgame_config: poker_solver_core::blueprint::SubgameConfig::default(),
        solve_cache: Arc::new(RwLock::new(HashMap::new())),
    });

    Ok(info)
}

/// Load a blueprint for subgame solving (Tauri wrapper).
#[tauri::command]
pub async fn load_subgame_source(
    state: State<'_, ExplorationState>,
    blueprint_path: String,
) -> Result<BundleInfo, String> {
    load_subgame_source_core(&state, blueprint_path).await
}

/// Load a Blueprint V2 bundle from a directory (core logic, no Tauri dependency).
///
/// Expects:
///   `dir_path/config.yaml` — `BlueprintV2Config`
///   `dir_path/strategy.bin` or `dir_path/final/strategy.bin` — `BlueprintV2Strategy`
pub async fn load_blueprint_v2_core(
    state: &ExplorationState,
    dir_path: String,
) -> Result<BundleInfo, String> {
    let dir = PathBuf::from(&dir_path);
    let (config, strategy) = tokio::task::spawn_blocking(move || {
        let cfg = v2_bundle::load_config(&dir)
            .map_err(|e| format!("Failed to load config.yaml: {e}"))?;

        // Try final/strategy.bin first, then strategy.bin at root.
        let strat_path = if dir.join("final/strategy.bin").exists() {
            dir.join("final/strategy.bin")
        } else if dir.join("strategy.bin").exists() {
            dir.join("strategy.bin")
        } else {
            // Look for the latest snapshot_NNNN directory.
            let mut snapshots: Vec<_> = std::fs::read_dir(&dir)
                .map_err(|e| format!("Cannot read directory: {e}"))?
                .filter_map(Result::ok)
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.starts_with("snapshot_"))
                })
                .collect();
            snapshots.sort_by_key(|e| e.file_name());
            match snapshots.last() {
                Some(entry) => entry.path().join("strategy.bin"),
                None => {
                    return Err(
                        "No strategy.bin found in bundle directory".to_string()
                    )
                }
            }
        };

        let strat = BlueprintV2Strategy::load(&strat_path)
            .map_err(|e| format!("Failed to load strategy.bin: {e}"))?;

        Ok::<_, String>((cfg, strat))
    })
    .await
    .map_err(|e| format!("Load task panicked: {e}"))??;

    let aa = &config.action_abstraction;
    let tree = V2GameTree::build(
        config.game.stack_depth,
        config.game.small_blind,
        config.game.big_blind,
        &aa.preflop,
        &aa.flop,
        &aa.turn,
        &aa.river,
    );
    let decision_map = tree.decision_index_map();

    let (decision_nodes, _, _) = tree.node_counts();
    let info = BundleInfo {
        name: Some("Blueprint V2".to_string()),
        stack_depth: config.game.stack_depth as u32,
        bet_sizes: vec![],
        info_sets: decision_nodes,
        iterations: strategy.iterations,
        preflop_only: false,
    };

    *state.source.write() = Some(StrategySource::BlueprintV2 {
        config: Box::new(config),
        strategy: Box::new(strategy),
        tree: Box::new(tree),
        decision_map,
    });
    state.bucket_cache.write().clear();
    *state.suit_mapping.write() = None;

    Ok(info)
}

/// Load a Blueprint V2 bundle (Tauri wrapper).
#[tauri::command]
pub async fn load_blueprint_v2(
    state: State<'_, ExplorationState>,
    path: String,
) -> Result<BundleInfo, String> {
    load_blueprint_v2_core(&state, path).await
}

/// Get the strategy matrix for a given position (core logic, no Tauri dependency).
///
/// `threshold` filters out hands whose prior action probabilities fell below
/// this value (range narrowing).  `street_histories` supplies the action
/// sequences of all completed streets so the filter can replay the game.
pub fn get_strategy_matrix_core(
    state: &ExplorationState,
    position: ExplorationPosition,
    threshold: Option<f32>,
    street_histories: Option<Vec<Vec<String>>>,
) -> Result<StrategyMatrix, String> {
    let source_guard = state.source.read();
    let source = source_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    let sh = street_histories.unwrap_or_default();
    let thresh = threshold.unwrap_or(0.0);

    match source {
        StrategySource::Bundle { config, blueprint } => {
            get_strategy_matrix_bundle(config, blueprint, state, &position, thresh, &sh)
        }
        StrategySource::Agent(agent) => get_strategy_matrix_agent(agent, &position),
        StrategySource::PreflopSolve { config, strategy, .. } => {
            get_strategy_matrix_preflop(config, strategy, &position)
        }
        StrategySource::SubgameSolve {
            blueprint,
            blueprint_config,
            ..
        } => get_strategy_matrix_bundle(
            blueprint_config,
            blueprint,
            state,
            &position,
            thresh,
            &sh,
        ),
        StrategySource::BlueprintV2 {
            config,
            strategy,
            tree,
            decision_map,
        } => get_strategy_matrix_v2(config, strategy, tree, decision_map, &position),
    }
}

/// Get the strategy matrix for a given position (Tauri wrapper).
#[tauri::command]
pub fn get_strategy_matrix(
    state: State<'_, ExplorationState>,
    position: ExplorationPosition,
    threshold: Option<f32>,
    street_histories: Option<Vec<Vec<String>>>,
) -> Result<StrategyMatrix, String> {
    get_strategy_matrix_core(&state, position, threshold, street_histories)
}

fn get_strategy_matrix_bundle(
    config: &BundleConfig,
    blueprint: &poker_solver_core::blueprint::BlueprintStrategy,
    state: &ExplorationState,
    position: &ExplorationPosition,
    threshold: f32,
    street_histories: &[Vec<String>],
) -> Result<StrategyMatrix, String> {
    let board = parse_board(&position.board)?;
    let street = street_from_board_len(board.len())?;
    let action_codes = build_action_codes(&position.history);
    let mode = config.abstraction_mode;
    let use_hand_class = mode.is_hand_class();

    let ranks = RANKS;
    let mut cells = Vec::with_capacity(13);

    let pos_state = compute_position_state(&config.game.bet_sizes, position);
    let actions =
        get_actions_for_position(config.game.stack_depth, &config.game.bet_sizes, position);

    // For EHS2 mode, use bucket cache; for hand_class mode, use classify() directly
    let bucket_cache: Option<HashMap<(char, char, bool), u16>> =
        if !use_hand_class && street != Street::Preflop && !board.is_empty() {
            let board_key = position.board.join("");
            let cache = state.bucket_cache.read();
            cache.get(&board_key).cloned()
        } else {
            None
        };

    // Range filtering: only for hand_class bundles with a positive threshold
    let apply_filter = use_hand_class && threshold > 0.0;

    for (row, &rank1) in ranks.iter().enumerate() {
        let mut row_cells = Vec::with_capacity(13);
        for (col, &rank2) in ranks.iter().enumerate() {
            let (hand_label, suited, pair) = hand_label_from_matrix(row, col, rank1, rank2);

            let filtered = apply_filter
                && !is_hand_in_range(
                    config,
                    blueprint,
                    &board,
                    street_histories,
                    &position.history,
                    position.to_act,
                    threshold,
                    rank1,
                    rank2,
                    suited,
                );

            let probabilities = if filtered {
                vec![]
            } else {
                let hand_bits = if mode == AbstractionModeConfig::HandClassV2 {
                    hand_bits_hand_class_v2(config, street, &board, rank1, rank2, suited)
                } else {
                    hand_bits_ehs2(street, rank1, rank2, suited, bucket_cache.as_ref())?
                };
                lookup_blueprint_strategy(
                    blueprint,
                    hand_bits,
                    street,
                    &action_codes,
                    pos_state.pot,
                    pos_state.eff_stack,
                    &actions,
                )?
            };

            row_cells.push(MatrixCell {
                hand: hand_label,
                suited,
                pair,
                probabilities,
                filtered,
            });
        }
        cells.push(row_cells);
    }

    Ok(StrategyMatrix {
        cells,
        actions,
        street: format!("{street:?}"),
        pot: pos_state.pot,
        stack: if position.to_act == 0 {
            pos_state.stack_p1
        } else {
            pos_state.stack_p2
        },
        to_call: pos_state.to_call,
        stack_p1: pos_state.stack_p1,
        stack_p2: pos_state.stack_p2,
        stacks: vec![pos_state.stack_p1, pos_state.stack_p2],
    })
}

fn get_strategy_matrix_agent(
    agent: &AgentConfig,
    position: &ExplorationPosition,
) -> Result<StrategyMatrix, String> {
    let board = parse_board(&position.board)?;
    let street = street_from_board_len(board.len())?;
    let is_preflop = street == Street::Preflop;

    let actions = get_actions_for_position(agent.game.stack_depth, &agent.game.bet_sizes, position);

    // For heads-up: to_act 0 (SB) = btn, to_act 1 (BB) = bb
    let position_name = if position.to_act == 0 { "btn" } else { "bb" };

    let ranks = RANKS;
    let mut cells = Vec::with_capacity(13);

    for (row, &rank1) in ranks.iter().enumerate() {
        let mut row_cells = Vec::with_capacity(13);
        for (col, &rank2) in ranks.iter().enumerate() {
            let (hand_label, suited, pair) = hand_label_from_matrix(row, col, rank1, rank2);

            let freq = if is_preflop {
                resolve_preflop_frequency(agent, position_name, rank1, rank2, suited)
            } else {
                resolve_postflop_frequency(agent, rank1, rank2, suited, &board)
            };

            let probabilities = map_frequencies_to_actions(freq, &actions);

            row_cells.push(MatrixCell {
                hand: hand_label,
                suited,
                pair,
                probabilities,
                filtered: false,
            });
        }
        cells.push(row_cells);
    }

    let pos_state = compute_position_state(&agent.game.bet_sizes, position);

    Ok(StrategyMatrix {
        cells,
        actions,
        street: format!("{street:?}"),
        pot: pos_state.pot,
        stack: if position.to_act == 0 {
            pos_state.stack_p1
        } else {
            pos_state.stack_p2
        },
        to_call: pos_state.to_call,
        stack_p1: pos_state.stack_p1,
        stack_p2: pos_state.stack_p2,
        stacks: vec![pos_state.stack_p1, pos_state.stack_p2],
    })
}

fn get_strategy_matrix_preflop(
    config: &poker_solver_core::preflop::PreflopConfig,
    strategy: &poker_solver_core::preflop::PreflopStrategy,
    position: &ExplorationPosition,
) -> Result<StrategyMatrix, String> {
    use poker_solver_core::preflop::{PreflopNode, PreflopTree};

    let tree = PreflopTree::build(config);

    // Walk the tree following the action history, tracking pot/stacks
    let walk = walk_preflop_tree_with_state(config, &tree, &position.history)?;

    // Get available actions at this node
    let (action_labels, _children) = match &tree.nodes[walk.node_idx as usize] {
        PreflopNode::Decision {
            action_labels,
            children,
            ..
        } => (action_labels.clone(), children.clone()),
        PreflopNode::Terminal { .. } => {
            return Err("Position is at a terminal node".to_string());
        }
    };

    // Build ActionInfo from preflop action labels (check vs call aware)
    let actions: Vec<ActionInfo> = action_labels
        .iter()
        .enumerate()
        .map(|(i, a)| preflop_action_info(a, i, walk.to_call))
        .collect();

    let ranks = RANKS;
    let mut cells = Vec::with_capacity(13);

    for (row, &rank1) in ranks.iter().enumerate() {
        let mut row_cells = Vec::with_capacity(13);
        for (col, &rank2) in ranks.iter().enumerate() {
            let (hand_label, suited, pair) = hand_label_from_matrix(row, col, rank1, rank2);

            let hand_idx = canonical_hand_index_from_ranks(rank1, rank2, suited);
            let probs_f64 = strategy.get_probs(walk.node_idx, hand_idx);

            #[allow(clippy::cast_possible_truncation)]
            let probabilities: Vec<ActionProb> = actions
                .iter()
                .zip(probs_f64.iter())
                .map(|(a, &p)| ActionProb {
                    action: a.label.clone(),
                    probability: p as f32,
                })
                .collect();

            row_cells.push(MatrixCell {
                hand: hand_label,
                suited,
                pair,
                probabilities,
                filtered: false,
            });
        }
        cells.push(row_cells);
    }

    Ok(StrategyMatrix {
        cells,
        actions,
        street: "Preflop".to_string(),
        pot: walk.pot,
        stack: walk.stacks.get(walk.to_act as usize).copied().unwrap_or(0),
        to_call: walk.to_call,
        stack_p1: walk.stacks.first().copied().unwrap_or(0),
        stack_p2: walk.stacks.get(1).copied().unwrap_or(0),
        stacks: walk.stacks,
    })
}

/// Result of walking the preflop tree with full state tracking.
struct PreflopWalkState {
    node_idx: u32,
    pot: u32,
    stacks: Vec<u32>,
    to_call: u32,
    to_act: u8,
}

/// Walk the preflop tree following an action history, tracking pot/stacks/to_call.
///
/// Mirrors the `BuildState` transitions from tree construction so that
/// pot, stacks, and to_call are accurate at each decision point.
fn walk_preflop_tree_with_state(
    config: &poker_solver_core::preflop::PreflopConfig,
    tree: &poker_solver_core::preflop::PreflopTree,
    history: &[String],
) -> Result<PreflopWalkState, String> {
    use poker_solver_core::preflop::{PreflopAction, PreflopNode};

    let num_players = config.num_players() as usize;
    let mut invested = vec![0u32; num_players];
    let mut stacks = config.stacks.clone();

    // Post blinds
    for &(pos, amount) in &config.blinds {
        let actual = amount.min(stacks[pos]);
        invested[pos] = actual;
        stacks[pos] -= actual;
    }
    // Post antes
    for &(pos, amount) in &config.antes {
        let actual = amount.min(stacks[pos]);
        invested[pos] += actual;
        stacks[pos] -= actual;
    }

    let mut current_bet = invested.iter().copied().max().unwrap_or(0);
    let mut node_idx = 0u32;

    for action_str in history {
        let node = &tree.nodes[node_idx as usize];
        let (action_labels, children, position) = match node {
            PreflopNode::Decision {
                action_labels,
                children,
                position,
            } => (action_labels, children, *position),
            PreflopNode::Terminal { .. } => {
                return Err(format!(
                    "Reached terminal before processing action '{action_str}'"
                ));
            }
        };

        let child_pos = find_action_position(action_str, action_labels)
            .ok_or_else(|| {
                format!(
                    "Action '{action_str}' not available at node {node_idx}. \
                     Available: {action_labels:?}"
                )
            })?;

        let tree_action = &action_labels[child_pos];
        let p = position as usize;
        let to_call = current_bet.saturating_sub(invested[p]);

        match tree_action {
            PreflopAction::Fold => {}
            PreflopAction::Call => {
                let actual = to_call.min(stacks[p]);
                invested[p] += actual;
                stacks[p] -= actual;
            }
            PreflopAction::Raise(size) => {
                let pot: u32 = invested.iter().sum();
                let to_call = current_bet.saturating_sub(invested[p]);
                let new_bet = size.resolve(current_bet, pot, to_call);
                let total = new_bet.saturating_sub(invested[p]);
                let actual = total.min(stacks[p]);
                invested[p] += actual;
                stacks[p] -= actual;
                current_bet = invested[p];
            }
            PreflopAction::AllIn => {
                let all_chips = stacks[p];
                invested[p] += all_chips;
                stacks[p] = 0;
                if invested[p] > current_bet {
                    current_bet = invested[p];
                }
            }
        }

        node_idx = children[child_pos];
    }

    // Determine to_act and to_call at the current node
    let (to_act, to_call) = match &tree.nodes[node_idx as usize] {
        PreflopNode::Decision { position, .. } => {
            let tc = current_bet.saturating_sub(invested[*position as usize]);
            (*position, tc)
        }
        PreflopNode::Terminal { .. } => (0, 0),
    };

    let pot: u32 = invested.iter().sum();

    Ok(PreflopWalkState {
        node_idx,
        pot,
        stacks,
        to_call,
        to_act,
    })
}

/// Find the position of a history action string within the node's action labels.
///
/// For raise actions with `r:{idx}` format, uses the index directly to select
/// the correct raise among multiple raise sizes at the same node.
fn find_action_position(
    action_str: &str,
    action_labels: &[poker_solver_core::preflop::PreflopAction],
) -> Option<usize> {
    use poker_solver_core::preflop::PreflopAction;

    if action_str == "f" || action_str == "fold" {
        return action_labels.iter().position(|a| matches!(a, PreflopAction::Fold));
    }
    if action_str == "c" || action_str == "call" || action_str == "x" || action_str == "check" {
        return action_labels.iter().position(|a| matches!(a, PreflopAction::Call));
    }

    // Extract the suffix after the first ':'
    let suffix = action_str.split(':').nth(1).unwrap_or("");

    if suffix == "A" {
        // All-in
        return action_labels.iter().position(|a| matches!(a, PreflopAction::AllIn));
    }

    // Raise with index — r:{idx} where idx is the position in the action_labels array
    if let Ok(idx) = suffix.parse::<usize>() {
        if idx < action_labels.len() && matches!(action_labels[idx], PreflopAction::Raise(_)) {
            return Some(idx);
        }
    }

    None
}

/// Build an `ActionInfo` from a preflop tree action.
///
/// `to_call` distinguishes check (0) from call (>0) for `PreflopAction::Call`.
fn preflop_action_info(
    action: &poker_solver_core::preflop::PreflopAction,
    idx: usize,
    to_call: u32,
) -> ActionInfo {
    use poker_solver_core::preflop::PreflopAction;
    match action {
        PreflopAction::Fold => ActionInfo {
            id: "fold".to_string(),
            label: "Fold".to_string(),
            action_type: "fold".to_string(),
            size_key: None,
        },
        PreflopAction::Call if to_call == 0 => ActionInfo {
            id: "check".to_string(),
            label: "Check".to_string(),
            action_type: "check".to_string(),
            size_key: None,
        },
        PreflopAction::Call => ActionInfo {
            id: "call".to_string(),
            label: format!("Call {to_call}"),
            action_type: "call".to_string(),
            size_key: None,
        },
        PreflopAction::Raise(size) => ActionInfo {
            id: format!("r:{idx}"),
            label: format!("Raise {size}"),
            action_type: "raise".to_string(),
            size_key: Some(format!("{size}")),
        },
        PreflopAction::AllIn => ActionInfo {
            id: "r:A".to_string(),
            label: "All-in".to_string(),
            action_type: "allin".to_string(),
            size_key: Some("allin".to_string()),
        },
    }
}

// ── Blueprint V2 helpers ─────────────────────────────────────────────

/// Result of walking the V2 game tree following an action history.
struct V2WalkState {
    node_idx: u32,
    #[allow(dead_code)]
    pot: f64,
    #[allow(dead_code)]
    stacks: [f64; 2],
    #[allow(dead_code)]
    to_call: f64,
    #[allow(dead_code)]
    to_act: u8,
}

/// Walk the V2 game tree following an action history.
///
/// Action strings: `"f"` fold, `"c"` call/check, `"r:{idx}"` raise by
/// action index, `"r:A"` all-in.
fn walk_v2_tree(
    tree: &V2GameTree,
    history: &[String],
) -> Result<V2WalkState, String> {
    let mut node_idx = tree.root;

    for action_str in history {
        let node = &tree.nodes[node_idx as usize];
        let (actions, children) = match node {
            V2GameNode::Decision {
                actions, children, ..
            } => (actions, children),
            V2GameNode::Chance { child, .. } => {
                // Skip chance nodes automatically.
                node_idx = *child;
                let node2 = &tree.nodes[node_idx as usize];
                match node2 {
                    V2GameNode::Decision {
                        actions, children, ..
                    } => (actions, children),
                    _ => {
                        return Err(format!(
                            "Expected decision after chance, got terminal at node {node_idx}"
                        ));
                    }
                }
            }
            V2GameNode::Terminal { .. } => {
                return Err(format!(
                    "Reached terminal before processing action '{action_str}'"
                ));
            }
        };

        let child_pos = find_v2_action_position(action_str, actions)
            .ok_or_else(|| {
                format!(
                    "Action '{action_str}' not available at node {node_idx}. \
                     Available: {actions:?}"
                )
            })?;

        node_idx = children[child_pos];

        // If we land on a Chance node, advance through it automatically.
        if let V2GameNode::Chance { child, .. } = &tree.nodes[node_idx as usize] {
            node_idx = *child;
        }
    }

    // Compute pot, stacks, to_call from the current node.
    let (to_act, pot, stacks, to_call) = v2_node_state(tree, node_idx);

    Ok(V2WalkState {
        node_idx,
        pot,
        stacks,
        to_call,
        to_act,
    })
}

/// Compute pot/stacks/to_call at a V2 tree node.
///
/// For terminal and chance nodes, returns zeros for to_call.
fn v2_node_state(tree: &V2GameTree, node_idx: u32) -> (u8, f64, [f64; 2], f64) {
    match &tree.nodes[node_idx as usize] {
        V2GameNode::Decision {
            player, ..
        } => {
            // Walk up to find invested amounts from the nearest terminal child
            // or from root state. For simplicity in the explorer, recompute
            // from the tree structure: invested amounts are tracked in terminals.
            //
            // Since we don't have parent pointers, we use a simpler approach:
            // we pass the position's pot/stacks from the ExplorationPosition.
            // This function returns defaults that get overridden.
            (*player, 0.0, [0.0; 2], 0.0)
        }
        V2GameNode::Terminal { pot, invested, .. } => (0, *pot, *invested, 0.0),
        V2GameNode::Chance { child, .. } => v2_node_state(tree, *child),
    }
}

/// Find the position of a history action string within V2 tree actions.
fn find_v2_action_position(action_str: &str, actions: &[TreeAction]) -> Option<usize> {
    match action_str {
        "f" | "fold" => actions.iter().position(|a| matches!(a, TreeAction::Fold)),
        "c" | "call" | "x" | "check" => actions
            .iter()
            .position(|a| matches!(a, TreeAction::Check | TreeAction::Call)),
        _ => {
            let suffix = action_str.split(':').nth(1).unwrap_or("");
            if suffix == "A" {
                return actions
                    .iter()
                    .position(|a| matches!(a, TreeAction::AllIn));
            }
            if let Ok(idx) = suffix.parse::<usize>() {
                if idx < actions.len() {
                    return Some(idx);
                }
            }
            None
        }
    }
}

/// Build `ActionInfo` items for actions at a V2 tree node.
fn v2_actions_at_node(tree: &V2GameTree, node_idx: u32) -> Vec<ActionInfo> {
    match &tree.nodes[node_idx as usize] {
        V2GameNode::Decision { actions, .. } => actions
            .iter()
            .enumerate()
            .map(|(i, a)| v2_action_info(a, i))
            .collect(),
        _ => vec![],
    }
}

/// Convert a V2 `TreeAction` to an `ActionInfo`.
#[allow(clippy::cast_possible_truncation)]
fn v2_action_info(action: &TreeAction, idx: usize) -> ActionInfo {
    match action {
        TreeAction::Fold => ActionInfo {
            id: "fold".to_string(),
            label: "Fold".to_string(),
            action_type: "fold".to_string(),
            size_key: None,
        },
        TreeAction::Check => ActionInfo {
            id: "check".to_string(),
            label: "Check".to_string(),
            action_type: "check".to_string(),
            size_key: None,
        },
        TreeAction::Call => ActionInfo {
            id: "call".to_string(),
            label: "Call".to_string(),
            action_type: "call".to_string(),
            size_key: None,
        },
        TreeAction::Bet(amount) => ActionInfo {
            id: format!("r:{idx}"),
            label: format!("{:.1}bb", amount),
            action_type: "bet".to_string(),
            size_key: Some(format!("{amount:.2}")),
        },
        TreeAction::Raise(amount) => ActionInfo {
            id: format!("r:{idx}"),
            label: format!("{:.1}bb", amount),
            action_type: "raise".to_string(),
            size_key: Some(format!("{amount:.2}")),
        },
        TreeAction::AllIn => ActionInfo {
            id: "r:A".to_string(),
            label: "All-in".to_string(),
            action_type: "allin".to_string(),
            size_key: Some("allin".to_string()),
        },
    }
}

/// Get the strategy matrix for a BlueprintV2 source.
#[allow(clippy::cast_possible_truncation)]
fn get_strategy_matrix_v2(
    _config: &BlueprintV2Config,
    strategy: &BlueprintV2Strategy,
    tree: &V2GameTree,
    decision_map: &[u32],
    position: &ExplorationPosition,
) -> Result<StrategyMatrix, String> {
    let walk = walk_v2_tree(tree, &position.history)?;
    let actions = v2_actions_at_node(tree, walk.node_idx);

    if actions.is_empty() {
        return Err("Position is at a terminal node".to_string());
    }

    let decision_idx = decision_map
        .get(walk.node_idx as usize)
        .copied()
        .unwrap_or(u32::MAX);

    if decision_idx == u32::MAX {
        return Err(format!(
            "Node {} is not a decision node",
            walk.node_idx
        ));
    }

    // Determine the street from the node.
    let street_name = match &tree.nodes[walk.node_idx as usize] {
        V2GameNode::Decision { street, .. } => match street {
            poker_solver_core::blueprint_v2::Street::Preflop => "Preflop",
            poker_solver_core::blueprint_v2::Street::Flop => "Flop",
            poker_solver_core::blueprint_v2::Street::Turn => "Turn",
            poker_solver_core::blueprint_v2::Street::River => "River",
        },
        _ => "Unknown",
    };

    let is_preflop = street_name == "Preflop";
    let num_buckets = strategy.bucket_counts[
        strategy.node_street_indices[decision_idx as usize] as usize
    ] as usize;

    let ranks = RANKS;
    let mut cells = Vec::with_capacity(13);

    for (row, &rank1) in ranks.iter().enumerate() {
        let mut row_cells = Vec::with_capacity(13);
        for (col, &rank2) in ranks.iter().enumerate() {
            let (hand_label, suited, pair) = hand_label_from_matrix(row, col, rank1, rank2);

            let probabilities = if is_preflop {
                // For preflop, canonical hand index maps directly to bucket.
                let hand_idx = canonical_hand_index_from_ranks(rank1, rank2, suited);
                let bucket = if num_buckets == 169 {
                    hand_idx as u16
                } else {
                    // If fewer buckets, use modulo as a placeholder.
                    (hand_idx % num_buckets) as u16
                };
                let probs = strategy.get_action_probs(decision_idx as usize, bucket);
                actions
                    .iter()
                    .zip(probs.iter().chain(std::iter::repeat(&0.0f32)))
                    .map(|(a, &p)| ActionProb {
                        action: a.label.clone(),
                        probability: p,
                    })
                    .collect()
            } else {
                // Postflop: bucket assignment requires cluster data which
                // is not yet loaded. Return uniform distribution.
                let n = actions.len();
                let uniform = 1.0 / n as f32;
                actions
                    .iter()
                    .map(|a| ActionProb {
                        action: a.label.clone(),
                        probability: uniform,
                    })
                    .collect()
            };

            row_cells.push(MatrixCell {
                hand: hand_label,
                suited,
                pair,
                probabilities,
                filtered: false,
            });
        }
        cells.push(row_cells);
    }

    // Compute pot/stacks from the ExplorationPosition (the canonical source).
    let pot = position.pot;
    let stack_p1 = position.stacks.first().copied().unwrap_or(0);
    let stack_p2 = position.stacks.get(1).copied().unwrap_or(0);
    let to_call = 0; // The frontend computes this from position data.

    Ok(StrategyMatrix {
        cells,
        actions,
        street: street_name.to_string(),
        pot,
        stack: if position.to_act == 0 { stack_p1 } else { stack_p2 },
        to_call,
        stack_p1,
        stack_p2,
        stacks: position.stacks.clone(),
    })
}

/// Get the canonical hand index (0..169) from rank characters.
fn canonical_hand_index_from_ranks(rank1: char, rank2: char, suited: bool) -> usize {
    let v1 = char_to_value(rank1);
    let v2 = char_to_value(rank2);
    let canonical = CanonicalHand::new(v1, v2, suited);
    canonical.index()
}

/// Get available actions at the current position (core logic, no Tauri dependency).
pub fn get_available_actions_core(
    state: &ExplorationState,
    position: ExplorationPosition,
) -> Result<Vec<ActionInfo>, String> {
    let source_guard = state.source.read();
    let source = source_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    // BlueprintV2 derives actions from the game tree rather than bet sizes.
    if let StrategySource::BlueprintV2 { tree, .. } = source {
        let walk = walk_v2_tree(tree, &position.history)?;
        return Ok(v2_actions_at_node(tree, walk.node_idx));
    }

    let (stack_depth, bet_sizes) = match source {
        StrategySource::Bundle { config, .. } | StrategySource::SubgameSolve { blueprint_config: config, .. } => {
            (config.game.stack_depth, config.game.bet_sizes.as_slice())
        }
        StrategySource::Agent(agent) => (agent.game.stack_depth, agent.game.bet_sizes.as_slice()),
        StrategySource::PreflopSolve { config, .. } => {
            let depth = config.stacks.first().copied().unwrap_or(0) / 2;
            // Preflop uses tree-based actions, return empty bet_sizes
            (depth, [].as_slice())
        }
        StrategySource::BlueprintV2 { .. } => unreachable!(),
    };

    Ok(get_actions_for_position(stack_depth, bet_sizes, &position))
}

/// Get available actions at the current position (Tauri wrapper).
#[tauri::command]
pub fn get_available_actions(
    state: State<'_, ExplorationState>,
    position: ExplorationPosition,
) -> Result<Vec<ActionInfo>, String> {
    get_available_actions_core(&state, position)
}

/// Check if a bundle or agent is loaded (core logic, no Tauri dependency).
pub fn is_bundle_loaded_core(state: &ExplorationState) -> bool {
    state.source.read().is_some()
}

/// Check if a bundle or agent is loaded (Tauri wrapper).
#[tauri::command]
pub fn is_bundle_loaded(state: State<'_, ExplorationState>) -> bool {
    is_bundle_loaded_core(&state)
}

/// Get info about currently loaded bundle or agent (core logic, no Tauri dependency).
pub fn get_bundle_info_core(state: &ExplorationState) -> Result<BundleInfo, String> {
    let source_guard = state.source.read();
    let source = source_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    Ok(match source {
        StrategySource::Bundle { config, blueprint } => BundleInfo {
            name: Some("Trained Bundle".to_string()),
            stack_depth: config.game.stack_depth,
            bet_sizes: config.game.bet_sizes.clone(),
            info_sets: blueprint.len(),
            iterations: blueprint.iterations_trained(),
            preflop_only: false,
        },
        StrategySource::Agent(agent) => BundleInfo {
            name: agent.game.name.clone(),
            stack_depth: agent.game.stack_depth,
            bet_sizes: agent.game.bet_sizes.clone(),
            info_sets: 0,
            iterations: 0,
            preflop_only: false,
        },
        StrategySource::PreflopSolve { config, strategy, .. } => BundleInfo {
            name: Some("Preflop Solve".to_string()),
            stack_depth: config.stacks.first().copied().unwrap_or(0) / 2,
            bet_sizes: vec![],
            info_sets: strategy.len(),
            iterations: 0,
            preflop_only: true,
        },
        StrategySource::SubgameSolve {
            blueprint,
            blueprint_config,
            ..
        } => BundleInfo {
            name: Some("Subgame Solve".to_string()),
            stack_depth: blueprint_config.game.stack_depth,
            bet_sizes: blueprint_config.game.bet_sizes.clone(),
            info_sets: blueprint.len(),
            iterations: blueprint.iterations_trained(),
            preflop_only: false,
        },
        StrategySource::BlueprintV2 {
            config,
            strategy,
            tree,
            ..
        } => {
            let (decision_nodes, _, _) = tree.node_counts();
            BundleInfo {
                name: Some("Blueprint V2".to_string()),
                stack_depth: config.game.stack_depth as u32,
                bet_sizes: vec![],
                info_sets: decision_nodes,
                iterations: strategy.iterations,
                preflop_only: false,
            }
        }
    })
}

/// Get info about currently loaded bundle or agent (Tauri wrapper).
#[tauri::command]
pub fn get_bundle_info(state: State<'_, ExplorationState>) -> Result<BundleInfo, String> {
    get_bundle_info_core(&state)
}

/// EV for a specific hero-vs-villain matchup.
#[derive(Debug, Clone, Serialize)]
pub struct MatchupEquity {
    pub villain_hand: String,
    pub ev_pos0: f64,
    pub ev_pos1: f64,
    pub ev_avg: f64,
}

/// Per-hand postflop equity data returned by `get_hand_equity`.
#[derive(Debug, Clone, Serialize)]
pub struct HandEquity {
    /// Average postflop EV (pot fractions) when hero is position 0 (SB).
    pub ev_pos0: f64,
    /// Average postflop EV (pot fractions) when hero is position 1 (BB).
    pub ev_pos1: f64,
    /// Overall average across both positions (pot fractions).
    pub ev_avg: f64,
    /// Optional matchup EV against a specific villain hand.
    pub ev_vs_hand: Option<MatchupEquity>,
}

/// Return the average postflop EV for a canonical hand (e.g. "AKs", "QQ", "72o").
///
/// Returns pot-fraction EV (where 1.0 = the initial postflop pot) averaged
/// uniformly across all 169 opponent hands. Returns `None` if no
/// postflop data is loaded or the hand string is unrecognised.
#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn get_hand_equity_core(
    state: &ExplorationState,
    hand: &str,
    villain_hand: Option<&str>,
) -> Result<Option<HandEquity>, String> {
    let hand_index = match canonical_hand_index_from_str(hand) {
        Some(idx) => idx as usize,
        None => return Ok(None),
    };

    let source_guard = state.source.read();
    let source = source_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    let hand_avg = match source {
        StrategySource::PreflopSolve { hand_avg_values: Some(v), .. } => v,
        _ => return Ok(None),
    };

    // Table layout: [pos0: N×N, pos1: N×N]
    let half = hand_avg.len() / 2;
    if half == 0 {
        return Ok(None);
    }
    let n = (half as f64).sqrt() as usize;
    if n == 0 || n * n != half || hand_index >= n {
        return Ok(None);
    }

    // Average EV (pot fractions) across all opponent hands for each position.
    let avg_for_pos = |pos: usize| -> f64 {
        let base = pos * n * n + hand_index * n;
        let slice = &hand_avg[base..base + n];
        let sum: f64 = slice.iter().sum();
        sum / n as f64
    };

    let ev_pos0 = avg_for_pos(0);
    let ev_pos1 = avg_for_pos(1);
    let ev_avg = (ev_pos0 + ev_pos1) / 2.0;

    let ev_vs_hand = villain_hand.and_then(|vh| {
        let v_idx = canonical_hand_index_from_str(vh)? as usize;
        if v_idx >= n { return None; }
        let vp0 = hand_avg[0 * n * n + hand_index * n + v_idx];
        let vp1 = hand_avg[1 * n * n + hand_index * n + v_idx];
        Some(MatchupEquity {
            villain_hand: vh.to_string(),
            ev_pos0: vp0,
            ev_pos1: vp1,
            ev_avg: (vp0 + vp1) / 2.0,
        })
    });

    Ok(Some(HandEquity { ev_pos0, ev_pos1, ev_avg, ev_vs_hand }))
}

/// Get postflop equity for a canonical hand (Tauri wrapper).
#[tauri::command]
pub fn get_hand_equity(
    state: State<'_, ExplorationState>,
    hand: String,
    villain_hand: Option<String>,
) -> Result<Option<HandEquity>, String> {
    get_hand_equity_core(&state, &hand, villain_hand.as_deref())
}

/// Computation progress status.
#[derive(Debug, Clone, Serialize)]
pub struct ComputationStatus {
    pub computing: bool,
    pub progress: usize,
    pub total: usize,
    pub board_key: Option<String>,
}

/// Get the current bucket computation status (core logic, no Tauri dependency).
pub fn get_computation_status_core(state: &ExplorationState) -> ComputationStatus {
    ComputationStatus {
        computing: state.computing.load(Ordering::SeqCst),
        progress: state.computation_progress.load(Ordering::SeqCst),
        total: state.computation_total.load(Ordering::SeqCst),
        board_key: state.computing_board_key.read().clone(),
    }
}

/// Get the current bucket computation status (Tauri wrapper).
#[tauri::command]
pub fn get_computation_status(state: State<'_, ExplorationState>) -> ComputationStatus {
    get_computation_status_core(&state)
}

/// Start async bucket computation for a board (core logic, no Tauri dependency).
///
/// Returns immediately for agents (no computation needed).
/// For bundles, computation happens in background.
/// An optional `on_progress` callback receives `(completed, total, board_key)`.
pub fn start_bucket_computation_core(
    state: &ExplorationState,
    board: Vec<String>,
    on_progress: Option<ProgressCallback>,
) -> Result<String, String> {
    let board_key = board.join("");

    // Agents, hand_class bundles, preflop solves, and subgame solves don't need bucket computation
    {
        let source_guard = state.source.read();
        match source_guard.as_ref() {
            Some(
                StrategySource::Agent(_)
                | StrategySource::PreflopSolve { .. }
                | StrategySource::BlueprintV2 { .. },
            ) => {
                return Ok(board_key);
            }
            Some(StrategySource::Bundle { config, .. })
                if config.abstraction_mode.is_hand_class() =>
            {
                return Ok(board_key);
            }
            Some(StrategySource::SubgameSolve { blueprint_config, .. })
                if blueprint_config.abstraction_mode.is_hand_class() =>
            {
                return Ok(board_key);
            }
            _ => {}
        }
    }

    // Parse board
    let board_cards = parse_board(&board)?;

    // Check if already cached
    {
        let cache = state.bucket_cache.read();
        if cache.contains_key(&board_key) {
            return Ok(board_key);
        }
    }

    // Check if already computing
    if state.computing.load(Ordering::SeqCst) {
        let current_key = state.computing_board_key.read();
        if current_key.as_ref() == Some(&board_key) {
            // Already computing this board
            return Ok(board_key);
        }
        // Different board - stop current computation
        state.computing.store(false, Ordering::SeqCst);
    }

    // Start computation
    state.computing.store(true, Ordering::SeqCst);
    state.computation_progress.store(0, Ordering::SeqCst);
    state.computation_total.store(169, Ordering::SeqCst); // 169 hand combos
    *state.computing_board_key.write() = Some(board_key.clone());

    // Get abstraction boundaries for thread
    let boundaries_guard = state.abstraction_boundaries.read();
    let boundaries = boundaries_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?
        .clone();
    drop(boundaries_guard);

    let abstraction = CardAbstraction::from_boundaries(boundaries);

    // Clone Arc references for the thread
    let computing = Arc::clone(&state.computing);
    let progress = Arc::clone(&state.computation_progress);
    let bucket_cache = Arc::clone(&state.bucket_cache);
    let computing_board_key = Arc::clone(&state.computing_board_key);
    let board_key_clone = board_key.clone();

    // Spawn background thread
    std::thread::spawn(move || {
        let ranks = [
            'A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2',
        ];
        let mut completed = 0;
        let mut local_cache: HashMap<(char, char, bool), u16> = HashMap::new();

        for &rank1 in &ranks {
            for &rank2 in &ranks {
                for suited in [true, false] {
                    // Check if cancelled
                    if !computing.load(Ordering::SeqCst) {
                        return;
                    }

                    // Skip invalid combos (pairs can't be suited)
                    if rank1 == rank2 && suited {
                        continue;
                    }

                    // Compute bucket
                    let (card1, card2) =
                        make_representative_hand(rank1, rank2, suited, &board_cards);
                    if let Ok(bucket) = abstraction.get_bucket(&board_cards, (card1, card2)) {
                        local_cache.insert((rank1, rank2, suited), bucket);
                    }

                    completed += 1;
                    progress.store(completed, Ordering::SeqCst);

                    // Emit progress event every 10 hands
                    if completed % 10 == 0 {
                        if let Some(ref cb) = on_progress {
                            cb(completed, 169, &board_key_clone);
                        }
                    }
                }
            }
        }

        // Store in global cache
        bucket_cache
            .write()
            .insert(board_key_clone.clone(), local_cache);

        // Computation complete - emit final event
        if let Some(ref cb) = on_progress {
            cb(169, 169, &board_key_clone);
        }

        computing.store(false, Ordering::SeqCst);
        *computing_board_key.write() = None;
    });

    Ok(board_key)
}

/// Start async bucket computation for a board (Tauri wrapper).
#[tauri::command]
pub fn start_bucket_computation(
    app: AppHandle,
    state: State<'_, ExplorationState>,
    board: Vec<String>,
) -> Result<String, String> {
    let on_progress = Box::new(move |completed: usize, total: usize, board_key: &str| {
        let _ = app.emit(
            "bucket-progress",
            BucketProgressEvent {
                completed,
                total,
                board_key: board_key.to_string(),
            },
        );
    });
    start_bucket_computation_core(&state, board, Some(on_progress))
}

/// Check if bucket computation for a board is complete (core logic, no Tauri dependency).
pub fn is_board_cached_core(state: &ExplorationState, board: Vec<String>) -> bool {
    let board_key = board.join("");
    let cache = state.bucket_cache.read();
    cache.contains_key(&board_key)
}

/// Check if bucket computation for a board is complete (Tauri wrapper).
#[tauri::command]
pub fn is_board_cached(state: State<'_, ExplorationState>, board: Vec<String>) -> bool {
    is_board_cached_core(&state, board)
}

/// Result of board canonicalization.
#[derive(Debug, Clone, Serialize)]
pub struct CanonicalizeResult {
    /// Canonical card strings (e.g., ["As", "Kh", "7d"])
    pub canonical_cards: Vec<String>,
    /// Whether the cards were remapped (false if already canonical)
    pub remapped: bool,
    /// Suit substitution map (original → canonical), only present when remapped
    pub suit_map: Option<HashMap<String, String>>,
}

/// Canonicalize board cards to their suit-isomorphic equivalent (core logic, no Tauri dependency).
///
/// On flop (3 cards): establishes a `SuitMapping` stored in state for reuse.
/// On turn/river (1 card): applies the stored flop mapping to the new card.
/// Returns canonical card strings and substitution info.
pub fn canonicalize_board_core(
    state: &ExplorationState,
    cards: Vec<String>,
) -> Result<CanonicalizeResult, String> {
    let parsed: Vec<Card> = cards.iter().map(|s| parse_card(s)).collect::<Result<_, _>>()?;

    if parsed.len() == 3 {
        // Flop: establish canonical mapping
        let canonical = CanonicalBoard::from_cards(&parsed)
            .map_err(|e| format!("Canonicalization failed: {e}"))?;

        let remapped = canonical.cards != parsed;
        let suit_map = if remapped {
            Some(build_suit_map(&parsed, &canonical.cards))
        } else {
            None
        };

        let canonical_strs: Vec<String> = canonical.cards.iter().map(format_card_short).collect();
        *state.suit_mapping.write() = Some(canonical.mapping);

        Ok(CanonicalizeResult {
            canonical_cards: canonical_strs,
            remapped,
            suit_map,
        })
    } else {
        // Turn/river: apply stored mapping
        let mapping_guard = state.suit_mapping.read();
        let mapping = mapping_guard
            .as_ref()
            .ok_or_else(|| "No flop mapping established".to_string())?;

        let canonical_strs: Vec<String> = parsed
            .iter()
            .map(|c| format_card_short(&mapping.map_card(*c)))
            .collect();

        let remapped = canonical_strs.iter().zip(cards.iter()).any(|(c, o)| c != o);
        let suit_map = if remapped {
            let mapped: Vec<Card> = parsed.iter().map(|c| mapping.map_card(*c)).collect();
            Some(build_suit_map(&parsed, &mapped))
        } else {
            None
        };

        Ok(CanonicalizeResult {
            canonical_cards: canonical_strs,
            remapped,
            suit_map,
        })
    }
}

/// Canonicalize board cards (Tauri wrapper).
#[tauri::command]
pub fn canonicalize_board(
    state: State<'_, ExplorationState>,
    cards: Vec<String>,
) -> Result<CanonicalizeResult, String> {
    canonicalize_board_core(&state, cards)
}

/// Build a suit substitution map from original cards to canonical cards.
fn build_suit_map(original: &[Card], canonical: &[Card]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for (orig, canon) in original.iter().zip(canonical.iter()) {
        if orig.suit != canon.suit {
            map.insert(suit_char(orig.suit), suit_char(canon.suit));
        }
    }
    map
}

/// Format a suit as its single-character abbreviation.
fn suit_char(suit: Suit) -> String {
    match suit {
        Suit::Spade => "s".to_string(),
        Suit::Heart => "h".to_string(),
        Suit::Diamond => "d".to_string(),
        Suit::Club => "c".to_string(),
    }
}

/// Format a card as a short string (e.g., "As", "Kh").
fn format_card_short(card: &Card) -> String {
    let rank = value_to_char(card.value);
    let suit = suit_char(card.suit);
    format!("{rank}{suit}")
}

/// List available agent configs from the agents/ directory.
///
/// Searches for `agents/` in the current directory and up to 4 parent
/// directories, so it works regardless of which subdirectory the process
/// was started from (e.g. `crates/tauri-app/` during `cargo tauri dev`).
#[tauri::command]
pub fn list_agents() -> Result<Vec<AgentInfo>, String> {
    let agents_dir = match find_agents_dir() {
        Some(dir) => dir,
        None => return Ok(vec![]),
    };

    let entries = std::fs::read_dir(&agents_dir)
        .map_err(|e| format!("Failed to read agents directory: {e}"))?;

    let mut agents = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let name = match AgentConfig::load(&path) {
            Ok(config) => config.game.name.unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Unknown")
                    .to_string()
            }),
            Err(_) => continue,
        };
        agents.push(AgentInfo {
            name,
            path: path.to_string_lossy().to_string(),
        });
    }

    agents.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(agents)
}

/// Walk up from CWD looking for an `agents/` directory (max 4 levels).
fn find_agents_dir() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    for _ in 0..5 {
        let candidate = dir.join("agents");
        if candidate.is_dir() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

/// A group of combos sharing the same hand classification.
#[derive(Debug, Clone, Serialize)]
pub struct ComboGroup {
    /// Raw `HandClassification` bits.
    pub bits: u32,
    /// Human-readable class names (e.g. `["Pair", "FlushDraw"]`).
    pub class_names: Vec<String>,
    /// Formatted combo strings (e.g. `["A♠K♠", "A♣K♣"]`).
    pub combos: Vec<String>,
    /// Action probabilities from the blueprint.
    pub strategy: Vec<f32>,
}

/// Per-cell combo classification breakdown.
#[derive(Debug, Clone, Serialize)]
pub struct ComboGroupInfo {
    /// Hand label (e.g. "AKs").
    pub hand: String,
    /// Combo groups, each with a distinct classification.
    pub groups: Vec<ComboGroup>,
    /// Number of valid (non-blocked) combos.
    pub total_combos: usize,
    /// Number of board-blocked combos.
    pub blocked_combos: usize,
    /// Current street name.
    pub street: String,
    /// SPR bucket used for key construction.
    pub spr_bucket: u32,
}

/// Get combo-level classification breakdown for a selected cell (core logic, no Tauri dependency).
///
/// Enumerates all specific combos of a canonical hand (e.g. "AKs"),
/// classifies each on the current board, groups by classification,
/// and looks up the blueprint strategy for each group.
///
/// Only meaningful for hand_class bundles on postflop streets.
pub fn get_combo_classes_core(
    state: &ExplorationState,
    position: ExplorationPosition,
    hand: String,
) -> Result<ComboGroupInfo, String> {
    let source_guard = state.source.read();
    let source = source_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    let (config, blueprint) = match source {
        StrategySource::Bundle { config, blueprint } => (config, blueprint),
        StrategySource::SubgameSolve {
            blueprint_config,
            blueprint,
            ..
        } => (blueprint_config, blueprint.as_ref()),
        StrategySource::Agent(_)
        | StrategySource::PreflopSolve { .. }
        | StrategySource::BlueprintV2 { .. } => {
            return Ok(empty_combo_info(&hand));
        }
    };

    if !config.abstraction_mode.is_hand_class() {
        return Ok(empty_combo_info(&hand));
    }

    let board = parse_board(&position.board)?;
    let street = street_from_board_len(board.len())?;

    if street == Street::Preflop {
        return Ok(empty_combo_info(&hand));
    }

    let (rank1, rank2, suited) = parse_hand_label(&hand)?;
    let max_combos = max_combo_count(rank1, rank2, suited);
    let combos = enumerate_combos(rank1, rank2, suited, &board);
    let blocked_combos = max_combos - combos.len();

    let action_codes = build_action_codes(&position.history);
    let pos_state = compute_position_state(&config.game.bet_sizes, &position);
    let street_num = street_to_num(street);
    let spr = spr_bucket(pos_state.pot, pos_state.eff_stack);

    // Group combos by hand bits (v2 encoding: class_id + strength + equity + draw flags)
    let mut groups_map: std::collections::BTreeMap<u32, Vec<String>> =
        std::collections::BTreeMap::new();
    for combo in &combos {
        let bits = match classify(*combo, &board) {
            Ok(classification) => {
                let made_id = classification.strongest_made_id();
                let strength = if HandClass::is_made_hand_id(made_id) {
                    intra_class_strength(
                        *combo,
                        &board,
                        HandClass::ALL[made_id as usize],
                    )
                } else {
                    1
                };
                let equity = showdown_equity::compute_equity(*combo, &board);
                let eq_bin = showdown_equity::equity_bin(
                    equity,
                    1u8 << config.equity_bits,
                );
                encode_hand_v2(
                    made_id,
                    strength,
                    eq_bin,
                    classification.draw_flags(),
                    config.strength_bits,
                    config.equity_bits,
                )
            }
            Err(_) => 0,
        };
        groups_map
            .entry(bits)
            .or_default()
            .push(format_combo(*combo));
    }

    // Build ComboGroups with blueprint lookup
    let groups: Vec<ComboGroup> = groups_map
        .into_iter()
        .map(|(bits, combo_strs)| {
            let key = InfoKey::new(
                bits,
                street_num,
                spr,
                &action_codes,
            )
            .as_u64();
            let strategy = blueprint
                .lookup(key)
                .map(|s| s.to_vec())
                .unwrap_or_default();
            let class_names = decode_v2_class_names(bits);
            ComboGroup {
                bits,
                class_names,
                combos: combo_strs,
                strategy,
            }
        })
        .collect();

    Ok(ComboGroupInfo {
        hand,
        groups,
        total_combos: combos.len(),
        blocked_combos,
        street: format!("{street:?}"),
        spr_bucket: spr,
    })
}

/// Get combo-level classification breakdown (Tauri wrapper).
#[tauri::command]
pub fn get_combo_classes(
    state: State<'_, ExplorationState>,
    position: ExplorationPosition,
    hand: String,
) -> Result<ComboGroupInfo, String> {
    get_combo_classes_core(&state, position, hand)
}

fn empty_combo_info(hand: &str) -> ComboGroupInfo {
    ComboGroupInfo {
        hand: hand.to_string(),
        groups: vec![],
        total_combos: 0,
        blocked_combos: 0,
        street: String::new(),
        spr_bucket: 0,
    }
}

/// Parse a hand label like "AKs", "QQ", "T9o" into (rank1, rank2, suited).
fn parse_hand_label(label: &str) -> Result<(char, char, bool), String> {
    let chars: Vec<char> = label.chars().collect();
    if chars.len() < 2 {
        return Err(format!("Invalid hand label: {label}"));
    }
    let rank1 = chars[0];
    let rank2 = chars[1];
    let suited = chars.get(2) == Some(&'s');
    Ok((rank1, rank2, suited))
}

/// Maximum number of combos for a canonical hand type.
fn max_combo_count(rank1: char, rank2: char, suited: bool) -> usize {
    if rank1 == rank2 {
        6 // pairs: C(4,2)
    } else if suited {
        4 // suited: 4 suits
    } else {
        12 // offsuit: 4×3
    }
}

/// Enumerate all specific combos of a canonical hand, excluding board-blocked cards.
fn enumerate_combos(rank1: char, rank2: char, suited: bool, board: &[Card]) -> Vec<[Card; 2]> {
    let v1 = char_to_value(rank1);
    let v2 = char_to_value(rank2);
    let suits = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];
    let board_set: std::collections::HashSet<Card> = board.iter().copied().collect();

    let mut combos = Vec::new();

    if rank1 == rank2 {
        // Pairs: all suit pairs where s1 < s2
        for (i, &s1) in suits.iter().enumerate() {
            let c1 = Card::new(v1, s1);
            if board_set.contains(&c1) {
                continue;
            }
            for &s2 in &suits[i + 1..] {
                let c2 = Card::new(v2, s2);
                if !board_set.contains(&c2) {
                    combos.push([c1, c2]);
                }
            }
        }
    } else if suited {
        // Suited: same suit for both cards
        for &s in &suits {
            let c1 = Card::new(v1, s);
            let c2 = Card::new(v2, s);
            if !board_set.contains(&c1) && !board_set.contains(&c2) {
                combos.push([c1, c2]);
            }
        }
    } else {
        // Offsuit: different suits
        for &s1 in &suits {
            let c1 = Card::new(v1, s1);
            if board_set.contains(&c1) {
                continue;
            }
            for &s2 in &suits {
                if s1 == s2 {
                    continue;
                }
                let c2 = Card::new(v2, s2);
                if !board_set.contains(&c2) {
                    combos.push([c1, c2]);
                }
            }
        }
    }

    combos
}

/// Decode v2 packed bits into human-readable class names.
///
/// v2 layout: class_id(5) | strength(4) | equity(4) | draw_flags(NUM_DRAWS) | spare
fn decode_v2_class_names(bits: u32) -> Vec<String> {
    let class_id = (bits >> 23) & 0x1F;
    let draw_mask = (1u32 << HandClass::NUM_DRAWS) - 1;
    let draw_flags = (bits >> 8) & draw_mask;

    let mut names = Vec::new();

    // Made hand class
    if HandClass::is_made_hand_id(class_id as u8) {
        if let Some(class) = HandClass::from_discriminant(class_id as u8) {
            names.push(class.to_string());
        }
    }

    // Draw classes (mapped from draw_flags)
    for i in 0u8..(HandClass::NUM_DRAWS as u8) {
        if draw_flags & (1 << i) != 0 {
            if let Some(class) = HandClass::from_discriminant(HandClass::DRAW_ONLY_ID + i) {
                names.push(class.to_string());
            }
        }
    }

    names
}

/// Format a card as a unicode string (e.g. "A♠").
fn format_card(card: Card) -> String {
    let rank = value_to_char(card.value);
    let suit = match card.suit {
        Suit::Spade => '♠',
        Suit::Heart => '♥',
        Suit::Diamond => '♦',
        Suit::Club => '♣',
    };
    format!("{rank}{suit}")
}

/// Format a two-card combo (e.g. "A♠K♠").
fn format_combo(cards: [Card; 2]) -> String {
    format!("{}{}", format_card(cards[0]), format_card(cards[1]))
}

/// Convert a Value back to its rank character.
fn value_to_char(v: Value) -> char {
    match v {
        Value::Ace => 'A',
        Value::King => 'K',
        Value::Queen => 'Q',
        Value::Jack => 'J',
        Value::Ten => 'T',
        Value::Nine => '9',
        Value::Eight => '8',
        Value::Seven => '7',
        Value::Six => '6',
        Value::Five => '5',
        Value::Four => '4',
        Value::Three => '3',
        Value::Two => '2',
    }
}

// ============================================================================
// Range filtering
// ============================================================================

/// Check if a hand would be in the viewing player's range at the current position.
///
/// Replays the full game history (completed streets + current street) and
/// checks that at each prior decision point of the viewing player, the action
/// taken had at least `threshold` probability in the blueprint.
#[allow(clippy::too_many_arguments)]
fn is_hand_in_range(
    config: &BundleConfig,
    blueprint: &poker_solver_core::blueprint::BlueprintStrategy,
    full_board: &[Card],
    street_histories: &[Vec<String>],
    current_history: &[String],
    viewing_player: u8,
    threshold: f32,
    rank1: char,
    rank2: char,
    suited: bool,
) -> bool {
    let bet_sizes = &config.game.bet_sizes;
    let mut stacks = [
        config.game.stack_depth * 2 - 1,
        config.game.stack_depth * 2 - 2,
    ];
    let mut pot = 3u32;

    // Process completed streets, then the current street
    let all_streets: Vec<&[String]> = street_histories
        .iter()
        .map(|v| v.as_slice())
        .chain(std::iter::once(current_history))
        .collect();

    for (street_idx, street_actions) in all_streets.iter().enumerate() {
        let board_for_street = board_at_street(full_board, street_idx);
        let mut to_call = initial_to_call(street_idx, &stacks);
        let mut action_codes: Vec<u8> = Vec::new();

        for (i, action) in street_actions.iter().enumerate() {
            let acting_player = (i % 2) as u8;

            if acting_player == viewing_player
                && !action_meets_threshold(
                    blueprint,
                    config,
                    board_for_street,
                    street_idx,
                    &action_codes,
                    pot,
                    &stacks,
                    to_call,
                    action,
                    threshold,
                    rank1,
                    rank2,
                    suited,
                )
            {
                return false;
            }

            action_codes.push(action_to_code(action));
            apply_action(action, i % 2, &mut stacks, &mut pot, &mut to_call, bet_sizes);
        }
    }

    true
}

/// Board cards visible during a given street.
fn board_at_street(full_board: &[Card], street_idx: usize) -> &[Card] {
    let n = match street_idx {
        0 => 0,
        1 => 3,
        2 => 4,
        3 => 5,
        _ => 0,
    };
    &full_board[..n.min(full_board.len())]
}

/// Initial to-call amount at the start of a street.
fn initial_to_call(street_idx: usize, stacks: &[u32; 2]) -> u32 {
    if street_idx == 0 {
        stacks[0].saturating_sub(stacks[1])
    } else {
        0
    }
}

/// Check whether a single action's blueprint probability meets the threshold.
#[allow(clippy::too_many_arguments)]
fn action_meets_threshold(
    blueprint: &poker_solver_core::blueprint::BlueprintStrategy,
    config: &BundleConfig,
    board: &[Card],
    street_idx: usize,
    action_codes: &[u8],
    pot: u32,
    stacks: &[u32; 2],
    to_call: u32,
    action: &str,
    threshold: f32,
    rank1: char,
    rank2: char,
    suited: bool,
) -> bool {
    let hand_bits = hand_bits_at_street(config, rank1, rank2, suited, board, street_idx);
    let street_num = street_idx.min(3) as u8;
    let eff_stack = stacks[0].min(stacks[1]);
    let key = InfoKey::new(hand_bits, street_num, spr_bucket(pot, eff_stack), action_codes).as_u64();

    let strategy = match blueprint.lookup(key) {
        Some(s) => s,
        None => return true, // no data → assume in range
    };

    let code = action_to_code(action);
    let idx = action_code_to_strategy_index(code, to_call, config.game.bet_sizes.len());

    match idx.and_then(|i| strategy.get(i)) {
        Some(&prob) => prob >= threshold,
        None => true,
    }
}

/// Compute hand bits for a street (preflop uses canonical rank index,
/// postflop uses hand-class or hand-class-v2 classification).
fn hand_bits_at_street(
    config: &BundleConfig,
    rank1: char,
    rank2: char,
    suited: bool,
    board: &[Card],
    street_idx: usize,
) -> u32 {
    if street_idx == 0 {
        return hand_bits_from_ranks(rank1, rank2, suited);
    }
    let (c1, c2) = make_representative_hand(rank1, rank2, suited, board);
    let classification = match classify([c1, c2], board) {
        Ok(c) => c,
        Err(_) => return 0,
    };
    if config.abstraction_mode == AbstractionModeConfig::HandClassV2 {
        let made_id = classification.strongest_made_id();
        let strength = if HandClass::is_made_hand_id(made_id) {
            intra_class_strength([c1, c2], board, HandClass::ALL[made_id as usize])
        } else {
            1
        };
        let equity = showdown_equity::compute_equity([c1, c2], board);
        let eq_bin = showdown_equity::equity_bin(equity, 1u8 << config.equity_bits);
        encode_hand_v2(
            made_id,
            strength,
            eq_bin,
            classification.draw_flags(),
            config.strength_bits,
            config.equity_bits,
        )
    } else {
        classification.bits()
    }
}

/// Map an action code to its index in the strategy probability vector.
fn action_code_to_strategy_index(code: u8, to_call: u32, num_bet_sizes: usize) -> Option<usize> {
    match code {
        1 => {
            // fold — only valid when facing a bet
            if to_call > 0 { Some(0) } else { None }
        }
        2 => Some(if to_call > 0 { 1 } else { 0 }), // check
        3 => {
            // call
            if to_call > 0 { Some(1) } else { None }
        }
        4..=8 => Some(1 + (code - 4) as usize),                 // bet:idx (to_call == 0)
        9..=13 => Some(2 + (code - 9) as usize),               // raise:idx (to_call > 0)
        14 => Some(1 + num_bet_sizes),                          // bet all-in
        15 => Some(2 + num_bet_sizes),                          // raise all-in
        _ => None,
    }
}

/// Update game state after one action (shared by replay and range filter).
fn apply_action(
    action: &str,
    player: usize,
    stacks: &mut [u32; 2],
    pot: &mut u32,
    to_call: &mut u32,
    bet_sizes: &[f32],
) {
    if action == "c" || action == "call" {
        let amt = (*to_call).min(stacks[player]);
        stacks[player] -= amt;
        *pot += amt;
        *to_call = 0;
    } else if let Some(idx_str) = action
        .strip_prefix("bet:")
        .or_else(|| action.strip_prefix("raise:"))
        .or_else(|| action.strip_prefix("b:"))
        .or_else(|| action.strip_prefix("r:"))
    {
        let effective = stacks[player].saturating_sub(*to_call);
        let bet_portion = resolve_bet_index(idx_str, bet_sizes, *pot, effective);
        let total = *to_call + bet_portion;
        let actual = total.min(stacks[player]);
        stacks[player] -= actual;
        *pot += actual;
        *to_call = actual.saturating_sub(*to_call);
    }
    // fold, check: no state change
}

// ============================================================================
// Helper functions
// ============================================================================

fn parse_board(board_strs: &[String]) -> Result<Vec<Card>, String> {
    board_strs.iter().map(|s| parse_card(s)).collect()
}

fn parse_card(s: &str) -> Result<Card, String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() != 2 {
        return Err(format!("Invalid card: {s}"));
    }

    let value = match chars[0].to_ascii_uppercase() {
        'A' => Value::Ace,
        'K' => Value::King,
        'Q' => Value::Queen,
        'J' => Value::Jack,
        'T' => Value::Ten,
        '9' => Value::Nine,
        '8' => Value::Eight,
        '7' => Value::Seven,
        '6' => Value::Six,
        '5' => Value::Five,
        '4' => Value::Four,
        '3' => Value::Three,
        '2' => Value::Two,
        c => return Err(format!("Invalid rank: {c}")),
    };

    let suit = match chars[1].to_ascii_lowercase() {
        's' => Suit::Spade,
        'h' => Suit::Heart,
        'd' => Suit::Diamond,
        'c' => Suit::Club,
        c => return Err(format!("Invalid suit: {c}")),
    };

    Ok(Card::new(value, suit))
}

fn street_from_board_len(len: usize) -> Result<Street, String> {
    match len {
        0 => Ok(Street::Preflop),
        3 => Ok(Street::Flop),
        4 => Ok(Street::Turn),
        5 => Ok(Street::River),
        n => Err(format!("Invalid board length: {n}")),
    }
}

/// Convert history action strings to 4-bit action codes for `InfoKey`.
fn build_action_codes(history: &[String]) -> Vec<u8> {
    history.iter().map(|a| action_to_code(a)).collect()
}

/// Convert a history action string to a 4-bit action code.
///
/// Matches `encode_action` from `info_key`:
/// 1=fold, 2=check, 3=call, 4-8=bet idx 0-4, 9-13=raise idx 0-4,
/// 14=bet all-in, 15=raise all-in.
fn action_to_code(action: &str) -> u8 {
    if action == "f" || action == "fold" {
        1
    } else if action == "x" || action == "check" {
        2
    } else if action == "c" || action == "call" {
        3
    } else if let Some(idx_str) = action
        .strip_prefix("bet:")
        .or_else(|| action.strip_prefix("b:"))
    {
        if idx_str == "A" {
            14 // bet all-in
        } else {
            let idx: u8 = idx_str.parse().unwrap_or(0);
            4 + idx.min(4) // bet(idx)
        }
    } else if let Some(idx_str) = action
        .strip_prefix("raise:")
        .or_else(|| action.strip_prefix("r:"))
    {
        if idx_str == "A" {
            15 // raise all-in
        } else {
            let idx: u8 = idx_str.parse().unwrap_or(0);
            9 + idx.min(4) // raise(idx)
        }
    } else {
        0 // unknown → empty
    }
}

fn get_actions_for_position(
    _stack_depth: u32,
    bet_sizes: &[f32],
    position: &ExplorationPosition,
) -> Vec<ActionInfo> {
    let mut actions = Vec::new();
    let pos_state = compute_position_state(bet_sizes, position);
    let to_call = pos_state.to_call;
    let stack = position.stacks.get(position.to_act as usize).copied().unwrap_or(0);

    // Fold if facing a bet
    if to_call > 0 {
        actions.push(ActionInfo {
            id: "fold".to_string(),
            label: "Fold".to_string(),
            action_type: "fold".to_string(),
            size_key: None,
        });
    }

    // Check or call
    if to_call == 0 {
        actions.push(ActionInfo {
            id: "check".to_string(),
            label: "Check".to_string(),
            action_type: "check".to_string(),
            size_key: None,
        });
    } else if stack >= to_call {
        actions.push(ActionInfo {
            id: "call".to_string(),
            label: format!("Call {to_call}"),
            action_type: "call".to_string(),
            size_key: None,
        });
    }

    // Bet/raise sizes from config — always include ALL indices + ALL_IN,
    // matching HunlPostflop::actions() order exactly so blueprint
    // probability indices align.
    let effective_stack = stack.saturating_sub(to_call);
    if effective_stack > 0 {
        for (idx, &fraction) in bet_sizes.iter().enumerate() {
            let action_type = if to_call == 0 { "bet" } else { "raise" };

            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let size = (f64::from(position.pot) * f64::from(fraction)).round() as u32;
            let capped = size.min(effective_stack);
            let display_total = if to_call == 0 { capped } else { to_call + capped };

            actions.push(ActionInfo {
                id: format!("{action_type}:{idx}"),
                label: format!(
                    "{} {display_total}",
                    if to_call == 0 { "Bet" } else { "Raise to" }
                ),
                action_type: action_type.to_string(),
                size_key: Some(bet_size_key(fraction)),
            });
        }

        // All-in (uses "A" as the index sentinel)
        let action_type = if to_call == 0 { "bet" } else { "raise" };
        actions.push(ActionInfo {
            id: format!("{action_type}:A"),
            label: format!("All-in {}", stack),
            action_type: "allin".to_string(),
            size_key: Some("allin".to_string()),
        });
    }

    actions
}

/// Format a bet size fraction as a string key matching TOML config keys.
/// Ensures `1.0` → `"1.0"` (not `"1"`).
fn bet_size_key(fraction: f32) -> String {
    let s = format!("{fraction}");
    if s.contains('.') { s } else { format!("{fraction:.1}") }
}

/// Replayed position state — pot, stacks, and to-call at the current node.
struct PositionState {
    to_call: u32,
    pot: u32,
    eff_stack: u32,
    stack_p1: u32,
    stack_p2: u32,
}

/// Replay the action history to determine pot, stacks, and to-call.
///
/// Tracks remaining stacks directly, matching `HunlPostflop::next_state`.
/// Position stacks already have blinds deducted (stack_p1 = depth-1,
/// stack_p2 = depth-2), so we replay from there without re-subtracting.
fn compute_position_state(bet_sizes: &[f32], position: &ExplorationPosition) -> PositionState {
    let stack_p1 = position.stacks.first().copied().unwrap_or(0);
    let stack_p2 = position.stacks.get(1).copied().unwrap_or(0);
    let mut stacks = [stack_p1, stack_p2];
    let mut pot = position.pot;

    // Preflop: SB owes 1 more to match BB's blind.
    // Postflop: no outstanding bet.
    let mut to_call = if position.board.is_empty() {
        stack_p1.saturating_sub(stack_p2)
    } else {
        0u32
    };

    for (i, action) in position.history.iter().enumerate() {
        let player = i % 2;

        if action == "c" || action == "call" {
            let amt = to_call.min(stacks[player]);
            stacks[player] -= amt;
            pot += amt;
            to_call = 0;
        } else if let Some(idx_str) = action
            .strip_prefix("bet:")
            .or_else(|| action.strip_prefix("raise:"))
            .or_else(|| action.strip_prefix("b:"))
            .or_else(|| action.strip_prefix("r:"))
        {
            let effective = stacks[player].saturating_sub(to_call);
            let bet_portion = resolve_bet_index(idx_str, bet_sizes, pot, effective);
            let total = to_call + bet_portion;
            let actual = total.min(stacks[player]);
            stacks[player] -= actual;
            pot += actual;
            to_call = actual.saturating_sub(to_call);
        }
        // "f"/"fold", "x"/"check" don't change stacks
    }

    PositionState {
        to_call,
        pot,
        eff_stack: stacks[0].min(stacks[1]),
        stack_p1: stacks[0],
        stack_p2: stacks[1],
    }
}

/// Resolve a bet-size index string to a cent amount.
///
/// `"A"` maps to all-in (effective stack). Numeric indices look up the
/// fraction in `bet_sizes` and compute the pot-relative amount.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn resolve_bet_index(idx_str: &str, bet_sizes: &[f32], pot: u32, effective_stack: u32) -> u32 {
    if idx_str == "A" {
        return effective_stack;
    }
    if let Ok(idx) = idx_str.parse::<usize>() {
        if let Some(&fraction) = bet_sizes.get(idx) {
            let size = (f64::from(pot) * f64::from(fraction)).round() as u32;
            return size.min(effective_stack);
        }
    }
    // Fallback: try parsing as raw cent amount (backwards compat)
    idx_str.parse::<u32>().unwrap_or(0)
}

/// Look up blueprint strategy for a hand, given precomputed hand bits.
///
/// Shared implementation for all abstraction modes. Each mode computes
/// `hand_bits` differently, but the InfoKey construction, blueprint lookup,
/// and ActionProb mapping are identical.
fn lookup_blueprint_strategy(
    blueprint: &poker_solver_core::blueprint::BlueprintStrategy,
    hand_bits: u32,
    street: Street,
    action_codes: &[u8],
    pot: u32,
    eff_stack: u32,
    actions: &[ActionInfo],
) -> Result<Vec<ActionProb>, String> {
    let street_num = street_to_num(street);
    let spr = spr_bucket(pot, eff_stack);

    let info_set_key =
        InfoKey::new(hand_bits, street_num, spr, action_codes).as_u64();

    let probs = blueprint.lookup(info_set_key).ok_or_else(|| {
        format!(
            "Blueprint lookup failed for key {info_set_key:#018x} \
             (blueprint has {} info sets)",
            blueprint.len()
        )
    })?;

    Ok(actions
        .iter()
        .enumerate()
        .map(|(i, a)| ActionProb {
            action: a.label.clone(),
            probability: probs.get(i).copied().unwrap_or(0.0),
        })
        .collect())
}

/// Compute hand bits for EHS2 mode (requires precomputed bucket cache).
fn hand_bits_ehs2(
    street: Street,
    rank1: char,
    rank2: char,
    suited: bool,
    bucket_cache: Option<&std::collections::HashMap<(char, char, bool), u16>>,
) -> Result<u32, String> {
    if street == Street::Preflop {
        return Ok(hand_bits_from_ranks(rank1, rank2, suited));
    }
    let cache = bucket_cache.ok_or_else(|| {
        format!(
            "Bucket cache not available for postflop street {street:?}. \
             Start bucket computation before requesting postflop strategy."
        )
    })?;
    let &bucket = cache.get(&(rank1, rank2, suited)).ok_or_else(|| {
        format!("No bucket in cache for hand ({rank1}, {rank2}, suited={suited})")
    })?;
    Ok(u32::from(bucket))
}

/// Compute hand bits for hand_class_v2 mode (classify + strength + equity).
fn hand_bits_hand_class_v2(
    config: &BundleConfig,
    street: Street,
    board: &[Card],
    rank1: char,
    rank2: char,
    suited: bool,
) -> u32 {
    if street == Street::Preflop {
        return hand_bits_from_ranks(rank1, rank2, suited);
    }
    let (card1, card2) = make_representative_hand(rank1, rank2, suited, board);
    match classify([card1, card2], board) {
        Ok(classification) => {
            let made_id = classification.strongest_made_id();
            let strength = if HandClass::is_made_hand_id(made_id) {
                intra_class_strength(
                    [card1, card2],
                    board,
                    HandClass::ALL[made_id as usize],
                )
            } else {
                1
            };
            let equity = showdown_equity::compute_equity([card1, card2], board);
            let eq_bin =
                showdown_equity::equity_bin(equity, 1u8 << config.equity_bits);
            encode_hand_v2(
                made_id,
                strength,
                eq_bin,
                classification.draw_flags(),
                config.strength_bits,
                config.equity_bits,
            )
        }
        Err(_) => 0,
    }
}

/// Compute hand bits from rank characters and suited flag for `InfoKey`.
fn hand_bits_from_ranks(rank1: char, rank2: char, suited: bool) -> u32 {
    if let Some(cards) = cards_from_rank_chars(rank1, rank2, suited) {
        u32::from(canonical_hand_index(cards))
    } else {
        0
    }
}

/// Convert a `Street` to numeric encoding for `InfoKey`.
fn street_to_num(street: Street) -> u8 {
    match street {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    }
}

const RANKS: [char; 13] = [
    'A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2',
];

fn hand_label_from_matrix(
    row: usize,
    col: usize,
    rank1: char,
    rank2: char,
) -> (String, bool, bool) {
    if row == col {
        (format!("{rank1}{rank1}"), false, true)
    } else if row < col {
        (format!("{rank1}{rank2}s"), true, false)
    } else {
        (format!("{rank2}{rank1}o"), false, false)
    }
}

fn resolve_preflop_frequency<'a>(
    agent: &'a AgentConfig,
    position_name: &str,
    rank1: char,
    rank2: char,
    suited: bool,
) -> &'a FrequencyMap {
    let v1 = char_to_value(rank1);
    let v2 = char_to_value(rank2);
    let canonical = CanonicalHand::new(v1, v2, suited);

    if agent.ranges.is_empty() {
        &agent.default
    } else {
        agent.preflop_frequency(position_name, &canonical)
    }
}

fn resolve_postflop_frequency<'a>(
    agent: &'a AgentConfig,
    rank1: char,
    rank2: char,
    suited: bool,
    board: &[Card],
) -> &'a FrequencyMap {
    let (card1, card2) = make_representative_hand(rank1, rank2, suited, board);
    match classify([card1, card2], board) {
        Ok(classification) => agent.resolve(&classification),
        Err(_) => &agent.default,
    }
}

fn map_frequencies_to_actions(freq: &FrequencyMap, actions: &[ActionInfo]) -> Vec<ActionProb> {
    let has_fold = actions.iter().any(|a| a.action_type == "fold");
    let raise_count = actions
        .iter()
        .filter(|a| matches!(a.action_type.as_str(), "bet" | "raise" | "allin"))
        .count();

    // Assign raw probability per action
    let mut probs: Vec<f32> = actions
        .iter()
        .map(|a| raw_action_freq(freq, a, raise_count))
        .collect();

    // Redistribute fold frequency when fold is not available (check scenario)
    if !has_fold {
        redistribute(&mut probs, actions, |a| a.action_type == "fold");
    }

    // Redistribute raise frequencies when no raise actions exist
    if raise_count == 0 {
        redistribute(&mut probs, actions, |a| {
            matches!(a.action_type.as_str(), "bet" | "raise" | "allin")
        });
    }

    actions
        .iter()
        .zip(probs.iter())
        .map(|(a, &p)| ActionProb {
            action: a.label.clone(),
            probability: p,
        })
        .collect()
}

fn raw_action_freq(freq: &FrequencyMap, action: &ActionInfo, raise_count: usize) -> f32 {
    match action.action_type.as_str() {
        "fold" => freq.fold,
        "check" | "call" => freq.call,
        "bet" | "raise" | "allin" => {
            if let Some(ref sizes) = freq.raise_sizes {
                action
                    .size_key
                    .as_ref()
                    .and_then(|key| sizes.get(key))
                    .copied()
                    .unwrap_or(0.0)
            } else if raise_count > 0 {
                freq.raise / raise_count as f32
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

/// Redistribute frequency from removed actions to remaining actions proportionally.
fn redistribute(probs: &mut [f32], actions: &[ActionInfo], is_removed: impl Fn(&ActionInfo) -> bool) {
    let removed_sum: f32 = probs
        .iter()
        .zip(actions.iter())
        .filter(|(_, a)| is_removed(a))
        .map(|(p, _)| *p)
        .sum();

    if removed_sum <= 0.0 {
        return;
    }

    let kept_sum: f32 = probs
        .iter()
        .zip(actions.iter())
        .filter(|(_, a)| !is_removed(a))
        .map(|(p, _)| *p)
        .sum();

    if kept_sum > 0.0 {
        let scale = (kept_sum + removed_sum) / kept_sum;
        for (p, a) in probs.iter_mut().zip(actions.iter()) {
            if is_removed(a) {
                *p = 0.0;
            } else {
                *p *= scale;
            }
        }
    } else {
        // Fallback: assign everything to check/call
        for (p, a) in probs.iter_mut().zip(actions.iter()) {
            *p = if a.action_type == "check" || a.action_type == "call" {
                1.0
            } else {
                0.0
            };
        }
    }
}

fn make_representative_hand(
    rank1: char,
    rank2: char,
    suited: bool,
    board: &[Card],
) -> (Card, Card) {
    let v1 = char_to_value(rank1);
    let v2 = char_to_value(rank2);

    // Find suits not on board for these ranks
    let board_cards: std::collections::HashSet<_> = board.iter().collect();
    let suits = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];

    let mut suit1 = Suit::Spade;
    let mut suit2 = if suited { Suit::Spade } else { Suit::Heart };

    // Try to find valid suits
    'outer: for &s1 in &suits {
        let c1 = Card::new(v1, s1);
        if board_cards.contains(&c1) {
            continue;
        }

        for &s2 in &suits {
            if suited && s2 != s1 {
                continue;
            }
            if !suited && s2 == s1 && v1 != v2 {
                continue;
            }

            let c2 = Card::new(v2, s2);
            if !board_cards.contains(&c2) && (v1 != v2 || c1 != c2) {
                suit1 = s1;
                suit2 = s2;
                break 'outer;
            }
        }
    }

    (Card::new(v1, suit1), Card::new(v2, suit2))
}

fn char_to_value(c: char) -> Value {
    match c {
        'A' => Value::Ace,
        'K' => Value::King,
        'Q' => Value::Queen,
        'J' => Value::Jack,
        'T' => Value::Ten,
        '9' => Value::Nine,
        '8' => Value::Eight,
        '7' => Value::Seven,
        '6' => Value::Six,
        '5' => Value::Five,
        '4' => Value::Four,
        '3' => Value::Three,
        '2' => Value::Two,
        _ => Value::Two,
    }
}

// ---------------------------------------------------------------------------
// Preflop range extraction (169 → 1326 expansion)
// ---------------------------------------------------------------------------

/// Preflop ranges for both players at a given position in the game tree.
#[derive(Debug, Clone, Serialize)]
pub struct PreflopRanges {
    /// 1326-element combo weights for OOP (player 0).
    pub oop_weights: Vec<f32>,
    /// 1326-element combo weights for IP (player 1).
    pub ip_weights: Vec<f32>,
    /// Pot size at the current position (in BB).
    pub pot: f64,
    /// Effective stack at the current position (in BB).
    pub effective_stack: f64,
    /// OOP bet sizes for flop in range-solver format (e.g. "33%,67%,100%,e").
    pub oop_bet_sizes: String,
    /// OOP raise sizes for flop in range-solver format.
    pub oop_raise_sizes: String,
    /// IP bet sizes for flop in range-solver format.
    pub ip_bet_sizes: String,
    /// IP raise sizes for flop in range-solver format.
    pub ip_raise_sizes: String,
}

/// Convert an `rs_poker::core::Card` to a range-solver `u8` card.
///
/// Range-solver encoding: `card = 4 * rank + suit`
/// where rank 0 = deuce .. 12 = ace, suit 0 = club, 1 = diamond, 2 = heart, 3 = spade.
///
/// `rs_poker` encoding: `Value::Two = 0 .. Ace = 12`,
/// `Suit::Spade = 0, Club = 1, Heart = 2, Diamond = 3`.
fn rs_card_to_range_solver(card: Card) -> u8 {
    let rank = card.value as u8; // Two=0 .. Ace=12, matches range-solver
    let suit = match card.suit {
        Suit::Club => 0,
        Suit::Diamond => 1,
        Suit::Heart => 2,
        Suit::Spade => 3,
    };
    4 * rank + suit
}

/// Build a lookup table mapping each canonical hand index (0..169) to its
/// combo indices (0..1326) in range-solver encoding.
fn build_canonical_to_combo_map() -> Vec<Vec<usize>> {
    let mut map = Vec::with_capacity(169);
    for i in 0..169 {
        // INVARIANT: index is always 0..169, so `from_index` always returns Some
        let hand = CanonicalHand::from_index(i).expect("valid canonical index");
        let combos = hand.combos();
        let indices: Vec<usize> = combos
            .iter()
            .map(|(c1, c2)| {
                let r1 = rs_card_to_range_solver(*c1);
                let r2 = rs_card_to_range_solver(*c2);
                range_solver::card::card_pair_to_index(r1, r2)
            })
            .collect();
        map.push(indices);
    }
    map
}

/// Expand 169 canonical hand weights to 1326 combo weights.
///
/// Each combo inherits the weight of its canonical hand.
fn expand_169_to_1326(weights_169: &[f32; 169]) -> Vec<f32> {
    let combo_map = build_canonical_to_combo_map();
    let mut weights_1326 = vec![0.0f32; 1326];
    for (hand_idx, combo_indices) in combo_map.iter().enumerate() {
        let w = weights_169[hand_idx];
        for &combo_idx in combo_indices {
            weights_1326[combo_idx] = w;
        }
    }
    weights_1326
}

/// Format pot-fraction bet sizes as a range-solver size string.
///
/// Each fraction is formatted as a percentage (e.g. 0.33 → "33%").
/// Includes "e" (all-in) at the end since the blueprint tree always has all-in.
fn format_bet_sizes(fractions: &[f64]) -> String {
    if fractions.is_empty() {
        return "e".to_string();
    }
    let mut parts: Vec<String> = fractions
        .iter()
        .map(|f| format!("{}%", (f * 100.0).round() as u32))
        .collect();
    parts.push("e".to_string());
    parts.join(",")
}

/// Get the invested amounts at a V2 decision node by inspecting its fold child.
///
/// Every decision node that has a Fold action will have a fold terminal child
/// whose `invested` field reflects the amounts put in by each player *before*
/// that fold. For nodes without a fold (e.g. check-only), we look at the Call
/// child's terminal instead and back out the opponent's investment.
fn invested_at_v2_node(tree: &V2GameTree, node_idx: u32) -> [f64; 2] {
    if let V2GameNode::Decision {
        actions, children, ..
    } = &tree.nodes[node_idx as usize]
    {
        // Try the fold terminal first — its invested is exactly what we want.
        if let Some(fold_pos) = actions.iter().position(|a| matches!(a, TreeAction::Fold)) {
            let child_idx = children[fold_pos] as usize;
            if let V2GameNode::Terminal { invested, .. } = &tree.nodes[child_idx] {
                return *invested;
            }
        }
        // Fall back: use the check child or any terminal we can find.
        for &child in children {
            if let V2GameNode::Terminal { invested, .. } = &tree.nodes[child as usize] {
                return *invested;
            }
        }
    }
    [0.0; 2]
}

/// Compute preflop ranges at a given history position (core logic).
///
/// Walks the V2 game tree, multiplying each canonical hand's range weight by
/// the strategy probability of the chosen action, then expands the 169
/// canonical weights into 1326 combo weights.
#[allow(clippy::cast_possible_truncation)]
pub fn get_preflop_ranges_core(
    state: &ExplorationState,
    history: Vec<String>,
) -> Result<PreflopRanges, String> {
    let source_guard = state.source.read();
    let source = source_guard
        .as_ref()
        .ok_or_else(|| "No strategy source loaded".to_string())?;

    let (config, strategy, tree, decision_map) = match source {
        StrategySource::BlueprintV2 {
            config,
            strategy,
            tree,
            decision_map,
        } => (config, strategy, tree, decision_map),
        _ => return Err("get_preflop_ranges requires a BlueprintV2 source".to_string()),
    };

    // 169-element weight arrays for both players, initialized to 1.0.
    let mut oop_weights = [1.0f32; 169];
    let mut ip_weights = [1.0f32; 169];

    // Walk the tree node by node, applying strategy weights.
    let mut node_idx = tree.root;

    for action_str in &history {
        let node = &tree.nodes[node_idx as usize];
        let (player, actions, children) = match node {
            V2GameNode::Decision {
                player,
                actions,
                children,
                ..
            } => (*player, actions, children),
            V2GameNode::Chance { child, .. } => {
                node_idx = *child;
                match &tree.nodes[node_idx as usize] {
                    V2GameNode::Decision {
                        player,
                        actions,
                        children,
                        ..
                    } => (*player, actions, children),
                    _ => {
                        return Err(format!(
                            "Expected decision after chance at node {node_idx}"
                        ));
                    }
                }
            }
            V2GameNode::Terminal { .. } => {
                return Err(format!(
                    "Reached terminal before processing action '{action_str}'"
                ));
            }
        };

        let action_pos = find_v2_action_position(action_str, actions).ok_or_else(|| {
            format!(
                "Action '{action_str}' not available at node {node_idx}. Available: {actions:?}"
            )
        })?;

        // Get the decision index for strategy lookup.
        let decision_idx = decision_map
            .get(node_idx as usize)
            .copied()
            .unwrap_or(u32::MAX);

        if decision_idx != u32::MAX {
            let num_buckets = strategy.bucket_counts
                [strategy.node_street_indices[decision_idx as usize] as usize]
                as usize;

            // Select which player's weights to update.
            let weights = if player == 0 {
                &mut oop_weights
            } else {
                &mut ip_weights
            };

            for (hand_idx, w) in weights.iter_mut().enumerate() {
                let bucket = if num_buckets == 169 {
                    hand_idx as u16
                } else {
                    (hand_idx % num_buckets) as u16
                };

                let probs = strategy.get_action_probs(decision_idx as usize, bucket);
                let prob = probs.get(action_pos).copied().unwrap_or(0.0);
                *w *= prob;
            }
        }

        node_idx = children[action_pos];

        // Advance through chance nodes automatically.
        if let V2GameNode::Chance { child, .. } = &tree.nodes[node_idx as usize] {
            node_idx = *child;
        }
    }

    // Compute pot and stacks at the current position.
    let invested = invested_at_v2_node(tree.as_ref(), node_idx);
    let stack_depth = config.game.stack_depth;
    let pot = invested[0] + invested[1];
    let remaining = [
        stack_depth - invested[0],
        stack_depth - invested[1],
    ];
    let effective_stack = remaining[0].min(remaining[1]);

    // Build flop bet size strings from the config.
    let flop_sizes = &config.action_abstraction.flop;
    // OOP uses depth 0, IP uses depth 0 for bets; raise uses depth 1 if available.
    let oop_bet_sizes = format_bet_sizes(flop_sizes.first().map_or(&[], Vec::as_slice));
    let ip_bet_sizes = format_bet_sizes(flop_sizes.first().map_or(&[], Vec::as_slice));
    let oop_raise_sizes = format_bet_sizes(flop_sizes.get(1).map_or(&[], Vec::as_slice));
    let ip_raise_sizes = format_bet_sizes(flop_sizes.get(1).map_or(&[], Vec::as_slice));

    // Expand 169 → 1326.
    let oop_1326 = expand_169_to_1326(&oop_weights);
    let ip_1326 = expand_169_to_1326(&ip_weights);

    Ok(PreflopRanges {
        oop_weights: oop_1326,
        ip_weights: ip_1326,
        pot,
        effective_stack,
        oop_bet_sizes,
        oop_raise_sizes,
        ip_bet_sizes,
        ip_raise_sizes,
    })
}

/// Get preflop ranges at a given history position (Tauri wrapper).
#[tauri::command]
pub fn get_preflop_ranges(
    state: State<'_, ExplorationState>,
    history: Vec<String>,
) -> Result<PreflopRanges, String> {
    get_preflop_ranges_core(&state, history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn exploration_position_default_has_two_stacks() {
        let pos = ExplorationPosition::default();
        assert_eq!(pos.stacks.len(), 2);
        assert_eq!(pos.num_players, 2);
        assert_eq!(pos.stacks[0], 199);
        assert_eq!(pos.stacks[1], 198);
    }

    #[timed_test]
    fn exploration_position_default_active_players() {
        let pos = ExplorationPosition::default();
        assert_eq!(pos.active_players, vec![true, true]);
        assert_eq!(pos.to_act, 0);
        assert_eq!(pos.pot, 3);
    }

    #[timed_test]
    fn compute_position_state_default() {
        let pos = ExplorationPosition::default();
        let state = compute_position_state(&[], &pos);
        // Preflop: SB owes 1 to match BB
        assert_eq!(state.to_call, 1);
        assert_eq!(state.pot, 3);
        assert_eq!(state.stack_p1, 199);
        assert_eq!(state.stack_p2, 198);
    }

    #[timed_test]
    fn subgame_progress_event_serializes() {
        let event = SubgameProgressEvent {
            iteration: 10,
            max_iterations: 100,
            board_key: "AsKhQd".to_string(),
            elapsed_ms: 42,
        };
        let json = serde_json::to_string(&event).expect("should serialize");
        assert!(json.contains("\"iteration\":10"));
        assert!(json.contains("\"max_iterations\":100"));
    }

    #[timed_test]
    fn walk_preflop_tree_empty_history() {
        let config = poker_solver_core::preflop::PreflopConfig::heads_up(100);
        let tree = poker_solver_core::preflop::PreflopTree::build(&config);
        let walk = walk_preflop_tree_with_state(&config, &tree, &[])
            .expect("should walk empty history");
        assert_eq!(walk.node_idx, 0);
        // Initial pot = SB(1) + BB(2) = 3
        assert_eq!(walk.pot, 3);
        assert_eq!(walk.to_call, 1);
    }

    #[timed_test]
    fn walk_preflop_tree_fold() {
        let config = poker_solver_core::preflop::PreflopConfig::heads_up(100);
        let tree = poker_solver_core::preflop::PreflopTree::build(&config);
        let history = vec!["fold".to_string()];
        let result = walk_preflop_tree_with_state(&config, &tree, &history);
        // Fold leads to a terminal node, but the walk itself should succeed
        assert!(result.is_ok());
        let walk = result.unwrap();
        assert_eq!(walk.pot, 3); // pot unchanged by fold
    }

    #[timed_test]
    fn walk_preflop_tree_raise_tracks_state() {
        let config = poker_solver_core::preflop::PreflopConfig::heads_up(50);
        let tree = poker_solver_core::preflop::PreflopTree::build(&config);
        // SB raises — Raise(2.5) is at index 3: [Fold, Call, AllIn, Raise(2.5)]
        let history = vec!["r:3".to_string()];
        let walk = walk_preflop_tree_with_state(&config, &tree, &history)
            .expect("should walk raise history");
        // After SB raise: pot should be > 3, SB stack should decrease
        assert!(walk.pot > 3, "pot should increase after raise");
        assert!(walk.stacks[0] < 99, "SB stack should decrease (was 99 after blind)");
    }

    #[timed_test]
    fn canonical_hand_index_from_ranks_aces() {
        let idx = canonical_hand_index_from_ranks('A', 'A', false);
        // AA should be index 0 in the canonical ordering
        assert!(idx < 169);
    }

    // -------------------------------------------------------------------
    // 169 → 1326 expansion tests
    // -------------------------------------------------------------------

    #[timed_test]
    fn canonical_to_combo_map_total_combos() {
        let map = build_canonical_to_combo_map();
        assert_eq!(map.len(), 169);
        let total: usize = map.iter().map(Vec::len).sum();
        assert_eq!(total, 1326, "Total combos across all canonical hands");
    }

    #[timed_test]
    fn canonical_to_combo_map_no_duplicates() {
        let map = build_canonical_to_combo_map();
        let mut seen = std::collections::HashSet::new();
        for combos in &map {
            for &idx in combos {
                assert!(
                    seen.insert(idx),
                    "Duplicate combo index {idx} in canonical-to-combo map"
                );
            }
        }
        assert_eq!(seen.len(), 1326);
    }

    #[timed_test]
    fn canonical_to_combo_map_hand_sizes() {
        let map = build_canonical_to_combo_map();
        for i in 0..169 {
            let hand = CanonicalHand::from_index(i).unwrap();
            let expected = hand.num_combos() as usize;
            assert_eq!(
                map[i].len(),
                expected,
                "Hand index {i} ({hand}) expected {expected} combos, got {}",
                map[i].len()
            );
        }
    }

    #[timed_test]
    fn expand_169_to_1326_uniform_weights() {
        let weights_169 = [1.0f32; 169];
        let weights_1326 = expand_169_to_1326(&weights_169);
        assert_eq!(weights_1326.len(), 1326);
        for (i, &w) in weights_1326.iter().enumerate() {
            assert!(
                (w - 1.0).abs() < f32::EPSILON,
                "Combo {i} should have weight 1.0, got {w}"
            );
        }
    }

    #[timed_test]
    fn expand_169_to_1326_zero_one_hand() {
        let mut weights_169 = [1.0f32; 169];
        // Set AA (index 0) to zero.
        weights_169[0] = 0.0;
        let weights_1326 = expand_169_to_1326(&weights_169);

        // AA has 6 combos — all should be zero.
        let aa_combos = &build_canonical_to_combo_map()[0];
        assert_eq!(aa_combos.len(), 6);
        for &idx in aa_combos {
            assert!(
                weights_1326[idx].abs() < f32::EPSILON,
                "AA combo {idx} should be zero"
            );
        }

        // All other combos should be 1.0.
        let non_zero_count = weights_1326.iter().filter(|&&w| w > 0.5).count();
        assert_eq!(non_zero_count, 1326 - 6);
    }

    #[timed_test]
    fn rs_card_to_range_solver_ace_of_spades() {
        let card = Card::new(Value::Ace, Suit::Spade);
        let rs_card = rs_card_to_range_solver(card);
        // Ace = rank 12, Spade = suit 3 in range-solver
        assert_eq!(rs_card, 4 * 12 + 3, "As should be card 51");
    }

    #[timed_test]
    fn rs_card_to_range_solver_deuce_of_clubs() {
        let card = Card::new(Value::Two, Suit::Club);
        let rs_card = rs_card_to_range_solver(card);
        // Two = rank 0, Club = suit 0 in range-solver
        assert_eq!(rs_card, 0, "2c should be card 0");
    }

    #[timed_test]
    fn format_bet_sizes_basic() {
        let sizes = vec![0.33, 0.67, 1.0];
        let result = format_bet_sizes(&sizes);
        assert_eq!(result, "33%,67%,100%,e");
    }

    #[timed_test]
    fn format_bet_sizes_empty() {
        let result = format_bet_sizes(&[]);
        assert_eq!(result, "e");
    }
}
