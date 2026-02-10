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
use poker_solver_core::agent::{AgentConfig, FrequencyMap};
use poker_solver_core::blueprint::{BundleConfig, StrategyBundle};
use poker_solver_core::hand_class::{classify, HandClassification};
use poker_solver_core::hands::CanonicalHand;
use poker_solver_core::info_key::{canonical_hand_index, cards_from_rank_chars, InfoKey};
use poker_solver_core::poker::{Card, Suit, Value};

/// Event payload for bucket computation progress.
#[derive(Debug, Clone, Serialize)]
pub struct BucketProgressEvent {
    pub completed: usize,
    pub total: usize,
    pub board_key: String,
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
}

/// A loaded strategy source — either a trained bundle or a rule-based agent.
enum StrategySource {
    Bundle {
        config: BundleConfig,
        blueprint: poker_solver_core::blueprint::BlueprintStrategy,
    },
    Agent(AgentConfig),
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
    /// Player 1 (SB) stack
    pub stack_p1: u32,
    /// Player 2 (BB) stack
    pub stack_p2: u32,
    /// Whose turn (0 = P1/SB, 1 = P2/BB)
    pub to_act: u8,
}

impl Default for ExplorationPosition {
    fn default() -> Self {
        Self {
            board: vec![],
            history: vec![],
            pot: 3,        // SB + BB posted (internal units)
            stack_p1: 199, // 100BB default: 100*2 - 1 (SB posted 1)
            stack_p2: 198, // 100BB default: 100*2 - 2 (BB posted 2)
            to_act: 0,    // SB acts first preflop
        }
    }
}

/// Load a strategy bundle from a directory, or an agent from a `.toml` file.
///
/// Bundle loading (which can deserialize ~1 GB of blueprint data) runs on a
/// blocking thread so the Tauri main thread stays responsive.
#[tauri::command]
pub async fn load_bundle(
    state: State<'_, ExplorationState>,
    path: String,
) -> Result<BundleInfo, String> {
    let bundle_path = PathBuf::from(&path);

    let (info, source, boundaries) = if path.ends_with(".toml") {
        let (info, source) = load_agent(&bundle_path)?;
        (info, source, None)
    } else {
        // Heavy I/O on a blocking thread to avoid freezing the UI
        let bundle = tauri::async_runtime::spawn_blocking(move || {
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
        };
        let boundaries = bundle.boundaries;
        let source = StrategySource::Bundle {
            config: bundle.config,
            blueprint: bundle.blueprint,
        };
        (info, source, boundaries)
    };

    *state.abstraction_boundaries.write() = boundaries;
    *state.source.write() = Some(source);
    state.bucket_cache.write().clear();

    Ok(info)
}

fn load_agent(path: &Path) -> Result<(BundleInfo, StrategySource), String> {
    let agent = AgentConfig::load(path).map_err(|e| format!("Failed to load agent: {e}"))?;

    let info = BundleInfo {
        name: agent.game.name.clone(),
        stack_depth: agent.game.stack_depth,
        bet_sizes: agent.game.bet_sizes.clone(),
        info_sets: 0,
        iterations: 0,
    };

    Ok((info, StrategySource::Agent(agent)))
}

/// Get the strategy matrix for a given position.
///
/// `threshold` filters out hands whose prior action probabilities fell below
/// this value (range narrowing).  `street_histories` supplies the action
/// sequences of all completed streets so the filter can replay the game.
#[tauri::command]
pub fn get_strategy_matrix(
    state: State<'_, ExplorationState>,
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
            get_strategy_matrix_bundle(config, blueprint, &state, &position, thresh, &sh)
        }
        StrategySource::Agent(agent) => get_strategy_matrix_agent(agent, &position),
    }
}

fn get_strategy_matrix_bundle(
    config: &BundleConfig,
    blueprint: &poker_solver_core::blueprint::BlueprintStrategy,
    state: &State<'_, ExplorationState>,
    position: &ExplorationPosition,
    threshold: f32,
    street_histories: &[Vec<String>],
) -> Result<StrategyMatrix, String> {
    let board = parse_board(&position.board)?;
    let street = street_from_board_len(board.len())?;
    let action_codes = build_action_codes(&position.history);
    let use_hand_class = config.abstraction_mode == "hand_class";

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
            } else if use_hand_class {
                get_hand_strategy_hand_class(
                    blueprint,
                    &board,
                    street,
                    &action_codes,
                    pos_state.pot,
                    pos_state.eff_stack,
                    rank1,
                    rank2,
                    suited,
                    &actions,
                )?
            } else {
                get_hand_strategy(
                    blueprint,
                    &board,
                    street,
                    &action_codes,
                    pos_state.pot,
                    pos_state.eff_stack,
                    rank1,
                    rank2,
                    suited,
                    &actions,
                    bucket_cache.as_ref(),
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
    })
}

/// Get available actions at the current position.
#[tauri::command]
pub fn get_available_actions(
    state: State<'_, ExplorationState>,
    position: ExplorationPosition,
) -> Result<Vec<ActionInfo>, String> {
    let source_guard = state.source.read();
    let source = source_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    let (stack_depth, bet_sizes) = match source {
        StrategySource::Bundle { config, .. } => {
            (config.game.stack_depth, config.game.bet_sizes.as_slice())
        }
        StrategySource::Agent(agent) => (agent.game.stack_depth, agent.game.bet_sizes.as_slice()),
    };

    Ok(get_actions_for_position(stack_depth, bet_sizes, &position))
}

/// Check if a bundle or agent is loaded.
#[tauri::command]
pub fn is_bundle_loaded(state: State<'_, ExplorationState>) -> bool {
    state.source.read().is_some()
}

/// Get info about currently loaded bundle or agent.
#[tauri::command]
pub fn get_bundle_info(state: State<'_, ExplorationState>) -> Result<BundleInfo, String> {
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
        },
        StrategySource::Agent(agent) => BundleInfo {
            name: agent.game.name.clone(),
            stack_depth: agent.game.stack_depth,
            bet_sizes: agent.game.bet_sizes.clone(),
            info_sets: 0,
            iterations: 0,
        },
    })
}

/// Computation progress status.
#[derive(Debug, Clone, Serialize)]
pub struct ComputationStatus {
    pub computing: bool,
    pub progress: usize,
    pub total: usize,
    pub board_key: Option<String>,
}

/// Get the current bucket computation status.
#[tauri::command]
pub fn get_computation_status(state: State<'_, ExplorationState>) -> ComputationStatus {
    ComputationStatus {
        computing: state.computing.load(Ordering::SeqCst),
        progress: state.computation_progress.load(Ordering::SeqCst),
        total: state.computation_total.load(Ordering::SeqCst),
        board_key: state.computing_board_key.read().clone(),
    }
}

/// Start async bucket computation for a board.
/// Returns immediately for agents (no computation needed).
/// For bundles, computation happens in background.
#[tauri::command]
pub fn start_bucket_computation(
    app: AppHandle,
    state: State<'_, ExplorationState>,
    board: Vec<String>,
) -> Result<String, String> {
    let board_key = board.join("");

    // Agents and hand_class bundles don't need bucket computation
    {
        let source_guard = state.source.read();
        match source_guard.as_ref() {
            Some(StrategySource::Agent(_)) => return Ok(board_key),
            Some(StrategySource::Bundle { config, .. })
                if config.abstraction_mode == "hand_class" =>
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
                        let _ = app.emit(
                            "bucket-progress",
                            BucketProgressEvent {
                                completed,
                                total: 169,
                                board_key: board_key_clone.clone(),
                            },
                        );
                    }
                }
            }
        }

        // Store in global cache
        bucket_cache
            .write()
            .insert(board_key_clone.clone(), local_cache);

        // Computation complete - emit final event
        let _ = app.emit(
            "bucket-progress",
            BucketProgressEvent {
                completed: 169,
                total: 169,
                board_key: board_key_clone,
            },
        );

        computing.store(false, Ordering::SeqCst);
        *computing_board_key.write() = None;
    });

    Ok(board_key)
}

/// Check if bucket computation for a board is complete.
#[tauri::command]
pub fn is_board_cached(state: State<'_, ExplorationState>, board: Vec<String>) -> bool {
    let board_key = board.join("");
    let cache = state.bucket_cache.read();
    cache.contains_key(&board_key)
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
    /// Human-readable class names (e.g. `["TopPair", "FlushDraw"]`).
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
    /// Pot bucket used for key construction.
    pub pot_bucket: u32,
    /// Stack bucket used for key construction.
    pub stack_bucket: u32,
}

/// Get combo-level classification breakdown for a selected cell.
///
/// Enumerates all specific combos of a canonical hand (e.g. "AKs"),
/// classifies each on the current board, groups by classification,
/// and looks up the blueprint strategy for each group.
///
/// Only meaningful for hand_class bundles on postflop streets.
#[tauri::command]
pub fn get_combo_classes(
    state: State<'_, ExplorationState>,
    position: ExplorationPosition,
    hand: String,
) -> Result<ComboGroupInfo, String> {
    let source_guard = state.source.read();
    let source = source_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    let (config, blueprint) = match source {
        StrategySource::Bundle { config, blueprint } => (config, blueprint),
        StrategySource::Agent(_) => {
            return Ok(empty_combo_info(&hand));
        }
    };

    if config.abstraction_mode != "hand_class" {
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
    let pot_bucket = pos_state.pot / 20;
    let stack_bucket = pos_state.eff_stack / 20;

    // Group combos by classification bits
    let mut groups_map: std::collections::BTreeMap<u32, Vec<String>> =
        std::collections::BTreeMap::new();
    for combo in &combos {
        let bits = classify(*combo, &board)
            .map(|c| c.bits())
            .unwrap_or(0);
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
                pot_bucket,
                stack_bucket,
                &action_codes,
            )
            .as_u64();
            let strategy = blueprint
                .lookup(key)
                .map(|s| s.to_vec())
                .unwrap_or_default();
            ComboGroup {
                bits,
                class_names: HandClassification::from_bits(bits).to_strings(),
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
        pot_bucket,
        stack_bucket,
    })
}

fn empty_combo_info(hand: &str) -> ComboGroupInfo {
    ComboGroupInfo {
        hand: hand.to_string(),
        groups: vec![],
        total_combos: 0,
        blocked_combos: 0,
        street: String::new(),
        pot_bucket: 0,
        stack_bucket: 0,
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
    let hand_bits = hand_bits_at_street(rank1, rank2, suited, board, street_idx);
    let street_num = street_idx.min(3) as u8;
    let eff_stack = stacks[0].min(stacks[1]);
    let key = InfoKey::new(hand_bits, street_num, pot / 20, eff_stack / 20, action_codes).as_u64();

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
/// postflop uses hand-class classification).
fn hand_bits_at_street(
    rank1: char,
    rank2: char,
    suited: bool,
    board: &[Card],
    street_idx: usize,
) -> u32 {
    if street_idx == 0 {
        hand_bits_from_ranks(rank1, rank2, suited)
    } else {
        let (c1, c2) = make_representative_hand(rank1, rank2, suited, board);
        classify([c1, c2], board).map(|c| c.bits()).unwrap_or(0)
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
        4..=7 => Some(1 + (code - 4) as usize),                // bet:idx (to_call == 0)
        8..=11 => Some(2 + (code - 8) as usize),               // raise:idx (to_call > 0)
        12 => Some(1 + num_bet_sizes),                          // bet all-in
        13 => Some(2 + num_bet_sizes),                          // raise all-in
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
/// 1=fold, 2=check, 3=call, 4-7=bet idx 0-3, 8-11=raise idx 0-3,
/// 12=bet all-in, 13=raise all-in.
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
            12 // bet all-in
        } else {
            let idx: u8 = idx_str.parse().unwrap_or(0);
            4 + idx.min(3) // bet(idx)
        }
    } else if let Some(idx_str) = action
        .strip_prefix("raise:")
        .or_else(|| action.strip_prefix("r:"))
    {
        if idx_str == "A" {
            13 // raise all-in
        } else {
            let idx: u8 = idx_str.parse().unwrap_or(0);
            8 + idx.min(3) // raise(idx)
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
    let stack = if position.to_act == 0 {
        position.stack_p1
    } else {
        position.stack_p2
    };

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
    let mut stacks = [position.stack_p1, position.stack_p2];
    let mut pot = position.pot;

    // Preflop: SB owes 1 more to match BB's blind.
    // Postflop: no outstanding bet.
    let mut to_call = if position.board.is_empty() {
        position.stack_p1.saturating_sub(position.stack_p2)
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

#[allow(clippy::too_many_arguments)]
fn get_hand_strategy(
    blueprint: &poker_solver_core::blueprint::BlueprintStrategy,
    _board: &[Card],
    street: Street,
    action_codes: &[u8],
    pot: u32,
    eff_stack: u32,
    rank1: char,
    rank2: char,
    suited: bool,
    actions: &[ActionInfo],
    bucket_cache: Option<&std::collections::HashMap<(char, char, bool), u16>>,
) -> Result<Vec<ActionProb>, String> {
    let hand_bits: u32 = if street == Street::Preflop {
        hand_bits_from_ranks(rank1, rank2, suited)
    } else {
        let cache = bucket_cache.ok_or_else(|| {
            format!(
                "Bucket cache not available for postflop street {street:?}. \
                 Start bucket computation before requesting postflop strategy."
            )
        })?;
        let &bucket = cache.get(&(rank1, rank2, suited)).ok_or_else(|| {
            format!("No bucket in cache for hand ({rank1}, {rank2}, suited={suited})")
        })?;
        u32::from(bucket)
    };

    let street_num = street_to_num(street);
    let pot_bucket = pot / 20;
    let stack_bucket = eff_stack / 20;

    let info_set_key =
        InfoKey::new(hand_bits, street_num, pot_bucket, stack_bucket, action_codes).as_u64();

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

/// Lookup strategy for a hand using classify()-based bucket keys (hand_class mode).
///
/// No bucket cache needed — `classify()` is O(1) and deterministic.
#[allow(clippy::too_many_arguments)]
fn get_hand_strategy_hand_class(
    blueprint: &poker_solver_core::blueprint::BlueprintStrategy,
    board: &[Card],
    street: Street,
    action_codes: &[u8],
    pot: u32,
    eff_stack: u32,
    rank1: char,
    rank2: char,
    suited: bool,
    actions: &[ActionInfo],
) -> Result<Vec<ActionProb>, String> {
    let hand_bits: u32 = if street == Street::Preflop {
        hand_bits_from_ranks(rank1, rank2, suited)
    } else {
        let (card1, card2) = make_representative_hand(rank1, rank2, suited, board);
        match classify([card1, card2], board) {
            Ok(classification) => classification.bits(),
            Err(_) => 0,
        }
    };

    let street_num = street_to_num(street);
    let pot_bucket = pot / 20;
    let stack_bucket = eff_stack / 20;

    let info_set_key =
        InfoKey::new(hand_bits, street_num, pot_bucket, stack_bucket, action_codes).as_u64();

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
