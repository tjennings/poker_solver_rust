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

use poker_solver_core::abstraction::isomorphism::{CanonicalBoard, SuitMapping};
use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::agent::{AgentConfig, FrequencyMap};
use poker_solver_core::blueprint_v2::bundle::{self as v2_bundle, BlueprintV2Strategy};
use poker_solver_core::blueprint_v2::bucket_file::BucketFile;
use poker_solver_core::blueprint_v2::cbv::CbvTable;
use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::blueprint_v2::game_tree::{GameNode as V2GameNode, GameTree as V2GameTree, TreeAction};
use poker_solver_core::hand_class::classify;
use poker_solver_core::hands::CanonicalHand;
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

/// Summary of a blueprint bundle found by directory scan.
#[derive(Debug, Clone, Serialize)]
pub struct BlueprintListEntry {
    pub name: String,
    pub path: String,
    pub stack_depth: f64,
    pub has_strategy: bool,
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
    /// Suit mapping established by flop canonicalization, applied to turn/river cards
    suit_mapping: RwLock<Option<SuitMapping>>,
}

/// A loaded strategy source — either a trained bundle, rule-based agent,
/// preflop solve, or subgame solve backed by a blueprint.
enum StrategySource {
    Agent(AgentConfig),
    BlueprintV2 {
        config: Box<BlueprintV2Config>,
        strategy: Box<BlueprintV2Strategy>,
        tree: Box<V2GameTree>,
        /// Arena-index to decision-node-index mapping.
        decision_map: Vec<u32>,
        /// Precomputed CBV table (player 0) for depth-limited subgame solving.
        /// `None` if the bundle doesn't include CBV files.
        cbv_table: Option<CbvTable>,
        /// Directory the bundle was loaded from (for resolving relative paths).
        bundle_dir: PathBuf,
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
    pub rake_rate: f64,
    pub rake_cap: f64,
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
    /// Reaching probability for the acting player (0.0–1.0).
    pub weight: f32,
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
    /// Per-hand reaching probability for player 1 (169 canonical hands)
    pub reaching_p1: Vec<f32>,
    /// Per-hand reaching probability for player 2 (169 canonical hands)
    pub reaching_p2: Vec<f32>,
    /// Which player is currently acting (0 = P1/SB, 1 = P2/BB)
    pub to_act: u8,
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

/// Load a strategy source from a directory or agent `.toml` file
/// (core logic, no Tauri dependency).
///
/// Supports agent configs (`.toml`) and BlueprintV2 bundles (`config.yaml`).
pub async fn load_bundle_core(
    state: &ExplorationState,
    path: String,
) -> Result<BundleInfo, String> {
    let bundle_path = PathBuf::from(&path);

    if path.ends_with(".toml") {
        let (info, source) = load_agent(&bundle_path)?;
        *state.source.write() = Some(source);
        state.bucket_cache.write().clear();
        *state.suit_mapping.write() = None;
        return Ok(info);
    }
    if bundle_path.join("config.yaml").exists() {
        // Blueprint V2 bundle — delegate to the dedicated loader.
        return load_blueprint_v2_core(state, path).await;
    }

    Err("Directory does not contain config.yaml (BlueprintV2 bundle) or .toml (agent)".to_string())
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
        rake_rate: 0.0,
        rake_cap: 0.0,
    };

    Ok((info, StrategySource::Agent(agent)))
}

/// Load a Blueprint V2 bundle from a directory (core logic, no Tauri dependency).
///
/// Expects:
///   `dir_path/config.yaml` — `BlueprintV2Config`
///   `dir_path/strategy.bin` or `dir_path/final/strategy.bin` — `BlueprintV2Strategy`
///
/// If `postflop_state` is provided and the bundle contains CBV tables and
/// bucket files, automatically populates the CBV context for depth-limited
/// solving.
pub async fn load_blueprint_v2_core(
    state: &ExplorationState,
    dir_path: String,
) -> Result<BundleInfo, String> {
    let dir = PathBuf::from(&dir_path);
    let (config, strategy, cbv_table) = tokio::task::spawn_blocking(move || {
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

        // Load CBV table (player 0) if present. Search the same directory as
        // strategy.bin, then the bundle root, then the latest snapshot.
        let strat_dir = strat_path.parent().unwrap_or(&dir);
        let cbv_dir = if strat_dir.join("cbv_p0.bin").exists() {
            strat_dir.to_path_buf()
        } else if dir.join("cbv_p0.bin").exists() {
            dir.clone()
        } else {
            dir.clone() // fallback; will check existence below
        };

        let cbv_table = if cbv_dir.join("cbv_p0.bin").exists() {
            let p0 = CbvTable::load(&cbv_dir.join("cbv_p0.bin"))
                .map_err(|e| format!("Failed to load cbv_p0.bin: {e}"))?;
            Some(p0)
        } else {
            None
        };

        Ok::<_, String>((cfg, strat, cbv_table))
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
        name: Some(config.game.name.clone()),
        stack_depth: config.game.stack_depth as u32,
        bet_sizes: vec![],
        info_sets: decision_nodes,
        iterations: strategy.iterations,
        preflop_only: true,
        rake_rate: config.game.rake_rate,
        rake_cap: config.game.rake_cap,
    };

    *state.source.write() = Some(StrategySource::BlueprintV2 {
        config: Box::new(config),
        strategy: Box::new(strategy),
        tree: Box::new(tree),
        decision_map,
        cbv_table,
        bundle_dir: {
            // The bundle root is where config.yaml lives. dir_path might
            // point to a snapshot subdir, so walk up to find config.yaml.
            let mut d = PathBuf::from(&dir_path);
            while !d.join("config.yaml").exists() {
                if !d.pop() { break; }
            }
            d
        },
    });
    state.bucket_cache.write().clear();
    *state.suit_mapping.write() = None;

    Ok(info)
}

/// Load a Blueprint V2 bundle (Tauri wrapper).
///
/// After loading the bundle, automatically populates the CBV context on
/// the postflop solver state if the bundle contains CBV tables.
#[tauri::command]
pub async fn load_blueprint_v2(
    state: State<'_, ExplorationState>,
    postflop_state: State<'_, Arc<crate::postflop::PostflopState>>,
    path: String,
) -> Result<BundleInfo, String> {
    let info = load_blueprint_v2_core(&state, path).await?;
    populate_cbv_context(&state, &postflop_state);
    Ok(info)
}

/// Populate the CBV context on `PostflopState` from the currently loaded
/// blueprint bundle, if it contains CBV tables and bucket files.
///
/// This is called after `load_blueprint_v2_core()` completes. If the bundle
/// has no CBV tables, the CBV context is cleared.
pub fn populate_cbv_context(
    exploration: &ExplorationState,
    postflop: &crate::postflop::PostflopState,
) {
    let source_guard = exploration.source.read();
    let Some(source) = source_guard.as_ref() else {
        crate::postflop::set_cbv_context(postflop, None);
        return;
    };

    let StrategySource::BlueprintV2 {
        config,
        strategy,
        tree,
        cbv_table: Some(cbv_table),
        bundle_dir,
        ..
    } = source
    else {
        crate::postflop::set_cbv_context(postflop, None);
        return;
    };

    // Expect bucket files at <bundle_dir>/buckets/ or ../buckets/.
    // For distribution, symlink or copy the cluster output into the bundle root.
    let cluster_dir = if bundle_dir.join("buckets").exists() {
        bundle_dir.join("buckets")
    } else if bundle_dir.parent().is_some_and(|p| p.join("buckets").exists()) {
        bundle_dir.parent().unwrap().join("buckets")
    } else {
        eprintln!(
            "CBV table found but no buckets/ directory in bundle (checked {} and parent). \
             Symlink your cluster output there.",
            bundle_dir.join("buckets").display()
        );
        crate::postflop::set_cbv_context(postflop, None);
        return;
    };
    eprintln!("[cbv] buckets dir: {}", cluster_dir.display());
    let bucket_counts = [
        config.clustering.preflop.buckets,
        config.clustering.flop.buckets,
        config.clustering.turn.buckets,
        config.clustering.river.buckets,
    ];

    let bucket_files: [Option<BucketFile>; 4] = {
        const NAMES: [&str; 4] = [
            "preflop.buckets",
            "flop.buckets",
            "turn.buckets",
            "river.buckets",
        ];
        let mut files: [Option<BucketFile>; 4] = [None, None, None, None];
        for (i, name) in NAMES.iter().enumerate() {
            let path = cluster_dir.join(name);
            if path.exists() {
                match BucketFile::load(&path) {
                    Ok(bf) => files[i] = Some(bf),
                    Err(e) => eprintln!("Warning: failed to load {}: {e}", path.display()),
                }
            }
        }
        files
    };

    let loaded_count = bucket_files.iter().filter(|f| f.is_some()).count();
    eprintln!("[cbv] loaded {loaded_count}/4 bucket files");
    let all_buckets = AllBuckets::new(bucket_counts, bucket_files);

    // Enable per-flop bucket files if present.
    let all_buckets = {
        let per_flop_marker = cluster_dir.join("flop_0000.buckets");
        if per_flop_marker.exists() {
            all_buckets.with_per_flop_dir(cluster_dir.to_path_buf())
        } else {
            all_buckets
        }
    };

    let ctx = crate::postflop::CbvContext {
        cbv_table: cbv_table.clone(),
        abstract_tree: (**tree).clone(),
        all_buckets: Arc::new(all_buckets),
        strategy: Arc::new((**strategy).clone()),
    };

    eprintln!("CBV context populated for depth-limited solving");
    crate::postflop::set_cbv_context(postflop, Some(ctx));
}

/// Scan a directory for blueprint bundles (subdirectories containing `config.yaml`).
///
/// Returns a sorted list of discovered blueprints with metadata.
pub fn list_blueprints_core(dir: String) -> Result<Vec<BlueprintListEntry>, String> {
    let base = PathBuf::from(&dir);
    let entries = std::fs::read_dir(&base)
        .map_err(|e| format!("Failed to read directory {dir}: {e}"))?;

    let mut blueprints = Vec::new();

    // Check if the directory itself is a blueprint.
    if let Some(entry) = try_make_blueprint_entry(&base) {
        blueprints.push(entry);
    }

    // Check subdirectories.
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
        let sub = entry.path();
        if !sub.is_dir() {
            continue;
        }
        if let Some(entry) = try_make_blueprint_entry(&sub) {
            blueprints.push(entry);
        }
    }

    blueprints.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(blueprints)
}

/// Try to interpret `dir` as a blueprint directory.
/// Returns `Some(BlueprintListEntry)` if `config.yaml` exists.
/// If config parsing fails, the entry is still created with defaults so
/// the user can see it in the picker (with stack_depth = 0).
fn try_make_blueprint_entry(dir: &Path) -> Option<BlueprintListEntry> {
    if !dir.join("config.yaml").exists() {
        return None;
    }

    let (name, stack_depth) = match v2_bundle::load_config(dir) {
        Ok(config) => (config.game.name.clone(), config.game.stack_depth),
        Err(e) => {
            eprintln!("Warning: failed to parse config in {}: {e}", dir.display());
            let fallback = dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
            (fallback, 0.0)
        }
    };

    let has_strategy = dir.join("final/strategy.bin").exists()
        || dir.join("strategy.bin").exists()
        || std::fs::read_dir(dir)
            .ok()
            .map(|rd| {
                rd.filter_map(Result::ok).any(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.starts_with("snapshot_"))
                })
            })
            .unwrap_or(false);

    Some(BlueprintListEntry {
        name,
        path: dir.to_string_lossy().to_string(),
        stack_depth,
        has_strategy,
    })
}

/// List available blueprint bundles in a directory (Tauri wrapper).
#[tauri::command]
pub async fn list_blueprints(dir: String) -> Result<Vec<BlueprintListEntry>, String> {
    list_blueprints_core(dir)
}

/// Convert blueprint action abstraction sizes to range-solver format strings.
///
/// Returns `(bet_sizes_str, raise_sizes_str)` for one street.
/// Depth 0 maps to bet sizes, depth 1 maps to raise sizes. If only one depth
/// is provided, the same sizes are used for both. An all-in option is always
/// appended.
pub fn blueprint_sizes_to_range_solver(depths: &[Vec<f64>]) -> (String, String) {
    let format_depth = |sizes: &[f64]| -> String {
        sizes
            .iter()
            .map(|&f| {
                let pct = (f * 100.0).round() as u32;
                format!("{pct}%")
            })
            .collect::<Vec<_>>()
            .join(",")
    };

    let bet_str = depths.first().map(|d| format_depth(d)).unwrap_or_default();
    let raise_str = depths
        .get(1)
        .map(|d| format_depth(d))
        .unwrap_or_else(|| bet_str.clone());

    let add_allin =
        |s: String| if s.is_empty() { "a".to_string() } else { format!("{s},a") };

    (add_allin(bet_str), add_allin(raise_str))
}

/// Get the strategy matrix for a given position (core logic, no Tauri dependency).
///
/// `threshold` filters out hands whose prior action probabilities fell below
/// this value (range narrowing).  `street_histories` supplies the action
/// sequences of all completed streets so the filter can replay the game.
/// `postflop_state` provides CbvContext for real postflop bucket lookups.
pub fn get_strategy_matrix_core(
    state: &ExplorationState,
    position: ExplorationPosition,
    _threshold: Option<f32>,
    _street_histories: Option<Vec<Vec<String>>>,
    postflop_state: Option<&crate::postflop::PostflopState>,
) -> Result<StrategyMatrix, String> {
    let source_guard = state.source.read();
    let source = source_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    match source {
        StrategySource::Agent(agent) => get_strategy_matrix_agent(agent, &position),
        StrategySource::BlueprintV2 {
            config,
            strategy,
            tree,
            decision_map,
            ..
        } => {
            let cbv_guard = postflop_state.map(|ps| ps.cbv_context.read());
            let cbv_ref = cbv_guard
                .as_ref()
                .and_then(|g| g.as_deref());
            get_strategy_matrix_v2(config, strategy, tree, decision_map, &position, cbv_ref)
        }
    }
}

/// Get the strategy matrix for a given position (Tauri wrapper).
#[tauri::command(rename_all = "snake_case")]
pub fn get_strategy_matrix(
    state: State<'_, ExplorationState>,
    postflop_state: State<'_, Arc<crate::postflop::PostflopState>>,
    position: ExplorationPosition,
    threshold: Option<f32>,
    street_histories: Option<Vec<Vec<String>>>,
) -> Result<StrategyMatrix, String> {
    get_strategy_matrix_core(&state, position, threshold, street_histories, Some(&postflop_state))
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
                weight: 1.0,
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
        reaching_p1: vec![],
        reaching_p2: vec![],
        to_act: position.to_act,
    })
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
        V2GameNode::Terminal { pot, .. } => (0, *pot, [0.0; 2], 0.0),
        V2GameNode::Chance { child, .. } => v2_node_state(tree, *child),
    }
}

/// Find the position of a history action string within V2 tree actions.
fn find_v2_action_position(action_str: &str, actions: &[TreeAction]) -> Option<usize> {
    // Action IDs are array indices (e.g. "0", "1", "2").
    if let Ok(idx) = action_str.parse::<usize>() {
        if idx < actions.len() {
            return Some(idx);
        }
    }
    None
}

/// Build `ActionInfo` items for actions at a V2 tree node.
fn v2_actions_at_node(tree: &V2GameTree, node_idx: u32) -> Vec<ActionInfo> {
    match &tree.nodes[node_idx as usize] {
        V2GameNode::Decision { actions, .. } => actions
            .iter()
            .enumerate()
            .map(|(i, a)| v2_action_info(a, i, 1.0, 0.0))
            .collect(),
        _ => vec![],
    }
}

/// Convert a V2 `TreeAction` to an `ActionInfo`.
///
/// `bb_scale` converts internal tree units to BB for display labels.
/// Use `1.0` when the tree uses BB units (blueprint preflop tree).
/// Use `0.5` when the tree uses chip units (subgame solver, 1BB = 2 chips).
///
/// `invested_offset` is subtracted from Bet/Raise amounts before display.
/// For postflop subgame trees this is the per-player investment entering
/// the street (pot / 2), so labels show the bet/raise amount on THIS
/// street rather than total invested across all streets.
/// Use `0.0` for preflop where invested starts at the blind.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn v2_action_info(
    action: &TreeAction,
    idx: usize,
    bb_scale: f64,
    invested_offset: f64,
) -> ActionInfo {
    let id = idx.to_string();
    match action {
        TreeAction::Fold => ActionInfo {
            id,
            label: "Fold".to_string(),
            action_type: "fold".to_string(),
            size_key: None,
        },
        TreeAction::Check => ActionInfo {
            id,
            label: "Check".to_string(),
            action_type: "check".to_string(),
            size_key: None,
        },
        TreeAction::Call => ActionInfo {
            id,
            label: "Call".to_string(),
            action_type: "call".to_string(),
            size_key: None,
        },
        TreeAction::Bet(amount) => {
            let display = (amount - invested_offset) * bb_scale;
            ActionInfo {
                id,
                label: format!("{display:.1}bb"),
                action_type: "bet".to_string(),
                size_key: Some(format!("{amount:.2}")),
            }
        }
        TreeAction::Raise(amount) => {
            let display = (amount - invested_offset) * bb_scale;
            ActionInfo {
                id,
                label: format!("{display:.1}bb"),
                action_type: "raise".to_string(),
                size_key: Some(format!("{amount:.2}")),
            }
        }
        TreeAction::AllIn => ActionInfo {
            id,
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
    cbv_context: Option<&crate::postflop::CbvContext>,
) -> Result<StrategyMatrix, String> {
    // Compute reaching probabilities by replaying the history step-by-step.
    let mut reaching_p1 = vec![1.0_f32; 169];
    let mut reaching_p2 = vec![1.0_f32; 169];
    let num_buckets_preflop = strategy.bucket_counts[0] as usize;
    let board_cards = parse_board(&position.board).ok();

    {
        let mut node_idx = tree.root;
        for action_str in &position.history {
            // Skip past chance nodes.
            if let V2GameNode::Chance { child, .. } = &tree.nodes[node_idx as usize] {
                node_idx = *child;
            }

            if let V2GameNode::Decision {
                player, actions, children, street,
                ..
            } = &tree.nodes[node_idx as usize]
            {
                let dec_idx = decision_map
                    .get(node_idx as usize)
                    .copied()
                    .unwrap_or(u32::MAX);

                if dec_idx != u32::MAX {
                    let action_pos = find_v2_action_position(action_str, actions);
                    if let Some(action_pos) = action_pos {
                        let reach = if *player == 0 {
                            &mut reaching_p1
                        } else {
                            &mut reaching_p2
                        };

                        let is_preflop_node = *street == Street::Preflop;

                        if is_preflop_node {
                            // Preflop: canonical hand index maps to bucket.
                            for (hand_idx, r) in reach.iter_mut().enumerate() {
                                let bucket = if num_buckets_preflop == 169 {
                                    hand_idx as u16
                                } else {
                                    (hand_idx % num_buckets_preflop) as u16
                                };
                                let probs =
                                    strategy.get_action_probs(dec_idx as usize, bucket);
                                let p = probs.get(action_pos).copied().unwrap_or(0.0);
                                *r *= p;
                            }
                        } else if let (Some(ctx), Some(board)) = (cbv_context, board_cards.as_ref()) {
                            // Postflop with bucket data: for each canonical hand,
                            // enumerate combos, look up each combo's bucket,
                            // get the action probability, and average.
                            // Slice the board to the correct length for this
                            // node's street (e.g. 3 cards for flop even if the
                            // full board has 4+ cards at a later street).
                            let board_slice = board_for_street_slice(board, *street);
                            let lookup = BucketLookup {
                                all_buckets: &ctx.all_buckets,
                                strategy: &ctx.strategy,
                                decision_idx: dec_idx as usize,
                            };
                            for (hand_idx, r) in reach.iter_mut().enumerate() {
                                if let Some(hand) = CanonicalHand::from_index(hand_idx) {
                                    let avg_p = avg_action_prob_for_hand(
                                        &hand, board_slice, *street,
                                        &lookup, action_pos,
                                    );
                                    *r *= avg_p;
                                }
                            }
                        } else if !is_preflop_node {
                            panic!(
                                "Postflop reaching probability computation requires CbvContext. \
                                 cbv_context: {}, board_cards: {}",
                                cbv_context.is_some(),
                                board_cards.is_some(),
                            );
                        }

                        node_idx = children[action_pos];
                    }
                }
            }

            // Skip past chance nodes after taking the action.
            if let V2GameNode::Chance { child, .. } = &tree.nodes[node_idx as usize] {
                node_idx = *child;
            }
        }
    }

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

    // Determine the street from the node (single match).
    let street_enum = match &tree.nodes[walk.node_idx as usize] {
        V2GameNode::Decision { street, .. } => *street,
        _ => Street::Preflop,
    };
    let street_name = match street_enum {
        Street::Preflop => "Preflop",
        Street::Flop => "Flop",
        Street::Turn => "Turn",
        Street::River => "River",
    };
    let is_preflop = street_enum == Street::Preflop;

    // Check that the board has enough cards for the tree node's street.
    // walk_v2_tree auto-skips chance nodes, so the tree can advance to
    // Turn while the board only has 3 cards. Return the street name so
    // the frontend knows to prompt for the next card.
    let required_cards = match street_enum {
        Street::Preflop => 0,
        Street::Flop => 3,
        Street::Turn => 4,
        Street::River => 5,
    };
    if position.board.len() < required_cards {
        return Err(format!(
            "street_transition:{street_name}:Board has {} cards but {street_name} requires {required_cards}",
            position.board.len(),
        ));
    }
    let num_buckets = strategy.bucket_counts[
        strategy.node_street_indices[decision_idx as usize] as usize
    ] as usize;

    // Reuse board_cards parsed above for postflop bucket lookups.
    let board = if !is_preflop { board_cards.as_ref() } else { None };

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
            } else if let (Some(ctx), Some(board)) = (cbv_context, board) {
                // Postflop with bucket data: enumerate combos, look up buckets,
                // average strategy probabilities.
                let lookup = BucketLookup {
                    all_buckets: &ctx.all_buckets,
                    strategy: &ctx.strategy,
                    decision_idx: decision_idx as usize,
                };
                postflop_cell_probs(
                    rank1, rank2, suited, board, street_enum,
                    &lookup, &actions,
                )
            } else {
                panic!(
                    "Postflop strategy matrix requested but CbvContext is missing. \
                     Board: {:?}, cbv_context: {}, board_parsed: {}",
                    position.board,
                    cbv_context.is_some(),
                    board.is_some(),
                );
            };

            // Compute reaching weight for the acting player.
            let hand_idx = canonical_hand_index_from_ranks(rank1, rank2, suited);
            let reach = if walk.to_act == 0 {
                reaching_p1[hand_idx]
            } else {
                reaching_p2[hand_idx]
            };

            row_cells.push(MatrixCell {
                hand: hand_label,
                suited,
                pair,
                probabilities,
                filtered: false,
                weight: reach,
            });
        }
        cells.push(row_cells);
    }

    // Compute pot/stacks from the tree.
    let node_pot = pot_at_v2_node(tree, walk.node_idx);
    let stack_depth = _config.game.stack_depth;
    let pot = (node_pot * 2.0) as u32; // convert BB to half-BB units
    // Approximate remaining stacks: each player started with stack_depth,
    // and roughly half the pot has come from each player.
    let each_invested = node_pot / 2.0;
    let stack_p1 = ((stack_depth - each_invested) * 2.0) as u32;
    let stack_p2 = ((stack_depth - each_invested) * 2.0) as u32;
    let to_call = 0u32; // approximation; exact to_call requires action tracking

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
        reaching_p1,
        reaching_p2,
        to_act: walk.to_act,
    })
}

// ---------------------------------------------------------------------------
// Blueprint range propagation
// ---------------------------------------------------------------------------

/// Propagate 1326-combo range weights through the blueprint's action history.
///
/// Walks the abstract game tree following `action_history`, and at each
/// decision node, multiplies the acting player's combo weights by the
/// blueprint strategy probability for each combo's bucket. This is the
/// blueprint equivalent of `postflop_close_street_core` — it updates
/// `PostflopState.filtered_{oop,ip}_weights` so the subgame solver can
/// use blueprint ranges as its starting point.
///
/// Also sets `PostflopState.config.abstract_node_idx` to the tree node
/// after walking the history, so the subgame solver starts from the
/// correct position in the abstract tree.
pub fn blueprint_propagate_ranges(
    exploration: &ExplorationState,
    postflop: &crate::postflop::PostflopState,
    board: &[String],
    action_history: &[String],
) -> Result<(), String> {
    use poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id;
    use poker_solver_core::hands::all_hands;
    use range_solver::card::card_pair_to_index;
    use range_solver::range::Range;

    let source_guard = exploration.source.read();
    let source = source_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    let StrategySource::BlueprintV2 {
        strategy, tree, decision_map, ..
    } = source
    else {
        return Err("Blueprint range propagation requires a BlueprintV2 source".to_string());
    };

    let cbv_guard = postflop.cbv_context.read();
    let ctx = cbv_guard
        .as_ref()
        .ok_or("CbvContext is required for blueprint range propagation")?;

    let board_cards = parse_board(board)?;

    // Start from current filtered weights (set by preflop explorer).
    let config = postflop.config.read().clone();
    let mut oop_weights: Vec<f32> = postflop
        .filtered_oop_weights
        .read()
        .clone()
        .unwrap_or_else(|| {
            let range: Range = config.oop_range.parse().unwrap_or_default();
            range.raw_data().to_vec()
        });
    let mut ip_weights: Vec<f32> = postflop
        .filtered_ip_weights
        .read()
        .clone()
        .unwrap_or_else(|| {
            let range: Range = config.ip_range.parse().unwrap_or_default();
            range.raw_data().to_vec()
        });

    // Walk the abstract tree through the action history.
    let mut node_idx = tree.root;
    for action_str in action_history {
        // Skip chance nodes.
        if let V2GameNode::Chance { child, .. } = &tree.nodes[node_idx as usize] {
            node_idx = *child;
        }

        let (player, street, actions, children) = match &tree.nodes[node_idx as usize] {
            V2GameNode::Decision { player, street, actions, children, .. } => {
                (*player, *street, actions, children)
            }
            _ => break,
        };

        let dec_idx = decision_map
            .get(node_idx as usize)
            .copied()
            .unwrap_or(u32::MAX);
        if dec_idx == u32::MAX {
            break;
        }

        let action_pos = find_v2_action_position(action_str, actions)
            .ok_or_else(|| format!("Action '{action_str}' not found at node {node_idx}"))?;

        // Slice board to the correct length for this node's street.
        let board_slice = board_for_street_slice(&board_cards, street);

        let weights = if player == 0 {
            &mut oop_weights
        } else {
            &mut ip_weights
        };

        let lookup = BucketLookup {
            all_buckets: &ctx.all_buckets,
            strategy: &ctx.strategy,
            decision_idx: dec_idx as usize,
        };

        // For each combo, look up its bucket and multiply by action probability.
        for hand in all_hands() {
            for (c0, c1) in hand.combos() {
                if board_slice.iter().any(|b| *b == c0 || *b == c1) {
                    continue; // blocked by board
                }
                let id0 = rs_poker_card_to_id(c0);
                let id1 = rs_poker_card_to_id(c1);
                let ci = card_pair_to_index(id0, id1);

                if street == Street::Preflop {
                    let bucket = if strategy.bucket_counts[0] == 169 {
                        hand.index() as u16
                    } else {
                        (hand.index() % strategy.bucket_counts[0] as usize) as u16
                    };
                    let probs = strategy.get_action_probs(dec_idx as usize, bucket);
                    let p = probs.get(action_pos).copied().unwrap_or(0.0);
                    weights[ci] *= p;
                } else {
                    let bucket = ctx.all_buckets.get_bucket(street, [c0, c1], board_slice);
                    let probs = ctx.strategy.get_action_probs(dec_idx as usize, bucket);
                    let p = probs.get(action_pos).copied().unwrap_or(0.0);
                    weights[ci] *= p;
                }
            }
        }

        node_idx = children[action_pos];

        // Skip chance nodes after action.
        if let V2GameNode::Chance { child, .. } = &tree.nodes[node_idx as usize] {
            node_idx = *child;
        }
    }

    // Store propagated weights and the abstract tree position.
    *postflop.filtered_oop_weights.write() = Some(oop_weights);
    *postflop.filtered_ip_weights.write() = Some(ip_weights);

    // Update abstract_node_idx so the subgame solver starts from the right place.
    postflop.config.write().abstract_node_idx = Some(node_idx);

    eprintln!(
        "[blueprint_propagate_ranges] propagated through {} actions, abstract_node_idx={}",
        action_history.len(),
        node_idx
    );

    Ok(())
}

/// Tauri command wrapper for blueprint range propagation.
#[tauri::command(rename_all = "snake_case")]
pub fn blueprint_propagate_ranges_cmd(
    state: State<'_, ExplorationState>,
    postflop_state: State<'_, Arc<crate::postflop::PostflopState>>,
    board: Vec<String>,
    action_history: Vec<String>,
) -> Result<(), String> {
    blueprint_propagate_ranges(&state, &postflop_state, &board, &action_history)
}

/// Get the canonical hand index (0..169) from rank characters.
fn canonical_hand_index_from_ranks(rank1: char, rank2: char, suited: bool) -> usize {
    let v1 = char_to_value(rank1);
    let v2 = char_to_value(rank2);
    let canonical = CanonicalHand::new(v1, v2, suited);
    canonical.index()
}

/// Groups the bucket-lookup parameters shared by postflop helpers.
struct BucketLookup<'a> {
    all_buckets: &'a AllBuckets,
    strategy: &'a BlueprintV2Strategy,
    decision_idx: usize,
}

/// Enumerate all combos for a canonical hand, filter board blockers, look up
/// each combo's bucket, and accumulate action probabilities.
///
/// Returns `(sum_of_probs_per_action, combo_count)`.
fn bucket_probs_for_hand(
    hand: &CanonicalHand,
    board: &[Card],
    street: Street,
    lookup: &BucketLookup,
) -> (Vec<f32>, u32) {
    let combos = hand.combos();
    let num_actions = lookup
        .strategy
        .get_action_probs(lookup.decision_idx, 0)
        .len();
    let mut sum_probs = vec![0.0_f32; num_actions];
    let mut count = 0u32;

    for (c1, c2) in &combos {
        // Linear scan over <=5 board elements is faster than HashSet.
        if board.iter().any(|b| *b == *c1 || *b == *c2) {
            continue;
        }

        let bucket = lookup.all_buckets.get_bucket(street, [*c1, *c2], board);
        let probs = lookup
            .strategy
            .get_action_probs(lookup.decision_idx, bucket);

        for (i, &p) in probs.iter().enumerate().take(num_actions) {
            sum_probs[i] += p;
        }
        count += 1;
    }

    (sum_probs, count)
}

/// Compute postflop action probabilities for a single 13x13 cell by
/// enumerating all specific Card combos, filtering board blockers,
/// looking up each combo's bucket, querying the strategy, and averaging.
fn postflop_cell_probs(
    rank1: char,
    rank2: char,
    suited: bool,
    board: &[Card],
    street: Street,
    lookup: &BucketLookup,
    actions: &[ActionInfo],
) -> Vec<ActionProb> {
    let v1 = char_to_value(rank1);
    let v2 = char_to_value(rank2);
    let canonical = CanonicalHand::new(v1, v2, suited);

    let (sum_probs, count) = bucket_probs_for_hand(&canonical, board, street, lookup);

    if count == 0 {
        // All combos blocked by board — zero probability (hand is impossible).
        return actions
            .iter()
            .map(|a| ActionProb {
                action: a.label.clone(),
                probability: 0.0,
            })
            .collect();
    }

    actions
        .iter()
        .enumerate()
        .map(|(i, a)| ActionProb {
            action: a.label.clone(),
            probability: sum_probs.get(i).copied().unwrap_or(0.0) / count as f32,
        })
        .collect()
}

/// Compute the average probability of a specific action for a canonical hand
/// at a postflop decision node. Uses `bucket_probs_for_hand` and indexes the
/// result.
fn avg_action_prob_for_hand(
    hand: &CanonicalHand,
    board: &[Card],
    street: Street,
    lookup: &BucketLookup,
    action_pos: usize,
) -> f32 {
    let (sum_probs, count) = bucket_probs_for_hand(hand, board, street, lookup);

    if count == 0 {
        // All combos blocked — return 0 (won't contribute to reach).
        return 0.0;
    }

    sum_probs.get(action_pos).copied().unwrap_or(0.0) / count as f32
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

    match source {
        StrategySource::BlueprintV2 { tree, .. } => {
            let walk = walk_v2_tree(tree, &position.history)?;
            Ok(v2_actions_at_node(tree, walk.node_idx))
        }
        StrategySource::Agent(agent) => {
            Ok(get_actions_for_position(agent.game.stack_depth, &agent.game.bet_sizes, &position))
        }
    }
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
        StrategySource::Agent(agent) => BundleInfo {
            name: agent.game.name.clone(),
            stack_depth: agent.game.stack_depth,
            bet_sizes: agent.game.bet_sizes.clone(),
            info_sets: 0,
            iterations: 0,
            preflop_only: false,
            rake_rate: 0.0,
            rake_cap: 0.0,
        },
        StrategySource::BlueprintV2 {
            config,
            strategy,
            tree,
            ..
        } => {
            let (decision_nodes, _, _) = tree.node_counts();
            BundleInfo {
                name: Some(config.game.name.clone()),
                stack_depth: config.game.stack_depth as u32,
                bet_sizes: vec![],
                info_sets: decision_nodes,
                iterations: strategy.iterations,
                preflop_only: false,
                rake_rate: config.game.rake_rate,
                rake_cap: config.game.rake_cap,
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
pub fn get_hand_equity_core(
    _state: &ExplorationState,
    _hand: &str,
    _villain_hand: Option<&str>,
) -> Result<Option<HandEquity>, String> {
    // Hand equity data was only available from the old PreflopSolve source.
    // Neither Agent nor BlueprintV2 sources carry per-hand EV tables.
    Ok(None)
}

/// Get postflop equity for a canonical hand (Tauri wrapper).
#[tauri::command(rename_all = "snake_case")]
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
    _state: &ExplorationState,
    board: Vec<String>,
    _on_progress: Option<ProgressCallback>,
) -> Result<String, String> {
    // Bucket computation was only needed for old EHS2 bundles.
    // Agent and BlueprintV2 sources do not require it.
    Ok(board.join(""))
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
    _state: &ExplorationState,
    _position: ExplorationPosition,
    hand: String,
) -> Result<ComboGroupInfo, String> {
    // Combo-level classification was only supported by old Bundle/SubgameSolve
    // sources with BundleConfig. Agent and BlueprintV2 do not support it.
    Ok(empty_combo_info(&hand))
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

/// Return the board slice visible at the given street, clamping to `board.len()`.
///
/// Mirrors [`AllBuckets::board_for_street`] but accepts a dynamically-sized slice.
fn board_for_street_slice(board: &[Card], street: Street) -> &[Card] {
    let n = match street {
        Street::Preflop => 0,
        Street::Flop => 3,
        Street::Turn => 4,
        Street::River => board.len(),
    };
    &board[..n.min(board.len())]
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
    /// Abstract tree node index at the postflop starting position.
    /// Used by the CBV evaluator to navigate the abstract tree.
    pub abstract_node_idx: u32,
    /// Rake rate (0.0–1.0 fraction of pot).
    pub rake_rate: f64,
    /// Rake cap in chips.
    pub rake_cap: f64,
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
/// Walk the V2 tree to find the pot at a given node.
///
/// For fold terminals, the pot is stored directly. For decision/chance nodes,
/// we look at the fold terminal child's pot (which equals the current pot before
/// any new action). Falls back to recursive search.
fn pot_at_v2_node(tree: &V2GameTree, node_idx: u32) -> f64 {
    match &tree.nodes[node_idx as usize] {
        V2GameNode::Terminal { pot, .. } => *pot,
        V2GameNode::Chance { child, .. } => pot_at_v2_node(tree, *child),
        V2GameNode::Decision {
            actions, children, ..
        } => {
            // Fold terminal's pot reflects the pot before the fold action.
            if let Some(fold_pos) = actions.iter().position(|a| matches!(a, TreeAction::Fold)) {
                let child_idx = children[fold_pos] as usize;
                if let V2GameNode::Terminal { pot, .. } = &tree.nodes[child_idx] {
                    return *pot;
                }
            }
            // Check child preserves the pot.
            if let Some(check_pos) = actions.iter().position(|a| matches!(a, TreeAction::Check)) {
                return pot_at_v2_node(tree, children[check_pos]);
            }
            pot_at_v2_node(tree, children[0])
        }
    }
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
            ..
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
    let pot = pot_at_v2_node(tree.as_ref(), node_idx);
    let stack_depth = config.game.stack_depth;
    // Approximate remaining stacks from pot.
    let each_invested = pot / 2.0;
    let remaining = [
        stack_depth - each_invested,
        stack_depth - each_invested,
    ];
    let effective_stack = remaining[0].min(remaining[1]);

    // Build flop bet size strings from the config.
    let flop_sizes = &config.action_abstraction.flop;
    // OOP uses depth 0, IP uses depth 0 for bets; raise uses depth 1 if available.
    let oop_bet_sizes = format_bet_sizes(flop_sizes.first().map_or(&[], Vec::as_slice));
    let ip_bet_sizes = format_bet_sizes(flop_sizes.first().map_or(&[], Vec::as_slice));
    let oop_raise_sizes = format_bet_sizes(flop_sizes.get(1).map_or(&[], Vec::as_slice));
    let ip_raise_sizes = format_bet_sizes(flop_sizes.get(1).map_or(&[], Vec::as_slice));

    // Zero out negligible weights (CFR never outputs exact 0, so threshold needed).
    let threshold = 0.005;
    for w in oop_weights.iter_mut() {
        if *w < threshold { *w = 0.0; }
    }
    for w in ip_weights.iter_mut() {
        if *w < threshold { *w = 0.0; }
    }

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
        rake_rate: config.game.rake_rate,
        rake_cap: config.game.rake_cap,
        abstract_node_idx: node_idx,
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
    fn canonical_hand_index_from_ranks_aces() {
        let idx = canonical_hand_index_from_ranks('A', 'A', false);
        // AA should be index 0 in the canonical ordering
        assert!(idx < 169);
    }

    #[timed_test]
    fn blueprint_sizes_two_depths() {
        let depths = vec![vec![0.33, 0.67, 1.0], vec![0.5, 1.0]];
        let (bet, raise) = blueprint_sizes_to_range_solver(&depths);
        assert_eq!(bet, "33%,67%,100%,a");
        assert_eq!(raise, "50%,100%,a");
    }

    #[timed_test]
    fn blueprint_sizes_single_depth() {
        let depths = vec![vec![0.5]];
        let (bet, raise) = blueprint_sizes_to_range_solver(&depths);
        assert_eq!(bet, "50%,a");
        assert_eq!(raise, "50%,a");
    }

    #[timed_test]
    fn blueprint_sizes_empty() {
        let (bet, raise) = blueprint_sizes_to_range_solver(&[]);
        assert_eq!(bet, "a");
        assert_eq!(raise, "a");
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

    // -------------------------------------------------------------------
    // Postflop strategy matrix tests
    // -------------------------------------------------------------------

    /// Build a minimal V2 game tree with one preflop decision node (node 0)
    /// and one flop decision node (node 2, after chance node 1).
    /// The preflop node has 2 actions (fold, call) leading to terminal and chance.
    /// The flop node has 2 actions (check, bet) leading to terminals.
    fn build_minimal_postflop_tree() -> V2GameTree {
        let nodes = vec![
            // Node 0: preflop decision (player 0, SB)
            V2GameNode::Decision {
                player: 0,
                street: Street::Preflop,
                actions: vec![TreeAction::Fold, TreeAction::Call],
                children: vec![3, 1], // fold -> terminal, call -> chance
                blueprint_decision_idx: None,
            },
            // Node 1: chance node (flop deal)
            V2GameNode::Chance {
                next_street: Street::Flop,
                child: 2,
            },
            // Node 2: flop decision (player 1, BB)
            V2GameNode::Decision {
                player: 1,
                street: Street::Flop,
                actions: vec![TreeAction::Check, TreeAction::Bet(2.0)],
                children: vec![4, 5],
                blueprint_decision_idx: None,
            },
            // Node 3: terminal (fold)
            V2GameNode::Terminal {
                kind: poker_solver_core::blueprint_v2::game_tree::TerminalKind::Fold { winner: 1 },
                pot: 1.5,
            },
            // Node 4: terminal (check through)
            V2GameNode::Terminal {
                kind: poker_solver_core::blueprint_v2::game_tree::TerminalKind::Showdown,
                pot: 2.0,
            },
            // Node 5: terminal (bet, then fold assumed)
            V2GameNode::Terminal {
                kind: poker_solver_core::blueprint_v2::game_tree::TerminalKind::Fold { winner: 1 },
                pot: 4.0,
            },
        ];
        V2GameTree {
            nodes,
            root: 0,
            dealer: 0,
        }
    }

    /// Build a matching strategy for the minimal postflop tree.
    /// 2 decision nodes: node 0 (preflop, 2 actions) and node 2 (flop, 2 actions).
    /// decision_map: node 0 -> decision 0, node 2 -> decision 1.
    fn build_minimal_postflop_strategy() -> (BlueprintV2Strategy, Vec<u32>) {
        // bucket_counts: preflop=169, flop=10, turn=10, river=10
        let bucket_counts: [u16; 4] = [169, 10, 10, 10];

        // Decision 0: preflop, 2 actions, 169 buckets -> 169*2 = 338 probs
        // Decision 1: flop, 2 actions, 10 buckets -> 10*2 = 20 probs
        let mut action_probs = Vec::new();
        // Preflop: fold=0.3, call=0.7 for all buckets
        for _ in 0..169 {
            action_probs.push(0.3);
            action_probs.push(0.7);
        }
        // Flop: check=0.6, bet=0.4 for all buckets
        for _ in 0..10 {
            action_probs.push(0.6);
            action_probs.push(0.4);
        }

        let node_action_counts = vec![2u16, 2u16];
        let node_street_indices = vec![0u8, 1u8]; // preflop=0, flop=1

        // Build via serde_json -> save -> load roundtrip to populate private node_offsets
        let json = serde_json::json!({
            "action_probs": action_probs,
            "node_action_counts": node_action_counts,
            "node_street_indices": node_street_indices,
            "bucket_counts": bucket_counts,
            "iterations": 100u64,
            "elapsed_minutes": 1u64,
        });
        let strat_raw: BlueprintV2Strategy = serde_json::from_value(json).expect("from_value");
        let tmp = std::env::temp_dir().join(format!(
            "test_strategy_postflop_{:?}.bin",
            std::thread::current().id()
        ));
        strat_raw.save(&tmp).expect("save");
        let strat = BlueprintV2Strategy::load(&tmp).expect("load");
        let _ = std::fs::remove_file(&tmp);

        // decision_map: 6 nodes, node 0 -> decision 0, node 2 -> decision 1, rest -> MAX
        let decision_map = vec![0, u32::MAX, 1, u32::MAX, u32::MAX, u32::MAX];

        (strat, decision_map)
    }

    fn build_minimal_config() -> BlueprintV2Config {
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
                algorithm: ClusteringAlgorithm::default(),
                preflop: StreetClusterConfig { buckets: 169, delta_bins: None, expected_delta: false, sample_boards: None },
                flop: StreetClusterConfig { buckets: 10, delta_bins: None, expected_delta: false, sample_boards: None },
                turn: StreetClusterConfig { buckets: 10, delta_bins: None, expected_delta: false, sample_boards: None },
                river: StreetClusterConfig { buckets: 10, delta_bins: None, expected_delta: false, sample_boards: None },
                seed: 42,
                kmeans_iterations: 10,
                cfvnet_river_data: None,
                per_flop: None,
            },
            action_abstraction: ActionAbstractionConfig {
                preflop: vec![vec!["2.5bb".to_string()]],
                flop: vec![vec![0.5]],
                turn: vec![vec![0.5]],
                river: vec![vec![0.5]],
            },
            training: TrainingConfig {
                cluster_path: None,
                iterations: Some(100),
                time_limit_minutes: None,
                lcfr_warmup_iterations: 0,
                lcfr_discount_interval: 1,
                prune_after_iterations: 0,
                prune_threshold: -250,
                prune_explore_pct: 0.05,
                print_every_minutes: 10,
                batch_size: 200,
                dcfr_alpha: 1.5,
                dcfr_beta: 0.5,
                dcfr_gamma: 2.0,
                dcfr_epoch_cap: None,
                target_strategy_delta: None,
                purify_threshold: 0.0,
                equity_cache_path: None,
            },
            snapshots: SnapshotConfig {
                warmup_minutes: 1,
                snapshot_every_minutes: 10,
                output_dir: "/tmp".to_string(),
                resume: false,
                max_snapshots: None,
            },
        }
    }

    /// Build a CbvContext with the same strategy and tree, and AllBuckets
    /// using equity-based fallback (no bucket files).
    fn build_test_cbv_context(
        strategy: &BlueprintV2Strategy,
        tree: &V2GameTree,
    ) -> crate::postflop::CbvContext {
        let bucket_counts = strategy.bucket_counts;
        let all_buckets = AllBuckets::new(bucket_counts, [None, None, None, None]);

        // Build via serde_json -> save -> load roundtrip
        let json = serde_json::json!({
            "action_probs": strategy.action_probs,
            "node_action_counts": strategy.node_action_counts,
            "node_street_indices": strategy.node_street_indices,
            "bucket_counts": strategy.bucket_counts,
            "iterations": strategy.iterations,
            "elapsed_minutes": strategy.elapsed_minutes,
        });
        let strat_raw: BlueprintV2Strategy = serde_json::from_value(json).expect("from_value");
        let tmp = std::env::temp_dir().join(format!(
            "test_cbv_strategy_{:?}.bin",
            std::thread::current().id()
        ));
        strat_raw.save(&tmp).expect("save");
        let cbv_strategy = BlueprintV2Strategy::load(&tmp).expect("load");
        let _ = std::fs::remove_file(&tmp);

        crate::postflop::CbvContext {
            cbv_table: CbvTable { values: vec![], node_offsets: vec![], buckets_per_node: vec![] },
            abstract_tree: tree.clone(),
            all_buckets: Arc::new(all_buckets),
            strategy: Arc::new(cbv_strategy),
        }
    }

    #[timed_test]
    #[should_panic(expected = "CbvContext is missing")]
    fn v2_postflop_without_cbv_context_panics() {
        let tree = build_minimal_postflop_tree();
        let (strategy, decision_map) = build_minimal_postflop_strategy();
        let config = build_minimal_config();

        let position = ExplorationPosition {
            board: vec!["As".into(), "Kh".into(), "7d".into()],
            history: vec!["1".to_string()],
            pot: 4,
            stacks: vec![198, 198],
            to_act: 1,
            num_players: 2,
            active_players: vec![true, true],
        };

        // Must panic — no silent fallback to uniform
        let _ = get_strategy_matrix_v2(
            &config,
            &strategy,
            &tree,
            &decision_map,
            &position,
            None,
        );
    }

    #[timed_test]
    fn get_strategy_matrix_core_accepts_postflop_state() {
        let state = ExplorationState::default();
        let postflop_state = crate::postflop::PostflopState::default();

        let position = ExplorationPosition::default();
        // No bundle loaded, should return an error
        let result = get_strategy_matrix_core(
            &state,
            position,
            None,
            None,
            Some(&postflop_state),
        );
        assert!(result.is_err(), "Should fail because no bundle is loaded");
    }

    #[timed_test]
    fn postflop_cell_probs_all_blocked_returns_zero() {
        let (strategy, _) = build_minimal_postflop_strategy();
        let all_buckets = AllBuckets::new([169, 10, 10, 10], [None, None, None, None]);
        let actions = vec![
            ActionInfo { id: "0".into(), label: "Check".into(), action_type: "check".into(), size_key: None },
            ActionInfo { id: "1".into(), label: "Bet".into(), action_type: "bet".into(), size_key: None },
        ];

        // Board contains all 4 aces — any AA combo is fully blocked
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Ace, Suit::Diamond),
            Card::new(Value::Ace, Suit::Club),
            Card::new(Value::King, Suit::Spade),
        ];

        let lookup = BucketLookup {
            all_buckets: &all_buckets,
            strategy: &strategy,
            decision_idx: 1, // flop node
        };
        let probs = postflop_cell_probs(
            'A', 'A', false, &board, Street::River,
            &lookup, &actions,
        );

        // All AA combos blocked by board — zero probability (hand impossible)
        assert_eq!(probs.len(), 2);
        assert!((probs[0].probability).abs() < 0.01, "expected 0.0, got {}", probs[0].probability);
        assert!((probs[1].probability).abs() < 0.01, "expected 0.0, got {}", probs[1].probability);
    }

    #[timed_test]
    #[ignore] // Requires bucket files loaded; run with --ignored when real data available
    fn v2_postflop_with_cbv_context_returns_strategy_probs() {
        let tree = build_minimal_postflop_tree();
        let (strategy, decision_map) = build_minimal_postflop_strategy();
        let config = build_minimal_config();
        let ctx = build_test_cbv_context(&strategy, &tree);

        // Position at the flop decision node: history = ["1"] (call = action index 1)
        let position = ExplorationPosition {
            board: vec!["As".into(), "Kh".into(), "7d".into()],
            history: vec!["1".to_string()],
            pot: 4,
            stacks: vec![198, 198],
            to_act: 1,
            num_players: 2,
            active_players: vec![true, true],
        };

        let result = get_strategy_matrix_v2(
            &config,
            &strategy,
            &tree,
            &decision_map,
            &position,
            Some(&ctx),
        );

        let matrix = result.expect("should succeed");
        assert_eq!(matrix.street, "Flop");

        // With CbvContext, postflop should return strategy probs (0.6/0.4),
        // not uniform (0.5/0.5). Since the strategy has check=0.6, bet=0.4
        // for all buckets, every cell should show approximately those values.
        let mut found_non_uniform = false;
        for row in &matrix.cells {
            for cell in row {
                assert_eq!(cell.probabilities.len(), 2, "should have 2 actions");
                // Check that at least some cells differ from 0.5
                let p0 = cell.probabilities[0].probability;
                if (p0 - 0.5).abs() > 0.01 {
                    found_non_uniform = true;
                }
            }
        }
        assert!(
            found_non_uniform,
            "Expected at least some cells to have non-uniform probabilities \
             when CbvContext is provided"
        );
    }

    #[timed_test]
    fn v2_preflop_still_works_with_new_signature() {
        let tree = build_minimal_postflop_tree();
        let (strategy, decision_map) = build_minimal_postflop_strategy();
        let config = build_minimal_config();

        // Position at preflop decision node: empty history
        let position = ExplorationPosition {
            board: vec![],
            history: vec![],
            pot: 3,
            stacks: vec![199, 198],
            to_act: 0,
            num_players: 2,
            active_players: vec![true, true],
        };

        let result = get_strategy_matrix_v2(
            &config,
            &strategy,
            &tree,
            &decision_map,
            &position,
            None,
        );

        let matrix = result.expect("should succeed");
        assert_eq!(matrix.street, "Preflop");

        // Preflop should return the strategy probs (0.3/0.7), not uniform
        for row in &matrix.cells {
            for cell in row {
                assert_eq!(cell.probabilities.len(), 2);
                let p0 = cell.probabilities[0].probability;
                let p1 = cell.probabilities[1].probability;
                assert!(
                    (p0 - 0.3).abs() < 0.01,
                    "Expected fold=0.3, got {} for hand {}",
                    p0, cell.hand
                );
                assert!(
                    (p1 - 0.7).abs() < 0.01,
                    "Expected call=0.7, got {} for hand {}",
                    p1, cell.hand
                );
            }
        }
    }

    #[timed_test]
    fn avg_action_prob_all_blocked_returns_zero() {
        let (strategy, _) = build_minimal_postflop_strategy();
        let all_buckets = AllBuckets::new([169, 10, 10, 10], [None, None, None, None]);

        // Board contains all 4 aces — AA is fully blocked
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Ace, Suit::Diamond),
            Card::new(Value::Ace, Suit::Club),
            Card::new(Value::King, Suit::Spade),
        ];

        let hand = CanonicalHand::new(Value::Ace, Value::Ace, false);
        let lookup = BucketLookup {
            all_buckets: &all_buckets,
            strategy: &strategy,
            decision_idx: 1,
        };
        let p = avg_action_prob_for_hand(
            &hand, &board, Street::River,
            &lookup, 0,
        );

        assert!(p.abs() < f32::EPSILON, "Expected 0.0 for fully blocked hand, got {p}");
    }

    #[timed_test]
    fn board_for_street_slice_returns_correct_length() {
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Queen, Suit::Diamond),
            Card::new(Value::Jack, Suit::Club),
            Card::new(Value::Ten, Suit::Spade),
        ];

        assert_eq!(board_for_street_slice(&board, Street::Preflop).len(), 0);
        assert_eq!(board_for_street_slice(&board, Street::Flop).len(), 3);
        assert_eq!(board_for_street_slice(&board, Street::Turn).len(), 4);
        assert_eq!(board_for_street_slice(&board, Street::River).len(), 5);
    }

    #[timed_test]
    fn board_for_street_slice_short_board_clamps() {
        // A 3-card board (flop) should clamp for turn/river queries
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Queen, Suit::Diamond),
        ];

        assert_eq!(board_for_street_slice(&board, Street::Flop).len(), 3);
        assert_eq!(board_for_street_slice(&board, Street::Turn).len(), 3); // clamped
        assert_eq!(board_for_street_slice(&board, Street::River).len(), 3); // clamped
    }

    #[timed_test]
    fn board_for_street_slice_empty_board() {
        let board: Vec<Card> = vec![];
        assert_eq!(board_for_street_slice(&board, Street::Preflop).len(), 0);
        assert_eq!(board_for_street_slice(&board, Street::Flop).len(), 0);
    }

    #[timed_test]
    fn postflop_cell_probs_no_hashset_still_filters_blockers() {
        // Verify that blocker filtering works correctly (same behavior
        // after replacing HashSet with linear scan).
        let (strategy, _) = build_minimal_postflop_strategy();
        let all_buckets = AllBuckets::new([169, 10, 10, 10], [None, None, None, None]);
        let actions = vec![
            ActionInfo { id: "0".into(), label: "Check".into(), action_type: "check".into(), size_key: None },
            ActionInfo { id: "1".into(), label: "Bet".into(), action_type: "bet".into(), size_key: None },
        ];

        // Board with 3 cards (flop) -- KsKhKd blocks all KK combos
        let board = vec![
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::King, Suit::Diamond),
        ];

        let lookup = BucketLookup {
            all_buckets: &all_buckets,
            strategy: &strategy,
            decision_idx: 1,
        };
        let probs = postflop_cell_probs(
            'K', 'K', false, &board, Street::Flop,
            &lookup, &actions,
        );

        // Only 1 KK combo remains (Kc isn't on the board) -- but KK needs 2 kings,
        // and we only have 1 unblocked king. So all KK combos are blocked.
        // Actually: KK combos are (Ks,Kh), (Ks,Kd), (Ks,Kc), (Kh,Kd), (Kh,Kc), (Kd,Kc)
        // Board has Ks, Kh, Kd -- blocks any combo containing those. Only (Kc, X) where
        // X is not on board... but all other Ks are on board. So only if both cards are Kc
        // which is impossible. All 6 combos are blocked.
        assert_eq!(probs.len(), 2);
        assert!((probs[0].probability).abs() < 0.01, "expected 0.0 for all-blocked KK");
    }

    #[timed_test]
    fn avg_action_prob_no_hashset_still_filters_blockers() {
        // Verify blocker filtering works after HashSet removal
        let (strategy, _) = build_minimal_postflop_strategy();
        let all_buckets = AllBuckets::new([169, 10, 10, 10], [None, None, None, None]);

        // Board blocks all AA combos
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Ace, Suit::Diamond),
        ];

        let hand = CanonicalHand::new(Value::Ace, Value::Ace, false);
        let lookup = BucketLookup {
            all_buckets: &all_buckets,
            strategy: &strategy,
            decision_idx: 1,
        };
        let p = avg_action_prob_for_hand(
            &hand, &board, Street::Flop,
            &lookup, 0,
        );

        // All AA combos blocked (3 aces on board block all 6 AA combos
        // since each combo needs 2 aces from 4, but 3 are on board)
        // Wait: (Ac, X) where X not on board -- only Ac is off board.
        // So only combo not fully blocked would need 2 off-board aces.
        // Only 1 ace (Ac) is off board. All combos blocked.
        assert!(p.abs() < f32::EPSILON, "Expected 0.0 for blocked hand");
    }

    #[timed_test]
    fn bucket_probs_for_hand_returns_sum_and_count() {
        let (strategy, _) = build_minimal_postflop_strategy();
        let all_buckets = AllBuckets::new([169, 10, 10, 10], [None, None, None, None]);

        // Use preflop (no bucket files needed) with an unblocked hand.
        let hand = CanonicalHand::new(Value::King, Value::Queen, true);
        let board: Vec<Card> = vec![];

        let lookup = BucketLookup {
            all_buckets: &all_buckets,
            strategy: &strategy,
            decision_idx: 0, // preflop node
        };

        let (sum_probs, count) = bucket_probs_for_hand(
            &hand, &board, Street::Preflop, &lookup,
        );

        // KQs has 4 combos, none blocked by empty board
        assert_eq!(count, 4, "Expected 4 unblocked KQs combos");
        // sum_probs should have entries for each action
        assert_eq!(sum_probs.len(), 2, "Expected 2 actions in sum_probs");
        // Average should equal sum / count and be in [0,1]
        let avg = sum_probs[0] / count as f32;
        assert!(avg >= 0.0 && avg <= 1.0, "Average prob should be in [0,1], got {avg}");
    }

    #[timed_test]
    fn bucket_probs_for_hand_all_blocked_returns_zero_count() {
        let (strategy, _) = build_minimal_postflop_strategy();
        let all_buckets = AllBuckets::new([169, 10, 10, 10], [None, None, None, None]);

        let hand = CanonicalHand::new(Value::Ace, Value::Ace, false);
        // Board blocks all AA combos
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Ace, Suit::Diamond),
            Card::new(Value::Ace, Suit::Club),
            Card::new(Value::King, Suit::Spade),
        ];

        let lookup = BucketLookup {
            all_buckets: &all_buckets,
            strategy: &strategy,
            decision_idx: 1,
        };

        let (sum_probs, count) = bucket_probs_for_hand(
            &hand, &board, Street::River, &lookup,
        );

        assert_eq!(count, 0, "All combos should be blocked");
        // sum_probs should be all zeros
        for p in &sum_probs {
            assert!(p.abs() < f32::EPSILON, "Expected 0.0 in sum_probs when all blocked");
        }
    }

    #[timed_test]
    #[ignore] // Requires real bucket files for postflop cell computation
    fn v2_postflop_reaching_probs_preflop_only_still_computed() {
        let tree = build_minimal_postflop_tree();
        let (strategy, decision_map) = build_minimal_postflop_strategy();
        let config = build_minimal_config();
        let ctx = build_test_cbv_context(&strategy, &tree);

        // Navigate to flop (through preflop call). The preflop reaching
        // probs should reflect the call action (0.7 for all hands).
        let position = ExplorationPosition {
            board: vec!["Ts".into(), "9h".into(), "2d".into()],
            history: vec!["1".to_string()], // call
            pot: 4,
            stacks: vec![198, 198],
            to_act: 1,
            num_players: 2,
            active_players: vec![true, true],
        };

        let result = get_strategy_matrix_v2(
            &config,
            &strategy,
            &tree,
            &decision_map,
            &position,
            Some(&ctx),
        );

        let matrix = result.expect("should succeed");
        // Player 0 (SB) called preflop with prob 0.7 for all hands.
        for &r in &matrix.reaching_p1 {
            assert!(
                (r - 0.7).abs() < 0.01,
                "Expected reaching prob 0.7 after call, got {r}"
            );
        }
        // Player 1 (BB) didn't act yet, reaching should be 1.0.
        for &r in &matrix.reaching_p2 {
            assert!(
                (r - 1.0).abs() < 0.01,
                "Expected reaching prob 1.0, got {r}"
            );
        }
    }
}
