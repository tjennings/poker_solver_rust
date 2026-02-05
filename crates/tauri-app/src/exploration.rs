//! Exploration commands for navigating trained HUNL strategies.
//!
//! Allows users to load a strategy bundle and explore the game tree,
//! viewing optimal strategies at each decision point.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use poker_solver_core::abstraction::{CardAbstraction, Street};
use poker_solver_core::blueprint::{BundleConfig, StrategyBundle};
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
    /// Currently loaded bundle
    bundle: RwLock<Option<LoadedBundle>>,
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

/// A loaded strategy bundle with derived components.
struct LoadedBundle {
    config: BundleConfig,
    blueprint: poker_solver_core::blueprint::BlueprintStrategy,
}

impl Default for ExplorationState {
    fn default() -> Self {
        Self {
            bundle: RwLock::new(None),
            bucket_cache: Arc::new(RwLock::new(HashMap::new())),
            computation_progress: Arc::new(AtomicUsize::new(0)),
            computation_total: Arc::new(AtomicUsize::new(0)),
            computing: Arc::new(AtomicBool::new(false)),
            computing_board_key: Arc::new(RwLock::new(None)),
            abstraction_boundaries: Arc::new(RwLock::new(None)),
        }
    }
}

/// Information about a loaded bundle.
#[derive(Debug, Clone, Serialize)]
pub struct BundleInfo {
    pub stack_depth: u32,
    pub bet_sizes: Vec<f32>,
    pub info_sets: usize,
    pub iterations: u64,
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
    /// Current pot size
    pub pot: u32,
    /// Current player's stack
    pub stack: u32,
    /// Amount to call (0 if check is available)
    pub to_call: u32,
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
            pot: 3, // SB + BB posted
            stack_p1: 99, // SB posted 1
            stack_p2: 98, // BB posted 2
            to_act: 0, // SB acts first preflop
        }
    }
}

/// Load a strategy bundle from a directory.
#[tauri::command]
pub fn load_bundle(
    state: State<'_, ExplorationState>,
    path: String,
) -> Result<BundleInfo, String> {
    let bundle_path = PathBuf::from(&path);

    let bundle = StrategyBundle::load(&bundle_path)
        .map_err(|e| format!("Failed to load bundle: {e}"))?;

    let info = BundleInfo {
        stack_depth: bundle.config.game.stack_depth,
        bet_sizes: bundle.config.game.bet_sizes.clone(),
        info_sets: bundle.blueprint.len(),
        iterations: bundle.blueprint.iterations_trained(),
    };

    // Store boundaries for thread-safe bucket computation
    *state.abstraction_boundaries.write() = Some(bundle.boundaries.clone());

    let loaded = LoadedBundle {
        config: bundle.config,
        blueprint: bundle.blueprint,
    };

    *state.bundle.write() = Some(loaded);

    // Clear bucket cache when loading new bundle
    state.bucket_cache.write().clear();

    Ok(info)
}

/// Get the strategy matrix for a given position.
/// Non-blocking: uses cached buckets for postflop, returns default probabilities if not cached.
#[tauri::command]
pub fn get_strategy_matrix(
    state: State<'_, ExplorationState>,
    position: ExplorationPosition,
) -> Result<StrategyMatrix, String> {
    let bundle_guard = state.bundle.read();
    let bundle = bundle_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    let board = parse_board(&position.board)?;
    let street = street_from_board_len(board.len())?;
    let history_str = build_history_string(&position.history);

    // Build the 13x13 matrix
    let ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
    let mut cells = Vec::with_capacity(13);

    // Get available actions for this position
    let actions = get_actions_for_position(bundle, &position);

    // For postflop, try to use cached buckets (non-blocking)
    let bucket_cache: Option<HashMap<(char, char, bool), u16>> =
        if street != Street::Preflop && !board.is_empty() {
            let board_key = position.board.join("");
            let cache = state.bucket_cache.read();
            cache.get(&board_key).cloned()
        } else {
            None
        };

    for (row, &rank1) in ranks.iter().enumerate() {
        let mut row_cells = Vec::with_capacity(13);
        for (col, &rank2) in ranks.iter().enumerate() {
            let (hand_label, suited, pair) = if row == col {
                // Pocket pair
                (format!("{rank1}{rank1}"), false, true)
            } else if row < col {
                // Suited (above diagonal)
                (format!("{rank1}{rank2}s"), true, false)
            } else {
                // Offsuit (below diagonal)
                (format!("{rank2}{rank1}o"), false, false)
            };

            // Get strategy for this hand
            let probabilities = get_hand_strategy(
                bundle,
                &board,
                street,
                &history_str,
                rank1,
                rank2,
                suited,
                &actions,
                bucket_cache.as_ref(),
            )?;

            row_cells.push(MatrixCell {
                hand: hand_label,
                suited,
                pair,
                probabilities,
            });
        }
        cells.push(row_cells);
    }

    Ok(StrategyMatrix {
        cells,
        actions,
        street: format!("{street:?}"),
        pot: position.pot,
        stack: if position.to_act == 0 { position.stack_p1 } else { position.stack_p2 },
        to_call: calculate_to_call(&position),
    })
}

/// Get available actions at the current position.
#[tauri::command]
pub fn get_available_actions(
    state: State<'_, ExplorationState>,
    position: ExplorationPosition,
) -> Result<Vec<ActionInfo>, String> {
    let bundle_guard = state.bundle.read();
    let bundle = bundle_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    Ok(get_actions_for_position(bundle, &position))
}

/// Check if a bundle is loaded.
#[tauri::command]
pub fn is_bundle_loaded(state: State<'_, ExplorationState>) -> bool {
    state.bundle.read().is_some()
}

/// Get info about currently loaded bundle.
#[tauri::command]
pub fn get_bundle_info(state: State<'_, ExplorationState>) -> Result<BundleInfo, String> {
    let bundle_guard = state.bundle.read();
    let bundle = bundle_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    Ok(BundleInfo {
        stack_depth: bundle.config.game.stack_depth,
        bet_sizes: bundle.config.game.bet_sizes.clone(),
        info_sets: bundle.blueprint.len(),
        iterations: bundle.blueprint.iterations_trained(),
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
/// Returns immediately, computation happens in background.
#[tauri::command]
pub fn start_bucket_computation(
    app: AppHandle,
    state: State<'_, ExplorationState>,
    board: Vec<String>,
) -> Result<String, String> {
    // Parse board
    let board_cards = parse_board(&board)?;
    let board_key = board.join("");

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
        let ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
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
                    let (card1, card2) = make_representative_hand(rank1, rank2, suited, &board_cards);
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
        bucket_cache.write().insert(board_key_clone.clone(), local_cache);

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
pub fn is_board_cached(
    state: State<'_, ExplorationState>,
    board: Vec<String>,
) -> bool {
    let board_key = board.join("");
    let cache = state.bucket_cache.read();
    cache.contains_key(&board_key)
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

fn build_history_string(history: &[String]) -> String {
    history.iter().map(|a| action_to_key_char(a)).collect()
}

fn action_to_key_char(action: &str) -> String {
    if action == "f" || action == "fold" {
        "f".to_string()
    } else if action == "x" || action == "check" {
        "x".to_string()
    } else if action == "c" || action == "call" {
        "c".to_string()
    } else if let Some(amt) = action.strip_prefix("b:") {
        format!("b{amt}")
    } else if let Some(amt) = action.strip_prefix("r:") {
        format!("r{amt}")
    } else {
        action.to_string()
    }
}

fn get_actions_for_position(bundle: &LoadedBundle, position: &ExplorationPosition) -> Vec<ActionInfo> {
    let mut actions = Vec::new();
    let to_call = calculate_to_call(position);
    let stack = if position.to_act == 0 { position.stack_p1 } else { position.stack_p2 };

    // Fold if facing a bet
    if to_call > 0 {
        actions.push(ActionInfo {
            id: "fold".to_string(),
            label: "Fold".to_string(),
            action_type: "fold".to_string(),
        });
    }

    // Check or call
    if to_call == 0 {
        actions.push(ActionInfo {
            id: "check".to_string(),
            label: "Check".to_string(),
            action_type: "check".to_string(),
        });
    } else if stack >= to_call {
        actions.push(ActionInfo {
            id: "call".to_string(),
            label: format!("Call {to_call}"),
            action_type: "call".to_string(),
        });
    }

    // Bet/raise sizes from config
    let effective_stack = stack.saturating_sub(to_call);
    if effective_stack > 0 {
        for &fraction in &bundle.config.game.bet_sizes {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let size = (f64::from(position.pot) * f64::from(fraction)).round() as u32;
            if size > 0 && size <= effective_stack {
                let action_type = if to_call == 0 { "bet" } else { "raise" };
                let total = if to_call == 0 { size } else { to_call + size };
                actions.push(ActionInfo {
                    id: format!("{action_type}:{total}"),
                    label: format!("{} {total}", if to_call == 0 { "Bet" } else { "Raise to" }),
                    action_type: action_type.to_string(),
                });
            }
        }

        // All-in
        let all_in = if to_call == 0 { effective_stack } else { stack };
        if !actions.iter().any(|a| a.id.ends_with(&format!(":{all_in}"))) {
            let action_type = if to_call == 0 { "bet" } else { "raise" };
            actions.push(ActionInfo {
                id: format!("{action_type}:{all_in}"),
                label: format!("All-in {all_in}"),
                action_type: "allin".to_string(),
            });
        }
    }

    actions
}

fn calculate_to_call(position: &ExplorationPosition) -> u32 {
    // Simple heuristic based on history
    // In a full implementation, we'd track bets properly
    let mut p1_invested = 1u32; // SB
    let mut p2_invested = 2u32; // BB

    for (i, action) in position.history.iter().enumerate() {
        let is_p1 = (i % 2) == 0;
        if action == "c" || action == "call" {
            if is_p1 {
                p1_invested = p2_invested;
            } else {
                p2_invested = p1_invested;
            }
        } else if let Some(amt) = action.strip_prefix("r:").or_else(|| action.strip_prefix("b:")) {
            if let Ok(amount) = amt.parse::<u32>() {
                if is_p1 {
                    p1_invested = amount;
                } else {
                    p2_invested = amount;
                }
            }
        }
    }

    if position.to_act == 0 {
        p2_invested.saturating_sub(p1_invested)
    } else {
        p1_invested.saturating_sub(p2_invested)
    }
}

#[allow(clippy::too_many_arguments)]
fn get_hand_strategy(
    bundle: &LoadedBundle,
    _board: &[Card],
    street: Street,
    history_str: &str,
    rank1: char,
    rank2: char,
    suited: bool,
    actions: &[ActionInfo],
    bucket_cache: Option<&std::collections::HashMap<(char, char, bool), u16>>,
) -> Result<Vec<ActionProb>, String> {
    // For preflop, use hand string directly
    // For postflop, need to pick a representative hand and get bucket
    let bucket_or_hand = if street == Street::Preflop {
        // Canonical format: AA (pairs), AKs (suited), AKo (offsuit)
        if rank1 == rank2 {
            format!("{rank1}{rank2}")
        } else if suited {
            format!("{rank1}{rank2}s")
        } else {
            format!("{rank1}{rank2}o")
        }
    } else {
        // Postflop requires bucket cache to have been computed
        let cache = bucket_cache.ok_or_else(|| {
            format!(
                "Bucket cache not available for postflop street {street:?}. \
                 Start bucket computation before requesting postflop strategy."
            )
        })?;
        let &bucket = cache.get(&(rank1, rank2, suited)).ok_or_else(|| {
            format!("No bucket in cache for hand ({rank1}, {rank2}, suited={suited})")
        })?;
        bucket.to_string()
    };

    let street_char = match street {
        Street::Preflop => 'P',
        Street::Flop => 'F',
        Street::Turn => 'T',
        Street::River => 'R',
    };

    let info_set_key = format!("{bucket_or_hand}|{street_char}|{history_str}");

    let probs = bundle.blueprint.lookup(&info_set_key).ok_or_else(|| {
        format!(
            "Blueprint lookup failed for key '{info_set_key}' \
             (blueprint has {} info sets)",
            bundle.blueprint.len()
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

fn make_representative_hand(rank1: char, rank2: char, suited: bool, board: &[Card]) -> (Card, Card) {
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

