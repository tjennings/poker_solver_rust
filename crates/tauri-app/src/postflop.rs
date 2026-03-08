use parking_lot::{Mutex, RwLock};
use range_solver::action_tree::{Action, ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{card_from_str, card_pair_to_index, card_to_string, CardConfig, NOT_DEALT};
use range_solver::interface::Game;
use range_solver::range::Range;
use range_solver::{compute_exploitability, finalize, solve_step, PostFlopGame};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use crate::exploration::ActionInfo;

// ---------------------------------------------------------------------------
// Shared types
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
            ip_range: "TT+,AQs+,AKo".to_string(),
            pot: 30,
            effective_stack: 170,
            oop_bet_sizes: "25%,33%,75%,a".to_string(),
            oop_raise_sizes: "a".to_string(),
            ip_bet_sizes: "25%,33%,75%,a".to_string(),
            ip_raise_sizes: "a".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PostflopConfigSummary {
    pub config: PostflopConfig,
    pub oop_combos: usize,
    pub ip_combos: usize,
}


#[derive(Debug, Clone, Serialize)]
pub struct PostflopComboDetail {
    pub cards: String,
    pub probabilities: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PostflopMatrixCell {
    pub hand: String,
    pub suited: bool,
    pub pair: bool,
    pub probabilities: Vec<f32>,
    pub combo_count: usize,
    pub ev: Option<f32>,
    pub combos: Vec<PostflopComboDetail>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PostflopStrategyMatrix {
    pub cells: Vec<Vec<PostflopMatrixCell>>,
    pub actions: Vec<ActionInfo>,
    pub player: usize,
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
    pub filtered_oop_range: Vec<f32>,
    pub filtered_ip_range: Vec<f32>,
    pub pot: i32,
    pub effective_stack: i32,
}

#[derive(Debug, Clone, Serialize)]
pub struct PostflopPlayResult {
    pub matrix: Option<PostflopStrategyMatrix>,
    pub is_terminal: bool,
    pub is_chance: bool,
    pub current_player: Option<usize>,
    pub pot: i32,
    pub stacks: [i32; 2],
}

// ---------------------------------------------------------------------------
// Card helpers
// ---------------------------------------------------------------------------

/// Rank names for the 13x13 matrix, row 0 = Ace, row 12 = Deuce.
const RANK_NAMES: [char; 13] = [
    'A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2',
];

/// Maps a range-solver card encoding (card = 4*rank + suit, rank 0=deuce..12=ace)
/// to a 13x13 matrix position. Returns (row, col, is_suited).
///
/// - Pairs sit on the diagonal (row == col).
/// - Suited hands go above the diagonal (smaller row first).
/// - Offsuit hands go below the diagonal (larger row first).
fn card_pair_to_matrix(c1: u8, c2: u8) -> (usize, usize, bool) {
    let rank1 = c1 >> 2; // 0=deuce..12=ace
    let suit1 = c1 & 3;
    let rank2 = c2 >> 2;
    let suit2 = c2 & 3;

    // Convert from range-solver rank (0=deuce, 12=ace) to matrix row (0=ace, 12=deuce).
    let row1 = 12 - rank1 as usize;
    let row2 = 12 - rank2 as usize;

    if rank1 == rank2 {
        // Pair: on diagonal, row order doesn't matter.
        (row1, row2, false)
    } else {
        let is_suited = suit1 == suit2;
        // Suited above diagonal: smaller row index first.
        // Offsuit below diagonal: larger row index first.
        let (high_row, low_row) = if row1 < row2 {
            (row1, row2)
        } else {
            (row2, row1)
        };
        if is_suited {
            (high_row, low_row, true)
        } else {
            (low_row, high_row, false)
        }
    }
}

/// Returns (label, is_suited, is_pair) for a given matrix cell.
fn matrix_cell_label(row: usize, col: usize) -> (String, bool, bool) {
    let r1 = RANK_NAMES[row];
    let r2 = RANK_NAMES[col];
    if row == col {
        (format!("{r1}{r2}"), false, true)
    } else if row < col {
        // Above diagonal = suited.
        (format!("{r1}{r2}s"), true, false)
    } else {
        // Below diagonal = offsuit.
        (format!("{r1}{r2}o"), false, false)
    }
}

/// Format a chip amount as a pot-percentage string (e.g. "33%" or "120%").
fn format_pot_pct(amt: i32, pot: i32) -> String {
    if pot > 0 {
        let pct = (amt as f64 / pot as f64 * 100.0).round() as i32;
        format!("{pct}%")
    } else {
        format!("{amt}")
    }
}

/// Converts a range-solver `Action` to a serializable `ActionInfo`.
fn action_to_info(action: &Action, index: usize, pot: i32) -> ActionInfo {
    match action {
        Action::Fold => ActionInfo {
            id: index.to_string(),
            label: "Fold".to_string(),
            action_type: "fold".to_string(),
            size_key: None,
        },
        Action::Check => ActionInfo {
            id: index.to_string(),
            label: "Check".to_string(),
            action_type: "check".to_string(),
            size_key: None,
        },
        Action::Call => ActionInfo {
            id: index.to_string(),
            label: "Call".to_string(),
            action_type: "call".to_string(),
            size_key: None,
        },
        Action::Bet(amt) => ActionInfo {
            id: index.to_string(),
            label: format!("Bet {}", format_pot_pct(*amt, pot)),
            action_type: "bet".to_string(),
            size_key: Some(amt.to_string()),
        },
        Action::Raise(amt) => ActionInfo {
            id: index.to_string(),
            label: format!("Raise {}", format_pot_pct(*amt, pot)),
            action_type: "raise".to_string(),
            size_key: Some(amt.to_string()),
        },
        Action::AllIn(amt) => ActionInfo {
            id: index.to_string(),
            label: format!("All-in {amt}"),
            action_type: "allin".to_string(),
            size_key: Some(amt.to_string()),
        },
        _ => ActionInfo {
            id: index.to_string(),
            label: format!("{action}"),
            action_type: "other".to_string(),
            size_key: None,
        },
    }
}

// ---------------------------------------------------------------------------
// Solve cache
// ---------------------------------------------------------------------------

const CACHE_MAGIC: u32 = 0x50534343; // "PSCC"
const CACHE_VERSION: u32 = 1;

fn compute_cache_key(
    config: &PostflopConfig,
    board: &[String],
    prior_actions: &[Vec<usize>],
) -> u64 {
    let mut hasher = DefaultHasher::new();
    config.oop_range.hash(&mut hasher);
    config.ip_range.hash(&mut hasher);
    config.pot.hash(&mut hasher);
    config.effective_stack.hash(&mut hasher);
    config.oop_bet_sizes.hash(&mut hasher);
    config.oop_raise_sizes.hash(&mut hasher);
    config.ip_bet_sizes.hash(&mut hasher);
    config.ip_raise_sizes.hash(&mut hasher);
    for card in board {
        card.hash(&mut hasher);
    }
    for street_actions in prior_actions {
        for &a in street_actions {
            a.hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn write_cache_file(
    path: &Path,
    game: &PostFlopGame,
    exploitability: f32,
    iterations: u32,
) -> std::io::Result<()> {
    let (s1, s2, s_ip, s_ch) = game.storage_buffers();
    let mut f = std::fs::File::create(path)?;
    f.write_all(&CACHE_MAGIC.to_le_bytes())?;
    f.write_all(&CACHE_VERSION.to_le_bytes())?;
    f.write_all(&exploitability.to_le_bytes())?;
    f.write_all(&iterations.to_le_bytes())?;
    f.write_all(&(s1.len() as u64).to_le_bytes())?;
    f.write_all(&(s2.len() as u64).to_le_bytes())?;
    f.write_all(&(s_ip.len() as u64).to_le_bytes())?;
    f.write_all(&(s_ch.len() as u64).to_le_bytes())?;
    f.write_all(s1)?;
    f.write_all(s2)?;
    f.write_all(s_ip)?;
    f.write_all(s_ch)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// PostflopState
// ---------------------------------------------------------------------------

pub struct PostflopState {
    pub config: RwLock<PostflopConfig>,
    pub game: Mutex<Option<PostFlopGame>>,
    pub current_iteration: AtomicU32,
    pub max_iterations: AtomicU32,
    pub exploitability_bits: AtomicU32,
    pub solving: AtomicBool,
    pub solve_complete: AtomicBool,
    pub matrix_snapshot: RwLock<Option<PostflopStrategyMatrix>>,
    pub filtered_oop_weights: RwLock<Option<Vec<f32>>>,
    pub filtered_ip_weights: RwLock<Option<Vec<f32>>>,
    pub cache_dir: RwLock<Option<PathBuf>>,
}

impl Default for PostflopState {
    fn default() -> Self {
        Self {
            config: RwLock::new(PostflopConfig::default()),
            game: Mutex::new(None),
            current_iteration: AtomicU32::new(0),
            max_iterations: AtomicU32::new(0),
            exploitability_bits: AtomicU32::new(0),
            solving: AtomicBool::new(false),
            solve_complete: AtomicBool::new(false),
            matrix_snapshot: RwLock::new(None),
            filtered_oop_weights: RwLock::new(None),
            filtered_ip_weights: RwLock::new(None),
            cache_dir: RwLock::new(None),
        }
    }
}

// ---------------------------------------------------------------------------
// build_strategy_matrix
// ---------------------------------------------------------------------------

/// Builds a 13x13 strategy matrix from the current game state.
///
/// The game must be at a non-terminal, non-chance node with memory allocated.
pub fn build_strategy_matrix(game: &mut PostFlopGame) -> PostflopStrategyMatrix {
    let player = game.current_player();
    let actions = game.available_actions();
    let strategy = game.strategy();
    let private_cards = game.private_cards(player).to_vec();
    let num_hands = game.num_private_hands(player);
    let num_actions = actions.len();

    // Compute pot and stacks before mutable borrows.
    let (pot, stacks) = {
        let tc = game.tree_config();
        let ba = game.total_bet_amount();
        let pot = tc.starting_pot + ba[0] + ba[1];
        let stacks = [tc.effective_stack - ba[0], tc.effective_stack - ba[1]];
        (pot, stacks)
    };

    let action_infos: Vec<ActionInfo> = actions
        .iter()
        .enumerate()
        .map(|(i, a)| action_to_info(a, i, pot))
        .collect();

    // Compute per-hand EV if game is solved.
    let hand_evs: Option<Vec<f32>> = if game.is_solved() {
        game.cache_normalized_weights();
        Some(game.expected_values(player))
    } else {
        None
    };

    // Initialize the 13x13 matrix with zeroed probability accumulators.
    let mut prob_sums = vec![vec![vec![0.0f64; num_actions]; 13]; 13];
    let mut combo_counts = vec![vec![0usize; 13]; 13];
    let mut ev_sums = vec![vec![0.0f64; 13]; 13];
    let mut combo_details: Vec<Vec<Vec<PostflopComboDetail>>> =
        vec![vec![Vec::new(); 13]; 13];

    for (hand_idx, &(c1, c2)) in private_cards.iter().enumerate() {
        let (row, col, _) = card_pair_to_matrix(c1, c2);
        combo_counts[row][col] += 1;
        if let Some(ref evs) = hand_evs {
            ev_sums[row][col] += evs[hand_idx] as f64;
        }
        let mut probs = Vec::with_capacity(num_actions);
        for action_idx in 0..num_actions {
            let prob = strategy[action_idx * num_hands + hand_idx];
            prob_sums[row][col][action_idx] += prob as f64;
            probs.push(prob);
        }
        let s1 = card_to_string(c1).unwrap_or_default();
        let s2 = card_to_string(c2).unwrap_or_default();
        combo_details[row][col].push(PostflopComboDetail {
            cards: format!("{s1}{s2}"),
            probabilities: probs,
        });
    }

    let cells: Vec<Vec<PostflopMatrixCell>> = (0..13)
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
                    PostflopMatrixCell {
                        hand: label,
                        suited,
                        pair,
                        probabilities,
                        combo_count: count,
                        ev,
                        combos,
                    }
                })
                .collect()
        })
        .collect();

    // Board cards — read from card_config to avoid relying on interpreter state.
    let cc = game.card_config();
    let mut board_cards: Vec<u8> = cc.flop.to_vec();
    if cc.turn != NOT_DEALT {
        board_cards.push(cc.turn);
    }
    if cc.river != NOT_DEALT {
        board_cards.push(cc.river);
    }
    let board: Vec<String> = board_cards
        .iter()
        .filter_map(|&c| card_to_string(c).ok())
        .collect();

    PostflopStrategyMatrix {
        cells,
        actions: action_infos,
        player,
        pot,
        stacks,
        board,
    }
}

// ---------------------------------------------------------------------------
// postflop_set_config
// ---------------------------------------------------------------------------

fn count_combos(range: &Range) -> usize {
    let raw = range.raw_data();
    raw.iter().filter(|&&w| w > 0.0).count()
}

pub fn postflop_set_config_core(
    state: &PostflopState,
    config: PostflopConfig,
) -> Result<PostflopConfigSummary, String> {
    // Parse and validate ranges.
    let oop_range: Range = config
        .oop_range
        .parse()
        .map_err(|e: String| format!("Invalid OOP range: {e}"))?;
    let ip_range: Range = config
        .ip_range
        .parse()
        .map_err(|e: String| format!("Invalid IP range: {e}"))?;

    if oop_range.is_empty() {
        return Err("OOP range is empty".to_string());
    }
    if ip_range.is_empty() {
        return Err("IP range is empty".to_string());
    }

    // Validate bet sizes.
    BetSizeOptions::try_from((
        config.oop_bet_sizes.as_str(),
        config.oop_raise_sizes.as_str(),
    ))
    .map_err(|e| format!("Invalid OOP bet sizes: {e}"))?;
    BetSizeOptions::try_from((
        config.ip_bet_sizes.as_str(),
        config.ip_raise_sizes.as_str(),
    ))
    .map_err(|e| format!("Invalid IP bet sizes: {e}"))?;

    if config.pot <= 0 {
        return Err("Pot must be positive".to_string());
    }
    if config.effective_stack <= 0 {
        return Err("Effective stack must be positive".to_string());
    }

    let oop_combos = count_combos(&oop_range);
    let ip_combos = count_combos(&ip_range);

    // Store config and clear stale state.
    *state.config.write() = config.clone();
    *state.game.lock() = None;
    *state.matrix_snapshot.write() = None;
    *state.filtered_oop_weights.write() = None;
    *state.filtered_ip_weights.write() = None;
    state.current_iteration.store(0, Ordering::Relaxed);
    state.max_iterations.store(0, Ordering::Relaxed);
    state.exploitability_bits.store(0, Ordering::Relaxed);
    state.solving.store(false, Ordering::Relaxed);
    state.solve_complete.store(false, Ordering::Relaxed);

    Ok(PostflopConfigSummary {
        config: config.clone(),
        oop_combos,
        ip_combos,
    })
}

#[tauri::command]
pub fn postflop_set_config(
    state: tauri::State<'_, Arc<PostflopState>>,
    config: PostflopConfig,
) -> Result<PostflopConfigSummary, String> {
    postflop_set_config_core(&state, config)
}

// ---------------------------------------------------------------------------
// postflop_solve_street
// ---------------------------------------------------------------------------

/// Determines the `BoardState` and parses board cards from a list of card strings.
///
/// Returns `(flop, turn, river, initial_state)`.
fn parse_board(board: &[String]) -> Result<([u8; 3], u8, u8, BoardState), String> {
    match board.len() {
        3 => {
            let flop_str = format!("{}{}{}", board[0], board[1], board[2]);
            let flop = range_solver::card::flop_from_str(&flop_str)
                .map_err(|e| format!("Invalid flop: {e}"))?;
            Ok((flop, NOT_DEALT, NOT_DEALT, BoardState::Flop))
        }
        4 => {
            let flop_str = format!("{}{}{}", board[0], board[1], board[2]);
            let flop = range_solver::card::flop_from_str(&flop_str)
                .map_err(|e| format!("Invalid flop: {e}"))?;
            let turn =
                card_from_str(&board[3]).map_err(|e| format!("Invalid turn card: {e}"))?;
            Ok((flop, turn, NOT_DEALT, BoardState::Turn))
        }
        5 => {
            let flop_str = format!("{}{}{}", board[0], board[1], board[2]);
            let flop = range_solver::card::flop_from_str(&flop_str)
                .map_err(|e| format!("Invalid flop: {e}"))?;
            let turn =
                card_from_str(&board[3]).map_err(|e| format!("Invalid turn card: {e}"))?;
            let river =
                card_from_str(&board[4]).map_err(|e| format!("Invalid river card: {e}"))?;
            Ok((flop, turn, river, BoardState::River))
        }
        n => Err(format!("Board must have 3-5 cards, got {n}")),
    }
}

/// Builds the range-solver `PostFlopGame` from the current config, board, and
/// optional filtered weights.
fn build_game(
    config: &PostflopConfig,
    board: &[String],
    filtered_oop: &Option<Vec<f32>>,
    filtered_ip: &Option<Vec<f32>>,
) -> Result<PostFlopGame, String> {
    let (flop, turn, river, initial_state) = parse_board(board)?;

    // Parse ranges (or use filtered weights if available from multi-street).
    let oop_range = match filtered_oop {
        Some(weights) => {
            Range::from_raw_data(weights).map_err(|e| format!("Bad OOP weights: {e}"))?
        }
        None => config
            .oop_range
            .parse()
            .map_err(|e: String| format!("Invalid OOP range: {e}"))?,
    };
    let ip_range = match filtered_ip {
        Some(weights) => {
            Range::from_raw_data(weights).map_err(|e| format!("Bad IP weights: {e}"))?
        }
        None => config
            .ip_range
            .parse()
            .map_err(|e: String| format!("Invalid IP range: {e}"))?,
    };

    // Parse bet sizes.
    let oop_sizes = BetSizeOptions::try_from((
        config.oop_bet_sizes.as_str(),
        config.oop_raise_sizes.as_str(),
    ))
    .map_err(|e| format!("Invalid OOP bet sizes: {e}"))?;
    let ip_sizes = BetSizeOptions::try_from((
        config.ip_bet_sizes.as_str(),
        config.ip_raise_sizes.as_str(),
    ))
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
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        turn_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        river_bet_sizes: [oop_sizes, ip_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };

    let action_tree =
        ActionTree::new(tree_config).map_err(|e| format!("Failed to build tree: {e}"))?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;
    game.allocate_memory(false);
    Ok(game)
}

pub fn postflop_solve_street_core(
    state: &Arc<PostflopState>,
    board: Vec<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
) -> Result<(), String> {
    postflop_solve_street_impl(state, board, max_iterations, target_exploitability, vec![])
}

fn postflop_solve_street_impl(
    state: &Arc<PostflopState>,
    board: Vec<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
    prior_actions: Vec<Vec<usize>>,
) -> Result<(), String> {
    // Guard: reject if already solving.
    if state.solving.load(Ordering::Relaxed) {
        return Err("A solve is already in progress".to_string());
    }

    let max_iters = max_iterations.unwrap_or(200);

    // Snapshot config and filtered weights under their locks.
    let config = state.config.read().clone();
    let target_exp = target_exploitability.unwrap_or(3.0);
    let filtered_oop = state.filtered_oop_weights.read().clone();
    let filtered_ip = state.filtered_ip_weights.read().clone();

    // Build game (expensive but runs on the calling thread before spawn).
    let mut game = build_game(&config, &board, &filtered_oop, &filtered_ip)?;

    // Reset progress atomics.
    state.current_iteration.store(0, Ordering::Relaxed);
    state.max_iterations.store(max_iters, Ordering::Relaxed);
    state
        .exploitability_bits
        .store(f32::MAX.to_bits(), Ordering::Relaxed);
    state.solve_complete.store(false, Ordering::Relaxed);
    state.solving.store(true, Ordering::Release);

    // Take initial matrix snapshot.
    {
        let matrix = build_strategy_matrix(&mut game);
        *state.matrix_snapshot.write() = Some(matrix);
    }

    // Clone the Arc for the background thread.
    let shared = Arc::clone(state);
    let cache_dir = state.cache_dir.read().clone();

    std::thread::spawn(move || {
        let mut last_exp = f32::MAX;

        for t in 0..max_iters {
            // Check if we've been asked to stop (e.g. config reset clears `solving`).
            if !shared.solving.load(Ordering::Relaxed) {
                break;
            }

            solve_step(&game, t);

            last_exp = compute_exploitability(&game);
            shared.current_iteration.store(t + 1, Ordering::Relaxed);
            shared
                .exploitability_bits
                .store(last_exp.to_bits(), Ordering::Relaxed);

            // Snapshot the matrix every iteration.
            {
                let matrix = build_strategy_matrix(&mut game);
                *shared.matrix_snapshot.write() = Some(matrix);
            }

            if last_exp <= target_exp {
                break;
            }
        }

        let final_iters = shared.current_iteration.load(Ordering::Relaxed);

        // Finalize: compute EV / normalize strategy.
        finalize(&mut game);

        // Final matrix snapshot.
        {
            let matrix = build_strategy_matrix(&mut game);
            *shared.matrix_snapshot.write() = Some(matrix);
        }

        // Write cache if cache_dir is configured.
        if let Some(ref dir) = *shared.cache_dir.read() {
            let spots_dir = dir.join("spots");
            let _ = std::fs::create_dir_all(&spots_dir);
            let key = compute_cache_key(&config, &board, &prior_actions);
            let cache_path = spots_dir.join(format!("{key:016x}.bin"));
            if let Err(e) = write_cache_file(&cache_path, &game, last_exp, final_iters) {
                eprintln!("Failed to write solve cache: {e}");
            }
        }

        // Store the solved game so other commands can navigate it.
        *shared.game.lock() = Some(game);
        shared.solve_complete.store(true, Ordering::Relaxed);
        shared.solving.store(false, Ordering::Release);
    });

    Ok(())
}

#[tauri::command]
pub async fn postflop_solve_street(
    state: tauri::State<'_, Arc<PostflopState>>,
    board: Vec<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
) -> Result<(), String> {
    postflop_solve_street_core(&state, board, max_iterations, target_exploitability)
}

// ---------------------------------------------------------------------------
// postflop_get_progress
// ---------------------------------------------------------------------------

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

#[tauri::command]
pub fn postflop_get_progress(
    state: tauri::State<'_, Arc<PostflopState>>,
) -> PostflopProgress {
    postflop_get_progress_core(&state)
}

// ---------------------------------------------------------------------------
// postflop_play_action
// ---------------------------------------------------------------------------

pub fn postflop_play_action_core(
    state: &PostflopState,
    action: usize,
) -> Result<PostflopPlayResult, String> {
    let mut game_guard = state.game.lock();
    let game = game_guard.as_mut().ok_or("No game loaded")?;

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

#[tauri::command]
pub fn postflop_play_action(
    state: tauri::State<'_, Arc<PostflopState>>,
    action: usize,
) -> Result<PostflopPlayResult, String> {
    postflop_play_action_core(&state, action)
}

// ---------------------------------------------------------------------------
// postflop_navigate_to
// ---------------------------------------------------------------------------

/// Replay the game tree to a given action history and return the result.
pub fn postflop_navigate_to_core(
    state: &PostflopState,
    history: Vec<usize>,
) -> Result<PostflopPlayResult, String> {
    let mut game_guard = state.game.lock();
    let game = game_guard.as_mut().ok_or("No game loaded")?;

    game.apply_history(&history);

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

#[tauri::command]
pub fn postflop_navigate_to(
    state: tauri::State<'_, Arc<PostflopState>>,
    history: Vec<usize>,
) -> Result<PostflopPlayResult, String> {
    postflop_navigate_to_core(&state, history)
}

// ---------------------------------------------------------------------------
// postflop_close_street
// ---------------------------------------------------------------------------

/// Walks the action history of a completed street, multiplying each acting
/// player's range weights by the strategy frequency for the chosen action.
/// Stores the filtered weights for the next street's solve.
pub fn postflop_close_street_core(
    state: &PostflopState,
    action_history: Vec<usize>,
) -> Result<PostflopStreetResult, String> {
    let mut game_guard = state.game.lock();
    let game = game_guard.as_mut().ok_or("No game loaded")?;

    game.back_to_root();

    // Start from current filtered weights, or fall back to the config ranges.
    let config = state.config.read().clone();
    let oop_range: Range = config
        .oop_range
        .parse()
        .map_err(|e: String| format!("Invalid OOP range: {e}"))?;
    let ip_range: Range = config
        .ip_range
        .parse()
        .map_err(|e: String| format!("Invalid IP range: {e}"))?;

    let mut oop_weights: Vec<f32> = state
        .filtered_oop_weights
        .read()
        .clone()
        .unwrap_or_else(|| oop_range.raw_data().to_vec());
    let mut ip_weights: Vec<f32> = state
        .filtered_ip_weights
        .read()
        .clone()
        .unwrap_or_else(|| ip_range.raw_data().to_vec());

    // Walk each action, filtering the acting player's range at each step.
    for &action_idx in &action_history {
        if game.is_terminal_node() || game.is_chance_node() {
            break;
        }

        let player = game.current_player();
        let num_hands = game.num_private_hands(player);
        let strategy = game.strategy();
        let private_cards = game.private_cards(player);

        let weights = if player == 0 {
            &mut oop_weights
        } else {
            &mut ip_weights
        };
        for (hand_idx, &(c1, c2)) in private_cards.iter().enumerate().take(num_hands) {
            let ci = card_pair_to_index(c1, c2);
            let action_prob = strategy[action_idx * num_hands + hand_idx];
            weights[ci] *= action_prob;
        }

        game.play(action_idx);
    }

    // Compute pot/stacks at the final node.
    let bet_amounts = game.total_bet_amount();
    let tc = game.tree_config();
    let pot = tc.starting_pot + bet_amounts[0] + bet_amounts[1];
    let effective_stack = tc.effective_stack - bet_amounts[0].max(bet_amounts[1]);

    // Store filtered weights for the next street.
    *state.filtered_oop_weights.write() = Some(oop_weights.clone());
    *state.filtered_ip_weights.write() = Some(ip_weights.clone());

    Ok(PostflopStreetResult {
        filtered_oop_range: oop_weights,
        filtered_ip_range: ip_weights,
        pot,
        effective_stack,
    })
}

#[tauri::command]
pub fn postflop_close_street(
    state: tauri::State<'_, Arc<PostflopState>>,
    action_history: Vec<usize>,
) -> Result<PostflopStreetResult, String> {
    postflop_close_street_core(&state, action_history)
}

// ---------------------------------------------------------------------------
// postflop_set_cache_dir
// ---------------------------------------------------------------------------

/// Enables or disables the solve cache. When `dir` is `Some`, solved spots
/// will be written to `<dir>/spots/<hash>.bin`. When `None`, caching is off.
pub fn postflop_set_cache_dir_core(state: &PostflopState, dir: Option<String>) {
    *state.cache_dir.write() = dir.map(PathBuf::from);
}

#[tauri::command]
pub fn postflop_set_cache_dir(
    state: tauri::State<'_, Arc<PostflopState>>,
    dir: Option<String>,
) {
    postflop_set_cache_dir_core(&state, dir);
}

// -- Cache read types -------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct CacheInfo {
    pub exploitability: f32,
    pub iterations: u32,
}

struct CachedSolve {
    exploitability: f32,
    iterations: u32,
    storage1: Vec<u8>,
    storage2: Vec<u8>,
    storage_ip: Vec<u8>,
    storage_chance: Vec<u8>,
}

fn load_cache_file(path: &Path) -> Result<CachedSolve, String> {
    use std::io::Read;

    let mut f = std::fs::File::open(path).map_err(|e| format!("Cannot open cache: {e}"))?;
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    f.read_exact(&mut buf4)
        .map_err(|e| format!("Read magic: {e}"))?;
    let magic = u32::from_le_bytes(buf4);
    if magic != CACHE_MAGIC {
        return Err(format!("Bad cache magic: {magic:#x}"));
    }

    f.read_exact(&mut buf4)
        .map_err(|e| format!("Read version: {e}"))?;
    let version = u32::from_le_bytes(buf4);
    if version != CACHE_VERSION {
        return Err(format!("Unsupported cache version: {version}"));
    }

    f.read_exact(&mut buf4)
        .map_err(|e| format!("Read exploitability: {e}"))?;
    let exploitability = f32::from_le_bytes(buf4);

    f.read_exact(&mut buf4)
        .map_err(|e| format!("Read iterations: {e}"))?;
    let iterations = u32::from_le_bytes(buf4);

    f.read_exact(&mut buf8)
        .map_err(|e| format!("Read s1 len: {e}"))?;
    let s1_len = u64::from_le_bytes(buf8) as usize;
    f.read_exact(&mut buf8)
        .map_err(|e| format!("Read s2 len: {e}"))?;
    let s2_len = u64::from_le_bytes(buf8) as usize;
    f.read_exact(&mut buf8)
        .map_err(|e| format!("Read s_ip len: {e}"))?;
    let s_ip_len = u64::from_le_bytes(buf8) as usize;
    f.read_exact(&mut buf8)
        .map_err(|e| format!("Read s_ch len: {e}"))?;
    let s_ch_len = u64::from_le_bytes(buf8) as usize;

    let mut s1 = vec![0u8; s1_len];
    f.read_exact(&mut s1)
        .map_err(|e| format!("Read storage1: {e}"))?;
    let mut s2 = vec![0u8; s2_len];
    f.read_exact(&mut s2)
        .map_err(|e| format!("Read storage2: {e}"))?;
    let mut s_ip = vec![0u8; s_ip_len];
    f.read_exact(&mut s_ip)
        .map_err(|e| format!("Read storage_ip: {e}"))?;
    let mut s_ch = vec![0u8; s_ch_len];
    f.read_exact(&mut s_ch)
        .map_err(|e| format!("Read storage_chance: {e}"))?;

    Ok(CachedSolve {
        exploitability,
        iterations,
        storage1: s1,
        storage2: s2,
        storage_ip: s_ip,
        storage_chance: s_ch,
    })
}

// -- Cache check & load -----------------------------------------------------

pub fn postflop_check_cache_core(
    state: &PostflopState,
    board: Vec<String>,
    prior_actions: Vec<Vec<usize>>,
) -> Result<Option<CacheInfo>, String> {
    let cache_dir = state.cache_dir.read().clone();
    let cache_dir = match cache_dir {
        Some(d) => d,
        None => return Ok(None),
    };

    let config = state.config.read().clone();
    let key = compute_cache_key(&config, &board, &prior_actions);
    let path = cache_dir.join("spots").join(format!("{key:016x}.bin"));

    if !path.exists() {
        return Ok(None);
    }

    match load_cache_file(&path) {
        Ok(cached) => Ok(Some(CacheInfo {
            exploitability: cached.exploitability,
            iterations: cached.iterations,
        })),
        Err(e) => {
            eprintln!("Cache read error: {e}");
            Ok(None)
        }
    }
}

pub fn postflop_load_cached_core(
    state: &Arc<PostflopState>,
    board: Vec<String>,
    prior_actions: Vec<Vec<usize>>,
) -> Result<PostflopStrategyMatrix, String> {
    let cache_dir = state
        .cache_dir
        .read()
        .clone()
        .ok_or("No cache directory configured")?;

    let config = state.config.read().clone();
    let filtered_oop = state.filtered_oop_weights.read().clone();
    let filtered_ip = state.filtered_ip_weights.read().clone();

    let key = compute_cache_key(&config, &board, &prior_actions);
    let path = cache_dir.join("spots").join(format!("{key:016x}.bin"));

    let cached = load_cache_file(&path)?;

    let mut game = build_game(&config, &board, &filtered_oop, &filtered_ip)?;

    game.set_storage_buffers(
        cached.storage1,
        cached.storage2,
        cached.storage_ip,
        cached.storage_chance,
    );

    finalize(&mut game);

    let matrix = build_strategy_matrix(&mut game);

    state
        .current_iteration
        .store(cached.iterations, Ordering::Relaxed);
    state
        .exploitability_bits
        .store(cached.exploitability.to_bits(), Ordering::Relaxed);
    state.solve_complete.store(true, Ordering::Relaxed);
    *state.matrix_snapshot.write() = Some(matrix.clone());
    *state.game.lock() = Some(game);

    Ok(matrix)
}

#[tauri::command]
pub fn postflop_check_cache(
    state: tauri::State<'_, Arc<PostflopState>>,
    board: Vec<String>,
    prior_actions: Vec<Vec<usize>>,
) -> Result<Option<CacheInfo>, String> {
    postflop_check_cache_core(&state, board, prior_actions)
}

#[tauri::command]
pub fn postflop_load_cached(
    state: tauri::State<'_, Arc<PostflopState>>,
    board: Vec<String>,
    prior_actions: Vec<Vec<usize>>,
) -> Result<PostflopStrategyMatrix, String> {
    postflop_load_cached_core(&state, board, prior_actions)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_pair_to_matrix_pair() {
        // Ace-Ace: rank 12, row 0 (diagonal)
        let (r, c, suited) = card_pair_to_matrix(48, 51); // Ac, As
        assert_eq!(r, 0);
        assert_eq!(c, 0);
        assert!(!suited);
    }

    #[test]
    fn test_card_pair_to_matrix_suited() {
        // AKs: Ace spade=51 (rank12), King spade=47 (rank11) -> same suit
        let (r, c, suited) = card_pair_to_matrix(51, 47);
        // row for ace=0, row for king=1. Suited => above diagonal (0,1).
        assert_eq!(r, 0);
        assert_eq!(c, 1);
        assert!(suited);
    }

    #[test]
    fn test_card_pair_to_matrix_offsuit() {
        // AKo: Ace spade=51 (rank12), King heart=46 (rank11) -> diff suits
        let (r, c, suited) = card_pair_to_matrix(51, 46);
        // Offsuit => below diagonal (1,0).
        assert_eq!(r, 1);
        assert_eq!(c, 0);
        assert!(!suited);
    }

    #[test]
    fn test_matrix_cell_label_pair() {
        let (label, suited, pair) = matrix_cell_label(0, 0);
        assert_eq!(label, "AA");
        assert!(!suited);
        assert!(pair);
    }

    #[test]
    fn test_matrix_cell_label_suited() {
        let (label, suited, pair) = matrix_cell_label(0, 1);
        assert_eq!(label, "AKs");
        assert!(suited);
        assert!(!pair);
    }

    #[test]
    fn test_matrix_cell_label_offsuit() {
        let (label, suited, pair) = matrix_cell_label(1, 0);
        assert_eq!(label, "KAo");
        assert!(!suited);
        assert!(!pair);

        // More conventional: row=larger rank index, col=smaller.
        let (label, suited, pair) = matrix_cell_label(2, 0);
        assert_eq!(label, "QAo");
        assert!(!suited);
        assert!(!pair);
    }

    #[test]
    fn test_action_to_info() {
        let info = action_to_info(&Action::Fold, 0, 100);
        assert_eq!(info.label, "Fold");
        assert_eq!(info.action_type, "fold");

        let info = action_to_info(&Action::Bet(50), 1, 100);
        assert_eq!(info.label, "Bet 50%");
        assert_eq!(info.action_type, "bet");

        let info = action_to_info(&Action::AllIn(200), 2, 100);
        assert_eq!(info.label, "All-in 200");
        assert_eq!(info.action_type, "allin");
    }

    #[test]
    fn test_set_config_valid() {
        let state = PostflopState::default();
        let config = PostflopConfig::default();
        let result = postflop_set_config_core(&state, config);
        assert!(result.is_ok());
        let summary = result.unwrap();
        assert!(summary.oop_combos > 0);
        assert!(summary.ip_combos > 0);
    }

    #[test]
    fn test_set_config_empty_range() {
        let state = PostflopState::default();
        let mut config = PostflopConfig::default();
        config.oop_range = String::new();
        let result = postflop_set_config_core(&state, config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("OOP range is empty"));
    }

    #[test]
    fn test_set_config_invalid_range() {
        let state = PostflopState::default();
        let mut config = PostflopConfig::default();
        config.oop_range = "XYZ".to_string();
        let result = postflop_set_config_core(&state, config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid OOP range"));
    }

    #[test]
    fn test_set_config_invalid_pot() {
        let state = PostflopState::default();
        let mut config = PostflopConfig::default();
        config.pot = 0;
        let result = postflop_set_config_core(&state, config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Pot must be positive"));
    }

    #[test]
    fn test_set_config_clears_state() {
        let state = PostflopState::default();
        state.solving.store(true, Ordering::Relaxed);
        state.current_iteration.store(42, Ordering::Relaxed);

        let config = PostflopConfig::default();
        let result = postflop_set_config_core(&state, config);
        assert!(result.is_ok());

        assert!(!state.solving.load(Ordering::Relaxed));
        assert_eq!(state.current_iteration.load(Ordering::Relaxed), 0);
        assert!(state.game.lock().is_none());
    }

    #[test]
    fn test_count_combos() {
        let range: Range = "AA".parse().unwrap();
        assert_eq!(count_combos(&range), 6); // 4 choose 2 = 6
    }

    #[test]
    fn test_card_pair_to_matrix_deuce_deuce() {
        // 2c=0, 2d=1 -> rank 0 -> matrix row 12
        let (r, c, suited) = card_pair_to_matrix(0, 1);
        assert_eq!(r, 12);
        assert_eq!(c, 12);
        assert!(!suited);
    }

    #[test]
    fn test_card_pair_to_matrix_symmetric() {
        // Same result regardless of card order.
        let (r1, c1, s1) = card_pair_to_matrix(51, 47);
        let (r2, c2, s2) = card_pair_to_matrix(47, 51);
        assert_eq!((r1, c1, s1), (r2, c2, s2));
    }

    #[test]
    fn test_parse_board_flop() {
        let (flop, turn, river, state) =
            parse_board(&["Ah".into(), "Kd".into(), "7c".into()]).unwrap();
        assert_eq!(state, BoardState::Flop);
        assert_eq!(turn, NOT_DEALT);
        assert_eq!(river, NOT_DEALT);
        // flop_from_str sorts, so just check all three are valid cards.
        assert!(flop.iter().all(|&c| c < 52));
    }

    #[test]
    fn test_parse_board_turn() {
        let (_, turn, river, state) =
            parse_board(&["Ah".into(), "Kd".into(), "7c".into(), "2s".into()]).unwrap();
        assert_eq!(state, BoardState::Turn);
        assert!(turn < 52);
        assert_eq!(river, NOT_DEALT);
    }

    #[test]
    fn test_parse_board_river() {
        let (_, turn, river, state) = parse_board(&[
            "Ah".into(),
            "Kd".into(),
            "7c".into(),
            "2s".into(),
            "Ts".into(),
        ])
        .unwrap();
        assert_eq!(state, BoardState::River);
        assert!(turn < 52);
        assert!(river < 52);
    }

    #[test]
    fn test_parse_board_invalid_count() {
        assert!(parse_board(&["Ah".into(), "Kd".into()]).is_err());
    }

    #[test]
    fn test_build_game_flop() {
        let config = PostflopConfig::default();
        let board = vec!["Td".into(), "9d".into(), "6h".into()];
        let game = build_game(&config, &board, &None, &None);
        assert!(game.is_ok(), "build_game failed: {:?}", game.err());
    }

    #[test]
    fn test_solve_street_completes() {
        let state = Arc::new(PostflopState::default());
        let config = PostflopConfig {
            oop_range: "AA".to_string(),
            ip_range: "KK".to_string(),
            pot: 30,
            effective_stack: 170,
            oop_bet_sizes: "33%".to_string(),
            oop_raise_sizes: "a".to_string(),
            ip_bet_sizes: "33%".to_string(),
            ip_raise_sizes: "a".to_string(),
        };
        *state.config.write() = config;

        // River board — single street, fast even in debug mode
        let board = vec!["Td".into(), "9d".into(), "6h".into(), "2c".into(), "3s".into()];
        let result =
            postflop_solve_street_core(&state, board, Some(2), Some(f32::MAX));
        assert!(result.is_ok(), "solve_street failed: {:?}", result.err());

        // Wait for the background thread to finish (generous timeout for debug builds).
        for _ in 0..600 {
            if state.solve_complete.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        assert!(state.solve_complete.load(Ordering::Relaxed));
        assert!(!state.solving.load(Ordering::Relaxed));
        assert!(state.current_iteration.load(Ordering::Relaxed) > 0);
        assert!(state.game.lock().is_some());
        assert!(state.matrix_snapshot.read().is_some());
    }

    #[test]
    fn test_solve_street_rejects_double_solve() {
        let state = Arc::new(PostflopState::default());
        state.solving.store(true, Ordering::Relaxed);

        let board = vec!["Td".into(), "9d".into(), "6h".into()];
        let result =
            postflop_solve_street_core(&state, board, None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already in progress"));
    }

    #[test]
    fn test_play_action_no_game() {
        let state = PostflopState::default();
        let result = postflop_play_action_core(&state, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No game loaded"));
    }

    #[test]
    fn test_play_action_after_solve() {
        let state = Arc::new(PostflopState::default());
        let config = PostflopConfig {
            oop_range: "AA".to_string(),
            ip_range: "KK".to_string(),
            pot: 30,
            effective_stack: 170,
            oop_bet_sizes: "33%".to_string(),
            oop_raise_sizes: "a".to_string(),
            ip_bet_sizes: "33%".to_string(),
            ip_raise_sizes: "a".to_string(),
        };
        *state.config.write() = config;

        let board = vec!["Td".into(), "9d".into(), "6h".into(), "2c".into(), "3s".into()];
        postflop_solve_street_core(&state, board, Some(2), Some(f32::MAX)).unwrap();

        // Wait for the background thread to finish.
        for _ in 0..600 {
            if state.solve_complete.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(state.solve_complete.load(Ordering::Relaxed));

        // Play first available action.
        let result = postflop_play_action_core(&state, 0).unwrap();
        // First action should yield a terminal, chance, or player node.
        assert!(result.is_terminal || result.matrix.is_some() || result.is_chance);
    }

    #[test]
    fn test_get_progress_before_solve() {
        let state = PostflopState::default();
        let progress = postflop_get_progress_core(&state);
        assert_eq!(progress.iteration, 0);
        assert_eq!(progress.max_iterations, 0);
        assert!(!progress.is_complete);
        assert!(progress.matrix.is_none());
    }

    #[test]
    fn test_close_street_no_game() {
        let state = PostflopState::default();
        let result = postflop_close_street_core(&state, vec![0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No game loaded"));
    }

    #[test]
    fn test_close_street_filters_ranges() {
        let state = Arc::new(PostflopState::default());
        let config = PostflopConfig {
            oop_range: "AA".to_string(),
            ip_range: "KK".to_string(),
            pot: 30,
            effective_stack: 170,
            oop_bet_sizes: "33%".to_string(),
            oop_raise_sizes: "a".to_string(),
            ip_bet_sizes: "33%".to_string(),
            ip_raise_sizes: "a".to_string(),
        };
        postflop_set_config_core(&state, config).unwrap();

        let board = vec!["Td".into(), "9d".into(), "6h".into(), "2c".into(), "3s".into()];
        postflop_solve_street_core(&state, board, Some(5), Some(f32::MAX)).unwrap();

        // Wait for background solve to finish.
        for _ in 0..600 {
            if state.solve_complete.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(state.solve_complete.load(Ordering::Relaxed));

        // Find a non-fold action from the matrix snapshot.
        let progress = postflop_get_progress_core(&state);
        let matrix = progress.matrix.unwrap();
        let action_idx = matrix
            .actions
            .iter()
            .enumerate()
            .find(|(_, a)| a.action_type != "fold")
            .map(|(i, _)| i)
            .expect("should have at least one non-fold action");

        let result = postflop_close_street_core(&state, vec![action_idx]).unwrap();
        assert_eq!(result.filtered_oop_range.len(), 1326);
        assert_eq!(result.filtered_ip_range.len(), 1326);
        assert!(result.pot > 0);

        // Verify weights were stored in state.
        assert!(state.filtered_oop_weights.read().is_some());
        assert!(state.filtered_ip_weights.read().is_some());
    }

    #[test]
    fn test_cache_key_deterministic() {
        let config = PostflopConfig::default();
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qs".to_string()];
        let k1 = compute_cache_key(&config, &board, &[]);
        let k2 = compute_cache_key(&config, &board, &[]);
        assert_eq!(k1, k2);

        // Different board produces a different key.
        let board2 = vec!["Ah".to_string(), "Kd".to_string(), "Js".to_string()];
        let k3 = compute_cache_key(&config, &board2, &[]);
        assert_ne!(k1, k3);
    }

    #[test]
    fn test_cache_key_varies_with_actions() {
        let config = PostflopConfig::default();
        let board = vec!["Ah".to_string(), "Kd".to_string(), "Qs".to_string()];
        let k1 = compute_cache_key(&config, &board, &[]);
        let k2 = compute_cache_key(&config, &board, &[vec![0, 1]]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_write_cache_file_header() {
        let game = PostFlopGame::default();
        let dir = std::env::temp_dir().join("postflop_cache_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_header.bin");

        write_cache_file(&path, &game, 1.5_f32, 42).unwrap();

        let data = std::fs::read(&path).unwrap();
        // Header: magic(4) + version(4) + exploitability(4) + iterations(4) + 4×len(8) = 48 bytes
        assert!(data.len() >= 48);

        let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
        assert_eq!(magic, CACHE_MAGIC);

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        assert_eq!(version, CACHE_VERSION);

        let exp = f32::from_le_bytes(data[8..12].try_into().unwrap());
        assert!((exp - 1.5).abs() < f32::EPSILON);

        let iters = u32::from_le_bytes(data[12..16].try_into().unwrap());
        assert_eq!(iters, 42);

        // All storage lens should be 0 for a default game.
        for i in 0..4 {
            let offset = 16 + i * 8;
            let len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            assert_eq!(len, 0);
        }

        // Total file size: 48-byte header + 0 body = 48 bytes.
        assert_eq!(data.len(), 48);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_set_cache_dir() {
        let state = PostflopState::default();
        assert!(state.cache_dir.read().is_none());

        postflop_set_cache_dir_core(&state, Some("/tmp/test_cache".to_string()));
        assert_eq!(
            state.cache_dir.read().as_deref(),
            Some(Path::new("/tmp/test_cache"))
        );

        postflop_set_cache_dir_core(&state, None);
        assert!(state.cache_dir.read().is_none());
    }

    #[test]
    fn test_cache_round_trip() {
        let state = Arc::new(PostflopState::default());
        let config = PostflopConfig {
            oop_range: "AA,KK".to_string(),
            ip_range: "QQ,JJ".to_string(),
            pot: 100,
            effective_stack: 200,
            oop_bet_sizes: "50%,a".to_string(),
            oop_raise_sizes: "a".to_string(),
            ip_bet_sizes: "50%,a".to_string(),
            ip_raise_sizes: "a".to_string(),
        };
        postflop_set_config_core(&state, config).unwrap();

        let tmp = tempfile::tempdir().unwrap();
        postflop_set_cache_dir_core(
            &state,
            Some(tmp.path().to_string_lossy().to_string()),
        );

        // River board for fast solve in debug mode.
        let board = vec![
            "Ah".to_string(),
            "Kd".to_string(),
            "Qc".to_string(),
            "2s".to_string(),
            "3c".to_string(),
        ];

        postflop_solve_street_core(&state, board.clone(), Some(5), Some(f32::MAX))
            .unwrap();

        // Wait for solve to complete.
        for _ in 0..600 {
            if state.solve_complete.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(state.solve_complete.load(Ordering::Relaxed));

        // Check cache exists.
        let info =
            postflop_check_cache_core(&state, board.clone(), vec![]).unwrap();
        assert!(info.is_some(), "Cache should exist after solve");
        let info = info.unwrap();
        assert!(info.iterations > 0);

        // Clear game state.
        *state.game.lock() = None;
        state.solve_complete.store(false, Ordering::Relaxed);

        // Load from cache.
        let matrix =
            postflop_load_cached_core(&state, board, vec![]).unwrap();
        assert!(!matrix.actions.is_empty());
        assert!(state.solve_complete.load(Ordering::Relaxed));
    }
}
