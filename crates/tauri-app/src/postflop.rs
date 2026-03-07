use parking_lot::{Mutex, RwLock};
use range_solver::action_tree::Action;
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{card_to_string, NOT_DEALT};
use range_solver::interface::Game;
use range_solver::range::Range;
use range_solver::PostFlopGame;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

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
            ip_range: "JJ-66,AQs-ATs,AQo,KQs".to_string(),
            pot: 30,
            effective_stack: 170,
            oop_bet_sizes: "25%,33%,75%".to_string(),
            oop_raise_sizes: "a".to_string(),
            ip_bet_sizes: "25%,33%,75%".to_string(),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostflopActionInfo {
    pub index: usize,
    pub label: String,
    pub action_type: String,
    pub amount: Option<i32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PostflopMatrixCell {
    pub hand: String,
    pub suited: bool,
    pub pair: bool,
    pub probabilities: Vec<f32>,
    pub combo_count: usize,
    pub ev: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PostflopStrategyMatrix {
    pub cells: Vec<Vec<PostflopMatrixCell>>,
    pub actions: Vec<PostflopActionInfo>,
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

/// Converts a range-solver `Action` to a serializable `PostflopActionInfo`.
fn action_to_info(action: &Action, index: usize) -> PostflopActionInfo {
    match action {
        Action::Fold => PostflopActionInfo {
            index,
            label: "Fold".to_string(),
            action_type: "fold".to_string(),
            amount: None,
        },
        Action::Check => PostflopActionInfo {
            index,
            label: "Check".to_string(),
            action_type: "check".to_string(),
            amount: None,
        },
        Action::Call => PostflopActionInfo {
            index,
            label: "Call".to_string(),
            action_type: "call".to_string(),
            amount: None,
        },
        Action::Bet(amt) => PostflopActionInfo {
            index,
            label: format!("Bet {amt}"),
            action_type: "bet".to_string(),
            amount: Some(*amt),
        },
        Action::Raise(amt) => PostflopActionInfo {
            index,
            label: format!("Raise {amt}"),
            action_type: "raise".to_string(),
            amount: Some(*amt),
        },
        Action::AllIn(amt) => PostflopActionInfo {
            index,
            label: format!("All-in {amt}"),
            action_type: "allin".to_string(),
            amount: Some(*amt),
        },
        _ => PostflopActionInfo {
            index,
            label: format!("{action}"),
            action_type: "other".to_string(),
            amount: None,
        },
    }
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
        }
    }
}

// ---------------------------------------------------------------------------
// build_strategy_matrix
// ---------------------------------------------------------------------------

/// Builds a 13x13 strategy matrix from the current game state.
///
/// The game must be at a non-terminal, non-chance node with memory allocated.
pub fn build_strategy_matrix(game: &PostFlopGame) -> PostflopStrategyMatrix {
    let player = game.current_player();
    let actions = game.available_actions();
    let strategy = game.strategy();
    let private_cards = game.private_cards(player);
    let num_hands = game.num_private_hands(player);
    let num_actions = actions.len();

    let action_infos: Vec<PostflopActionInfo> = actions
        .iter()
        .enumerate()
        .map(|(i, a)| action_to_info(a, i))
        .collect();

    // Initialize the 13x13 matrix with zeroed probability accumulators.
    let mut prob_sums = vec![vec![vec![0.0f64; num_actions]; 13]; 13];
    let mut combo_counts = vec![vec![0usize; 13]; 13];

    for (hand_idx, &(c1, c2)) in private_cards.iter().enumerate() {
        let (row, col, _) = card_pair_to_matrix(c1, c2);
        combo_counts[row][col] += 1;
        for action_idx in 0..num_actions {
            let prob = strategy[action_idx * num_hands + hand_idx];
            prob_sums[row][col][action_idx] += prob as f64;
        }
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
                    PostflopMatrixCell {
                        hand: label,
                        suited,
                        pair,
                        probabilities,
                        combo_count: count,
                        ev: None,
                    }
                })
                .collect()
        })
        .collect();

    // Board cards.
    let board: Vec<String> = game
        .current_board()
        .iter()
        .filter_map(|&c| {
            if c == NOT_DEALT {
                None
            } else {
                // INVARIANT: board cards are valid (0..52)
                card_to_string(c).ok()
            }
        })
        .collect();

    // Pot and stacks.
    let tree_config = game.tree_config();
    let bet_amounts = game.total_bet_amount();
    let pot = tree_config.starting_pot + bet_amounts[0] + bet_amounts[1];
    let stacks = [
        tree_config.effective_stack - bet_amounts[0],
        tree_config.effective_stack - bet_amounts[1],
    ];

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
    state: tauri::State<'_, PostflopState>,
    config: PostflopConfig,
) -> Result<PostflopConfigSummary, String> {
    postflop_set_config_core(&state, config)
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
        let info = action_to_info(&Action::Fold, 0);
        assert_eq!(info.label, "Fold");
        assert_eq!(info.action_type, "fold");
        assert!(info.amount.is_none());

        let info = action_to_info(&Action::Bet(50), 1);
        assert_eq!(info.label, "Bet 50");
        assert_eq!(info.action_type, "bet");
        assert_eq!(info.amount, Some(50));

        let info = action_to_info(&Action::AllIn(200), 2);
        assert_eq!(info.label, "All-in 200");
        assert_eq!(info.action_type, "allin");
        assert_eq!(info.amount, Some(200));
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
}
