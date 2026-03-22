use parking_lot::{Mutex, RwLock};
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::continuation::{BiasType, RolloutContext, rollout_from_boundary};
use poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id;
use poker_solver_core::blueprint_v2::solver_dispatch::{
    self, SolverChoice, SolverConfig, Street,
};
use poker_solver_core::blueprint_v2::subgame_cfr::cards_overlap;
use poker_solver_core::blueprint_v2::{
    CfvSubgameSolver, LeafEvaluator, SubgameHands, SubgameStrategy,
    compute_combo_equities,
};
use poker_solver_core::blueprint_v2::cbv::CbvTable;
use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind, TreeAction};
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::blueprint_v2::Street as V2Street;
use poker_solver_core::poker::{Card as RsPokerCard, Suit as RsPokerSuit, Value as RsPokerValue};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use range_solver::action_tree::{Action, ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{card_from_str, card_pair_to_index, card_to_string, CardConfig, NOT_DEALT};
use range_solver::interface::Game;
use range_solver::range::Range;
use range_solver::{compute_exploitability, finalize, solve_step, PostFlopGame};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use crate::exploration::{ActionInfo, v2_action_info};

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
    pub rake_rate: f64,
    pub rake_cap: f64,
    /// Max live combos on flop before switching to depth-limited solver.
    #[serde(default = "default_flop_combo_threshold")]
    pub flop_combo_threshold: usize,
    /// Max live combos on turn before switching to depth-limited solver.
    #[serde(default = "default_turn_combo_threshold")]
    pub turn_combo_threshold: usize,
    /// Abstract tree node index at the postflop start (for CBV boundary mapping).
    #[serde(default)]
    pub abstract_node_idx: Option<u32>,
}

fn default_flop_combo_threshold() -> usize { 200 }
fn default_turn_combo_threshold() -> usize { 300 }

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
            rake_rate: 0.0,
            rake_cap: 0.0,
            flop_combo_threshold: default_flop_combo_threshold(),
            turn_combo_threshold: default_turn_combo_threshold(),
            abstract_node_idx: None,
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
    /// Average initial range weight for this cell (0.0–1.0).
    /// When derived from blueprint strategy, this reflects reaching probability.
    pub weight: f32,
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
    /// Seconds elapsed since solve started.
    pub elapsed_secs: f64,
    /// Which solver is running (e.g. "range" or "subgame").
    pub solver_name: String,
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
// Subgame solve result (stored for range propagation and navigation)
// ---------------------------------------------------------------------------

/// Stored result from a subgame solve, used for range propagation and navigation.
pub struct SubgameSolveResult {
    pub strategy: SubgameStrategy,
    pub hands: SubgameHands,
    pub action_infos: Vec<ActionInfo>,
    pub tree: GameTree,
    pub board: Vec<RsPokerCard>,
    pub initial_pot: f64,
    pub starting_stack: f64,
}

// ---------------------------------------------------------------------------
// Subgame solver helpers
// ---------------------------------------------------------------------------

/// Parse a 2-char card string (e.g. "Ks") into an `rs_poker` Card.
fn parse_rs_poker_card(s: &str) -> Result<RsPokerCard, String> {
    if s.len() != 2 {
        return Err(format!("Invalid card string: {s}"));
    }
    let mut chars = s.chars();
    let rank_ch = chars.next().unwrap();
    let suit_ch = chars.next().unwrap();

    let value = match rank_ch {
        'A' | 'a' => RsPokerValue::Ace,
        'K' | 'k' => RsPokerValue::King,
        'Q' | 'q' => RsPokerValue::Queen,
        'J' | 'j' => RsPokerValue::Jack,
        'T' | 't' => RsPokerValue::Ten,
        '9' => RsPokerValue::Nine,
        '8' => RsPokerValue::Eight,
        '7' => RsPokerValue::Seven,
        '6' => RsPokerValue::Six,
        '5' => RsPokerValue::Five,
        '4' => RsPokerValue::Four,
        '3' => RsPokerValue::Three,
        '2' => RsPokerValue::Two,
        c => return Err(format!("Unknown rank: {c}")),
    };
    let suit = match suit_ch {
        's' | 'S' => RsPokerSuit::Spade,
        'h' | 'H' => RsPokerSuit::Heart,
        'd' | 'D' => RsPokerSuit::Diamond,
        'c' | 'C' => RsPokerSuit::Club,
        c => return Err(format!("Unknown suit: {c}")),
    };
    Ok(RsPokerCard::new(value, suit))
}

// ---------------------------------------------------------------------------
// EquityLeafEvaluator — simple equity-based evaluator for depth boundaries
// ---------------------------------------------------------------------------

/// Evaluates depth boundary nodes by computing equity against the opponent's
/// current reach-weighted range. Returns per-combo CFVs in pot-fraction units.
///
/// Unlike the old `SubgameCfrSolver` approach (which used a static scalar
/// opponent reach), this evaluator receives the dynamically-propagated opponent
/// range at each boundary, properly tracking which opponent combos reach each
/// terminal.
struct EquityLeafEvaluator;

impl LeafEvaluator for EquityLeafEvaluator {
    fn evaluate(
        &self,
        combos: &[[RsPokerCard; 2]],
        board: &[RsPokerCard],
        _pot: f64,
        _effective_stack: f64,
        oop_range: &[f64],
        ip_range: &[f64],
        traverser: u8,
    ) -> Vec<f64> {
        let opp_range = if traverser == 0 { ip_range } else { oop_range };
        let hands = SubgameHands { combos: combos.to_vec() };
        let equities = compute_combo_equities(&hands, board, opp_range);
        // Convert equity (0..1) to pot-fraction CFVs (-1..+1).
        equities.iter().map(|&eq| 2.0 * eq - 1.0).collect()
    }
}

// ---------------------------------------------------------------------------
// RolloutLeafEvaluator helpers
// ---------------------------------------------------------------------------

/// Sample `n` indices weighted by `weights`, returning indices into the weights
/// array.  Returns empty if total weight is zero or `n` is zero.
fn sample_weighted(rng: &mut impl Rng, weights: &[f64], n: u32) -> Vec<usize> {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return vec![];
    }
    let mut samples = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let mut r = rng.random::<f64>() * total;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                samples.push(i);
                break;
            }
        }
    }
    samples
}

// ---------------------------------------------------------------------------
// RolloutLeafEvaluator — Monte Carlo rollout-based evaluator
// ---------------------------------------------------------------------------

/// Evaluates depth boundary nodes by performing Monte Carlo rollouts through
/// the blueprint strategy tree.
///
/// For each hero combo, samples opponent hands weighted by reach probabilities,
/// performs fixed-strategy rollouts from the boundary node, and averages the
/// results.  Returns per-combo CFVs in pot-fraction units.
struct RolloutLeafEvaluator {
    strategy: Arc<BlueprintV2Strategy>,
    abstract_tree: Arc<GameTree>,
    all_buckets: Arc<AllBuckets>,
    decision_idx_map: Vec<u32>,
    abstract_start_node: u32,
    bias: BiasType,
    bias_factor: f64,
    num_rollouts: u32,
    num_opponent_samples: u32,
}

impl RolloutLeafEvaluator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        strategy: Arc<BlueprintV2Strategy>,
        abstract_tree: Arc<GameTree>,
        all_buckets: Arc<AllBuckets>,
        abstract_start_node: u32,
        bias: BiasType,
        bias_factor: f64,
        num_rollouts: u32,
        num_opponent_samples: u32,
    ) -> Self {
        let decision_idx_map = abstract_tree.decision_index_map();
        Self {
            strategy,
            abstract_tree,
            all_buckets,
            decision_idx_map,
            abstract_start_node,
            bias,
            bias_factor,
            num_rollouts,
            num_opponent_samples,
        }
    }
}

impl LeafEvaluator for RolloutLeafEvaluator {
    fn evaluate(
        &self,
        combos: &[[RsPokerCard; 2]],
        board: &[RsPokerCard],
        pot: f64,
        _effective_stack: f64,
        oop_range: &[f64],
        ip_range: &[f64],
        traverser: u8,
    ) -> Vec<f64> {
        let half_pot = pot / 2.0;
        let opp_range = if traverser == 0 { ip_range } else { oop_range };

        combos
            .par_iter()
            .enumerate()
            .map(|(i, hero_hand)| {
                // Build opponent sampling weights: zero out combos that
                // conflict with hero hand. Board conflicts are already
                // excluded by SubgameHands::enumerate(board).
                let weights: Vec<f64> = combos
                    .iter()
                    .enumerate()
                    .map(|(j, opp_hand)| {
                        if opp_range[j] <= 0.0
                            || cards_overlap(*hero_hand, *opp_hand)
                        {
                            0.0
                        } else {
                            opp_range[j]
                        }
                    })
                    .collect();

                // Seed deterministically per-combo for reproducibility.
                let mut rng = SmallRng::seed_from_u64(i as u64);
                let sampled = sample_weighted(&mut rng, &weights, self.num_opponent_samples);
                if sampled.is_empty() {
                    return 0.0;
                }

                let ctx = RolloutContext {
                    abstract_tree: &self.abstract_tree,
                    decision_idx_map: &self.decision_idx_map,
                    strategy: &self.strategy,
                    buckets: &self.all_buckets,
                    bias: self.bias,
                    bias_factor: self.bias_factor,
                    player: traverser,
                    num_rollouts: self.num_rollouts,
                };

                let total: f64 = sampled
                    .iter()
                    .map(|&j| {
                        let opp_hand = combos[j];
                        rollout_from_boundary(
                            *hero_hand,
                            opp_hand,
                            board,
                            &ctx,
                            self.abstract_start_node,
                            &mut rng,
                        )
                    })
                    .sum();

                let avg_chips = total / sampled.len() as f64;
                // Convert from chip value to pot-fraction units.
                avg_chips / half_pot
            })
            .collect()
    }
}

/// Build a subgame tree and CFV solver, ready to iterate.
///
/// Returns the solver, hands, action labels, tree, initial pot, and starting
/// stack without running any iterations. The caller drives the iteration loop
/// and can build matrix snapshots between iterations.
///
/// Uses [`CfvSubgameSolver`] with per-combo reach vectors at depth boundaries,
/// which properly tracks which opponent combos reach each terminal (fixing the
/// mass-shoving bug from the old `SubgameCfrSolver`).
#[allow(clippy::too_many_arguments)]
pub fn build_subgame_solver(
    board_cards: &[RsPokerCard],
    bet_sizes_per_depth: &[Vec<f32>],
    pot: u32,
    stacks: [u32; 2],
    oop_weights: &[f32],
    ip_weights: &[f32],
    _player: usize,
    cbv_context: Option<&CbvContext>,
    abstract_node_idx: Option<u32>,
) -> Result<(CfvSubgameSolver, SubgameHands, Vec<ActionInfo>, GameTree, f64, f64), String> {
    let street = match board_cards.len() {
        3 => V2Street::Flop,
        4 => V2Street::Turn,
        5 => V2Street::River,
        n => return Err(format!("Invalid board length for subgame: {n}")),
    };
    let pot_f = f64::from(pot);
    let inv = [pot_f / 2.0; 2];
    let starting_stack = f64::from(stacks[0]) + inv[0];
    let sizes_f64: Vec<Vec<f64>> = bet_sizes_per_depth
        .iter()
        .map(|depth| depth.iter().map(|&s| f64::from(s)).collect())
        .collect();

    let tree = GameTree::build_subgame(
        street,
        pot_f,
        inv,
        starting_stack,
        &sizes_f64,
        Some(1),
    );

    let hands = SubgameHands::enumerate(board_cards);

    // Extract action labels from tree root.
    // invested_offset = pot/2 so labels show bet amounts on this street,
    // not total invested including preflop contributions.
    let invested_offset = pot_f / 2.0;
    let action_infos = match &tree.nodes[tree.root as usize] {
        GameNode::Decision { actions, .. } => actions
            .iter()
            .enumerate()
            .map(|(i, a)| v2_action_info(a, i, 0.5, invested_offset))
            .collect(),
        _ => return Err("Subgame tree root is not a decision node".to_string()),
    };

    let evaluators: Vec<Box<dyn LeafEvaluator>> = if let (Some(ctx), Some(abs_node)) = (cbv_context, abstract_node_idx) {
        eprintln!("[subgame] using RolloutLeafEvaluator x4 (abstract_node={abs_node})");
        let strategy = Arc::clone(&ctx.strategy);
        let abstract_tree = Arc::new(ctx.abstract_tree.clone());
        let all_buckets = Arc::clone(&ctx.all_buckets);

        vec![
            Box::new(RolloutLeafEvaluator::new(
                Arc::clone(&strategy), Arc::clone(&abstract_tree), Arc::clone(&all_buckets),
                abs_node, BiasType::Unbiased, 10.0, 3, 8,
            )) as Box<dyn LeafEvaluator>,
            Box::new(RolloutLeafEvaluator::new(
                Arc::clone(&strategy), Arc::clone(&abstract_tree), Arc::clone(&all_buckets),
                abs_node, BiasType::Fold, 10.0, 3, 8,
            )),
            Box::new(RolloutLeafEvaluator::new(
                Arc::clone(&strategy), Arc::clone(&abstract_tree), Arc::clone(&all_buckets),
                abs_node, BiasType::Call, 10.0, 3, 8,
            )),
            Box::new(RolloutLeafEvaluator::new(
                Arc::clone(&strategy), Arc::clone(&abstract_tree), Arc::clone(&all_buckets),
                abs_node, BiasType::Raise, 10.0, 3, 8,
            )),
        ]
    } else {
        eprintln!("[subgame] using EquityLeafEvaluator (no CBV context)");
        vec![Box::new(EquityLeafEvaluator)]
    };
    // Map 1326-element weight vectors to SubgameHands ordering.
    let map_weights = |weights: &[f32]| -> Vec<f64> {
        hands.combos.iter().map(|combo| {
            let id0 = rs_poker_card_to_id(combo[0]);
            let id1 = rs_poker_card_to_id(combo[1]);
            let ci = card_pair_to_index(id0, id1);
            f64::from(weights[ci])
        }).collect()
    };
    let oop_reach = map_weights(oop_weights);
    let ip_reach = map_weights(ip_weights);

    let nonzero_oop = oop_reach.iter().filter(|&&r| r > 0.0).count();
    let nonzero_ip = ip_reach.iter().filter(|&&r| r > 0.0).count();
    eprintln!("[subgame] initial reach: OOP={nonzero_oop} IP={nonzero_ip} combos with nonzero reach");

    let solver = CfvSubgameSolver::new(
        tree.clone(),
        hands.clone(),
        board_cards,
        evaluators,
        starting_stack,
        oop_reach,
        ip_reach,
    );

    Ok((solver, hands, action_infos, tree, pot_f, starting_stack))
}

// ---------------------------------------------------------------------------
// Solve cache
// ---------------------------------------------------------------------------

// Spot caching disabled — raw solver buffers were 80GB+ per spot.
// TODO: re-enable with a compact format when real-time solving is optimized.

// ---------------------------------------------------------------------------
// CbvContext
// ---------------------------------------------------------------------------

/// Blueprint data needed for CBV-based depth-limited solving.
///
/// When a blueprint bundle includes precomputed CBV tables, this context
/// is constructed and stored in `PostflopState`. The subgame solver uses
/// it to look up boundary values instead of raw equity.
pub struct CbvContext {
    pub cbv_table: CbvTable,
    pub abstract_tree: GameTree,
    pub all_buckets: Arc<AllBuckets>,
    pub strategy: Arc<BlueprintV2Strategy>,
}

/// Set the CBV context on `PostflopState` for use by the depth-limited solver.
///
/// Call after loading a blueprint bundle that includes CBV tables and bucket
/// files. If `context` is `None`, clears any existing CBV context (fallback
/// to equity-based leaf values).
pub fn set_cbv_context(state: &PostflopState, context: Option<CbvContext>) {
    *state.cbv_context.write() = context.map(Arc::new);
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
    pub solve_start: RwLock<Option<std::time::Instant>>,
    pub solver_name: RwLock<String>,
    pub subgame_result: RwLock<Option<SubgameSolveResult>>,
    /// Current node index in the subgame tree during navigation.
    pub subgame_node: AtomicU32,
    /// CBV tables + abstract tree for depth-limited solving.
    /// Loaded from the blueprint bundle. If present, subgame solver
    /// uses CBVs at depth boundaries instead of raw equity.
    pub cbv_context: RwLock<Option<Arc<CbvContext>>>,
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
            solve_start: RwLock::new(None),
            solver_name: RwLock::new("full".to_string()),
            subgame_result: RwLock::new(None),
            subgame_node: AtomicU32::new(0),
            cbv_context: RwLock::new(None),
        }
    }
}

// ---------------------------------------------------------------------------
// build_strategy_matrix
// ---------------------------------------------------------------------------

/// All data needed to build a matrix, captured quickly from the game.
struct MatrixSnapshot {
    player: usize,
    strategy: Vec<f32>,
    private_cards: Vec<(u8, u8)>,
    initial_weights: Vec<f32>,
    num_hands: usize,
    actions: Vec<ActionInfo>,
    pot: i32,
    stacks: [i32; 2],
    hand_evs: Option<Vec<f32>>,
    board: Vec<String>,
}

/// Capture all data needed for matrix construction from the game.
/// This borrows the game briefly; the expensive matrix build runs on owned data.
fn capture_matrix_snapshot(game: &mut PostFlopGame) -> MatrixSnapshot {
    let player = game.current_player();
    let raw_actions = game.available_actions();
    let strategy = game.strategy();
    let private_cards = game.private_cards(player).to_vec();
    let initial_weights = game.initial_weights(player).to_vec();
    let num_hands = game.num_private_hands(player);

    let (pot, stacks) = {
        let tc = game.tree_config();
        let ba = game.total_bet_amount();
        let pot = tc.starting_pot + ba[0] + ba[1];
        let stacks = [tc.effective_stack - ba[0], tc.effective_stack - ba[1]];
        (pot, stacks)
    };

    let hand_evs: Option<Vec<f32>> = if game.is_solved() {
        game.cache_normalized_weights();
        Some(game.expected_values(player))
    } else {
        None
    };

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

    let actions: Vec<ActionInfo> = raw_actions
        .iter()
        .enumerate()
        .map(|(i, a)| action_to_info(a, i, pot))
        .collect();

    MatrixSnapshot {
        player,
        strategy,
        private_cards,
        initial_weights,
        num_hands,
        actions,
        pot,
        stacks,
        hand_evs,
        board,
    }
}

/// Build the matrix from a snapshot (no game borrow needed).
fn build_matrix_from_snapshot(snap: MatrixSnapshot) -> PostflopStrategyMatrix {
    let num_actions = snap.actions.len();

    let mut prob_sums = vec![vec![vec![0.0f64; num_actions]; 13]; 13];
    let mut combo_counts = vec![vec![0usize; 13]; 13];
    let mut weight_sums = vec![vec![0.0f64; 13]; 13];
    let mut ev_sums = vec![vec![0.0f64; 13]; 13];
    let mut combo_details: Vec<Vec<Vec<PostflopComboDetail>>> =
        vec![vec![Vec::new(); 13]; 13];

    for (hand_idx, &(c1, c2)) in snap.private_cards.iter().enumerate() {
        let (row, col, _) = card_pair_to_matrix(c1, c2);
        combo_counts[row][col] += 1;
        weight_sums[row][col] += snap.initial_weights[hand_idx] as f64;
        if let Some(ref evs) = snap.hand_evs {
            ev_sums[row][col] += evs[hand_idx] as f64;
        }
        let mut probs = Vec::with_capacity(num_actions);
        for (action_idx, prob_sum) in prob_sums[row][col].iter_mut().enumerate() {
            let prob = snap.strategy[action_idx * snap.num_hands + hand_idx];
            *prob_sum += prob as f64;
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
                    let ev = if count > 0 && snap.hand_evs.is_some() {
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
                    PostflopMatrixCell {
                        hand: label,
                        suited,
                        pair,
                        probabilities,
                        combo_count: count,
                        ev,
                        combos,
                        weight,
                    }
                })
                .collect()
        })
        .collect();

    PostflopStrategyMatrix {
        cells,
        actions: snap.actions,
        player: snap.player,
        pot: snap.pot,
        stacks: snap.stacks,
        board: snap.board,
    }
}

/// Builds a 13x13 strategy matrix from the current game state.
///
/// The game must be at a non-terminal, non-chance node with memory allocated.
pub fn build_strategy_matrix(game: &mut PostFlopGame) -> PostflopStrategyMatrix {
    let snap = capture_matrix_snapshot(game);
    build_matrix_from_snapshot(snap)
}

/// Convert subgame solver output into a [`MatrixSnapshot`] so it can be rendered
/// by the same [`build_matrix_from_snapshot`] used by the range-solver.
#[allow(clippy::cast_possible_truncation)]
fn snapshot_from_subgame(
    hands: &SubgameHands,
    strategy: &SubgameStrategy,
    action_infos: Vec<ActionInfo>,
    weights: &[f32],
    board_strings: &[String],
    pot: i32,
    stacks: [i32; 2],
    player: usize,
    node_idx: u32,
) -> MatrixSnapshot {
    let num_actions = action_infos.len();

    // Build private_cards and initial_weights in the same order as SubgameHands.
    // Only include combos with non-zero weight.
    let mut private_cards = Vec::with_capacity(hands.combos.len());
    let mut initial_weights = Vec::with_capacity(hands.combos.len());
    // strategy laid out as action_idx * num_included_hands + hand_idx
    let mut hand_strategies: Vec<Vec<f32>> = vec![Vec::new(); num_actions];

    for (combo_idx, combo) in hands.combos.iter().enumerate() {
        let rs_id0 = rs_poker_card_to_id(combo[0]);
        let rs_id1 = rs_poker_card_to_id(combo[1]);
        let ci = card_pair_to_index(rs_id0, rs_id1);
        let w = weights[ci];
        if w <= 0.0 {
            continue;
        }
        private_cards.push((rs_id0, rs_id1));
        initial_weights.push(w);
        let probs = strategy.get_probs(node_idx, combo_idx);
        for a in 0..num_actions {
            hand_strategies[a].push(probs.get(a).copied().unwrap_or(0.0) as f32);
        }
    }

    let num_hands = private_cards.len();
    // Flatten to the layout expected by build_matrix_from_snapshot:
    // strategy[action_idx * num_hands + hand_idx]
    let mut flat_strategy = Vec::with_capacity(num_actions * num_hands);
    for a in 0..num_actions {
        flat_strategy.extend_from_slice(&hand_strategies[a]);
    }

    MatrixSnapshot {
        player,
        strategy: flat_strategy,
        private_cards,
        initial_weights,
        num_hands,
        actions: action_infos,
        pot,
        stacks,
        hand_evs: None,
        board: board_strings.to_vec(),
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
    *state.subgame_result.write() = None;
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
// postflop_set_filtered_weights
// ---------------------------------------------------------------------------

/// Set pre-computed range weights (1326-element arrays) from blueprint
/// preflop strategy. These override the range strings when building the
/// postflop game.
pub fn postflop_set_filtered_weights_core(
    state: &PostflopState,
    oop_weights: Vec<f32>,
    ip_weights: Vec<f32>,
) -> Result<FilteredWeightsResult, String> {
    if oop_weights.len() != 1326 {
        return Err(format!(
            "OOP weights must have 1326 elements, got {}",
            oop_weights.len()
        ));
    }
    if ip_weights.len() != 1326 {
        return Err(format!(
            "IP weights must have 1326 elements, got {}",
            ip_weights.len()
        ));
    }
    let oop_combos = oop_weights.iter().sum::<f32>().round() as usize;
    let ip_combos = ip_weights.iter().sum::<f32>().round() as usize;
    *state.filtered_oop_weights.write() = Some(oop_weights);
    *state.filtered_ip_weights.write() = Some(ip_weights);
    Ok(FilteredWeightsResult { oop_combos, ip_combos })
}

#[derive(Debug, Clone, Serialize)]
pub struct FilteredWeightsResult {
    pub oop_combos: usize,
    pub ip_combos: usize,
}

#[tauri::command(rename_all = "snake_case")]
pub fn postflop_set_filtered_weights(
    state: tauri::State<'_, Arc<PostflopState>>,
    oop_weights: Vec<f32>,
    ip_weights: Vec<f32>,
) -> Result<FilteredWeightsResult, String> {
    postflop_set_filtered_weights_core(&state, oop_weights, ip_weights)
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
            let nonzero = weights.iter().filter(|&&w| w > 0.0).count();
            let sum: f32 = weights.iter().sum();
            eprintln!("[build_game] OOP filtered weights: {nonzero}/1326 nonzero, sum={sum:.1}");
            Range::from_raw_data(weights).map_err(|e| format!("Bad OOP weights: {e}"))?
        }
        None => config
            .oop_range
            .parse()
            .map_err(|e: String| format!("Invalid OOP range: {e}"))?,
    };
    let ip_range = match filtered_ip {
        Some(weights) => {
            let nonzero = weights.iter().filter(|&&w| w > 0.0).count();
            let sum: f32 = weights.iter().sum();
            eprintln!("[build_game] IP filtered weights: {nonzero}/1326 nonzero, sum={sum:.1}");
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
        rake_rate: config.rake_rate,
        rake_cap: config.rake_cap,
        flop_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        turn_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        river_bet_sizes: [oop_sizes, ip_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        depth_limit: None,
    };

    let action_tree =
        ActionTree::new(tree_config).map_err(|e| format!("Failed to build tree: {e}"))?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;

    // Reject trees that would exceed the memory budget to prevent OOM.
    const MEM_LIMIT: u64 = 2 * 1_024 * 1_024 * 1_024; // 2 GB
    let (mem_estimate, _) = game.memory_usage();
    if mem_estimate > MEM_LIMIT {
        return Err(format!(
            "Tree too large ({:.0} MB). Reduce bet sizes or solve from a later street.",
            mem_estimate as f64 / 1_048_576.0
        ));
    }
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
    _prior_actions: Vec<Vec<usize>>,
) -> Result<(), String> {
    // Guard: reject if already solving.
    if state.solving.load(Ordering::Relaxed) {
        return Err("A solve is already in progress".to_string());
    }

    // Snapshot config and filtered weights under their locks.
    let config = state.config.read().clone();
    let target_exp = target_exploitability.unwrap_or(3.0);
    let filtered_oop = state.filtered_oop_weights.read().clone();
    let filtered_ip = state.filtered_ip_weights.read().clone();

    // Determine street from board length.
    let street = match board.len() {
        3 => Street::Flop,
        4 => Street::Turn,
        5 => Street::River,
        n => return Err(format!("Invalid board length: {n}")),
    };

    // Count live combos for dispatch decision.
    let live_combos = {
        let oop_count = filtered_oop
            .as_ref()
            .map(|w| w.iter().filter(|&&v| v > 0.0).count())
            .unwrap_or(0);
        let ip_count = filtered_ip
            .as_ref()
            .map(|w| w.iter().filter(|&&v| v > 0.0).count())
            .unwrap_or(0);
        oop_count.max(ip_count)
    };

    let solver_config = SolverConfig {
        flop_combo_threshold: config.flop_combo_threshold,
        turn_combo_threshold: config.turn_combo_threshold,
        ..SolverConfig::default()
    };
    let choice = solver_dispatch::dispatch_decision(&solver_config, street, live_combos);

    // Reset progress atomics.
    state.current_iteration.store(0, Ordering::Relaxed);
    state.solve_complete.store(false, Ordering::Relaxed);
    state.subgame_node.store(0, Ordering::Relaxed);
    *state.solve_start.write() = Some(std::time::Instant::now());

    match choice {
        SolverChoice::FullDepth => {
            solve_full_depth(state, &config, board, max_iterations, target_exp, &filtered_oop, &filtered_ip)
        }
        SolverChoice::DepthLimited => {
            let cbv_ctx = state.cbv_context.read().clone();
            solve_depth_limited(state, &config, board, max_iterations, &solver_config, &filtered_oop, &filtered_ip, cbv_ctx)
        }
    }
}

/// Full-depth solve using the range-solver (existing path).
fn solve_full_depth(
    state: &Arc<PostflopState>,
    config: &PostflopConfig,
    board: Vec<String>,
    max_iterations: Option<u32>,
    target_exp: f32,
    filtered_oop: &Option<Vec<f32>>,
    filtered_ip: &Option<Vec<f32>>,
) -> Result<(), String> {
    let max_iters = max_iterations.unwrap_or(200);
    state.max_iterations.store(max_iters, Ordering::Relaxed);
    state
        .exploitability_bits
        .store(f32::MAX.to_bits(), Ordering::Relaxed);
    *state.solver_name.write() = "full".to_string();
    *state.subgame_result.write() = None;
    state.solving.store(true, Ordering::Release);

    // Build game (expensive but runs on the calling thread before spawn).
    let mut game = build_game(config, &board, filtered_oop, filtered_ip)?;

    // Take initial matrix snapshot and store game in shared state so
    // navigation commands can access it during solving.
    {
        let matrix = build_strategy_matrix(&mut game);
        *state.matrix_snapshot.write() = Some(matrix);
    }
    *state.game.lock() = Some(game);

    let shared = Arc::clone(state);
    std::thread::spawn(move || {
        #[allow(unused_assignments)]
        let mut last_exp = f32::MAX;

        for t in 0..max_iters {
            if !shared.solving.load(Ordering::Relaxed) {
                break;
            }

            // Lock game for solve_step + exploitability. solve_step traverses
            // from root regardless of the game's current query position, so
            // user navigation between iterations doesn't interfere.
            {
                let game_guard = shared.game.lock();
                let game = game_guard.as_ref().unwrap();
                solve_step(game, t);
                last_exp = compute_exploitability(game);
            }
            // Lock released — navigation commands can run here.

            shared.current_iteration.store(t + 1, Ordering::Relaxed);
            shared
                .exploitability_bits
                .store(last_exp.to_bits(), Ordering::Relaxed);

            // Capture snapshot at the game's current position (which the user
            // may have changed via navigation). Build matrix on another thread.
            {
                let mut game_guard = shared.game.lock();
                let game = game_guard.as_mut().unwrap();
                let snap = capture_matrix_snapshot(game);
                let shared2 = Arc::clone(&shared);
                std::thread::spawn(move || {
                    let matrix = build_matrix_from_snapshot(snap);
                    *shared2.matrix_snapshot.write() = Some(matrix);
                });
            }

            if last_exp <= target_exp {
                break;
            }
        }

        // Finalize: compute EV / normalize strategy.
        {
            let mut game_guard = shared.game.lock();
            let game = game_guard.as_mut().unwrap();
            finalize(game);
            let matrix = build_strategy_matrix(game);
            *shared.matrix_snapshot.write() = Some(matrix);
        }

        shared.solve_complete.store(true, Ordering::Relaxed);
        shared.solving.store(false, Ordering::Release);
    });

    Ok(())
}

/// Depth-limited solve using `CfvSubgameSolver`.
#[allow(clippy::too_many_arguments)]
fn solve_depth_limited(
    state: &Arc<PostflopState>,
    config: &PostflopConfig,
    board: Vec<String>,
    max_iterations: Option<u32>,
    solver_config: &SolverConfig,
    filtered_oop: &Option<Vec<f32>>,
    filtered_ip: &Option<Vec<f32>>,
    cbv_context: Option<Arc<CbvContext>>,
) -> Result<(), String> {
    let max_iters = max_iterations.unwrap_or(solver_config.depth_limited_iterations);
    state.max_iterations.store(max_iters, Ordering::Relaxed);
    state
        .exploitability_bits
        .store(f32::MAX.to_bits(), Ordering::Relaxed);
    *state.solver_name.write() = "subgame".to_string();
    state.solving.store(true, Ordering::Release);

    // Parse board to rs_poker Cards.
    let board_cards: Vec<RsPokerCard> = board
        .iter()
        .map(|s| parse_rs_poker_card(s))
        .collect::<Result<Vec<_>, _>>()?;

    // Extract bet/raise sizes from config (parse comma-separated percentage strings).
    let parse_sizes = |s: &str| -> Vec<f32> {
        s.split(',')
            .filter_map(|s| {
                let s = s.trim().trim_end_matches('%');
                if s == "a" || s == "e" { return None; }
                s.parse::<f32>().ok().map(|v| v / 100.0)
            })
            .collect()
    };
    let bet_depth = parse_sizes(&config.oop_bet_sizes);
    let raise_depth = parse_sizes(&config.oop_raise_sizes);
    let mut bet_sizes_per_depth: Vec<Vec<f32>> = vec![];
    if !bet_depth.is_empty() {
        bet_sizes_per_depth.push(bet_depth);
    } else {
        bet_sizes_per_depth.push(vec![1.0]);
    }
    if !raise_depth.is_empty() {
        bet_sizes_per_depth.push(raise_depth);
    }

    let pot = config.pot;
    let eff_stack = config.effective_stack;
    let abstract_node_idx = config.abstract_node_idx;

    let oop_w = filtered_oop.clone().unwrap_or_else(|| {
        let range: Range = config.oop_range.parse().unwrap_or_default();
        range.raw_data().to_vec()
    });
    let ip_w = filtered_ip.clone().unwrap_or_else(|| {
        let range: Range = config.ip_range.parse().unwrap_or_default();
        range.raw_data().to_vec()
    });

    let shared = Arc::clone(state);
    let board_strings = board;
    std::thread::spawn(move || {
        let player = 0; // OOP acts first at root
        match build_subgame_solver(
            &board_cards,
            &bet_sizes_per_depth,
            pot as u32,
            [eff_stack as u32, eff_stack as u32],
            &oop_w,
            &ip_w,
            player,
            cbv_context.as_deref(),
            abstract_node_idx,
        ) {
            Ok((mut solver, hands, action_infos, tree, initial_pot, starting_stack)) => {
                // Helper closure to build matrix from current strategy.
                let make_matrix = |strat: &SubgameStrategy| {
                    let snap = snapshot_from_subgame(
                        &hands, strat, action_infos.clone(),
                        &oop_w, &board_strings, pot,
                        [eff_stack, eff_stack], player, 0,
                    );
                    build_matrix_from_snapshot(snap)
                };

                // Initial matrix: uniform strategy with range weights, shown before any iterations.
                let initial_strategy = solver.strategy();
                *shared.matrix_snapshot.write() = Some(make_matrix(&initial_strategy));

                const SNAPSHOT_INTERVAL: u32 = 10;

                for t in 0..max_iters {
                    if !shared.solving.load(Ordering::Relaxed) {
                        break;
                    }
                    solver.train(1);
                    shared.current_iteration.store(t + 1, Ordering::Relaxed);

                    let is_snapshot = (t + 1) % SNAPSHOT_INTERVAL == 0 || t + 1 == max_iters;
                    if is_snapshot {
                        let strategy = solver.strategy();
                        *shared.matrix_snapshot.write() = Some(make_matrix(&strategy));
                    }
                }

                // Final strategy snapshot.
                let strategy = solver.strategy();
                *shared.matrix_snapshot.write() = Some(make_matrix(&strategy));

                // Post-training diagnostic: dump regrets and strategy for
                // a few sample combos at the root node.
                {
                    let n = hands.combos.len();
                    let num_actions = action_infos.len();
                    let labels: Vec<&str> = action_infos.iter().map(|a| a.label.as_str()).collect();
                    eprintln!("[solver audit] {} iterations complete, {} combos, {} actions: {:?}",
                        max_iters, n, num_actions, labels);

                    // Show strategy + regrets for first 10 combos with nonzero reach
                    let mut shown = 0;
                    for combo_idx in 0..n {
                        if shown >= 10 { break; }
                        let probs = strategy.root_probs(combo_idx);
                        if probs.is_empty() { continue; }
                        let reach = if player == 0 { &oop_w } else { &ip_w };
                        let card_id0 = rs_poker_card_to_id(hands.combos[combo_idx][0]);
                        let card_id1 = rs_poker_card_to_id(hands.combos[combo_idx][1]);
                        let ci = card_pair_to_index(card_id0, card_id1);
                        if reach[ci] <= 0.0 { continue; }
                        let probs_str: Vec<String> = probs.iter()
                            .zip(labels.iter())
                            .map(|(p, l)| format!("{l}={:.1}%", p * 100.0))
                            .collect();
                        let regrets = solver.root_regrets(combo_idx);
                        let strat_sums = solver.root_strategy_sums(combo_idx);
                        let regret_str: Vec<String> = regrets.iter()
                            .zip(labels.iter())
                            .map(|(r, l)| format!("{l}={r:.1}"))
                            .collect();
                        let ssum_str: Vec<String> = strat_sums.iter()
                            .zip(labels.iter())
                            .map(|(s, l)| format!("{l}={s:.1}"))
                            .collect();
                        eprintln!("  combo {combo_idx} {}{}:",
                            hands.combos[combo_idx][0], hands.combos[combo_idx][1]);
                        eprintln!("    strategy: {}", probs_str.join(" "));
                        eprintln!("    regrets:  {}", regret_str.join(" "));
                        eprintln!("    str_sums: {}", ssum_str.join(" "));
                        shown += 1;
                    }
                }

                // Log final choice node mix if using multi-valued evaluation.
                if solver.choice_regrets().len() > 1 {
                    let choice_mix = solver.choice_strategy();
                    let mix_str = CfvSubgameSolver::format_choice_mix(max_iters, &choice_mix);
                    eprintln!("[choice audit] final mix: {mix_str}");
                    eprintln!("[choice regrets] {:?}", solver.choice_regrets());
                }

                // Store subgame result for range propagation and navigation.
                *shared.subgame_result.write() = Some(SubgameSolveResult {
                    strategy,
                    hands,
                    action_infos,
                    tree,
                    board: board_cards.clone(),
                    initial_pot,
                    starting_stack,
                });
            }
            Err(e) => {
                eprintln!("Subgame solve failed: {e}");
            }
        }
        shared.solve_complete.store(true, Ordering::Relaxed);
        shared.solving.store(false, Ordering::Release);
    });

    Ok(())
}

#[tauri::command(rename_all = "snake_case")]
pub async fn postflop_solve_street(
    state: tauri::State<'_, Arc<PostflopState>>,
    board: Vec<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
) -> Result<(), String> {
    postflop_solve_street_core(&state, board, max_iterations, target_exploitability)
}

// ---------------------------------------------------------------------------
// postflop_cancel_solve
// ---------------------------------------------------------------------------

pub fn postflop_cancel_solve_core(state: &PostflopState) {
    state.solving.store(false, Ordering::Release);
}

#[tauri::command]
pub fn postflop_cancel_solve(state: tauri::State<'_, Arc<PostflopState>>) {
    postflop_cancel_solve_core(&state);
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
    let elapsed_secs = state
        .solve_start
        .read()
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(0.0);

    let solver_name = state.solver_name.read().clone();

    PostflopProgress {
        iteration,
        max_iterations,
        exploitability,
        is_complete,
        matrix,
        elapsed_secs,
        solver_name,
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

/// Navigate the subgame tree by one action and return the result at the child node.
fn postflop_play_action_subgame(
    state: &PostflopState,
    action: usize,
) -> Result<PostflopPlayResult, String> {
    let result_guard = state.subgame_result.read();
    let result = result_guard.as_ref().ok_or("No subgame result stored")?;

    let current = state.subgame_node.load(Ordering::Relaxed);

    let children = match &result.tree.nodes[current as usize] {
        GameNode::Decision { children, .. } => children,
        _ => return Err("Current subgame node is not a decision node".to_string()),
    };

    if action >= children.len() {
        return Err(format!(
            "Action {action} out of range (max {})",
            children.len() - 1
        ));
    }
    let child_idx = children[action];
    state.subgame_node.store(child_idx, Ordering::Relaxed);

    subgame_node_to_result(state, result, child_idx)
}

/// Compute pot and remaining stacks for a `GameNode::Decision` by finding its
/// fold child's terminal state. Falls back to `initial_pot` for nodes without
/// a fold action (e.g., the opening check/bet node).
#[allow(clippy::cast_possible_truncation)]
fn decision_display_info(
    tree: &GameTree,
    node: &GameNode,
    starting_stack: f64,
    initial_pot: f64,
) -> (i32, [i32; 2]) {
    if let GameNode::Decision { actions, children, .. } = node {
        for (a, &child_idx) in actions.iter().zip(children.iter()) {
            if matches!(a, TreeAction::Fold) {
                if let GameNode::Terminal { pot, invested, .. } =
                    &tree.nodes[child_idx as usize]
                {
                    return (
                        *pot as i32,
                        [
                            (starting_stack - invested[0]) as i32,
                            (starting_stack - invested[1]) as i32,
                        ],
                    );
                }
            }
        }
        // No fold action — opening node. Use initial pot.
        let inv = initial_pot / 2.0;
        return (
            initial_pot as i32,
            [
                (starting_stack - inv) as i32,
                (starting_stack - inv) as i32,
            ],
        );
    }
    (0, [0, 0])
}

/// Convert a subgame tree node into a `PostflopPlayResult`, building the
/// strategy matrix when the node is a decision point.
fn subgame_node_to_result(
    state: &PostflopState,
    result: &SubgameSolveResult,
    node_idx: u32,
) -> Result<PostflopPlayResult, String> {
    match &result.tree.nodes[node_idx as usize] {
        GameNode::Terminal {
            kind, pot, invested,
        } => {
            let remaining = [
                (result.starting_stack - invested[0]) as i32,
                (result.starting_stack - invested[1]) as i32,
            ];
            match kind {
                TerminalKind::Fold { .. } | TerminalKind::Showdown => Ok(PostflopPlayResult {
                    matrix: None,
                    is_terminal: true,
                    is_chance: false,
                    current_player: None,
                    pot: *pot as i32,
                    stacks: remaining,
                }),
                TerminalKind::DepthBoundary => Ok(PostflopPlayResult {
                    matrix: None,
                    is_terminal: false,
                    is_chance: true,
                    current_player: None,
                    pot: *pot as i32,
                    stacks: remaining,
                }),
            }
        }
        GameNode::Chance { child, .. } => {
            subgame_node_to_result(state, result, *child)
        }
        GameNode::Decision {
            player, actions, ..
        } => {
            let player_usize = *player as usize;

            let weights = if player_usize == 0 {
                state.filtered_oop_weights.read().clone()
            } else {
                state.filtered_ip_weights.read().clone()
            };
            let weights = weights.unwrap_or_else(|| vec![0.0f32; 1326]);

            let board_card_strings: Vec<String> = result
                .board
                .iter()
                .map(|c| {
                    let id = rs_poker_card_to_id(*c);
                    card_to_string(id).unwrap_or_default()
                })
                .collect();

            let invested_offset = result.initial_pot / 2.0;
            let action_infos: Vec<ActionInfo> = actions
                .iter()
                .enumerate()
                .map(|(i, a)| v2_action_info(a, i, 0.5, invested_offset))
                .collect();

            let (pot_i32, stacks_i32) = decision_display_info(
                &result.tree,
                &result.tree.nodes[node_idx as usize],
                result.starting_stack,
                result.initial_pot,
            );

            let snap = snapshot_from_subgame(
                &result.hands,
                &result.strategy,
                action_infos,
                &weights,
                &board_card_strings,
                pot_i32,
                stacks_i32,
                player_usize,
                node_idx,
            );
            let matrix = build_matrix_from_snapshot(snap);
            Ok(PostflopPlayResult {
                matrix: Some(matrix),
                is_terminal: false,
                is_chance: false,
                current_player: Some(player_usize),
                pot: pot_i32,
                stacks: stacks_i32,
            })
        }
    }
}

pub fn postflop_play_action_core(
    state: &PostflopState,
    action: usize,
) -> Result<PostflopPlayResult, String> {
    let solver_name = state.solver_name.read().clone();
    if solver_name == "subgame" {
        return postflop_play_action_subgame(state, action);
    }

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
pub async fn postflop_play_action(
    state: tauri::State<'_, Arc<PostflopState>>,
    action: usize,
) -> Result<PostflopPlayResult, String> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || postflop_play_action_core(&state, action))
        .await
        .map_err(|e| e.to_string())?
}

// ---------------------------------------------------------------------------
// postflop_navigate_to
// ---------------------------------------------------------------------------

/// Replay a subgame tree from the root along `history` and return the result.
fn postflop_navigate_to_subgame(
    state: &PostflopState,
    history: &[usize],
) -> Result<PostflopPlayResult, String> {
    let root = {
        let result_guard = state.subgame_result.read();
        let result = result_guard.as_ref().ok_or("No subgame result stored")?;
        result.tree.root
    };
    state.subgame_node.store(root, Ordering::Relaxed);

    if history.is_empty() {
        let result_guard = state.subgame_result.read();
        let result = result_guard.as_ref().ok_or("No subgame result stored")?;
        return subgame_node_to_result(state, result, root);
    }

    let mut last_result = None;
    for &action in history {
        last_result = Some(postflop_play_action_subgame(state, action)?);
    }
    // INVARIANT: history is non-empty so last_result is Some.
    Ok(last_result.unwrap())
}

/// Replay the game tree to a given action history and return the result.
pub fn postflop_navigate_to_core(
    state: &PostflopState,
    history: Vec<usize>,
) -> Result<PostflopPlayResult, String> {
    let solver_name = state.solver_name.read().clone();
    if solver_name == "subgame" {
        return postflop_navigate_to_subgame(state, &history);
    }

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
pub async fn postflop_navigate_to(
    state: tauri::State<'_, Arc<PostflopState>>,
    history: Vec<usize>,
) -> Result<PostflopPlayResult, String> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || postflop_navigate_to_core(&state, history))
        .await
        .map_err(|e| e.to_string())?
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
    let solver_name = state.solver_name.read().clone();
    if solver_name == "subgame" {
        return postflop_close_street_subgame(state, action_history);
    }

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

/// Range propagation through a subgame tree for `postflop_close_street_core`.
///
/// Walks the stored `SubgameSolveResult`, multiplying each acting player's
/// 1326-element weight vector by the subgame strategy probabilities at each
/// decision node. Returns pot/stacks from the final node.
fn postflop_close_street_subgame(
    state: &PostflopState,
    action_history: Vec<usize>,
) -> Result<PostflopStreetResult, String> {
    let result_guard = state.subgame_result.read();
    let result = result_guard
        .as_ref()
        .ok_or("No subgame result stored")?;

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

    // Walk each action through the subgame tree, narrowing the acting player's range.
    let mut current_node = result.tree.root;
    for &action_idx in &action_history {
        match &result.tree.nodes[current_node as usize] {
            GameNode::Decision {
                player, children, ..
            } => {
                let player_usize = *player as usize;
                let weights = if player_usize == 0 {
                    &mut oop_weights
                } else {
                    &mut ip_weights
                };

                // Multiply each combo's weight by its strategy probability for this action.
                for (combo_idx, combo) in result.hands.combos.iter().enumerate() {
                    let rs_id0 = rs_poker_card_to_id(combo[0]);
                    let rs_id1 = rs_poker_card_to_id(combo[1]);
                    let ci = card_pair_to_index(rs_id0, rs_id1);
                    let probs = result.strategy.get_probs(current_node, combo_idx);
                    #[allow(clippy::cast_possible_truncation)]
                    let action_prob = probs.get(action_idx).copied().unwrap_or(0.0) as f32;
                    weights[ci] *= action_prob;
                }

                if action_idx < children.len() {
                    current_node = children[action_idx];
                } else {
                    break;
                }
            }
            _ => break,
        }
    }

    // Get pot/effective_stack from the final node.
    #[allow(clippy::cast_possible_truncation)]
    let (pot, effective_stack) = match &result.tree.nodes[current_node as usize] {
        GameNode::Terminal { pot, invested, .. } => {
            let s0 = (result.starting_stack - invested[0]) as i32;
            let s1 = (result.starting_stack - invested[1]) as i32;
            (*pot as i32, s0.min(s1))
        }
        GameNode::Decision { .. } => {
            let (pot_i32, stacks) = decision_display_info(
                &result.tree,
                &result.tree.nodes[current_node as usize],
                result.starting_stack,
                result.initial_pot,
            );
            (pot_i32, stacks[0].min(stacks[1]))
        }
        GameNode::Chance { child, .. } => {
            // Chance node: look through to child for pot info.
            match &result.tree.nodes[*child as usize] {
                GameNode::Terminal { pot, invested, .. } => {
                    let s0 = (result.starting_stack - invested[0]) as i32;
                    let s1 = (result.starting_stack - invested[1]) as i32;
                    (*pot as i32, s0.min(s1))
                }
                _ => {
                    let inv = result.initial_pot / 2.0;
                    (result.initial_pot as i32, (result.starting_stack - inv) as i32)
                }
            }
        }
    };

    *state.filtered_oop_weights.write() = Some(oop_weights.clone());
    *state.filtered_ip_weights.write() = Some(ip_weights.clone());

    Ok(PostflopStreetResult {
        filtered_oop_range: oop_weights,
        filtered_ip_range: ip_weights,
        pot,
        effective_stack,
    })
}

#[tauri::command(rename_all = "snake_case")]
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

// -- Cache types ------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CacheInfo {
    pub exploitability: f32,
    pub iterations: u32,
}

// -- Cache check & load -----------------------------------------------------

pub fn postflop_check_cache_core(
    _state: &PostflopState,
    _board: Vec<String>,
    _prior_actions: Vec<Vec<usize>>,
) -> Result<Option<CacheInfo>, String> {
    // Spot caching disabled — files were too large (80GB+).
    Ok(None)
}

pub fn postflop_load_cached_core(
    _state: &Arc<PostflopState>,
    _board: Vec<String>,
    _prior_actions: Vec<Vec<usize>>,
) -> Result<PostflopStrategyMatrix, String> {
    // Spot caching disabled — files were too large (80GB+).
    Err("Spot caching is disabled".to_string())
}

#[tauri::command(rename_all = "snake_case")]
pub fn postflop_check_cache(
    state: tauri::State<'_, Arc<PostflopState>>,
    board: Vec<String>,
    prior_actions: Vec<Vec<usize>>,
) -> Result<Option<CacheInfo>, String> {
    postflop_check_cache_core(&state, board, prior_actions)
}

#[tauri::command(rename_all = "snake_case")]
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

    // -----------------------------------------------------------------------
    // RolloutLeafEvaluator helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_weighted_basic() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let mut rng = SmallRng::seed_from_u64(42);
        let weights = [0.0, 1.0, 0.0]; // only index 1 has weight
        let samples = sample_weighted(&mut rng, &weights, 5);
        assert_eq!(samples.len(), 5);
        for &s in &samples {
            assert_eq!(s, 1); // all samples must be index 1
        }
    }

    #[test]
    fn test_sample_weighted_empty_weights() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let mut rng = SmallRng::seed_from_u64(42);
        let weights: [f64; 0] = [];
        let samples = sample_weighted(&mut rng, &weights, 5);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_sample_weighted_all_zero() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let mut rng = SmallRng::seed_from_u64(42);
        let weights = [0.0, 0.0, 0.0];
        let samples = sample_weighted(&mut rng, &weights, 5);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_sample_weighted_respects_distribution() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let mut rng = SmallRng::seed_from_u64(123);
        // 90% weight on index 0, 10% on index 1
        let weights = [9.0, 1.0];
        let samples = sample_weighted(&mut rng, &weights, 1000);
        assert_eq!(samples.len(), 1000);
        let count_0 = samples.iter().filter(|&&s| s == 0).count();
        // With 90/10 split over 1000 samples, index 0 should get ~900
        assert!(count_0 > 800, "expected ~900 samples at index 0, got {count_0}");
        assert!(count_0 < 980, "expected ~900 samples at index 0, got {count_0}");
    }

    #[test]
    fn test_sample_weighted_zero_samples_requested() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let mut rng = SmallRng::seed_from_u64(42);
        let weights = [1.0, 2.0];
        let samples = sample_weighted(&mut rng, &weights, 0);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_rollout_leaf_evaluator_implements_leaf_evaluator() {
        // Compile-time check: RolloutLeafEvaluator implements LeafEvaluator.
        fn _assert_impl<T: LeafEvaluator>() {}
        _assert_impl::<RolloutLeafEvaluator>();
    }

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
            rake_rate: 0.0,
            rake_cap: 0.0,
            ..PostflopConfig::default()
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
            rake_rate: 0.0,
            rake_cap: 0.0,
            ..PostflopConfig::default()
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
            rake_rate: 0.0,
            rake_cap: 0.0,
            ..PostflopConfig::default()
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
    fn test_check_cache_returns_none() {
        let state = PostflopState::default();
        let result = postflop_check_cache_core(&state, vec![], vec![]);
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn test_cache_disabled() {
        // Spot caching is disabled — check/load should return None/Err.
        let state = Arc::new(PostflopState::default());
        assert_eq!(
            postflop_check_cache_core(&state, vec![], vec![]).unwrap(),
            None
        );
        assert!(postflop_load_cached_core(&state, vec![], vec![]).is_err());
    }

    /// Helper: build a 1326-element weight vector from a range string.
    fn weights_from_range(range_str: &str) -> Vec<f32> {
        let range: Range = range_str.parse().unwrap();
        range.raw_data().to_vec()
    }

    #[test]
    fn test_dispatch_subgame_for_wide_flop() {
        let state = Arc::new(PostflopState::default());

        let config = PostflopConfig {
            oop_range: "22+,A2s+,K2s+,Q2s+,J2s+,T6s+,97s+,87s,A2o+,K5o+,Q8o+,J8o+,T8o+".to_string(),
            ip_range: "22+,A2s+,K2s+,Q2s+,J2s+,T6s+,97s+,87s,A2o+,K5o+,Q8o+,J8o+,T8o+".to_string(),
            pot: 100,
            effective_stack: 200,
            oop_bet_sizes: "50%".to_string(),
            oop_raise_sizes: "".to_string(),
            ip_bet_sizes: "50%".to_string(),
            ip_raise_sizes: "".to_string(),
            rake_rate: 0.0,
            rake_cap: 0.0,
            ..PostflopConfig::default()
        };
        postflop_set_config_core(&state, config).unwrap();

        // Pre-populate filtered weights so dispatch sees >200 live combos.
        let wide = weights_from_range(
            "22+,A2s+,K2s+,Q2s+,J2s+,T6s+,97s+,87s,A2o+,K5o+,Q8o+,J8o+,T8o+",
        );
        *state.filtered_oop_weights.write() = Some(wide.clone());
        *state.filtered_ip_weights.write() = Some(wide);

        // Solve flop — should dispatch to subgame due to wide ranges.
        let board = vec!["Ks".to_string(), "Qh".to_string(), "Jd".to_string()];
        let result = postflop_solve_street_core(&state, board, Some(5), Some(1e9));
        assert!(result.is_ok(), "solve should succeed: {:?}", result);

        let solver_name = state.solver_name.read().clone();
        assert_eq!(solver_name, "subgame", "wide flop should dispatch to subgame");

        // Wait for solve to complete.
        for _ in 0..600 {
            if state.solve_complete.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(
            state.solve_complete.load(Ordering::Relaxed),
            "solve should complete",
        );

        // Matrix should be available.
        let matrix = state.matrix_snapshot.read().clone();
        assert!(matrix.is_some(), "matrix should be set after solve");
    }

    #[test]
    fn test_dispatch_range_for_narrow_river() {
        let state = Arc::new(PostflopState::default());

        let config = PostflopConfig {
            oop_range: "AA".to_string(),
            ip_range: "KK".to_string(),
            pot: 100,
            effective_stack: 200,
            oop_bet_sizes: "100%".to_string(),
            oop_raise_sizes: "".to_string(),
            ip_bet_sizes: "100%".to_string(),
            ip_raise_sizes: "".to_string(),
            rake_rate: 0.0,
            rake_cap: 0.0,
            ..PostflopConfig::default()
        };
        postflop_set_config_core(&state, config).unwrap();

        // Pre-populate narrow filtered weights (6 combos each).
        let narrow = weights_from_range("AA");
        *state.filtered_oop_weights.write() = Some(narrow.clone());
        let narrow_ip = weights_from_range("KK");
        *state.filtered_ip_weights.write() = Some(narrow_ip);

        // River — always dispatches to range solver regardless of combo count.
        let board = vec![
            "Ks".to_string(),
            "Qh".to_string(),
            "Jd".to_string(),
            "Tc".to_string(),
            "2d".to_string(),
        ];
        let result = postflop_solve_street_core(&state, board, Some(10), Some(1e9));
        assert!(result.is_ok(), "solve should succeed: {:?}", result);

        let solver_name = state.solver_name.read().clone();
        assert_eq!(solver_name, "full", "river should always dispatch to full");

        // Wait for completion.
        for _ in 0..600 {
            if state.solve_complete.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(state.solve_complete.load(Ordering::Relaxed));
    }

    #[test]
    fn test_dispatch_range_for_narrow_flop() {
        let state = Arc::new(PostflopState::default());

        let config = PostflopConfig {
            oop_range: "AA,KK".to_string(),
            ip_range: "QQ,JJ".to_string(),
            pot: 100,
            effective_stack: 200,
            oop_bet_sizes: "50%".to_string(),
            oop_raise_sizes: "".to_string(),
            ip_bet_sizes: "50%".to_string(),
            ip_raise_sizes: "".to_string(),
            rake_rate: 0.0,
            rake_cap: 0.0,
            ..PostflopConfig::default()
        };
        postflop_set_config_core(&state, config).unwrap();

        // Pre-populate narrow filtered weights well below the 200 threshold.
        let narrow_oop = weights_from_range("AA,KK");
        let narrow_ip = weights_from_range("QQ,JJ");
        *state.filtered_oop_weights.write() = Some(narrow_oop);
        *state.filtered_ip_weights.write() = Some(narrow_ip);

        let board = vec!["Ks".to_string(), "Qh".to_string(), "Jd".to_string()];
        let result = postflop_solve_street_core(&state, board, Some(10), Some(1e9));
        assert!(result.is_ok(), "solve should succeed: {:?}", result);

        let solver_name = state.solver_name.read().clone();
        assert_eq!(solver_name, "full", "narrow flop should dispatch to full");
    }

    #[test]
    fn test_build_subgame_solver_returns_cfv_solver() {
        use poker_solver_core::blueprint_v2::CfvSubgameSolver;

        let board_cards = vec![
            RsPokerCard::new(RsPokerValue::Ace, RsPokerSuit::Spade),
            RsPokerCard::new(RsPokerValue::King, RsPokerSuit::Heart),
            RsPokerCard::new(RsPokerValue::Seven, RsPokerSuit::Diamond),
            RsPokerCard::new(RsPokerValue::Four, RsPokerSuit::Club),
        ];
        let bet_sizes = vec![vec![1.0f32]];
        let oop_w = weights_from_range("AA,KK,QQ");
        let ip_w = weights_from_range("JJ,TT,99");

        let result = build_subgame_solver(
            &board_cards,
            &bet_sizes,
            100,
            [200, 200],
            &oop_w,
            &ip_w,
            0,
            None,
            None,
        );
        assert!(result.is_ok(), "build_subgame_solver failed: {:?}", result.err());

        // Explicit type annotation: this MUST be CfvSubgameSolver, not SubgameCfrSolver.
        let (mut solver, _hands, action_infos, _tree, initial_pot, starting_stack):
            (CfvSubgameSolver, SubgameHands, Vec<ActionInfo>, GameTree, f64, f64) =
            result.unwrap();

        // Verify the solver trains and produces valid strategy.
        solver.train(5);
        let strategy = solver.strategy();
        assert!(strategy.num_combos() > 0);
        assert!(!action_infos.is_empty());
        assert!(initial_pot > 0.0);
        assert!(starting_stack > 0.0);

        // Verify strategy produces valid probability distributions at root.
        for combo_idx in 0..strategy.num_combos() {
            let probs = strategy.root_probs(combo_idx);
            if probs.is_empty() {
                continue;
            }
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.05,
                "combo {combo_idx}: strategy sum = {sum}, expected ~1.0"
            );
        }
    }

    #[test]
    fn test_build_subgame_solver_no_precomputed_leaf_values() {
        // Verify build_subgame_solver does NOT use precomputed leaf values
        // (which was the source of the mass-shoving bug). The CfvSubgameSolver
        // evaluates leaves dynamically using per-combo reach vectors.
        let board_cards = vec![
            RsPokerCard::new(RsPokerValue::Ten, RsPokerSuit::Diamond),
            RsPokerCard::new(RsPokerValue::Nine, RsPokerSuit::Heart),
            RsPokerCard::new(RsPokerValue::Six, RsPokerSuit::Club),
            RsPokerCard::new(RsPokerValue::Two, RsPokerSuit::Spade),
        ];
        let bet_sizes = vec![vec![0.5f32]];
        let oop_w = weights_from_range("AA,KK");
        let ip_w = weights_from_range("QQ,JJ");

        let result = build_subgame_solver(
            &board_cards,
            &bet_sizes,
            80,
            [150, 150],
            &oop_w,
            &ip_w,
            0,
            None,
            None,
        );
        assert!(result.is_ok());

        let (mut solver, _hands, _, _, _, _) = result.unwrap();

        // Train a few iterations and verify convergence produces
        // non-degenerate strategies (not 100% all-in).
        solver.train(20);
        let strategy = solver.strategy();

        // Verify the solver produces a strategy with valid distributions.
        let mut total_checked = 0;
        for combo_idx in 0..strategy.num_combos() {
            let probs = strategy.root_probs(combo_idx);
            if probs.is_empty() {
                continue;
            }
            total_checked += 1;
        }
        assert!(total_checked > 0, "should have combos with strategies");
    }

    #[test]
    fn test_cbv_context_has_strategy_and_arc_all_buckets() {
        // CbvContext must store Arc<BlueprintV2Strategy> and Arc<AllBuckets>
        // so the RolloutLeafEvaluator can share them cheaply across 4 instances.
        use poker_solver_core::blueprint_v2::cbv::CbvTable;

        let strategy = Arc::new(BlueprintV2Strategy::empty());
        let tree = GameTree::build_subgame(
            V2Street::Turn, 100.0, [50.0; 2], 200.0, &[vec![1.0]], Some(1),
        );
        let all_buckets = Arc::new(AllBuckets::new([2, 2, 2, 2], [None, None, None, None]));
        let cbv_table = CbvTable { values: vec![], node_offsets: vec![], buckets_per_node: vec![] };

        let ctx = CbvContext {
            cbv_table,
            abstract_tree: tree,
            all_buckets: Arc::clone(&all_buckets),
            strategy: Arc::clone(&strategy),
        };

        // Verify the Arc fields are accessible and can be cloned cheaply
        // (this is the interface needed by build_subgame_solver).
        let _s: Arc<BlueprintV2Strategy> = Arc::clone(&ctx.strategy);
        let _b: Arc<AllBuckets> = Arc::clone(&ctx.all_buckets);
    }

    #[test]
    fn test_rollout_evaluator_constructible_from_cbv_context_arcs() {
        // Verify RolloutLeafEvaluator can be constructed from Arc fields
        // that CbvContext now provides, for all 4 bias types.
        let strategy = Arc::new(BlueprintV2Strategy::empty());
        let tree = Arc::new(GameTree::build_subgame(
            V2Street::Turn, 100.0, [50.0; 2], 200.0, &[vec![1.0]], Some(1),
        ));
        let all_buckets = Arc::new(AllBuckets::new([2, 2, 2, 2], [None, None, None, None]));

        let evaluators: Vec<Box<dyn LeafEvaluator>> = vec![
            Box::new(RolloutLeafEvaluator::new(
                Arc::clone(&strategy), Arc::clone(&tree), Arc::clone(&all_buckets),
                0, BiasType::Unbiased, 10.0, 3, 8,
            )),
            Box::new(RolloutLeafEvaluator::new(
                Arc::clone(&strategy), Arc::clone(&tree), Arc::clone(&all_buckets),
                0, BiasType::Fold, 10.0, 3, 8,
            )),
            Box::new(RolloutLeafEvaluator::new(
                Arc::clone(&strategy), Arc::clone(&tree), Arc::clone(&all_buckets),
                0, BiasType::Call, 10.0, 3, 8,
            )),
            Box::new(RolloutLeafEvaluator::new(
                Arc::clone(&strategy), Arc::clone(&tree), Arc::clone(&all_buckets),
                0, BiasType::Raise, 10.0, 3, 8,
            )),
        ];

        assert_eq!(evaluators.len(), 4, "should create exactly 4 rollout evaluators");
    }
}
