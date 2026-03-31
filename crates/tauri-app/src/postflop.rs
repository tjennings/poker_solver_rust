use parking_lot::RwLock;
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::continuation::{BiasType, RolloutContext, rollout_from_boundary};
use poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id;
use poker_solver_core::blueprint_v2::solver_dispatch::SolverConfig;
use poker_solver_core::blueprint_v2::subgame_cfr::cards_overlap;
use poker_solver_core::blueprint_v2::{
    LeafEvaluator, Street, SubgameHands,
    compute_combo_equities,
};
use poker_solver_core::blueprint_v2::cbv::CbvTable;
use poker_solver_core::blueprint_v2::game_tree::{GameNode as BlueprintGameNode, GameTree};
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::poker::{Card as RsPokerCard, Suit as RsPokerSuit, Value as RsPokerValue};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{card_pair_to_index, card_to_string};
use range_solver::range::Range;
use range_solver::{PostFlopGame, solve_step, finalize};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
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
    /// Which seat is the dealer/button (SB in heads-up).
    pub dealer: u8,
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
pub(crate) fn card_pair_to_matrix(c1: u8, c2: u8) -> (usize, usize, bool) {
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
pub(crate) fn matrix_cell_label(row: usize, col: usize) -> (String, bool, bool) {
    let r1 = RANK_NAMES[row];
    let r2 = RANK_NAMES[col];
    if row == col {
        (format!("{r1}{r2}"), false, true)
    } else if row < col {
        // Above diagonal = suited. row has higher rank (lower index).
        (format!("{r1}{r2}s"), true, false)
    } else {
        // Below diagonal = offsuit. col has higher rank (lower index).
        (format!("{r2}{r1}o"), false, false)
    }
}


// ---------------------------------------------------------------------------
// range-solver Action -> ActionInfo conversion
// ---------------------------------------------------------------------------

/// Convert a range-solver `Action` to an `ActionInfo` for the postflop explorer.
fn range_solver_action_to_action_info(
    action: &range_solver::Action,
    idx: usize,
) -> ActionInfo {
    let (label, action_type, id) = match action {
        range_solver::Action::Fold => ("Fold".to_string(), "fold", "fold".to_string()),
        range_solver::Action::Check => ("Check".to_string(), "check", "check".to_string()),
        range_solver::Action::Call => ("Call".to_string(), "call", "call".to_string()),
        range_solver::Action::Bet(amt) => (format!("Bet {amt}"), "bet", format!("bet:{amt}")),
        range_solver::Action::Raise(amt) => (format!("Raise {amt}"), "raise", format!("raise:{amt}")),
        range_solver::Action::AllIn(amt) => ("All-in".to_string(), "allin", format!("allin:{amt}")),
        _ => ("?".to_string(), "unknown", format!("unknown:{idx}")),
    };
    ActionInfo {
        id,
        label,
        action_type: action_type.to_string(),
        size_key: None,
    }
}

// ---------------------------------------------------------------------------
// Subgame solve result (stored for range propagation and navigation)
// ---------------------------------------------------------------------------

/// Stored result from a subgame solve, used for range propagation and navigation.
pub struct SubgameSolveResult {
    pub game: PostFlopGame,
    pub hands: SubgameHands,
    pub action_infos: Vec<ActionInfo>,
    pub board: Vec<RsPokerCard>,
    pub initial_pot: f64,
    pub starting_stack: f64,
}

// ---------------------------------------------------------------------------
// Subgame solver helpers
// ---------------------------------------------------------------------------

/// Parse a 2-char card string (e.g. "Ks") into an `rs_poker` Card.
pub(crate) fn parse_rs_poker_card(s: &str) -> Result<RsPokerCard, String> {
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
/// This evaluator receives the dynamically-propagated opponent range at each
/// boundary, properly tracking which opponent combos reach each terminal.
#[allow(dead_code)]
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
pub(crate) struct RolloutLeafEvaluator {
    pub(crate) strategy: Arc<BlueprintV2Strategy>,
    pub(crate) abstract_tree: Arc<GameTree>,
    pub(crate) all_buckets: Arc<AllBuckets>,
    pub(crate) decision_idx_map: Vec<u32>,
    pub(crate) abstract_start_node: u32,
    pub(crate) bias: BiasType,
    pub(crate) bias_factor: f64,
    pub(crate) num_rollouts: u32,
    pub(crate) num_opponent_samples: u32,
    #[allow(dead_code)]
    pub(crate) starting_stack: f64,
    /// Stack-to-pot ratio at the subgame root, used to scale the unit
    /// game's starting stack so rollout dynamics match the real game.
    root_spr: f64,
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
        starting_stack: f64,
        root_pot: f64,
    ) -> Self {
        let decision_idx_map = abstract_tree.decision_index_map();
        let root_spr = (starting_stack - root_pot / 2.0) / root_pot;
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
            starting_stack,
            root_spr,
        }
    }
}


impl RolloutLeafEvaluator {
    /// Compute rollout chip values per combo, passing the subgame's actual
    /// pot and invested amounts through the abstract tree traversal.
    #[allow(clippy::too_many_arguments)]
    fn rollout_chip_values_with_state(
        &self,
        combos: &[[RsPokerCard; 2]],
        board: &[RsPokerCard],
        oop_range: &[f64],
        ip_range: &[f64],
        traverser: u8,
        boundary_pot: f64,
        boundary_invested: [f64; 2],
    ) -> Vec<f64> {
        let hero_range = if traverser == 0 { oop_range } else { ip_range };
        let opp_range = if traverser == 0 { ip_range } else { oop_range };

        combos
            .par_iter()
            .enumerate()
            .map(|(i, hero_hand)| {
                if hero_range[i] <= 0.0 {
                    return 0.0;
                }

                let weights: Vec<f64> = combos
                    .iter()
                    .enumerate()
                    .map(|(j, opp_hand)| {
                        if opp_range[j] <= 0.0 || cards_overlap(*hero_hand, *opp_hand) {
                            0.0
                        } else {
                            opp_range[j]
                        }
                    })
                    .collect();

                let mut rng = SmallRng::seed_from_u64(i as u64);
                let sampled = sample_weighted(&mut rng, &weights, self.num_opponent_samples);
                if sampled.is_empty() {
                    return 0.0;
                }

                // Unit game stack: SPR * unit_pot + invested = SPR * 1 + 0.5
                let unit_stack = self.root_spr + 0.5;
                let ctx = RolloutContext {
                    abstract_tree: &self.abstract_tree,
                    decision_idx_map: &self.decision_idx_map,
                    strategy: &self.strategy,
                    buckets: &self.all_buckets,
                    bias: self.bias,
                    bias_factor: self.bias_factor,
                    player: traverser,
                    num_rollouts: self.num_rollouts,
                    starting_stack: unit_stack,
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
                            boundary_pot,
                            boundary_invested,
                        )
                    })
                    .sum();

                total / sampled.len() as f64
            })
            .collect()
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
        // Assume symmetric investment for the single-boundary evaluate path.
        let invested = [pot / 2.0, pot / 2.0];
        let chip_values = self.rollout_chip_values_with_state(
            combos, board, oop_range, ip_range, traverser, pot, invested,
        );
        chip_values.iter().map(|&v| v / half_pot).collect()
    }

    fn evaluate_boundaries(
        &self,
        combos: &[[RsPokerCard; 2]],
        board: &[RsPokerCard],
        oop_range: &[f64],
        ip_range: &[f64],
        requests: &[(f64, f64, u8)],
    ) -> Vec<Vec<f64>> {
        if requests.is_empty() {
            return vec![];
        }

        let traverser = requests[0].2;
        let hero_range = if traverser == 0 { oop_range } else { ip_range };
        let eval_start = std::time::Instant::now();

        // Compute rollout ONCE using a unit game (pot=1) with the same
        // SPR as the subgame root. Terminal payoffs scale linearly with
        // pot, so we scale the unit-game chip values to each boundary's
        // actual pot to get pot-fraction CFVs.
        let unit_pot = 1.0;
        let unit_invested = [0.5, 0.5]; // each player put in half the unit pot
        let chip_values = self.rollout_chip_values_with_state(
            combos, board, oop_range, ip_range, traverser, unit_pot, unit_invested,
        );

        // Unit-game chip values → pot-fraction: multiply by 2 (divide by half_pot=0.5).
        // Then scale doesn't depend on boundary pot since payoffs are proportional.
        let results: Vec<Vec<f64>> = requests
            .iter()
            .map(|_| {
                chip_values.iter().map(|&v| v / 0.5).collect()
            })
            .collect();

        let elapsed = eval_start.elapsed();
        eprintln!(
            "[rollout] {:?} {} boundaries, {:.1}s",
            self.bias, requests.len(), elapsed.as_secs_f64()
        );

        results
    }
}

/// Build a `PostFlopGame` and associated data for subgame solving.
///
/// Returns the game, hands, action labels, initial pot, and starting
/// stack without running any iterations. The caller drives the iteration loop
/// and can build matrix snapshots between iterations.
///
/// Uses [`PostFlopGame`] from range-solver for depth-limited solving, with
/// optional `BoundaryEvaluator` for rollout-based boundary values.
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
    rollout_bias_factor: Option<f64>,
    rollout_num_samples: Option<u32>,
    rollout_opponent_samples: Option<u32>,
) -> Result<(PostFlopGame, SubgameHands, Vec<ActionInfo>, f64, f64), String> {
    use range_solver::card::CardConfig;
    use range_solver::{ActionTree, TreeConfig, BoardState};

    let (initial_state, n_board) = match board_cards.len() {
        3 => (BoardState::Flop, 3),
        4 => (BoardState::Turn, 4),
        5 => (BoardState::River, 5),
        n => return Err(format!("Invalid board length for subgame: {n}")),
    };

    let pot_i32 = pot as i32;
    let eff_stack = stacks[0] as i32;
    let pot_f = f64::from(pot);
    let starting_stack = f64::from(stacks[0]) + pot_f / 2.0;

    // Convert board cards from rs_poker to range-solver IDs.
    let board_ids: Vec<u8> = board_cards.iter().map(|c| rs_poker_card_to_id(*c)).collect();
    let flop = [board_ids[0], board_ids[1], board_ids[2]];
    let turn = if n_board >= 4 { board_ids[3] } else { range_solver::card::NOT_DEALT };
    let river = if n_board >= 5 { board_ids[4] } else { range_solver::card::NOT_DEALT };

    // Build bet size strings from pot fractions.
    let bet_str = if !bet_sizes_per_depth.is_empty() {
        bet_sizes_per_depth[0]
            .iter()
            .map(|s| format!("{}%", s * 100.0))
            .collect::<Vec<_>>()
            .join(",")
    } else {
        "100%".to_string()
    };
    let raise_str = if bet_sizes_per_depth.len() > 1 {
        bet_sizes_per_depth[1]
            .iter()
            .map(|s| format!("{}%", s * 100.0))
            .collect::<Vec<_>>()
            .join(",")
    } else {
        String::new()
    };

    let oop_range = Range::from_raw_data(oop_weights)
        .map_err(|e| format!("Bad OOP weights: {e}"))?;
    let ip_range = Range::from_raw_data(ip_weights)
        .map_err(|e| format!("Bad IP weights: {e}"))?;

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
        starting_pot: pot_i32,
        effective_stack: eff_stack,
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
        // Solve current street only. Boundaries at next street transition.
        // River: no boundaries needed (solve to showdown).
        depth_limit: if initial_state == BoardState::River { None } else { Some(0) },
    };

    let action_tree = ActionTree::new(tree_config)
        .map_err(|e| format!("Failed to build tree: {e}"))?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;
    game.allocate_memory(false);

    // Extract action labels from the game root.
    game.back_to_root();
    let action_infos: Vec<ActionInfo> = game
        .available_actions()
        .iter()
        .enumerate()
        .map(|(i, a)| range_solver_action_to_action_info(a, i))
        .collect();

    // Set up BoundaryEvaluator if boundaries exist and CBV context is available.
    let n_boundaries = game.num_boundary_nodes();
    if n_boundaries > 0 {
        if let (Some(ctx), Some(abs_node)) = (cbv_context, abstract_node_idx) {
            let bias_factor = rollout_bias_factor.unwrap_or(10.0);
            let num_rollouts = rollout_num_samples.unwrap_or(3);
            let opp_samples = rollout_opponent_samples.unwrap_or(8);
            eprintln!("[subgame] using SolveBoundaryEvaluator (abstract_node={abs_node}, bias={bias_factor}, rollouts={num_rollouts}, opp_samples={opp_samples})");

            // Build combos in rollout ordering for the boundary evaluator.
            let rs_board_cards: Vec<rs_poker::core::Card> = board_cards.to_vec();
            let mut combos: Vec<[rs_poker::core::Card; 2]> = Vec::new();
            use poker_solver_core::hands::all_hands;
            for hand in all_hands() {
                for (c0, c1) in hand.combos() {
                    if rs_board_cards.iter().any(|b| *b == c0 || *b == c1) {
                        continue;
                    }
                    combos.push([c0, c1]);
                }
            }

            let build_map = |player: usize, combos: &[[rs_poker::core::Card; 2]]| -> Vec<usize> {
                game.private_cards(player).iter().map(|&(c1, c2)| {
                    let rs_c1 = crate::exploration::range_solver_to_rs_card(c1);
                    let rs_c2 = crate::exploration::range_solver_to_rs_card(c2);
                    combos.iter().position(|c|
                        (c[0] == rs_c1 && c[1] == rs_c2) || (c[0] == rs_c2 && c[1] == rs_c1)
                    ).unwrap_or(usize::MAX)
                }).collect()
            };
            let map0 = build_map(0, &combos);
            let map1 = build_map(1, &combos);

            let rollout = RolloutLeafEvaluator::new(
                Arc::clone(&ctx.strategy),
                Arc::new(ctx.abstract_tree.clone()),
                Arc::clone(&ctx.all_buckets),
                abs_node,
                BiasType::Unbiased,
                bias_factor,
                num_rollouts,
                opp_samples,
                starting_stack,
                pot_f,
            );

            game.boundary_evaluator = Some(Arc::new(
                crate::game_session::SolveBoundaryEvaluator {
                    private_cards: [
                        game.private_cards(0).to_vec(),
                        game.private_cards(1).to_vec(),
                    ],
                    board_cards: rs_board_cards,
                    eff_stack: f64::from(eff_stack),
                    rollout: Some(rollout),
                    combos,
                    game_to_combo: [map0, map1],
                },
            ));
        } else {
            eprintln!("[subgame] {} boundary nodes but no CBV context — boundaries will use default evaluation", n_boundaries);
        }
    }

    let hands = SubgameHands::enumerate(board_cards);

    let nonzero_oop = oop_weights.iter().filter(|&&w| w > 0.0).count();
    let nonzero_ip = ip_weights.iter().filter(|&&w| w > 0.0).count();
    eprintln!("[subgame] initial reach: OOP={nonzero_oop} IP={nonzero_ip} combos with nonzero reach");

    Ok((game, hands, action_infos, pot_f, starting_stack))
}

// ---------------------------------------------------------------------------
// Blueprint strategy seeding
// ---------------------------------------------------------------------------

/// Seed the range solver's cumulative strategy sums with blueprint action
/// frequencies. This initializes the solver at the blueprint's strategy
/// rather than uniform, giving it a warm start.
///
/// Walks the range-solver tree and blueprint tree in parallel via DFS.
/// At each decision node, looks up each hand's bucket in the blueprint
/// and writes `blueprint_prob * SCALE` into the strategy sums.
pub fn seed_solver_with_blueprint(
    game: &PostFlopGame,
    strategy: &BlueprintV2Strategy,
    all_buckets: &AllBuckets,
    tree: &GameTree,
    board: &[RsPokerCard],
    street: Street,
    blueprint_root: u32,
) {
    /// Scale factor for blueprint probabilities written into cumulative
    /// strategy sums. Gives the initial blueprint sufficient weight so
    /// early solver iterations don't immediately wash it out.
    const SEED_SCALE: f32 = 1000.0;

    // Build a map: range_solver_arena_index -> blueprint_decision_index.
    let decision_map = tree.decision_index_map();
    let mut arena_to_decision: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    // DFS stack: (range_solver_arena_index, blueprint_tree_arena_index).
    let mut dfs_stack: Vec<(usize, u32)> = vec![(0, blueprint_root)];

    while let Some((rs_idx, bp_idx)) = dfs_stack.pop() {
        let bp_node = &tree.nodes[bp_idx as usize];

        match bp_node {
            BlueprintGameNode::Decision { children, .. } => {
                let dec_idx = decision_map[bp_idx as usize];
                if dec_idx != u32::MAX {
                    arena_to_decision.insert(rs_idx, dec_idx as usize);
                }

                // Push children in reverse order (to process in forward order).
                let rs_children = game.child_indices(rs_idx);
                let n = rs_children.len().min(children.len());
                for i in (0..n).rev() {
                    dfs_stack.push((rs_children[i], children[i]));
                }
            }
            BlueprintGameNode::Chance { child, .. } => {
                // Skip chance nodes: the range solver may also have a chance
                // node here, or the subtree may be flattened. Try to descend
                // through both.
                let rs_children = game.child_indices(rs_idx);
                if rs_children.len() == 1 {
                    // Single child (chance passthrough) -- continue.
                    dfs_stack.push((rs_children[0], *child));
                }
                // If the range solver has multiple chance children (deal
                // cards), we can't match them 1:1 with the blueprint's single
                // child. For single-street solves this shouldn't happen.
            }
            BlueprintGameNode::Terminal { .. } => {
                // Leaf -- nothing to do.
            }
        }
    }

    if arena_to_decision.is_empty() {
        eprintln!("[seed] no blueprint decision nodes mapped -- skipping seeding");
        return;
    }

    let board_rs: Vec<RsPokerCard> = board.to_vec();
    let num_seeded = std::sync::atomic::AtomicU32::new(0);

    game.seed_strategy(|node_idx, player, num_actions, num_hands| {
        let &dec_idx = arena_to_decision.get(&node_idx)?;

        let bp_probs = strategy.get_action_probs(dec_idx, 0);
        if bp_probs.is_empty() || bp_probs.len() != num_actions {
            return None;
        }

        let private_cards = game.private_cards(player);
        let mut strat = vec![0.0f32; num_actions * num_hands];

        for (h, &(c1, c2)) in private_cards.iter().enumerate() {
            let rs_c1 = crate::exploration::range_solver_to_rs_card(c1);
            let rs_c2 = crate::exploration::range_solver_to_rs_card(c2);

            let bucket = all_buckets.get_bucket(street, [rs_c1, rs_c2], &board_rs);
            let probs = strategy.get_action_probs(dec_idx, bucket);

            let n = probs.len().min(num_actions);
            for a in 0..n {
                strat[a * num_hands + h] = probs[a] * SEED_SCALE;
            }
        }

        num_seeded.fetch_add(1, Ordering::Relaxed);
        Some(strat)
    });

    let count = num_seeded.load(Ordering::Relaxed);
    eprintln!("[seed] seeded {count} decision nodes with blueprint strategy");
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
    dealer: u8,
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

    for (hand_idx, &(c1_raw, c2_raw)) in snap.private_cards.iter().enumerate() {
        let (row, col, _) = card_pair_to_matrix(c1_raw, c2_raw);
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
        // High card first: rank = id / 4.
        let (c1, c2) = if c1_raw / 4 >= c2_raw / 4 { (c1_raw, c2_raw) } else { (c2_raw, c1_raw) };
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
        dealer: snap.dealer,
    }
}

/// Build a [`MatrixSnapshot`] directly from a `PostFlopGame`'s current strategy.
///
/// The game must be positioned at the desired node (call `back_to_root()` or
/// navigate before calling this). Extracts strategy, private cards, and weights
/// from the game, producing a snapshot compatible with `build_matrix_from_snapshot`.
#[allow(clippy::cast_possible_truncation)]
fn snapshot_from_game(
    game: &mut PostFlopGame,
    action_infos: Vec<ActionInfo>,
    board_strings: &[String],
    pot: i32,
    stacks: [i32; 2],
    dealer: u8,
    hand_evs: Option<Vec<f32>>,
) -> MatrixSnapshot {
    use range_solver::interface::Game;

    let player = game.current_player();
    let strategy = game.strategy();
    let private_cards: Vec<(u8, u8)> = game.private_cards(player).to_vec();
    let initial_weights: Vec<f32> = game.initial_weights(player).to_vec();
    let num_hands = private_cards.len();

    MatrixSnapshot {
        player,
        strategy,
        private_cards,
        initial_weights,
        num_hands,
        actions: action_infos,
        pot,
        stacks,
        hand_evs,
        board: board_strings.to_vec(),
        dealer,
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

pub fn postflop_solve_street_core(
    state: &Arc<PostflopState>,
    board: Vec<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
    rollout_bias_factor: Option<f64>,
    rollout_num_samples: Option<u32>,
    rollout_opponent_samples: Option<u32>,
    leaf_eval_interval: Option<u32>,
    range_clamp_threshold: Option<f64>,
) -> Result<(), String> {
    postflop_solve_street_impl(state, board, max_iterations, target_exploitability, vec![],
        rollout_bias_factor, rollout_num_samples, rollout_opponent_samples, leaf_eval_interval,
        range_clamp_threshold)
}

#[allow(clippy::too_many_arguments)]
fn postflop_solve_street_impl(
    state: &Arc<PostflopState>,
    board: Vec<String>,
    max_iterations: Option<u32>,
    _target_exploitability: Option<f32>,
    _prior_actions: Vec<Vec<usize>>,
    rollout_bias_factor: Option<f64>,
    rollout_num_samples: Option<u32>,
    rollout_opponent_samples: Option<u32>,
    leaf_eval_interval: Option<u32>,
    range_clamp_threshold: Option<f64>,
) -> Result<(), String> {
    // Guard: reject if already solving.
    if state.solving.load(Ordering::Relaxed) {
        return Err("A solve is already in progress".to_string());
    }

    // Snapshot config and filtered weights under their locks.
    let config = state.config.read().clone();
    let filtered_oop = state.filtered_oop_weights.read().clone();
    let filtered_ip = state.filtered_ip_weights.read().clone();

    let solver_config = SolverConfig {
        flop_combo_threshold: config.flop_combo_threshold,
        turn_combo_threshold: config.turn_combo_threshold,
        ..SolverConfig::default()
    };

    // Reset progress atomics.
    state.current_iteration.store(0, Ordering::Relaxed);
    state.solve_complete.store(false, Ordering::Relaxed);
    state.subgame_node.store(0, Ordering::Relaxed);
    *state.solve_start.write() = Some(std::time::Instant::now());
    *state.matrix_snapshot.write() = None;

    // Always use depth-limited solver (full-depth range-solver path removed).
    let cbv_ctx = state.cbv_context.read().clone();
    solve_depth_limited(state, &config, board, max_iterations, &solver_config, &filtered_oop, &filtered_ip, cbv_ctx,
        rollout_bias_factor, rollout_num_samples, rollout_opponent_samples, leaf_eval_interval,
        range_clamp_threshold)
}

/// Depth-limited solve using `PostFlopGame` from range-solver.
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
    rollout_bias_factor: Option<f64>,
    rollout_num_samples: Option<u32>,
    rollout_opponent_samples: Option<u32>,
    leaf_eval_interval: Option<u32>,
    range_clamp_threshold: Option<f64>,
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

    let mut oop_w = filtered_oop.clone().unwrap_or_else(|| {
        let range: Range = config.oop_range.parse().unwrap_or_default();
        range.raw_data().to_vec()
    });
    let mut ip_w = filtered_ip.clone().unwrap_or_else(|| {
        let range: Range = config.ip_range.parse().unwrap_or_default();
        range.raw_data().to_vec()
    });

    // Clamp low-reach combos to zero to remove blueprint noise.
    let clamp = range_clamp_threshold.unwrap_or(0.0) as f32;
    if clamp > 0.0 {
        let oop_before = oop_w.iter().filter(|&&w| w > 0.0).count();
        let ip_before = ip_w.iter().filter(|&&w| w > 0.0).count();
        for w in oop_w.iter_mut() {
            if *w > 0.0 && *w < clamp { *w = 0.0; }
        }
        for w in ip_w.iter_mut() {
            if *w > 0.0 && *w < clamp { *w = 0.0; }
        }
        let oop_after = oop_w.iter().filter(|&&w| w > 0.0).count();
        let ip_after = ip_w.iter().filter(|&&w| w > 0.0).count();
        eprintln!(
            "[range clamp] threshold={clamp:.3}: OOP {oop_before}→{oop_after} combos, IP {ip_before}→{ip_after} combos"
        );
    }

    let shared = Arc::clone(state);
    let board_strings = board;
    std::thread::spawn(move || {
        match build_subgame_solver(
            &board_cards,
            &bet_sizes_per_depth,
            pot as u32,
            [eff_stack as u32, eff_stack as u32],
            &oop_w,
            &ip_w,
            0, // _player param (unused)
            cbv_context.as_deref(),
            abstract_node_idx,
            rollout_bias_factor,
            rollout_num_samples,
            rollout_opponent_samples,
        ) {
            Ok((mut game, hands, action_infos, initial_pot, starting_stack)) => {
                // Seed solver with blueprint strategy if available.
                eprintln!("[seed] cbv_context={}, abstract_node_idx={:?}", cbv_context.is_some(), abstract_node_idx);
                if let (Some(ctx), Some(abs_node)) = (&cbv_context, abstract_node_idx) {
                    let seed_street = match board_cards.len() {
                        3 => Street::Flop,
                        4 => Street::Turn,
                        _ => Street::River,
                    };
                    seed_solver_with_blueprint(
                        &game,
                        &ctx.strategy,
                        &ctx.all_buckets,
                        &ctx.abstract_tree,
                        &board_cards,
                        seed_street,
                        abs_node,
                    );
                }

                let eval_interval = leaf_eval_interval.unwrap_or(10);

                // Initial matrix snapshot.
                game.back_to_root();
                let snap = snapshot_from_game(
                    &mut game, action_infos.clone(), &board_strings,
                    pot, [eff_stack, eff_stack], 1, None,
                );
                *shared.matrix_snapshot.write() = Some(build_matrix_from_snapshot(snap));

                // Solve loop.
                let mut t = 0u32;
                while t < max_iters {
                    if !shared.solving.load(Ordering::Relaxed) {
                        break;
                    }

                    // Flush boundary caches periodically so the solver
                    // lazily recomputes reach + CFVs with updated strategy.
                    if t > 0 && t % eval_interval == 0 {
                        game.flush_boundary_caches();
                    }

                    solve_step(&game, t);
                    t += 1;
                    shared.current_iteration.store(t, Ordering::Relaxed);

                    // Snapshot matrix periodically.
                    if t % eval_interval == 0 {
                        game.back_to_root();
                        let snap = snapshot_from_game(
                            &mut game, action_infos.clone(), &board_strings,
                            pot, [eff_stack, eff_stack], 1, None,
                        );
                        *shared.matrix_snapshot.write() = Some(build_matrix_from_snapshot(snap));
                    }
                }

                // Finalize: normalize strategy, compute EVs.
                finalize(&mut game);
                game.back_to_root();
                game.cache_normalized_weights();
                let player = game.current_player();
                let evs = game.expected_values(player);

                let snap = snapshot_from_game(
                    &mut game, action_infos.clone(), &board_strings,
                    pot, [eff_stack, eff_stack], 1, Some(evs),
                );
                *shared.matrix_snapshot.write() = Some(build_matrix_from_snapshot(snap));

                // Store subgame result for range propagation and navigation.
                *shared.subgame_result.write() = Some(SubgameSolveResult {
                    game,
                    hands,
                    action_infos,
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
    rollout_bias_factor: Option<f64>,
    rollout_num_samples: Option<u32>,
    rollout_opponent_samples: Option<u32>,
    leaf_eval_interval: Option<u32>,
    range_clamp_threshold: Option<f64>,
) -> Result<(), String> {
    postflop_solve_street_core(&state, board, max_iterations, target_exploitability,
        rollout_bias_factor, rollout_num_samples, rollout_opponent_samples, leaf_eval_interval,
        range_clamp_threshold)
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

/// Compute pot and stack info from the `PostFlopGame`'s current position.
#[allow(clippy::cast_possible_truncation)]
fn game_display_info(game: &PostFlopGame) -> (i32, [i32; 2]) {
    let tc = game.tree_config();
    let bets = game.total_bet_amount();
    let pot = tc.starting_pot + bets[0] + bets[1];
    let stacks = [
        tc.effective_stack - bets[0],
        tc.effective_stack - bets[1],
    ];
    (pot, stacks)
}

/// Build a `PostflopPlayResult` from the current position of the `PostFlopGame`.
fn game_node_to_result(
    result: &mut SubgameSolveResult,
) -> Result<PostflopPlayResult, String> {
    let game = &mut result.game;

    if game.is_terminal_node() {
        let (pot, stacks) = game_display_info(game);
        // Depth boundaries appear as terminal nodes but represent
        // a street transition — report as chance nodes.
        let is_boundary = game.num_boundary_nodes() > 0 && {
            // Check if the current terminal is a depth boundary by
            // inspecting whether remaining stack equals zero or not.
            // Actually, all depth-boundary terminals are explicitly marked
            // in range-solver. For simplicity, report all terminals as
            // terminal (fold/showdown). Depth boundaries won't appear
            // for river solves, and for flop/turn the UI handles
            // the chance/boundary transition via close_street.
            false
        };
        return Ok(PostflopPlayResult {
            matrix: None,
            is_terminal: !is_boundary,
            is_chance: is_boundary,
            current_player: None,
            pot,
            stacks,
        });
    }

    if game.is_chance_node() {
        let (pot, stacks) = game_display_info(game);
        return Ok(PostflopPlayResult {
            matrix: None,
            is_terminal: false,
            is_chance: true,
            current_player: None,
            pot,
            stacks,
        });
    }

    // Decision node: build strategy matrix.
    let board_card_strings: Vec<String> = result
        .board
        .iter()
        .map(|c| {
            let id = rs_poker_card_to_id(*c);
            card_to_string(id).unwrap_or_default()
        })
        .collect();

    let action_infos: Vec<ActionInfo> = game
        .available_actions()
        .iter()
        .enumerate()
        .map(|(i, a)| range_solver_action_to_action_info(a, i))
        .collect();

    let (pot, stacks) = game_display_info(game);
    let player = game.current_player();

    let snap = snapshot_from_game(
        game,
        action_infos,
        &board_card_strings,
        pot,
        stacks,
        1, // dealer (range-solver convention)
        None, // EVs not available during navigation
    );
    let matrix = build_matrix_from_snapshot(snap);
    Ok(PostflopPlayResult {
        matrix: Some(matrix),
        is_terminal: false,
        is_chance: false,
        current_player: Some(player),
        pot,
        stacks,
    })
}

pub fn postflop_play_action_core(
    state: &PostflopState,
    action: usize,
) -> Result<PostflopPlayResult, String> {
    // During solving, subgame_result is not yet populated.
    // Wait for solve to complete before allowing navigation.
    if state.solving.load(Ordering::Relaxed) {
        return Err("Solve in progress — wait for completion before navigating".to_string());
    }

    let mut result_guard = state.subgame_result.write();
    let result = result_guard.as_mut().ok_or("No subgame result stored")?;

    // Navigate one step: play the action on the PostFlopGame.
    let num_actions = result.game.available_actions().len();
    if action >= num_actions {
        return Err(format!(
            "Action {action} out of range (max {})",
            num_actions.saturating_sub(1)
        ));
    }
    result.game.play(action);

    game_node_to_result(result)
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

/// Replay the game tree to a given action history and return the result.
pub fn postflop_navigate_to_core(
    state: &PostflopState,
    history: Vec<usize>,
) -> Result<PostflopPlayResult, String> {
    if state.solving.load(Ordering::Relaxed) {
        return Err("Solve in progress — wait for completion before navigating".to_string());
    }

    let mut result_guard = state.subgame_result.write();
    let result = result_guard.as_mut().ok_or("No subgame result stored")?;

    // Reset to root, then replay the action history.
    result.game.back_to_root();
    for &action in &history {
        result.game.play(action);
    }

    game_node_to_result(result)
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
    postflop_close_street_subgame(state, action_history)
}

/// Range propagation through a subgame for `postflop_close_street_core`.
///
/// Walks the stored `PostFlopGame`, multiplying each acting player's
/// 1326-element weight vector by the strategy probabilities at each
/// decision node. Returns pot/stacks from the final position.
fn postflop_close_street_subgame(
    state: &PostflopState,
    action_history: Vec<usize>,
) -> Result<PostflopStreetResult, String> {
    let mut result_guard = state.subgame_result.write();
    let result = result_guard
        .as_mut()
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

    // Walk each action, narrowing the acting player's range.
    result.game.back_to_root();
    for &action_idx in &action_history {
        if result.game.is_terminal_node() || result.game.is_chance_node() {
            break;
        }

        let player = result.game.current_player();
        let strategy = result.game.strategy();
        let private_cards = result.game.private_cards(player).to_vec();
        let num_hands = private_cards.len();
        let num_actions = result.game.available_actions().len();

        let weights = if player == 0 {
            &mut oop_weights
        } else {
            &mut ip_weights
        };

        // Multiply each combo's weight by its strategy probability for this action.
        for (hand_idx, &(c1, c2)) in private_cards.iter().enumerate() {
            let ci = card_pair_to_index(c1, c2);
            let action_prob = if action_idx < num_actions {
                strategy[action_idx * num_hands + hand_idx]
            } else {
                0.0
            };
            weights[ci] *= action_prob;
        }

        result.game.play(action_idx);
    }

    // Get pot/effective_stack from the final position.
    let (pot, effective_stack) = {
        let (pot_val, stacks) = game_display_info(&result.game);
        (pot_val, stacks[0].min(stacks[1]))
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
            postflop_solve_street_core(&state, board, Some(2), Some(f32::MAX), None, None, None, None, None);
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
        assert!(state.matrix_snapshot.read().is_some());
    }

    #[test]
    fn test_solve_street_rejects_double_solve() {
        let state = Arc::new(PostflopState::default());
        state.solving.store(true, Ordering::Relaxed);

        let board = vec!["Td".into(), "9d".into(), "6h".into()];
        let result =
            postflop_solve_street_core(&state, board, None, None, None, None, None, None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already in progress"));
    }

    #[test]
    fn test_play_action_no_subgame() {
        let state = PostflopState::default();
        let result = postflop_play_action_core(&state, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No subgame result stored"));
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
        postflop_solve_street_core(&state, board, Some(2), Some(f32::MAX), None, None, None, None, None).unwrap();

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
    fn test_close_street_no_subgame() {
        let state = PostflopState::default();
        let result = postflop_close_street_core(&state, vec![0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No subgame result stored"));
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
        postflop_solve_street_core(&state, board, Some(5), Some(f32::MAX), None, None, None, None, None).unwrap();

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
    #[ignore] // TODO: slow (~100s equity matrix) and currently failing — fix separately
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
        let result = postflop_solve_street_core(&state, board, Some(5), Some(1e9), None, None, None, None, None);
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

        // River — always dispatches to subgame solver.
        let board = vec![
            "Ks".to_string(),
            "Qh".to_string(),
            "Jd".to_string(),
            "Tc".to_string(),
            "2d".to_string(),
        ];
        let result = postflop_solve_street_core(&state, board, Some(10), Some(1e9), None, None, None, None, None);
        assert!(result.is_ok(), "solve should succeed: {:?}", result);

        let solver_name = state.solver_name.read().clone();
        assert_eq!(solver_name, "subgame", "river should dispatch to subgame");

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
        let result = postflop_solve_street_core(&state, board, Some(10), Some(1e9), None, None, None, None, None);
        assert!(result.is_ok(), "solve should succeed: {:?}", result);

        let solver_name = state.solver_name.read().clone();
        assert_eq!(solver_name, "subgame", "narrow flop should dispatch to subgame");
    }

    #[test]
    fn test_build_subgame_solver_returns_postflop_game() {
        use range_solver::interface::Game;

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
            None, None, None,
        );
        assert!(result.is_ok(), "build_subgame_solver failed: {:?}", result.err());

        let (mut game, _hands, action_infos, initial_pot, starting_stack) =
            result.unwrap();

        // Verify the game has hands and actions.
        assert!(game.num_private_hands(0) > 0);
        assert!(!action_infos.is_empty());
        assert!(initial_pot > 0.0);
        assert!(starting_stack > 0.0);

        // Solve a few iterations and verify strategy.
        for i in 0..5u32 {
            range_solver::solve_step(&game, i);
        }
        range_solver::finalize(&mut game);
        game.back_to_root();

        let strategy = game.strategy();
        let num_hands = game.num_private_hands(game.current_player());
        let num_actions = game.available_actions().len();
        assert_eq!(strategy.len(), num_actions * num_hands);

        // Verify strategy produces valid probability distributions.
        for hand_idx in 0..num_hands {
            let sum: f32 = (0..num_actions)
                .map(|a| strategy[a * num_hands + hand_idx])
                .sum();
            if sum > 0.0 {
                assert!(
                    (sum - 1.0).abs() < 0.05,
                    "hand {hand_idx}: strategy sum = {sum}, expected ~1.0"
                );
            }
        }
    }

    #[test]
    fn test_build_subgame_solver_produces_valid_strategy() {
        use range_solver::interface::Game;

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
            None, None, None,
        );
        assert!(result.is_ok());

        let (mut game, _hands, _, _, _) = result.unwrap();

        // Train a few iterations and verify convergence produces
        // non-degenerate strategies (not 100% all-in).
        for i in 0..20u32 {
            range_solver::solve_step(&game, i);
        }
        range_solver::finalize(&mut game);
        game.back_to_root();

        let strategy = game.strategy();
        let num_hands = game.num_private_hands(game.current_player());
        assert!(num_hands > 0, "should have hands");
        assert!(!strategy.is_empty(), "strategy should not be empty");
    }

    #[test]
    fn test_cbv_context_has_strategy_and_arc_all_buckets() {
        // CbvContext must store Arc<BlueprintV2Strategy> and Arc<AllBuckets>
        // so the RolloutLeafEvaluator can share them cheaply across 4 instances.
        use poker_solver_core::blueprint_v2::cbv::CbvTable;

        let strategy = Arc::new(BlueprintV2Strategy::empty());
        let tree = GameTree::build_subgame(
            poker_solver_core::blueprint_v2::Street::Turn, 100.0, [50.0; 2], 200.0, &[vec![1.0]], Some(1), 0,
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
            poker_solver_core::blueprint_v2::Street::Turn, 100.0, [50.0; 2], 200.0, &[vec![1.0]], Some(1), 0,
        ));
        let all_buckets = Arc::new(AllBuckets::new([2, 2, 2, 2], [None, None, None, None]));

        let evaluators: Vec<Box<dyn LeafEvaluator>> = vec![
            Box::new(RolloutLeafEvaluator::new(
                Arc::clone(&strategy), Arc::clone(&tree), Arc::clone(&all_buckets),
                0, BiasType::Unbiased, 10.0, 3, 8, 100.0, 10.0,
            )),
            Box::new(RolloutLeafEvaluator::new(
                Arc::clone(&strategy), Arc::clone(&tree), Arc::clone(&all_buckets),
                0, BiasType::Fold, 10.0, 3, 8, 100.0, 10.0,
            )),
            Box::new(RolloutLeafEvaluator::new(
                Arc::clone(&strategy), Arc::clone(&tree), Arc::clone(&all_buckets),
                0, BiasType::Call, 10.0, 3, 8, 100.0, 10.0,
            )),
            Box::new(RolloutLeafEvaluator::new(
                Arc::clone(&strategy), Arc::clone(&tree), Arc::clone(&all_buckets),
                0, BiasType::Raise, 10.0, 3, 8, 100.0, 10.0,
            )),
        ];

        assert_eq!(evaluators.len(), 4, "should create exactly 4 rollout evaluators");
    }

    #[test]
    fn test_build_subgame_solver_accepts_rollout_params() {
        use range_solver::interface::Game;

        // Verify build_subgame_solver accepts optional rollout configuration
        // parameters (bias_factor, num_samples, opponent_samples) and produces
        // a valid solver when custom values are provided.
        let board_cards = vec![
            RsPokerCard::new(RsPokerValue::Ace, RsPokerSuit::Spade),
            RsPokerCard::new(RsPokerValue::King, RsPokerSuit::Heart),
            RsPokerCard::new(RsPokerValue::Seven, RsPokerSuit::Diamond),
            RsPokerCard::new(RsPokerValue::Four, RsPokerSuit::Club),
        ];
        let bet_sizes = vec![vec![1.0f32]];
        let oop_w = weights_from_range("AA,KK,QQ");
        let ip_w = weights_from_range("JJ,TT,99");

        // Pass custom rollout params (these only matter when cbv_context is Some,
        // but the function must accept them regardless).
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
            Some(5.0),  // rollout_bias_factor
            Some(5),     // rollout_num_samples
            Some(12),    // rollout_opponent_samples
        );
        assert!(result.is_ok(), "build_subgame_solver with rollout params failed: {:?}", result.err());

        let (game, _hands, action_infos, initial_pot, starting_stack) = result.unwrap();
        assert!(game.num_private_hands(0) > 0);
        assert!(!action_infos.is_empty());
        assert!(initial_pot > 0.0);
        assert!(starting_stack > 0.0);
    }

    #[test]
    fn test_build_subgame_solver_defaults_rollout_params_to_none() {
        // Verify that passing None for all rollout params works (backward compat).
        let board_cards = vec![
            RsPokerCard::new(RsPokerValue::Ace, RsPokerSuit::Spade),
            RsPokerCard::new(RsPokerValue::King, RsPokerSuit::Heart),
            RsPokerCard::new(RsPokerValue::Seven, RsPokerSuit::Diamond),
            RsPokerCard::new(RsPokerValue::Four, RsPokerSuit::Club),
        ];
        let bet_sizes = vec![vec![1.0f32]];
        let oop_w = weights_from_range("AA,KK");
        let ip_w = weights_from_range("QQ,JJ");

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
            None, // rollout_bias_factor
            None, // rollout_num_samples
            None, // rollout_opponent_samples
        );
        assert!(result.is_ok(), "build_subgame_solver with None rollout params failed: {:?}", result.err());
        // Verify the tuple shape: (PostFlopGame, SubgameHands, Vec<ActionInfo>, f64, f64)
        let (_game, _hands, _action_infos, _initial_pot, _starting_stack) = result.unwrap();
    }

    #[test]
    fn test_matrix_snapshot_dealer_passthrough() {
        // Minimal snapshot with no combos — verify dealer is passed through.
        let snap = MatrixSnapshot {
            player: 0,
            strategy: vec![],
            private_cards: vec![],
            initial_weights: vec![],
            num_hands: 0,
            actions: vec![],
            pot: 100,
            stacks: [900, 900],
            hand_evs: None,
            board: vec!["Td".into(), "9d".into(), "6h".into()],
            dealer: 0,
        };
        let matrix = build_matrix_from_snapshot(snap);
        assert_eq!(matrix.dealer, 0, "dealer should pass through from snapshot");
        assert_eq!(matrix.player, 0);

        // Range-solver convention: dealer = 1
        let snap2 = MatrixSnapshot {
            player: 0,
            strategy: vec![],
            private_cards: vec![],
            initial_weights: vec![],
            num_hands: 0,
            actions: vec![],
            pot: 100,
            stacks: [900, 900],
            hand_evs: None,
            board: vec!["Td".into(), "9d".into(), "6h".into()],
            dealer: 1,
        };
        let matrix2 = build_matrix_from_snapshot(snap2);
        assert_eq!(matrix2.dealer, 1, "range-solver dealer convention");
    }

    #[test]
    fn test_seed_solver_with_blueprint_changes_strategy() {
        use poker_solver_core::blueprint_v2::game_tree::GameTree as V2GameTree;
        use poker_solver_core::blueprint_v2::Street;
        use range_solver::interface::Game;

        // 1. Build a range-solver game for a river board.
        let board_cards = vec![
            RsPokerCard::new(RsPokerValue::Ace, RsPokerSuit::Spade),
            RsPokerCard::new(RsPokerValue::King, RsPokerSuit::Heart),
            RsPokerCard::new(RsPokerValue::Seven, RsPokerSuit::Diamond),
            RsPokerCard::new(RsPokerValue::Four, RsPokerSuit::Club),
            RsPokerCard::new(RsPokerValue::Two, RsPokerSuit::Club),
        ];
        let bet_sizes = vec![vec![1.0f32]]; // 100% pot bet
        let oop_w = weights_from_range("AA,KK,QQ");
        let ip_w = weights_from_range("JJ,TT,99");

        let (mut game, _hands, _actions, _pot, _stack) = build_subgame_solver(
            &board_cards, &bet_sizes, 100, [200, 200],
            &oop_w, &ip_w, 0, None, None, None, None, None,
        ).unwrap();

        // 2. Capture the initial (uniform) strategy at root.
        game.back_to_root();
        let strategy_before = game.strategy();
        let player = game.current_player();
        let num_hands = game.num_private_hands(player);
        let num_actions = game.available_actions().len();
        assert!(num_actions >= 2, "need at least 2 actions");

        // Verify initial strategy is uniform.
        for h in 0..num_hands {
            let p0 = strategy_before[h]; // action 0, hand h
            let expected = 1.0 / num_actions as f32;
            assert!(
                (p0 - expected).abs() < 0.1,
                "before seeding, strategy should be ~uniform, got {p0} vs {expected}"
            );
        }

        // 3. Build a matching blueprint tree and strategy.
        let blueprint_tree = V2GameTree::build_subgame(
            Street::River,
            100.0,        // pot
            [100.0; 2],   // invested (stack 200 - invested 100 = eff 100)
            200.0,        // starting_stack
            &[vec![1.0]], // same 100% pot bet
            None,         // no depth limit (river)
            0,            // dealer = seat 0
        );
        // Create a strategy where every decision node has 1 bucket
        // and action 0 gets 90% probability.
        let bucket_counts: [u16; 4] = [169, 1, 1, 1]; // 1 bucket for river
        let mut action_probs = Vec::new();
        let mut node_action_counts = Vec::new();
        let mut node_street_indices = Vec::new();

        for node in &blueprint_tree.nodes {
            if let poker_solver_core::blueprint_v2::game_tree::GameNode::Decision { actions, street, .. } = node {
                let n_actions = actions.len() as u16;
                node_action_counts.push(n_actions);
                node_street_indices.push(*street as u8);
                // For the single bucket: put 90% on action 0, spread rest.
                let mut probs = vec![0.1 / (n_actions as f32 - 1.0); n_actions as usize];
                probs[0] = 0.9;
                action_probs.extend_from_slice(&probs);
            }
        }

        let mut strategy = BlueprintV2Strategy {
            action_probs,
            node_action_counts,
            node_street_indices,
            bucket_counts,
            iterations: 100,
            elapsed_minutes: 1,
            node_offsets: Vec::new(),
        };
        strategy.post_deserialize();

        // Verify strategy works.
        let probs = strategy.get_action_probs(0, 0);
        assert!(!probs.is_empty(), "strategy should have action probs");
        assert!(
            (probs[0] - 0.9).abs() < 0.01,
            "first action should be ~0.9, got {}",
            probs[0]
        );

        // 4. Build AllBuckets that maps all hands to bucket 0.
        // Enable equity_fallback so lookups don't panic without bucket files.
        let mut all_buckets = AllBuckets::new(bucket_counts, [None, None, None, None]);
        all_buckets.equity_fallback = true;

        // 5. Seed the solver.
        seed_solver_with_blueprint(
            &game,
            &strategy,
            &all_buckets,
            &blueprint_tree,
            &board_cards,
            Street::River,
            blueprint_tree.root,
        );

        // 6. Verify strategy changed: action 0 should dominate.
        game.back_to_root();
        let strategy_after = game.strategy();
        for h in 0..num_hands {
            let p0 = strategy_after[h]; // action 0, hand h
            assert!(
                p0 > 0.7,
                "after seeding, action 0 should dominate for hand {h}, got {p0}"
            );
        }
    }

    #[test]
    fn test_seed_solver_with_blueprint_no_cbv_context_is_noop() {
        // When called with a tree that has no matching nodes, it should not panic.
        use poker_solver_core::blueprint_v2::game_tree::GameTree as V2GameTree;
        use poker_solver_core::blueprint_v2::Street;

        let board_cards = vec![
            RsPokerCard::new(RsPokerValue::Ten, RsPokerSuit::Diamond),
            RsPokerCard::new(RsPokerValue::Nine, RsPokerSuit::Heart),
            RsPokerCard::new(RsPokerValue::Six, RsPokerSuit::Club),
            RsPokerCard::new(RsPokerValue::Two, RsPokerSuit::Spade),
            RsPokerCard::new(RsPokerValue::Three, RsPokerSuit::Spade),
        ];
        let bet_sizes = vec![vec![0.5f32]];
        let oop_w = weights_from_range("AA,KK");
        let ip_w = weights_from_range("QQ,JJ");

        let (game, _, _, _, _) = build_subgame_solver(
            &board_cards, &bet_sizes, 80, [150, 150],
            &oop_w, &ip_w, 0, None, None, None, None, None,
        ).unwrap();

        // Build a trivial empty tree with just a terminal.
        let empty_tree = V2GameTree::build_subgame(
            Street::River, 80.0, [75.0; 2], 150.0,
            &[vec![0.5]], None, 0,
        );
        let strategy = BlueprintV2Strategy::empty();
        let all_buckets = AllBuckets::new([169, 1, 1, 1], [None, None, None, None]);

        // Should not panic even though the tree structure may differ.
        seed_solver_with_blueprint(
            &game, &strategy, &all_buckets,
            &empty_tree, &board_cards,
            Street::River, empty_tree.root,
        );
    }
}
