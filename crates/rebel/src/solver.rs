// solver.rs — PBS → range-solver conversion and subgame solving.
//
// Converts a PBS (public belief state) into the data structures required by
// range-solver, solves the resulting subgame with DCFR, and extracts
// per-combo counterfactual values.
//
// Supports:
// - River PBSs (5 cards): direct solving, no depth limit needed.
// - Turn/Flop PBSs (4/3 cards): depth-limited solving with a LeafEvaluator
//   providing CFVs at street boundaries.

use crate::pbs::{Pbs, NUM_COMBOS};
use poker_solver_core::blueprint_v2::LeafEvaluator;
use poker_solver_core::poker::Card as RsPokerCard;
use range_solver::{
    action_tree::{ActionTree, BoardState, TreeConfig},
    bet_size::BetSizeOptions,
    card::{card_pair_to_index, Card, NOT_DEALT},
    range::Range,
    solve, CardConfig, PostFlopGame,
};

/// Configuration for the range-solver wrapper used in ReBeL subgame solving.
#[derive(Clone)]
pub struct SolveConfig {
    /// River bet size options (also used as the sole bet sizes for river-only solves).
    pub bet_sizes: BetSizeOptions,
    /// Turn bet size options (used when solving turn PBSs with depth_limit).
    pub turn_bet_sizes: Option<[BetSizeOptions; 2]>,
    /// Flop bet size options (used when solving flop PBSs with depth_limit).
    pub flop_bet_sizes: Option<[BetSizeOptions; 2]>,
    /// Maximum number of solver iterations.
    pub solver_iterations: u32,
    /// Target exploitability (pot-relative); solver stops early if reached.
    pub target_exploitability: f32,
    /// Add all-in if max bet / pot <= this threshold.
    pub add_allin_threshold: f64,
    /// Force all-in if SPR after call <= this threshold.
    pub force_allin_threshold: f64,
}

/// Result of solving a single river PBS.
pub struct SolveResult {
    /// Pot-relative counterfactual values for OOP, indexed by combo (0..1326).
    /// Board-blocked combos have CFV = 0.0.
    pub oop_cfvs: [f32; NUM_COMBOS],
    /// Pot-relative counterfactual values for IP, indexed by combo (0..1326).
    /// Board-blocked combos have CFV = 0.0.
    pub ip_cfvs: [f32; NUM_COMBOS],
    /// Weighted game value for OOP: sum(cfv[i] * reach[i]).
    pub oop_game_value: f32,
    /// Weighted game value for IP: sum(cfv[i] * reach[i]).
    pub ip_game_value: f32,
    /// Final exploitability returned by the solver.
    pub exploitability: f32,
}

/// Solve a river PBS using range-solver DCFR.
///
/// Converts PBS reach probabilities to range-solver ranges,
/// builds a river PostFlopGame (no depth limit), solves,
/// and extracts per-combo counterfactual values.
///
/// Board-blocked combos get CFV = 0.0.
/// If `effective_stack <= 0`, returns an error.
pub fn solve_river_pbs(pbs: &Pbs, config: &SolveConfig) -> Result<SolveResult, String> {
    if pbs.pot <= 0 {
        return Err("pot must be positive for solver".into());
    }

    if pbs.effective_stack <= 0 {
        return Err("effective_stack must be positive for solver".into());
    }

    if pbs.board.len() != 5 {
        return Err(format!(
            "river PBS requires exactly 5 board cards, got {}",
            pbs.board.len()
        ));
    }

    // Convert PBS reach probabilities ([f32; 1326]) to Range objects.
    // Range::from_raw_data expects a &[f32] of length 1326 with values in [0.0, 1.0].
    // PBS reach probs are already in [0.0, 1.0] since they come from probability products.
    let oop_range = Range::from_raw_data(&pbs.reach_probs[0])?;
    let ip_range = Range::from_raw_data(&pbs.reach_probs[1])?;

    // Build CardConfig — board cards are already u8 with encoding 4*rank + suit,
    // which matches range-solver's Card = u8 type exactly.
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [pbs.board[0], pbs.board[1], pbs.board[2]],
        turn: pbs.board[3],
        river: pbs.board[4],
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: pbs.pot,
        effective_stack: pbs.effective_stack,
        river_bet_sizes: [config.bet_sizes.clone(), config.bet_sizes.clone()],
        add_allin_threshold: config.add_allin_threshold,
        force_allin_threshold: config.force_allin_threshold,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config)?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)?;
    game.allocate_memory(false);

    let exploitability =
        solve(&mut game, config.solver_iterations, config.target_exploitability, false);

    // Must cache normalized weights before calling expected_values().
    game.cache_normalized_weights();

    // Extract per-combo EVs from the solver. These are in the solver's
    // internal combo ordering (private_cards), so we map them back to
    // canonical 1326 ordering.
    let raw_oop = game.expected_values(0);
    let raw_ip = game.expected_values(1);
    let oop_hands = game.private_cards(0);
    let ip_hands = game.private_cards(1);

    let pot = pbs.pot as f32;

    let mut oop_cfvs = [0.0f32; NUM_COMBOS];
    let mut ip_cfvs = [0.0f32; NUM_COMBOS];

    map_evs_to_combos(&raw_oop, oop_hands, pot, &mut oop_cfvs);
    map_evs_to_combos(&raw_ip, ip_hands, pot, &mut ip_cfvs);

    let oop_game_value = weighted_game_value(&oop_cfvs, &pbs.reach_probs[0]);
    let ip_game_value = weighted_game_value(&ip_cfvs, &pbs.reach_probs[1]);

    Ok(SolveResult {
        oop_cfvs,
        ip_cfvs,
        oop_game_value,
        ip_game_value,
        exploitability,
    })
}

/// Solve a PBS with depth-limited CFR using a LeafEvaluator at boundaries.
///
/// - For river PBSs (5 cards): delegates to `solve_river_pbs` (no boundaries needed).
/// - For turn PBSs (4 cards): builds turn tree with `depth_limit=0`, evaluator at river boundary.
/// - For flop PBSs (3 cards): builds flop tree with `depth_limit=0`, evaluator at turn boundary.
///
/// The evaluator provides CFVs at depth boundary nodes where the tree is truncated.
/// Board-blocked combos get CFV = 0.0.
pub fn solve_depth_limited_pbs(
    pbs: &Pbs,
    config: &SolveConfig,
    evaluator: &dyn LeafEvaluator,
) -> Result<SolveResult, String> {
    if pbs.pot <= 0 {
        return Err("pot must be positive for solver".into());
    }
    if pbs.effective_stack <= 0 {
        return Err("effective_stack must be positive for solver".into());
    }

    match pbs.board.len() {
        5 => solve_river_pbs(pbs, config),
        4 => solve_turn_pbs(pbs, config, evaluator),
        3 => solve_flop_pbs(pbs, config, evaluator),
        n => Err(format!(
            "depth-limited solving requires 3-5 board cards, got {n}"
        )),
    }
}

/// Solve a turn PBS (4 board cards) with depth_limit=0 and evaluator at river boundary.
fn solve_turn_pbs(
    pbs: &Pbs,
    config: &SolveConfig,
    evaluator: &dyn LeafEvaluator,
) -> Result<SolveResult, String> {
    let oop_range = Range::from_raw_data(&pbs.reach_probs[0])?;
    let ip_range = Range::from_raw_data(&pbs.reach_probs[1])?;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [pbs.board[0], pbs.board[1], pbs.board[2]],
        turn: pbs.board[3],
        river: NOT_DEALT,
    };

    // Use turn-specific bet sizes if available, otherwise fall back to river sizes.
    let turn_sizes = config
        .turn_bet_sizes
        .clone()
        .unwrap_or_else(|| [config.bet_sizes.clone(), config.bet_sizes.clone()]);

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: pbs.pot,
        effective_stack: pbs.effective_stack,
        turn_bet_sizes: turn_sizes,
        depth_limit: Some(0),
        add_allin_threshold: config.add_allin_threshold,
        force_allin_threshold: config.force_allin_threshold,
        ..Default::default()
    };

    solve_with_boundaries(pbs, config, evaluator, card_config, tree_config)
}

/// Solve a flop PBS (3 board cards) with depth_limit=0 and evaluator at turn boundary.
fn solve_flop_pbs(
    pbs: &Pbs,
    config: &SolveConfig,
    evaluator: &dyn LeafEvaluator,
) -> Result<SolveResult, String> {
    let oop_range = Range::from_raw_data(&pbs.reach_probs[0])?;
    let ip_range = Range::from_raw_data(&pbs.reach_probs[1])?;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [pbs.board[0], pbs.board[1], pbs.board[2]],
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    // Use flop-specific bet sizes if available, otherwise fall back to river sizes.
    let flop_sizes = config
        .flop_bet_sizes
        .clone()
        .unwrap_or_else(|| [config.bet_sizes.clone(), config.bet_sizes.clone()]);

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: pbs.pot,
        effective_stack: pbs.effective_stack,
        flop_bet_sizes: flop_sizes,
        depth_limit: Some(0),
        add_allin_threshold: config.add_allin_threshold,
        force_allin_threshold: config.force_allin_threshold,
        ..Default::default()
    };

    solve_with_boundaries(pbs, config, evaluator, card_config, tree_config)
}

/// A prepared game ready for boundary evaluation and solving.
/// Separates game construction from GPU evaluation to enable batching.
pub struct PreparedGame {
    pub game: PostFlopGame,
    pub pbs: Pbs,
    pub config: SolveConfig,
}

/// Prepare a depth-limited game for solving (build tree, allocate memory).
/// Does NOT evaluate boundaries — call `set_boundaries` or batch-evaluate separately.
pub fn prepare_game(
    pbs: &Pbs,
    config: &SolveConfig,
) -> Result<PreparedGame, String> {
    if pbs.pot <= 0 {
        return Err("pot must be positive for solver".into());
    }
    if pbs.effective_stack <= 0 {
        return Err("effective_stack must be positive for solver".into());
    }

    let oop_range = Range::from_raw_data(&pbs.reach_probs[0])?;
    let ip_range = Range::from_raw_data(&pbs.reach_probs[1])?;

    let (card_config, tree_config) = match pbs.board.len() {
        5 => {
            let cc = CardConfig {
                range: [oop_range, ip_range],
                flop: [pbs.board[0], pbs.board[1], pbs.board[2]],
                turn: pbs.board[3],
                river: pbs.board[4],
            };
            let tc = TreeConfig {
                initial_state: BoardState::River,
                starting_pot: pbs.pot,
                effective_stack: pbs.effective_stack,
                river_bet_sizes: [config.bet_sizes.clone(), config.bet_sizes.clone()],
                add_allin_threshold: config.add_allin_threshold,
                force_allin_threshold: config.force_allin_threshold,
                ..Default::default()
            };
            (cc, tc)
        }
        4 => {
            let cc = CardConfig {
                range: [oop_range, ip_range],
                flop: [pbs.board[0], pbs.board[1], pbs.board[2]],
                turn: pbs.board[3],
                river: NOT_DEALT,
            };
            let turn_sizes = config
                .turn_bet_sizes
                .clone()
                .unwrap_or_else(|| [config.bet_sizes.clone(), config.bet_sizes.clone()]);
            let tc = TreeConfig {
                initial_state: BoardState::Turn,
                starting_pot: pbs.pot,
                effective_stack: pbs.effective_stack,
                turn_bet_sizes: turn_sizes,
                depth_limit: Some(0),
                add_allin_threshold: config.add_allin_threshold,
                force_allin_threshold: config.force_allin_threshold,
                ..Default::default()
            };
            (cc, tc)
        }
        3 => {
            let cc = CardConfig {
                range: [oop_range, ip_range],
                flop: [pbs.board[0], pbs.board[1], pbs.board[2]],
                turn: NOT_DEALT,
                river: NOT_DEALT,
            };
            let flop_sizes = config
                .flop_bet_sizes
                .clone()
                .unwrap_or_else(|| [config.bet_sizes.clone(), config.bet_sizes.clone()]);
            let tc = TreeConfig {
                initial_state: BoardState::Flop,
                starting_pot: pbs.pot,
                effective_stack: pbs.effective_stack,
                flop_bet_sizes: flop_sizes,
                depth_limit: Some(0),
                add_allin_threshold: config.add_allin_threshold,
                force_allin_threshold: config.force_allin_threshold,
                ..Default::default()
            };
            (cc, tc)
        }
        n => return Err(format!("requires 3-5 board cards, got {n}")),
    };

    let action_tree = ActionTree::new(tree_config)?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)?;
    game.allocate_memory(false);

    Ok(PreparedGame {
        game,
        pbs: pbs.clone(),
        config: config.clone(),
    })
}

/// Collect boundary evaluation requests from a prepared game.
/// Returns (board_cards_rs, combos_rs, oop_range, ip_range, requests) per player.
/// Empty if the game has no boundary nodes (e.g., river).
pub fn boundary_requests(
    prepared: &PreparedGame,
) -> Vec<BoundaryRequest> {
    let game = &prepared.game;
    let pbs = &prepared.pbs;
    let n_boundary = game.num_boundary_nodes();
    if n_boundary == 0 {
        return Vec::new();
    }

    let board_cards: Vec<RsPokerCard> = pbs.board.iter().map(|&c| u8_to_rs_poker_card(c)).collect();
    let starting_pot = game.tree_config().starting_pot;
    let eff_stack = game.tree_config().effective_stack;

    let mut out = Vec::new();
    for player in 0..2usize {
        let hands = game.private_cards(player);
        let combos_rs: Vec<[RsPokerCard; 2]> = hands
            .iter()
            .map(|&(c1, c2)| [u8_to_rs_poker_card(c1), u8_to_rs_poker_card(c2)])
            .collect();
        let oop_range: Vec<f64> = hands
            .iter()
            .map(|&(c1, c2)| pbs.reach_probs[0][card_pair_to_index(c1, c2)] as f64)
            .collect();
        let ip_range: Vec<f64> = hands
            .iter()
            .map(|&(c1, c2)| pbs.reach_probs[1][card_pair_to_index(c1, c2)] as f64)
            .collect();
        let requests: Vec<(f64, f64, u8)> = (0..n_boundary)
            .map(|ordinal| {
                let bpot = game.boundary_pot(ordinal);
                let amount = (bpot - starting_pot) / 2;
                let boundary_eff_stack = eff_stack - amount;
                (bpot as f64, boundary_eff_stack as f64, player as u8)
            })
            .collect();

        out.push(BoundaryRequest {
            board_cards: board_cards.clone(),
            combos: combos_rs,
            oop_range,
            ip_range,
            requests,
            player,
        });
    }
    out
}

/// A batch of boundary evaluation requests for one player of one game.
pub struct BoundaryRequest {
    pub board_cards: Vec<RsPokerCard>,
    pub combos: Vec<[RsPokerCard; 2]>,
    pub oop_range: Vec<f64>,
    pub ip_range: Vec<f64>,
    pub requests: Vec<(f64, f64, u8)>,
    pub player: usize,
}

/// Set boundary CFVs on a prepared game from pre-computed evaluations.
/// `cfvs_per_player` should have one entry per player (0=OOP, 1=IP),
/// each containing one Vec<f64> per boundary node.
pub fn set_boundaries(
    prepared: &mut PreparedGame,
    cfvs_per_player: &[Vec<Vec<f64>>; 2],
) {
    for player in 0..2 {
        for (ordinal, cfvs) in cfvs_per_player[player].iter().enumerate() {
            let cfvs_f32: Vec<f32> = cfvs.iter().map(|&v| v as f32).collect();
            prepared.game.set_boundary_cfvs(ordinal, player, cfvs_f32);
        }
    }
}

/// Solve a prepared game (boundaries must already be set) and extract results.
pub fn solve_prepared(prepared: &mut PreparedGame) -> SolveResult {
    let exploitability = solve(
        &mut prepared.game,
        prepared.config.solver_iterations,
        prepared.config.target_exploitability,
        false,
    );

    prepared.game.cache_normalized_weights();

    let raw_oop = prepared.game.expected_values(0);
    let raw_ip = prepared.game.expected_values(1);
    let oop_hands = prepared.game.private_cards(0);
    let ip_hands = prepared.game.private_cards(1);
    let pot = prepared.pbs.pot as f32;

    let mut oop_cfvs = [0.0f32; NUM_COMBOS];
    let mut ip_cfvs = [0.0f32; NUM_COMBOS];
    map_evs_to_combos(&raw_oop, oop_hands, pot, &mut oop_cfvs);
    map_evs_to_combos(&raw_ip, ip_hands, pot, &mut ip_cfvs);

    let oop_game_value = weighted_game_value(&oop_cfvs, &prepared.pbs.reach_probs[0]);
    let ip_game_value = weighted_game_value(&ip_cfvs, &prepared.pbs.reach_probs[1]);

    SolveResult {
        oop_cfvs,
        ip_cfvs,
        oop_game_value,
        ip_game_value,
        exploitability,
    }
}

/// Common logic for depth-limited solving: build game, set boundary CFVs, solve, extract.
fn solve_with_boundaries(
    pbs: &Pbs,
    config: &SolveConfig,
    evaluator: &dyn LeafEvaluator,
    card_config: CardConfig,
    tree_config: TreeConfig,
) -> Result<SolveResult, String> {
    let action_tree = ActionTree::new(tree_config)?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)?;
    game.allocate_memory(false);

    let n_boundary = game.num_boundary_nodes();

    if n_boundary > 0 {
        // Convert PBS board to rs_poker Card format for the evaluator.
        let board_cards: Vec<RsPokerCard> = pbs
            .board
            .iter()
            .map(|&c| u8_to_rs_poker_card(c))
            .collect();

        let starting_pot = game.tree_config().starting_pot;
        let eff_stack = game.tree_config().effective_stack;

        // Evaluate boundary CFVs for each player separately, since each player
        // has its own combo ordering in the solver.
        for player in 0..2usize {
            let hands = game.private_cards(player);

            // Convert solver combos to rs_poker Card pairs.
            let combos_rs: Vec<[RsPokerCard; 2]> = hands
                .iter()
                .map(|&(c1, c2)| [u8_to_rs_poker_card(c1), u8_to_rs_poker_card(c2)])
                .collect();

            // Build range arrays in f64 for the evaluator, in solver combo ordering.
            let oop_range_f64: Vec<f64> = hands
                .iter()
                .map(|&(c1, c2)| pbs.reach_probs[0][card_pair_to_index(c1, c2)] as f64)
                .collect();
            let ip_range_f64: Vec<f64> = hands
                .iter()
                .map(|&(c1, c2)| pbs.reach_probs[1][card_pair_to_index(c1, c2)] as f64)
                .collect();

            // Build boundary requests: (pot, eff_stack, traverser).
            let requests: Vec<(f64, f64, u8)> = (0..n_boundary)
                .map(|ordinal| {
                    let bpot = game.boundary_pot(ordinal);
                    let amount = (bpot - starting_pot) / 2;
                    let boundary_eff_stack = eff_stack - amount;
                    (bpot as f64, boundary_eff_stack as f64, player as u8)
                })
                .collect();

            let batch_cfvs = evaluator.evaluate_boundaries(
                &combos_rs,
                &board_cards,
                &oop_range_f64,
                &ip_range_f64,
                &requests,
            );

            // Set boundary CFVs for this player.
            for (ordinal, cfvs_f64) in batch_cfvs.into_iter().enumerate() {
                let cfvs_f32: Vec<f32> = cfvs_f64.iter().map(|&v| v as f32).collect();
                game.set_boundary_cfvs(ordinal, player, cfvs_f32);
            }
        }
    }

    // Solve with DCFR.
    let exploitability = solve(
        &mut game,
        config.solver_iterations,
        config.target_exploitability,
        false,
    );

    // Extract CFVs back to canonical 1326 layout.
    game.cache_normalized_weights();

    let raw_oop = game.expected_values(0);
    let raw_ip = game.expected_values(1);
    let oop_hands = game.private_cards(0);
    let ip_hands = game.private_cards(1);

    let pot = pbs.pot as f32;

    let mut oop_cfvs = [0.0f32; NUM_COMBOS];
    let mut ip_cfvs = [0.0f32; NUM_COMBOS];

    map_evs_to_combos(&raw_oop, oop_hands, pot, &mut oop_cfvs);
    map_evs_to_combos(&raw_ip, ip_hands, pot, &mut ip_cfvs);

    let oop_game_value = weighted_game_value(&oop_cfvs, &pbs.reach_probs[0]);
    let ip_game_value = weighted_game_value(&ip_cfvs, &pbs.reach_probs[1]);

    Ok(SolveResult {
        oop_cfvs,
        ip_cfvs,
        oop_game_value,
        ip_game_value,
        exploitability,
    })
}

/// Convert a range-solver card (u8: 4*rank + suit) to an rs_poker Card.
///
/// range-solver: rank 0=2..12=A, suit 0=club,1=diamond,2=heart,3=spade
/// rs_poker: Value enum (Two..Ace), Suit enum (Club..Spade)
fn u8_to_rs_poker_card(card: u8) -> RsPokerCard {
    use poker_solver_core::poker::{Suit, Value};

    let rank = card / 4;
    let suit = card % 4;

    let value = match rank {
        0 => Value::Two,
        1 => Value::Three,
        2 => Value::Four,
        3 => Value::Five,
        4 => Value::Six,
        5 => Value::Seven,
        6 => Value::Eight,
        7 => Value::Nine,
        8 => Value::Ten,
        9 => Value::Jack,
        10 => Value::Queen,
        11 => Value::King,
        12 => Value::Ace,
        _ => panic!("invalid card rank: {rank}"),
    };

    let suit = match suit {
        0 => Suit::Club,
        1 => Suit::Diamond,
        2 => Suit::Heart,
        3 => Suit::Spade,
        _ => unreachable!(),
    };

    RsPokerCard::new(value, suit)
}

/// Map solver EVs (indexed by private_cards order) into the canonical 1326 layout,
/// dividing by pot to produce pot-relative values.
fn map_evs_to_combos(
    raw_evs: &[f32],
    hands: &[(Card, Card)],
    pot: f32,
    out: &mut [f32; NUM_COMBOS],
) {
    for (i, &(c1, c2)) in hands.iter().enumerate() {
        let idx = card_pair_to_index(c1, c2);
        out[idx] = raw_evs[i] / pot;
    }
}

/// Compute range-weighted game value: sum(cfv[i] * reach[i]).
///
/// This is a plain weighted sum with no normalization, matching the cfvnet
/// training pipeline convention.
pub fn weighted_game_value(cfvs: &[f32; NUM_COMBOS], reach: &[f32; NUM_COMBOS]) -> f32 {
    let mut weighted_sum = 0.0f64;
    for i in 0..NUM_COMBOS {
        weighted_sum += cfvs[i] as f64 * reach[i] as f64;
    }
    weighted_sum as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::card::index_to_card_pair;

    use poker_solver_core::blueprint_v2::LeafEvaluator;
    use poker_solver_core::poker::Card as RsPokerCard;

    /// Build a test solve config with reasonable defaults.
    fn test_solve_config() -> SolveConfig {
        let bet_sizes =
            BetSizeOptions::try_from(("50%,a", "")).expect("valid test bet sizes");
        SolveConfig {
            bet_sizes,
            turn_bet_sizes: None,
            flop_bet_sizes: None,
            solver_iterations: 200,
            target_exploitability: 0.01,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
        }
    }

    /// A trivial evaluator that returns zero CFVs at every boundary.
    struct ZeroEvaluator;

    impl LeafEvaluator for ZeroEvaluator {
        fn evaluate(
            &self,
            combos: &[[RsPokerCard; 2]],
            _board: &[RsPokerCard],
            _pot: f64,
            _effective_stack: f64,
            _oop_range: &[f64],
            _ip_range: &[f64],
            _traverser: u8,
        ) -> Vec<f64> {
            vec![0.0; combos.len()]
        }
    }

    /// Board: Qs Jh 2c 8d 3s (same as cfvnet tests for consistency).
    fn test_board() -> Vec<u8> {
        vec![
            4 * 10 + 3, // Qs = 43
            4 * 9 + 2,  // Jh = 38
            4 * 0 + 0,  // 2c = 0
            4 * 6 + 1,  // 8d = 25
            4 * 1 + 3,  // 3s = 7
        ]
    }

    /// Returns true if combo index `i` is blocked by any board card.
    fn is_blocked(i: usize, board: &[u8]) -> bool {
        let (c1, c2) = index_to_card_pair(i);
        board.contains(&c1) || board.contains(&c2)
    }

    #[test]
    fn test_weighted_game_value_known() {
        let mut cfvs = [0.0f32; NUM_COMBOS];
        let mut reach = [0.0f32; NUM_COMBOS];

        // Set up 3 combos with known values
        cfvs[0] = 10.0;
        reach[0] = 1.0;

        cfvs[1] = 20.0;
        reach[1] = 1.0;

        cfvs[2] = 30.0;
        reach[2] = 2.0;

        // Expected: 10*1 + 20*1 + 30*2 = 90.0 (plain weighted sum, no normalization)
        let gv = weighted_game_value(&cfvs, &reach);
        assert!(
            (gv - 90.0).abs() < 1e-4,
            "expected 90.0, got {}",
            gv
        );
    }

    #[test]
    fn test_weighted_game_value_zero_reach() {
        let cfvs = [1.0f32; NUM_COMBOS];
        let reach = [0.0f32; NUM_COMBOS];
        let gv = weighted_game_value(&cfvs, &reach);
        assert_eq!(gv, 0.0, "zero reach should return 0.0");
    }

    #[test]
    fn test_solve_river_pbs_uniform() {
        let board = test_board();
        let pbs = Pbs::new_uniform(board.clone(), 100, 100);
        let config = test_solve_config();

        let result = solve_river_pbs(&pbs, &config).unwrap();

        // 1326 CFVs returned for each player
        assert_eq!(result.oop_cfvs.len(), NUM_COMBOS);
        assert_eq!(result.ip_cfvs.len(), NUM_COMBOS);

        // Board-blocked combos have 0 CFV
        for i in 0..NUM_COMBOS {
            if is_blocked(i, &board) {
                assert_eq!(
                    result.oop_cfvs[i], 0.0,
                    "blocked combo {} should have OOP CFV 0.0",
                    i
                );
                assert_eq!(
                    result.ip_cfvs[i], 0.0,
                    "blocked combo {} should have IP CFV 0.0",
                    i
                );
            }
        }

        // Some non-blocked combos should have non-zero CFVs
        let has_nonzero_oop = result
            .oop_cfvs
            .iter()
            .enumerate()
            .any(|(i, &v)| !is_blocked(i, &board) && v != 0.0);
        let has_nonzero_ip = result
            .ip_cfvs
            .iter()
            .enumerate()
            .any(|(i, &v)| !is_blocked(i, &board) && v != 0.0);
        assert!(has_nonzero_oop, "expected some non-zero OOP CFVs");
        assert!(has_nonzero_ip, "expected some non-zero IP CFVs");

        // Game values are finite
        assert!(
            result.oop_game_value.is_finite(),
            "OOP game value not finite: {}",
            result.oop_game_value
        );
        assert!(
            result.ip_game_value.is_finite(),
            "IP game value not finite: {}",
            result.ip_game_value
        );

        // Exploitability is below a reasonable threshold for 200 iterations
        assert!(
            result.exploitability < 5.0,
            "exploitability {} too high for 200 iterations",
            result.exploitability
        );
    }

    #[test]
    fn test_solve_river_pbs_asymmetric_reach() {
        let board = test_board();

        // Create PBS with asymmetric reach: OOP has all combos at 1.0,
        // IP has half combos at 0.5, other half at 1.0.
        let mut pbs = Pbs::new_uniform(board.clone(), 100, 100);
        for i in 0..NUM_COMBOS {
            if !is_blocked(i, &board) && i % 2 == 0 {
                pbs.reach_probs[1][i] = 0.5;
            }
        }

        let config = test_solve_config();
        let result = solve_river_pbs(&pbs, &config).unwrap();

        // Should still produce valid results
        assert!(
            result.oop_game_value.is_finite(),
            "OOP game value not finite"
        );
        assert!(
            result.ip_game_value.is_finite(),
            "IP game value not finite"
        );
        assert!(
            result.exploitability >= 0.0,
            "exploitability should be non-negative"
        );

        // With asymmetric reach the game values should differ from uniform case
        // (at least verify they're not trivially zero)
        let oop_has_values = result.oop_cfvs.iter().any(|&v| v != 0.0);
        let ip_has_values = result.ip_cfvs.iter().any(|&v| v != 0.0);
        assert!(oop_has_values, "expected non-zero OOP CFVs with asymmetric reach");
        assert!(ip_has_values, "expected non-zero IP CFVs with asymmetric reach");
    }

    #[test]
    fn test_solve_river_pbs_zero_stack_error() {
        let board = test_board();
        let pbs = Pbs::new_uniform(board, 100, 0);
        let config = test_solve_config();
        let result = solve_river_pbs(&pbs, &config);
        assert!(result.is_err(), "expected error for zero effective_stack");
    }

    #[test]
    fn test_solve_river_pbs_zero_pot_error() {
        let board = test_board();
        let pbs = Pbs::new_uniform(board, 0, 100);
        let config = test_solve_config();
        let result = solve_river_pbs(&pbs, &config);
        assert!(result.is_err(), "expected error for zero pot");
    }

    #[test]
    fn test_solve_river_pbs_negative_pot_error() {
        let board = test_board();
        let pbs = Pbs::new_uniform(board, -50, 100);
        let config = test_solve_config();
        let result = solve_river_pbs(&pbs, &config);
        assert!(result.is_err(), "expected error for negative pot");
    }

    #[test]
    fn test_solve_river_pbs_wrong_board_size_error() {
        // Only 4 cards — not a river board
        let board = vec![43, 38, 0, 25];
        let pbs = Pbs::new_uniform(board, 100, 100);
        let config = test_solve_config();
        let result = solve_river_pbs(&pbs, &config);
        assert!(result.is_err(), "expected error for non-river board");
    }

    // -----------------------------------------------------------------------
    // Depth-limited solving tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_depth_limited_river_delegates() {
        // A 5-card board should work the same as solve_river_pbs.
        let board = test_board();
        let pbs = Pbs::new_uniform(board.clone(), 100, 100);
        let config = test_solve_config();
        let evaluator = ZeroEvaluator;

        let result = solve_depth_limited_pbs(&pbs, &config, &evaluator).unwrap();

        // Should produce valid output (same as river solver)
        assert_eq!(result.oop_cfvs.len(), NUM_COMBOS);
        assert_eq!(result.ip_cfvs.len(), NUM_COMBOS);

        // Board-blocked combos should be zero
        for i in 0..NUM_COMBOS {
            if is_blocked(i, &board) {
                assert_eq!(result.oop_cfvs[i], 0.0);
                assert_eq!(result.ip_cfvs[i], 0.0);
            }
        }

        // Some non-blocked combos should be non-zero
        let has_nonzero = result
            .oop_cfvs
            .iter()
            .enumerate()
            .any(|(i, &v)| !is_blocked(i, &board) && v != 0.0);
        assert!(has_nonzero, "expected non-zero CFVs from river delegation");

        assert!(result.exploitability.is_finite());
        assert!(result.oop_game_value.is_finite());
        assert!(result.ip_game_value.is_finite());
    }

    #[test]
    fn test_solve_depth_limited_turn() {
        // 4-card board: turn PBS
        // Board: Qs Jh 2c 8d
        let board = vec![
            4 * 10 + 3, // Qs = 43
            4 * 9 + 2,  // Jh = 38
            4 * 0 + 0,  // 2c = 0
            4 * 6 + 1,  // 8d = 25
        ];
        let pbs = Pbs::new_uniform(board.clone(), 100, 200);
        let config = test_solve_config();
        let evaluator = ZeroEvaluator;

        let result = solve_depth_limited_pbs(&pbs, &config, &evaluator).unwrap();

        // Should produce valid output
        assert_eq!(result.oop_cfvs.len(), NUM_COMBOS);
        assert_eq!(result.ip_cfvs.len(), NUM_COMBOS);

        // Board-blocked combos should be zero
        for i in 0..NUM_COMBOS {
            if is_blocked(i, &board) {
                assert_eq!(
                    result.oop_cfvs[i], 0.0,
                    "blocked combo {i} should have OOP CFV 0.0"
                );
                assert_eq!(
                    result.ip_cfvs[i], 0.0,
                    "blocked combo {i} should have IP CFV 0.0"
                );
            }
        }

        // Game values should be finite
        assert!(
            result.oop_game_value.is_finite(),
            "OOP game value not finite: {}",
            result.oop_game_value
        );
        assert!(
            result.ip_game_value.is_finite(),
            "IP game value not finite: {}",
            result.ip_game_value
        );
        assert!(
            result.exploitability.is_finite(),
            "exploitability not finite: {}",
            result.exploitability
        );
    }

    #[test]
    fn test_solve_depth_limited_flop() {
        // 3-card board: flop PBS
        // Board: Qs Jh 2c
        let board = vec![
            4 * 10 + 3, // Qs = 43
            4 * 9 + 2,  // Jh = 38
            4 * 0 + 0,  // 2c = 0
        ];
        let pbs = Pbs::new_uniform(board.clone(), 100, 200);
        let config = test_solve_config();
        let evaluator = ZeroEvaluator;

        let result = solve_depth_limited_pbs(&pbs, &config, &evaluator).unwrap();

        // Should produce valid output
        assert_eq!(result.oop_cfvs.len(), NUM_COMBOS);
        assert_eq!(result.ip_cfvs.len(), NUM_COMBOS);

        // Board-blocked combos should be zero
        for i in 0..NUM_COMBOS {
            if is_blocked(i, &board) {
                assert_eq!(result.oop_cfvs[i], 0.0);
                assert_eq!(result.ip_cfvs[i], 0.0);
            }
        }

        // Game values should be finite
        assert!(result.oop_game_value.is_finite());
        assert!(result.ip_game_value.is_finite());
        assert!(result.exploitability.is_finite());
    }

    #[test]
    fn test_solve_depth_limited_invalid_board_size() {
        // 2 cards — not a valid street
        let board = vec![43, 38];
        let pbs = Pbs::new_uniform(board, 100, 100);
        let config = test_solve_config();
        let evaluator = ZeroEvaluator;
        let result = solve_depth_limited_pbs(&pbs, &config, &evaluator);
        assert!(result.is_err(), "expected error for 2-card board");
    }

    #[test]
    fn test_solve_depth_limited_zero_pot_error() {
        let board = vec![43, 38, 0, 25]; // turn board
        let pbs = Pbs::new_uniform(board, 0, 100);
        let config = test_solve_config();
        let evaluator = ZeroEvaluator;
        let result = solve_depth_limited_pbs(&pbs, &config, &evaluator);
        assert!(result.is_err(), "expected error for zero pot");
    }

    #[test]
    fn test_solve_depth_limited_zero_stack_error() {
        let board = vec![43, 38, 0, 25]; // turn board
        let pbs = Pbs::new_uniform(board, 100, 0);
        let config = test_solve_config();
        let evaluator = ZeroEvaluator;
        let result = solve_depth_limited_pbs(&pbs, &config, &evaluator);
        assert!(result.is_err(), "expected error for zero stack");
    }

    #[test]
    fn test_u8_to_rs_poker_card_roundtrip() {
        // Verify card conversion for a few known cards
        use poker_solver_core::poker::{Suit, Value};

        // 2c = 0
        let card = u8_to_rs_poker_card(0);
        assert_eq!(card.value, Value::Two);
        assert_eq!(card.suit, Suit::Club);

        // As = 51
        let card = u8_to_rs_poker_card(51);
        assert_eq!(card.value, Value::Ace);
        assert_eq!(card.suit, Suit::Spade);

        // Kh = 4*11 + 2 = 46
        let card = u8_to_rs_poker_card(46);
        assert_eq!(card.value, Value::King);
        assert_eq!(card.suit, Suit::Heart);

        // 8d = 4*6 + 1 = 25
        let card = u8_to_rs_poker_card(25);
        assert_eq!(card.value, Value::Eight);
        assert_eq!(card.suit, Suit::Diamond);
    }
}
