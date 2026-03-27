// subgame_solve.rs — Subgame solving for live self-play (Algorithm 1).
//
// At each PBS during self-play, we solve a depth-limited subgame and extract:
// 1. Root CFVs (training targets for the value network)
// 2. Root strategy (for action sampling during self-play)
//
// This module wraps the range-solver, reusing the game-building and
// boundary-evaluation logic from solver.rs, but additionally extracts
// the average strategy at the root node.

use crate::pbs::{Pbs, NUM_COMBOS};
use crate::solver::SolveConfig;
use poker_solver_core::blueprint_v2::LeafEvaluator;
use poker_solver_core::poker::Card as RsPokerCard;
use range_solver::{
    action_tree::{ActionTree, BoardState, TreeConfig},
    card::{card_pair_to_index, Card, NOT_DEALT},
    range::Range,
    solve, CardConfig, PostFlopGame,
};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of solving a subgame at a PBS for self-play.
///
/// Contains everything the self-play loop needs: root CFVs for value network
/// training targets, and the root strategy for action sampling.
pub struct SubgameSolveResult {
    /// Pot-relative CFVs at root for each player.
    /// `root_cfvs[player][combo]` where combo is the canonical 0-1325 index.
    /// Board-blocked combos have CFV = 0.0.
    pub root_cfvs: Box<[[f32; NUM_COMBOS]; 2]>,

    /// Average strategy at root, stored as a flat array.
    /// Layout: `root_strategy[combo * num_actions + action] = probability`.
    /// Indexed by canonical combo index (0-1325).
    /// Board-blocked combos have uniform strategy (1/num_actions per action).
    pub root_strategy: Vec<f32>,

    /// Number of legal actions at the root decision node.
    pub num_actions: usize,
}

// ---------------------------------------------------------------------------
// Strategy access helpers
// ---------------------------------------------------------------------------

/// Get action probabilities for a specific combo from the solved strategy.
///
/// Returns a slice of length `result.num_actions` whose entries sum to ~1.0.
/// Used by the self-play loop to sample actions.
#[inline]
pub fn get_action_probs(result: &SubgameSolveResult, combo_idx: usize) -> &[f32] {
    let start = combo_idx * result.num_actions;
    &result.root_strategy[start..start + result.num_actions]
}

// ---------------------------------------------------------------------------
// Subgame solving
// ---------------------------------------------------------------------------

/// Solve a depth-limited subgame at a PBS and extract results for self-play.
///
/// Builds a `PostFlopGame` for the current street, sets boundary CFVs from
/// the evaluator (for non-river streets), runs T iterations of DCFR, then
/// extracts root CFVs and the average strategy.
///
/// This is the "search" component of Algorithm 1 in the ReBeL paper.
///
/// # Arguments
///
/// * `pbs` - The public belief state to solve at
/// * `config` - Solver configuration (iterations, bet sizes, etc.)
/// * `evaluator` - Leaf evaluator for depth boundaries (unused for river PBSs)
///
/// # Returns
///
/// A `SubgameSolveResult` containing root CFVs and strategy, or an error
/// if the PBS is invalid (e.g. non-positive pot/stack).
pub fn solve_subgame(
    pbs: &Pbs,
    config: &SolveConfig,
    evaluator: &dyn LeafEvaluator,
) -> Result<SubgameSolveResult, String> {
    if pbs.pot <= 0 {
        return Err("pot must be positive for solver".into());
    }
    if pbs.effective_stack <= 0 {
        return Err("effective_stack must be positive for solver".into());
    }

    match pbs.board.len() {
        5 => solve_subgame_river(pbs, config),
        4 => solve_subgame_with_boundaries(pbs, config, evaluator, Street::Turn),
        3 => solve_subgame_with_boundaries(pbs, config, evaluator, Street::Flop),
        n => Err(format!(
            "subgame solving requires 3-5 board cards, got {n}"
        )),
    }
}

/// Street indicator for selecting bet sizes and board state.
enum Street {
    Turn,
    Flop,
}

/// Solve a river subgame (5 board cards, no depth boundaries).
fn solve_subgame_river(pbs: &Pbs, config: &SolveConfig) -> Result<SubgameSolveResult, String> {
    let oop_range = Range::from_raw_data(&pbs.reach_probs[0])?;
    let ip_range = Range::from_raw_data(&pbs.reach_probs[1])?;

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

    solve(
        &mut game,
        config.solver_iterations,
        config.target_exploitability,
        false,
    );

    extract_result(pbs, &mut game)
}

/// Solve a turn or flop subgame with depth boundaries.
fn solve_subgame_with_boundaries(
    pbs: &Pbs,
    config: &SolveConfig,
    evaluator: &dyn LeafEvaluator,
    street: Street,
) -> Result<SubgameSolveResult, String> {
    let oop_range = Range::from_raw_data(&pbs.reach_probs[0])?;
    let ip_range = Range::from_raw_data(&pbs.reach_probs[1])?;

    let (card_config, tree_config) = match street {
        Street::Turn => {
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
        Street::Flop => {
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
    };

    let action_tree = ActionTree::new(tree_config)?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)?;
    game.allocate_memory(false);

    // Set boundary CFVs from the evaluator.
    set_boundary_cfvs(pbs, &mut game, evaluator);

    solve(
        &mut game,
        config.solver_iterations,
        config.target_exploitability,
        false,
    );

    extract_result(pbs, &mut game)
}

/// Set boundary CFVs on the game using the evaluator.
///
/// Mirrors the logic in solver.rs `solve_with_boundaries`.
fn set_boundary_cfvs(pbs: &Pbs, game: &mut PostFlopGame, evaluator: &dyn LeafEvaluator) {
    let n_boundary = game.num_boundary_nodes();
    if n_boundary == 0 {
        return;
    }

    let board_cards: Vec<RsPokerCard> = pbs.board.iter().map(|&c| u8_to_rs_poker_card(c)).collect();
    let starting_pot = game.tree_config().starting_pot;
    let eff_stack = game.tree_config().effective_stack;

    for player in 0..2usize {
        let hands = game.private_cards(player);

        let combos_rs: Vec<[RsPokerCard; 2]> = hands
            .iter()
            .map(|&(c1, c2)| [u8_to_rs_poker_card(c1), u8_to_rs_poker_card(c2)])
            .collect();

        let oop_range_f64: Vec<f64> = hands
            .iter()
            .map(|&(c1, c2)| pbs.reach_probs[0][card_pair_to_index(c1, c2)] as f64)
            .collect();
        let ip_range_f64: Vec<f64> = hands
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

        let batch_cfvs = evaluator.evaluate_boundaries(
            &combos_rs,
            &board_cards,
            &oop_range_f64,
            &ip_range_f64,
            &requests,
        );

        for (ordinal, cfvs_f64) in batch_cfvs.into_iter().enumerate() {
            let cfvs_f32: Vec<f32> = cfvs_f64.iter().map(|&v| v as f32).collect();
            game.set_boundary_cfvs(ordinal, player, cfvs_f32);
        }
    }
}

/// Extract CFVs and strategy from a solved game into a SubgameSolveResult.
fn extract_result(pbs: &Pbs, game: &mut PostFlopGame) -> Result<SubgameSolveResult, String> {
    // Extract CFVs.
    game.cache_normalized_weights();

    let raw_oop = game.expected_values(0);
    let raw_ip = game.expected_values(1);
    let oop_hands = game.private_cards(0);
    let ip_hands = game.private_cards(1);

    let pot = pbs.pot as f32;

    let mut root_cfvs = Box::new([[0.0f32; NUM_COMBOS]; 2]);
    map_evs_to_combos(&raw_oop, oop_hands, pot, &mut root_cfvs[0]);
    map_evs_to_combos(&raw_ip, ip_hands, pot, &mut root_cfvs[1]);

    // Extract strategy at the root node.
    // game.strategy() returns a flat array with layout:
    //   strategy[action * num_solver_hands + hand] = probability
    // We need to transpose this to canonical 1326 layout:
    //   root_strategy[combo * num_actions + action] = probability
    let num_actions = game.available_actions().len();
    let root_player = game.current_player();
    let solver_hands = game.private_cards(root_player);
    let solver_strategy = game.strategy();
    let num_solver_hands = solver_hands.len();

    // Initialize with uniform strategy (for blocked combos).
    let uniform = 1.0 / num_actions as f32;
    let mut root_strategy = vec![uniform; NUM_COMBOS * num_actions];

    // Map solver-ordered strategy to canonical combo ordering.
    for (hand_idx, &(c1, c2)) in solver_hands.iter().enumerate() {
        let combo_idx = card_pair_to_index(c1, c2);
        for action in 0..num_actions {
            let solver_idx = action * num_solver_hands + hand_idx;
            let canonical_idx = combo_idx * num_actions + action;
            root_strategy[canonical_idx] = solver_strategy[solver_idx];
        }
    }

    Ok(SubgameSolveResult {
        root_cfvs,
        root_strategy,
        num_actions,
    })
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

/// Convert a range-solver card (u8: 4*rank + suit) to an rs_poker Card.
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::blueprint_v2::LeafEvaluator;
    use poker_solver_core::poker::Card as RsPokerCard;
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::index_to_card_pair;

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

    /// Board: Qs Jh 2c 8d 3s
    fn test_river_board() -> Vec<u8> {
        vec![
            4 * 10 + 3, // Qs = 43
            4 * 9 + 2,  // Jh = 38
            4 * 0 + 0,  // 2c = 0
            4 * 6 + 1,  // 8d = 25
            4 * 1 + 3,  // 3s = 7
        ]
    }

    /// Board: Qs Jh 2c 8d (turn)
    fn test_turn_board() -> Vec<u8> {
        vec![
            4 * 10 + 3, // Qs = 43
            4 * 9 + 2,  // Jh = 38
            4 * 0 + 0,  // 2c = 0
            4 * 6 + 1,  // 8d = 25
        ]
    }

    /// Returns true if combo index `i` is blocked by any board card.
    fn is_blocked(i: usize, board: &[u8]) -> bool {
        let (c1, c2) = index_to_card_pair(i);
        board.contains(&c1) || board.contains(&c2)
    }

    // -----------------------------------------------------------------------
    // River subgame test
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_subgame_river() {
        let board = test_river_board();
        let pbs = Pbs::new_uniform(board.clone(), 100, 100);
        let config = test_solve_config();
        let evaluator = ZeroEvaluator;

        let result = solve_subgame(&pbs, &config, &evaluator).unwrap();

        // num_actions should be positive (check, bet, allin, etc.)
        assert!(
            result.num_actions > 0,
            "expected at least one action, got {}",
            result.num_actions
        );

        // root_cfvs should have values for both players
        let has_nonzero_oop = result.root_cfvs[0]
            .iter()
            .enumerate()
            .any(|(i, &v)| !is_blocked(i, &board) && v != 0.0);
        let has_nonzero_ip = result.root_cfvs[1]
            .iter()
            .enumerate()
            .any(|(i, &v)| !is_blocked(i, &board) && v != 0.0);
        assert!(has_nonzero_oop, "expected non-zero OOP CFVs");
        assert!(has_nonzero_ip, "expected non-zero IP CFVs");

        // Board-blocked combos should have 0 CFV
        for i in 0..NUM_COMBOS {
            if is_blocked(i, &board) {
                assert_eq!(
                    result.root_cfvs[0][i], 0.0,
                    "blocked combo {i} should have OOP CFV 0.0"
                );
                assert_eq!(
                    result.root_cfvs[1][i], 0.0,
                    "blocked combo {i} should have IP CFV 0.0"
                );
            }
        }

        // root_strategy should have correct length
        assert_eq!(
            result.root_strategy.len(),
            NUM_COMBOS * result.num_actions,
            "strategy length mismatch"
        );

        // Strategy probabilities should sum to ~1.0 for non-blocked combos
        let mut checked = 0;
        for i in 0..NUM_COMBOS {
            if is_blocked(i, &board) {
                continue;
            }
            let probs = get_action_probs(&result, i);
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "combo {i}: strategy sum = {sum}, expected ~1.0"
            );
            // All probs should be non-negative
            for (a, &p) in probs.iter().enumerate() {
                assert!(
                    p >= -1e-6,
                    "combo {i}, action {a}: negative probability {p}"
                );
            }
            checked += 1;
        }
        assert!(checked > 0, "expected to check at least one non-blocked combo");
    }

    // -----------------------------------------------------------------------
    // Turn subgame with zero evaluator
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_subgame_turn_with_zero_evaluator() {
        let board = test_turn_board();
        let pbs = Pbs::new_uniform(board.clone(), 100, 200);
        let config = test_solve_config();
        let evaluator = ZeroEvaluator;

        let result = solve_subgame(&pbs, &config, &evaluator).unwrap();

        // Should produce valid results
        assert!(result.num_actions > 0, "expected at least one action");

        // CFVs should be finite
        for player in 0..2 {
            for i in 0..NUM_COMBOS {
                assert!(
                    result.root_cfvs[player][i].is_finite(),
                    "player {player}, combo {i}: non-finite CFV {}",
                    result.root_cfvs[player][i]
                );
            }
        }

        // Strategy length should be correct
        assert_eq!(
            result.root_strategy.len(),
            NUM_COMBOS * result.num_actions
        );

        // Strategy should sum to ~1.0 for non-blocked combos
        for i in 0..NUM_COMBOS {
            if is_blocked(i, &board) {
                continue;
            }
            let probs = get_action_probs(&result, i);
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "combo {i}: strategy sum = {sum}, expected ~1.0"
            );
        }
    }

    // -----------------------------------------------------------------------
    // get_action_probs helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_action_probs() {
        // Construct a small synthetic result with 3 actions.
        let num_actions = 3;
        let mut root_strategy = vec![0.0f32; NUM_COMBOS * num_actions];

        // Set specific probabilities for combo 42: [0.2, 0.3, 0.5]
        root_strategy[42 * num_actions + 0] = 0.2;
        root_strategy[42 * num_actions + 1] = 0.3;
        root_strategy[42 * num_actions + 2] = 0.5;

        // Set specific probabilities for combo 100: [0.1, 0.8, 0.1]
        root_strategy[100 * num_actions + 0] = 0.1;
        root_strategy[100 * num_actions + 1] = 0.8;
        root_strategy[100 * num_actions + 2] = 0.1;

        let result = SubgameSolveResult {
            root_cfvs: Box::new([[0.0; NUM_COMBOS]; 2]),
            root_strategy,
            num_actions,
        };

        // Check combo 42
        let probs_42 = get_action_probs(&result, 42);
        assert_eq!(probs_42.len(), 3);
        assert!((probs_42[0] - 0.2).abs() < 1e-6);
        assert!((probs_42[1] - 0.3).abs() < 1e-6);
        assert!((probs_42[2] - 0.5).abs() < 1e-6);

        // Check combo 100
        let probs_100 = get_action_probs(&result, 100);
        assert_eq!(probs_100.len(), 3);
        assert!((probs_100[0] - 0.1).abs() < 1e-6);
        assert!((probs_100[1] - 0.8).abs() < 1e-6);
        assert!((probs_100[2] - 0.1).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_subgame_zero_pot_error() {
        let board = test_river_board();
        let pbs = Pbs::new_uniform(board, 0, 100);
        let config = test_solve_config();
        let evaluator = ZeroEvaluator;
        let result = solve_subgame(&pbs, &config, &evaluator);
        assert!(result.is_err(), "expected error for zero pot");
    }

    #[test]
    fn test_solve_subgame_zero_stack_error() {
        let board = test_river_board();
        let pbs = Pbs::new_uniform(board, 100, 0);
        let config = test_solve_config();
        let evaluator = ZeroEvaluator;
        let result = solve_subgame(&pbs, &config, &evaluator);
        assert!(result.is_err(), "expected error for zero stack");
    }

    #[test]
    fn test_solve_subgame_invalid_board_size() {
        let board = vec![43, 38]; // only 2 cards
        let pbs = Pbs::new_uniform(board, 100, 100);
        let config = test_solve_config();
        let evaluator = ZeroEvaluator;
        let result = solve_subgame(&pbs, &config, &evaluator);
        assert!(result.is_err(), "expected error for 2-card board");
    }
}
