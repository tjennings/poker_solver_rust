// solver.rs — PBS → range-solver conversion and river subgame solving.
//
// Converts a PBS (public belief state) into the data structures required by
// range-solver, solves the resulting river subgame with DCFR, and extracts
// per-combo counterfactual values.

use crate::pbs::{Pbs, NUM_COMBOS};
use range_solver::{
    action_tree::{ActionTree, BoardState, TreeConfig},
    bet_size::BetSizeOptions,
    card::{card_pair_to_index, Card},
    range::Range,
    solve, CardConfig, PostFlopGame,
};

/// Configuration for the range-solver wrapper used in ReBeL subgame solving.
pub struct SolveConfig {
    /// Pre-parsed bet size options (avoids re-parsing per solve call).
    pub bet_sizes: BetSizeOptions,
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

    /// Build a test solve config with reasonable defaults.
    fn test_solve_config() -> SolveConfig {
        let bet_sizes =
            BetSizeOptions::try_from(("50%,a", "")).expect("valid test bet sizes");
        SolveConfig {
            bet_sizes,
            solver_iterations: 200,
            target_exploitability: 0.01,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
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
}
