use crate::datagen::range_gen::NUM_COMBOS;
use crate::datagen::sampler::Situation;
use range_solver::{
    action_tree::{ActionTree, BoardState, TreeConfig},
    bet_size::BetSizeOptions,
    card::{card_pair_to_index, Card},
    range::Range,
    solve, solve_with_scheme, CardConfig, DiscountScheme, PostFlopGame,
};

/// Configuration for the range-solver wrapper.
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
    /// Discount scheme. Default = original range-solver scheme.
    /// DcfrPlus = Supremus DCFR+ with delayed linear strategy weighting.
    pub discount_scheme: DiscountScheme,
}

/// Result of solving a single river situation.
pub struct SolveResult {
    /// Pot-relative EVs for OOP, indexed by combo (0..1326).
    pub oop_evs: [f32; NUM_COMBOS],
    /// Pot-relative EVs for IP, indexed by combo (0..1326).
    pub ip_evs: [f32; NUM_COMBOS],
    /// Weighted game value for OOP (sum of range[i] * ev[i]).
    pub oop_game_value: f32,
    /// Weighted game value for IP (sum of range[i] * ev[i]).
    pub ip_game_value: f32,
    /// Which combos were actually present in the solution (not board-blocked).
    pub valid_mask: [bool; NUM_COMBOS],
    /// Final exploitability returned by the solver.
    pub exploitability: f32,
}

/// Solve a river `Situation` and return pot-relative counterfactual values.
///
/// Board-blocked combos get EV = 0.0 and `valid_mask = false`.
/// If `effective_stack == 0`, returns an error since the solver requires a positive stack.
pub fn solve_situation(situation: &Situation, config: &SolveConfig) -> Result<SolveResult, String> {
    if situation.effective_stack <= 0 {
        return Err("effective_stack must be positive for solver".into());
    }

    let oop_range = Range::from_raw_data(&situation.ranges[0])?;
    let ip_range = Range::from_raw_data(&situation.ranges[1])?;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [
            situation.board[0],
            situation.board[1],
            situation.board[2],
        ],
        turn: situation.board[3],
        river: situation.board[4],
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: situation.pot,
        effective_stack: situation.effective_stack,
        river_bet_sizes: [config.bet_sizes.clone(), config.bet_sizes.clone()],
        add_allin_threshold: config.add_allin_threshold,
        force_allin_threshold: config.force_allin_threshold,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config)?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)?;
    game.allocate_memory(false);

    let exploitability = solve_with_scheme(
        &mut game,
        config.solver_iterations,
        config.target_exploitability,
        false,
        config.discount_scheme,
    );

    game.cache_normalized_weights();

    let raw_oop = game.expected_values(0);
    let raw_ip = game.expected_values(1);
    let oop_hands = game.private_cards(0);
    let ip_hands = game.private_cards(1);

    let pot = situation.pot as f32;

    let mut oop_evs = [0.0f32; NUM_COMBOS];
    let mut ip_evs = [0.0f32; NUM_COMBOS];
    let mut valid_mask = [false; NUM_COMBOS];

    map_evs_to_combos(&raw_oop, oop_hands, pot, &mut oop_evs, &mut valid_mask);
    map_evs_to_combos(&raw_ip, ip_hands, pot, &mut ip_evs, &mut valid_mask);

    let oop_game_value = weighted_sum(&situation.ranges[0], &oop_evs);
    let ip_game_value = weighted_sum(&situation.ranges[1], &ip_evs);

    Ok(SolveResult {
        oop_evs,
        ip_evs,
        oop_game_value,
        ip_game_value,
        valid_mask,
        exploitability,
    })
}

/// Map solver EVs (indexed by private_cards order) into the canonical 1326 layout.
fn map_evs_to_combos(
    raw_evs: &[f32],
    hands: &[(Card, Card)],
    pot: f32,
    out: &mut [f32; NUM_COMBOS],
    mask: &mut [bool; NUM_COMBOS],
) {
    for (i, &(c1, c2)) in hands.iter().enumerate() {
        let idx = card_pair_to_index(c1, c2);
        out[idx] = raw_evs[i] / pot;
        mask[idx] = true;
    }
}

/// Compute weighted sum: sum(range[i] * ev[i]) for all combos.
fn weighted_sum(range: &[f32; NUM_COMBOS], evs: &[f32; NUM_COMBOS]) -> f32 {
    range
        .iter()
        .zip(evs.iter())
        .map(|(&r, &e)| r * e)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::card::index_to_card_pair;

    fn known_river_situation() -> Situation {
        // Board: Qs Jh 2c 8d 3s
        let board = [
            4 * 10 + 3, // Qs = 43
            4 * 9 + 2,  // Jh = 38
            4 * 0 + 0,  // 2c = 0
            4 * 6 + 1,  // 8d = 25
            4 * 1 + 3,  // 3s = 7
        ];
        let board_cards = &board[..5];
        // Uniform ranges over all valid combos
        let mut oop_range = [0.0f32; 1326];
        let mut ip_range = [0.0f32; 1326];
        let mut count = 0u32;
        for i in 0..1326 {
            let (c1, c2) = index_to_card_pair(i);
            if !board_cards.contains(&c1) && !board_cards.contains(&c2) && c1 != c2 {
                oop_range[i] = 1.0;
                ip_range[i] = 1.0;
                count += 1;
            }
        }
        // Normalize
        for v in oop_range.iter_mut().chain(ip_range.iter_mut()) {
            if *v > 0.0 {
                *v /= count as f32;
            }
        }
        Situation {
            board,
            board_size: 5,
            pot: 100,
            effective_stack: 100,
            ranges: [oop_range, ip_range],
        }
    }

    fn test_solve_config() -> SolveConfig {
        let bet_sizes = BetSizeOptions::try_from(("50%,a", ""))
            .expect("valid test bet sizes");
        SolveConfig {
            bet_sizes,
            solver_iterations: 200,
            target_exploitability: 0.01,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            discount_scheme: DiscountScheme::Default,
        }
    }

    #[test]
    fn solve_returns_correct_length_evs() {
        let sit = known_river_situation();
        let result = solve_situation(&sit, &test_solve_config()).unwrap();
        assert_eq!(result.oop_evs.len(), 1326);
        assert_eq!(result.ip_evs.len(), 1326);
    }

    #[test]
    fn solve_board_blocked_evs_are_zero() {
        let sit = known_river_situation();
        let result = solve_situation(&sit, &test_solve_config()).unwrap();
        for card in sit.board_cards() {
            for i in 0..1326 {
                let (c1, c2) = index_to_card_pair(i);
                if c1 == *card || c2 == *card {
                    assert_eq!(result.oop_evs[i], 0.0);
                    assert_eq!(result.ip_evs[i], 0.0);
                    assert!(!result.valid_mask[i]);
                }
            }
        }
    }

    #[test]
    fn solve_exploitability_below_threshold() {
        let sit = known_river_situation();
        let mut config = test_solve_config();
        config.solver_iterations = 500;
        config.target_exploitability = 0.01;
        let result = solve_situation(&sit, &config).unwrap();
        assert!(
            result.exploitability < 0.02,
            "exploitability {} too high",
            result.exploitability
        );
    }

    #[test]
    fn solve_evs_are_pot_relative() {
        let sit = known_river_situation();
        let result = solve_situation(&sit, &test_solve_config()).unwrap();
        for &ev in result.oop_evs.iter().chain(result.ip_evs.iter()) {
            if ev != 0.0 {
                assert!(
                    ev.abs() < 5.0,
                    "EV {} seems too large for pot-relative",
                    ev
                );
            }
        }
    }

    #[test]
    fn solve_game_values_finite() {
        let sit = known_river_situation();
        let mut config = test_solve_config();
        config.solver_iterations = 500;
        config.target_exploitability = 0.005;
        let result = solve_situation(&sit, &config).unwrap();
        assert!(result.oop_game_value.is_finite(), "OOP game value not finite");
        assert!(result.ip_game_value.is_finite(), "IP game value not finite");
    }
}
