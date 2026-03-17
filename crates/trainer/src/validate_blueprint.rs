//! Validate-blueprint logic: solve spots with exact DCFR and compare
//! strategies against a blueprint bundle.

/// A validation spot defines a postflop scenario to solve.
pub struct ValidationSpot {
    pub name: String,
    pub board: Vec<String>,
    pub oop_range: String,
    pub ip_range: String,
    pub pot: f64,
    pub effective_stack: f64,
}

/// Result of validating a single spot.
pub struct SpotValidationResult {
    pub name: String,
    pub num_hands: usize,
    pub num_actions: usize,
    pub exploitability: f32,
    /// Average L2 distance between exact and blueprint strategies per hand.
    /// `None` if no blueprint was loaded.
    pub avg_strategy_l2: Option<f64>,
}

/// Compute the average L2 distance between two strategy matrices.
///
/// Each matrix is flat: `action_probs[action_idx * num_hands + hand_idx]`.
/// Both must have the same shape: `num_actions * num_hands` elements.
///
/// Returns the average per-hand L2 distance across all hands.
pub fn compute_strategy_l2_distance(
    exact: &[f32],
    blueprint: &[f32],
    num_actions: usize,
    num_hands: usize,
) -> f64 {
    assert_eq!(exact.len(), num_actions * num_hands);
    assert_eq!(blueprint.len(), num_actions * num_hands);

    if num_hands == 0 {
        return 0.0;
    }

    let mut total_l2 = 0.0_f64;
    for h in 0..num_hands {
        let mut sq_sum = 0.0_f64;
        for a in 0..num_actions {
            let diff = f64::from(exact[a * num_hands + h]) - f64::from(blueprint[a * num_hands + h]);
            sq_sum += diff * diff;
        }
        total_l2 += sq_sum.sqrt();
    }
    total_l2 / num_hands as f64
}

/// Solve a validation spot using the range solver with default settings
/// (1000 iterations, 0.5% target exploitability).
pub fn solve_spot(
    spot: &ValidationSpot,
) -> Result<SolveSpotResult, Box<dyn std::error::Error>> {
    solve_spot_with_params(spot, 1000, 0.005)
}

/// Solve a validation spot using the range solver, returning the root
/// strategy and game metadata.
pub fn solve_spot_with_params(
    spot: &ValidationSpot,
    iterations: u32,
    target_exploitability: f32,
) -> Result<SolveSpotResult, Box<dyn std::error::Error>> {
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{card_from_str, flop_from_str, CardConfig, NOT_DEALT};
    use range_solver::range::Range;
    use range_solver::{solve, PostFlopGame};

    // Parse ranges
    let oop_range: Range = spot
        .oop_range
        .parse()
        .map_err(|e: String| format!("Invalid OOP range: {e}"))?;
    let ip_range: Range = spot
        .ip_range
        .parse()
        .map_err(|e: String| format!("Invalid IP range: {e}"))?;

    // Parse board
    let board_cards = &spot.board;
    if board_cards.len() < 3 {
        return Err("Board must have at least 3 cards (flop)".into());
    }
    let flop_str = format!("{} {} {}", board_cards[0], board_cards[1], board_cards[2]);
    let flop = flop_from_str(&flop_str).map_err(|e| format!("Invalid flop: {e}"))?;

    let turn = if board_cards.len() >= 4 {
        card_from_str(&board_cards[3]).map_err(|e| format!("Invalid turn: {e}"))?
    } else {
        NOT_DEALT
    };

    let river = if board_cards.len() >= 5 {
        card_from_str(&board_cards[4]).map_err(|e| format!("Invalid river: {e}"))?
    } else {
        NOT_DEALT
    };

    let initial_state = if river != NOT_DEALT {
        BoardState::River
    } else if turn != NOT_DEALT {
        BoardState::Turn
    } else {
        BoardState::Flop
    };

    // Default bet sizes: 33%, 67%, 100%
    let bet_sizes =
        BetSizeOptions::try_from(("33%,67%,100%", "33%,67%,100%"))
            .map_err(|e| format!("Invalid bet sizes: {e}"))?;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop,
        turn,
        river,
    };

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: spot.pot as i32,
        effective_stack: spot.effective_stack as i32,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes],
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config)
        .map_err(|e| format!("Failed to build action tree: {e}"))?;

    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;

    game.allocate_memory(false);

    let exploitability = solve(&mut game, iterations, target_exploitability, false);

    // Extract root strategy
    game.back_to_root();
    let player = game.current_player();
    let hands = game.private_cards(player).to_vec();
    let strategy = game.strategy();
    let actions = game.available_actions();
    let actions_display: Vec<String> = actions.iter().map(|a| a.to_string()).collect();

    Ok(SolveSpotResult {
        exploitability,
        player,
        num_hands: hands.len(),
        num_actions: actions.len(),
        strategy,
        actions_display,
    })
}

/// Output of solving a single spot.
pub struct SolveSpotResult {
    pub exploitability: f32,
    pub player: usize,
    pub num_hands: usize,
    pub num_actions: usize,
    pub strategy: Vec<f32>,
    pub actions_display: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_distance_identical_strategies_is_zero() {
        let strategy = vec![0.5, 0.3, 0.2, 0.5, 0.7, 0.8];
        // 2 actions, 3 hands
        let dist = compute_strategy_l2_distance(&strategy, &strategy, 2, 3);
        assert!(
            dist.abs() < 1e-10,
            "identical strategies should have zero L2 distance, got {dist}"
        );
    }

    #[test]
    fn l2_distance_known_values() {
        // 2 actions, 2 hands
        // exact:     action0=[1.0, 0.0], action1=[0.0, 1.0]
        // blueprint: action0=[0.5, 0.5], action1=[0.5, 0.5]
        let exact = vec![1.0_f32, 0.0, 0.0, 1.0];
        let blueprint = vec![0.5_f32, 0.5, 0.5, 0.5];
        let dist = compute_strategy_l2_distance(&exact, &blueprint, 2, 2);
        // hand 0: diff = [0.5, -0.5], L2 = sqrt(0.25+0.25) = sqrt(0.5)
        // hand 1: diff = [-0.5, 0.5], L2 = sqrt(0.5)
        // avg = sqrt(0.5) ~= 0.7071
        let expected = (0.5_f64).sqrt();
        assert!(
            (dist - expected).abs() < 1e-6,
            "expected {expected}, got {dist}"
        );
    }

    #[test]
    fn l2_distance_zero_hands_returns_zero() {
        let dist = compute_strategy_l2_distance(&[], &[], 2, 0);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn l2_distance_single_action() {
        // 1 action, 3 hands
        let exact = vec![1.0_f32, 1.0, 1.0];
        let blueprint = vec![0.8_f32, 0.9, 0.7];
        let dist = compute_strategy_l2_distance(&exact, &blueprint, 1, 3);
        // diffs: 0.2, 0.1, 0.3 => L2 per hand: 0.2, 0.1, 0.3
        // avg = (0.2+0.1+0.3)/3 = 0.2
        assert!(
            (dist - 0.2).abs() < 1e-6,
            "expected 0.2, got {dist}"
        );
    }

    #[test]
    #[should_panic]
    fn l2_distance_panics_on_mismatched_lengths() {
        compute_strategy_l2_distance(&[1.0], &[1.0, 2.0], 1, 1);
    }

    #[test]
    fn solve_spot_produces_valid_result() {
        let spot = ValidationSpot {
            name: "test-spot".to_string(),
            board: vec!["Ks".into(), "7d".into(), "2c".into()],
            oop_range: "QQ+,AKs".to_string(),
            ip_range: "TT+,AQs+".to_string(),
            pot: 100.0,
            effective_stack: 100.0,
        };

        // Use few iterations for fast test
        let result = solve_spot_with_params(&spot, 10, 10.0).expect("solve should succeed");

        assert!(result.num_hands > 0, "should have at least one hand");
        assert!(result.num_actions > 0, "should have at least one action");
        assert!(
            result.strategy.len() == result.num_hands * result.num_actions,
            "strategy length should be num_hands * num_actions"
        );
        assert!(
            result.exploitability >= 0.0,
            "exploitability should be non-negative"
        );
        // Check strategy sums to ~1.0 for each hand
        for h in 0..result.num_hands {
            let sum: f32 = (0..result.num_actions)
                .map(|a| result.strategy[a * result.num_hands + h])
                .sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "strategy for hand {h} should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn solve_spot_rejects_invalid_board() {
        let spot = ValidationSpot {
            name: "bad-board".to_string(),
            board: vec!["Ks".into(), "7d".into()], // only 2 cards
            oop_range: "QQ+".to_string(),
            ip_range: "TT+".to_string(),
            pot: 100.0,
            effective_stack: 100.0,
        };

        assert!(solve_spot(&spot).is_err());
    }
}
