use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{flop_from_str, NOT_DEALT};
use range_solver::range::Range;
use range_solver::{CardConfig, PostFlopGame};

/// Builds the Flop Poker game: QhJdTh, all combos, 20bb effective / 2bb pot.
///
/// Game definition:
/// - Board: QhJdTh (flop), turn/river dealt by solver
/// - Ranges: every non-conflicting combo (full deck, `Range::ones()`)
/// - Starting pot: 2, Effective stack: 20
/// - Bet sizes: 67% pot for bets and raises (both players, all streets)
/// - All-in available via `add_allin_threshold: 1.5` and `force_allin_threshold: 0.6`
/// - No rake, no merging, no depth limit
pub fn build_flop_poker_game() -> Result<PostFlopGame, String> {
    let card_config = CardConfig {
        range: [Range::ones(), Range::ones()],
        flop: flop_from_str("QhJdTh").unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from(("67%", "67%")).unwrap();

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 2,
        effective_stack: 20,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.6,
        merging_threshold: 0.0,
        depth_limit: None,
    };

    let action_tree = ActionTree::new(tree_config)?;
    PostFlopGame::with_config(card_config, action_tree)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_flop_poker_game() {
        let game = build_flop_poker_game().unwrap();
        // 49 remaining cards (52 - 3 board cards) => C(49,2) = 1176 combos per player
        assert_eq!(game.private_cards(0).len(), 1176);
        assert_eq!(game.private_cards(1).len(), 1176);
    }

    #[test]
    fn test_flop_poker_memory_is_tractable() {
        let game = build_flop_poker_game().unwrap();
        let (uncompressed, _compressed) = game.memory_usage();
        // Must be under 4GB uncompressed to be tractable
        assert!(
            uncompressed < 4 * 1024 * 1024 * 1024,
            "Memory usage too high: {} bytes",
            uncompressed
        );
    }
}
