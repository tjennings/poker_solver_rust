use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{flop_from_str, NOT_DEALT};
use range_solver::range::Range;
use range_solver::{CardConfig, PostFlopGame};

/// Configuration for the Flop Poker convergence test game.
#[derive(Debug, Clone)]
pub struct FlopPokerConfig {
    /// Flop board string, e.g. "QhJdTh"
    pub flop: String,
    /// Starting pot in bb
    pub starting_pot: i32,
    /// Effective stack in bb
    pub effective_stack: i32,
    /// Bet size string (pot-relative), e.g. "33%,67%"
    pub bet_sizes: String,
    /// Raise size string (pot-relative), e.g. "67%"
    pub raise_sizes: String,
    /// Threshold for adding all-in as a bet option (0.0 = never auto-add)
    pub add_allin_threshold: f64,
    /// Force all-in when SPR after call is below this (0.0 = disable)
    pub force_allin_threshold: f64,
}

impl Default for FlopPokerConfig {
    fn default() -> Self {
        Self {
            flop: "QhJdTh".into(),
            starting_pot: 2,
            effective_stack: 20,
            bet_sizes: "67%".into(),
            raise_sizes: "67%".into(),
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.6,
        }
    }
}

/// Builds the Flop Poker game with default configuration.
pub fn build_flop_poker_game() -> Result<PostFlopGame, String> {
    build_flop_poker_game_with_config(&FlopPokerConfig::default())
}

/// Builds the Flop Poker game with the given configuration.
///
/// - Board: configured flop, turn/river dealt by solver
/// - Ranges: every non-conflicting combo (full deck)
/// - All-in available via thresholds
/// - No rake, no merging, no depth limit
pub fn build_flop_poker_game_with_config(config: &FlopPokerConfig) -> Result<PostFlopGame, String> {
    let card_config = CardConfig {
        range: [Range::ones(), Range::ones()],
        flop: flop_from_str(&config.flop).map_err(|e| format!("Invalid flop '{}': {}", config.flop, e))?,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from((config.bet_sizes.as_str(), config.raise_sizes.as_str()))
        .map_err(|e| format!("Invalid bet sizes: {e}"))?;

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: config.starting_pot,
        effective_stack: config.effective_stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: config.add_allin_threshold,
        force_allin_threshold: config.force_allin_threshold,
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

    #[test]
    fn test_custom_config() {
        let config = FlopPokerConfig {
            effective_stack: 10,
            ..Default::default()
        };
        let game = build_flop_poker_game_with_config(&config).unwrap();
        assert_eq!(game.private_cards(0).len(), 1176);
    }

    #[test]
    fn test_invalid_flop_returns_error() {
        let config = FlopPokerConfig {
            flop: "ZzZzZz".into(),
            ..Default::default()
        };
        assert!(build_flop_poker_game_with_config(&config).is_err());
    }
}
