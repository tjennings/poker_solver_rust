use crate::config::GameDef;
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
}

impl FlopPokerConfig {
    /// Build a `FlopPokerConfig` from a `GameDef` for a specific flop.
    pub fn from_game_def(game: &GameDef, flop: &str) -> Self {
        Self {
            flop: flop.into(),
            starting_pot: game.starting_pot,
            effective_stack: game.effective_stack,
            bet_sizes: game.bet_sizes.clone(),
            raise_sizes: game.raise_sizes.clone(),
        }
    }
}

impl Default for FlopPokerConfig {
    fn default() -> Self {
        Self {
            flop: "QhJdTh".into(),
            starting_pot: 2,
            effective_stack: 20,
            bet_sizes: "67%".into(),
            raise_sizes: "67%".into(),
        }
    }
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
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        merging_threshold: 0.0,
        depth_limit: None,
    };

    let action_tree = ActionTree::new(tree_config)?;
    PostFlopGame::with_config(card_config, action_tree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GameDef;

    #[test]
    fn test_invalid_flop_returns_error() {
        let config = FlopPokerConfig {
            flop: "ZzZzZz".into(),
            ..Default::default()
        };
        assert!(build_flop_poker_game_with_config(&config).is_err());
    }

    #[test]
    fn test_from_game_def_sets_all_fields() {
        let game_def = GameDef {
            flops: vec!["QhJdTh".into(), "Ks7d2c".into()],
            starting_pot: 4,
            effective_stack: 15,
            bet_sizes: "50%,100%,a".into(),
            raise_sizes: "50%,100%,a".into(),
        };
        let config = FlopPokerConfig::from_game_def(&game_def, "QhJdTh");
        assert_eq!(config.flop, "QhJdTh");
        assert_eq!(config.starting_pot, 4);
        assert_eq!(config.effective_stack, 15);
        assert_eq!(config.bet_sizes, "50%,100%,a");
        assert_eq!(config.raise_sizes, "50%,100%,a");
    }

    #[test]
    fn test_from_game_def_uses_specified_flop() {
        let game_def = GameDef {
            flops: vec!["QhJdTh".into(), "Ks7d2c".into()],
            starting_pot: 2,
            effective_stack: 20,
            bet_sizes: "67%".into(),
            raise_sizes: "67%".into(),
        };
        let config = FlopPokerConfig::from_game_def(&game_def, "Ks7d2c");
        assert_eq!(config.flop, "Ks7d2c");
    }

    #[test]
    fn test_from_game_def_builds_valid_game() {
        let game_def = GameDef {
            flops: vec!["QhJdTh".into()],
            starting_pot: 2,
            effective_stack: 10,
            bet_sizes: "a".into(),
            raise_sizes: "a".into(),
        };
        let config = FlopPokerConfig::from_game_def(&game_def, "QhJdTh");
        let result = build_flop_poker_game_with_config(&config);
        assert!(result.is_ok(), "Should build valid game from GameDef");
    }
}
