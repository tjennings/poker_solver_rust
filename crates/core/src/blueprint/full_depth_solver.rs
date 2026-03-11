//! Full-depth solve bridge to the `range-solver` crate.
//!
//! Translates game state (board, pot, stacks, ranges as 1326-weight vectors)
//! into range-solver inputs, runs an exact postflop DCFR solve, and extracts
//! per-combo action probabilities.

use crate::poker::{Card, Suit};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during a full-depth solve.
#[derive(Debug, Error)]
pub enum FullDepthError {
    /// The board has an invalid number of cards (must be 3, 4, or 5).
    #[error("invalid board length {0}: expected 3, 4, or 5")]
    InvalidBoardLength(usize),

    /// A card could not be converted to the range-solver representation.
    #[error("card conversion failed: {0}")]
    CardConversion(String),

    /// The range-solver rejected the configuration.
    #[error("range-solver config error: {0}")]
    ConfigError(String),

    /// The range weight vector has the wrong length (expected 1326).
    #[error("range must have exactly 1326 elements, got {0}")]
    InvalidRangeLength(usize),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parameters for a full-depth postflop solve.
pub struct FullDepthConfig {
    /// Board cards (3, 4, or 5 cards for flop/turn/river).
    pub board: Vec<Card>,
    /// Total pot in chips.
    pub pot: i32,
    /// Remaining effective stack for each player.
    pub effective_stack: i32,
    /// Bet sizes as pot fractions (e.g., `[0.33, 0.67, 1.0]`).
    pub bet_sizes: Vec<f64>,
    /// Maximum DCFR iterations.
    pub iterations: u32,
    /// Target exploitability as fraction of pot.
    pub target_exploitability: f64,
}

// ---------------------------------------------------------------------------
// Solve result
// ---------------------------------------------------------------------------

/// Wraps a solved `PostFlopGame` and provides strategy extraction methods.
pub struct SolveResult {
    game: range_solver::PostFlopGame,
}

impl SolveResult {
    /// Returns the action probabilities for a specific hand index.
    ///
    /// The hand index corresponds to the ordering in `private_cards(player)`
    /// where player 0 is OOP. The returned `Vec` has length `num_actions()`;
    /// element `i` is the probability of taking action `i`.
    #[must_use]
    pub fn strategy_for_hand(&self, player: usize, hand_idx: usize) -> Vec<f64> {
        let strategy = self.game.strategy();
        let num_hands = self.game.private_cards(player).len();
        let num_actions = self.num_actions();
        let mut result = Vec::with_capacity(num_actions);
        for a in 0..num_actions {
            result.push(f64::from(strategy[a * num_hands + hand_idx]));
        }
        result
    }

    /// Returns the number of available actions at the root.
    #[must_use]
    pub fn num_actions(&self) -> usize {
        self.game.available_actions().len()
    }

    /// Returns human-readable labels for each available action.
    #[must_use]
    pub fn actions(&self) -> Vec<String> {
        self.game
            .available_actions()
            .iter()
            .map(std::string::ToString::to_string)
            .collect()
    }

    /// Returns the private card pairs for the given player.
    ///
    /// Card values are in range-solver encoding (`4 * rank + suit`).
    #[must_use]
    pub fn private_cards(&self, player: usize) -> &[(u8, u8)] {
        self.game.private_cards(player)
    }

    /// Finds the hand index for a specific card pair in the given player's
    /// private hands list. Returns `None` if the combo is not in range.
    #[must_use]
    pub fn hand_index(&self, player: usize, card1: u8, card2: u8) -> Option<usize> {
        let (lo, hi) = if card1 < card2 {
            (card1, card2)
        } else {
            (card2, card1)
        };
        self.game
            .private_cards(player)
            .iter()
            .position(|&(c1, c2)| c1 == lo && c2 == hi)
    }
}

// ---------------------------------------------------------------------------
// Card conversion
// ---------------------------------------------------------------------------

/// Converts an `rs_poker::core::Card` to a range-solver card ID.
///
/// Range-solver encoding: `card_id = 4 * rank + suit`
///   - rank: 2->0, 3->1, ..., A->12
///   - suit: club->0, diamond->1, heart->2, spade->3
///
/// `rs_poker` encoding:
///   - Value: Two=0 ... Ace=12 (same rank)
///   - Suit: Spade=0, Club=1, Heart=2, Diamond=3
fn card_to_rs_id(card: Card) -> u8 {
    let rank = card.value as u8;
    let suit = match card.suit {
        Suit::Club => 0,
        Suit::Diamond => 1,
        Suit::Heart => 2,
        Suit::Spade => 3,
    };
    4 * rank + suit
}

// ---------------------------------------------------------------------------
// Bet-size string builder
// ---------------------------------------------------------------------------

/// Builds a comma-separated bet-size string from pot fractions.
///
/// Always appends "a" (all-in) to ensure all-in is available.
fn bet_sizes_str(fracs: &[f64]) -> String {
    let mut parts: Vec<String> = fracs.iter().map(|f| format!("{}%", f * 100.0)).collect();
    if !fracs.iter().any(|&f| f >= 100.0) {
        parts.push("a".to_string());
    }
    parts.join(", ")
}

// ---------------------------------------------------------------------------
// Solver entry point
// ---------------------------------------------------------------------------

/// Solves a postflop spot using the range-solver crate.
///
/// `oop_range` and `ip_range` are 1326-element weight vectors indexed by
/// `card_pair_to_index(card1, card2)` using range-solver card encoding.
///
/// # Errors
///
/// Returns `FullDepthError` if the configuration is invalid or the
/// range-solver rejects the inputs.
#[allow(clippy::cast_possible_truncation)]
pub fn solve_full_depth(
    config: &FullDepthConfig,
    oop_range: &[f64],
    ip_range: &[f64],
) -> Result<SolveResult, FullDepthError> {
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{CardConfig, NOT_DEALT};
    use range_solver::range::Range;

    // -- Validate range lengths --
    if oop_range.len() != 1326 {
        return Err(FullDepthError::InvalidRangeLength(oop_range.len()));
    }
    if ip_range.len() != 1326 {
        return Err(FullDepthError::InvalidRangeLength(ip_range.len()));
    }

    // -- Convert board cards --
    let board_len = config.board.len();
    if !(3..=5).contains(&board_len) {
        return Err(FullDepthError::InvalidBoardLength(board_len));
    }
    let rs_board: Vec<u8> = config.board.iter().map(|c| card_to_rs_id(*c)).collect();
    let mut flop = [rs_board[0], rs_board[1], rs_board[2]];
    flop.sort_unstable();

    let turn = if board_len >= 4 {
        rs_board[3]
    } else {
        NOT_DEALT
    };
    let river = if board_len >= 5 {
        rs_board[4]
    } else {
        NOT_DEALT
    };

    // -- Build ranges (f64 -> f32 truncation is intentional; weights are [0,1]) --
    let oop_f32: Vec<f32> = oop_range.iter().map(|&w| w as f32).collect();
    let ip_f32: Vec<f32> = ip_range.iter().map(|&w| w as f32).collect();
    let oop_rs = Range::from_raw_data(&oop_f32).map_err(FullDepthError::ConfigError)?;
    let ip_rs = Range::from_raw_data(&ip_f32).map_err(FullDepthError::ConfigError)?;

    // -- Determine board state --
    let initial_state = match board_len {
        3 => BoardState::Flop,
        4 => BoardState::Turn,
        5 => BoardState::River,
        _ => unreachable!(),
    };

    // -- Build bet sizes --
    let bet_str = bet_sizes_str(&config.bet_sizes);
    let sizes =
        BetSizeOptions::try_from((bet_str.as_str(), "")).map_err(FullDepthError::ConfigError)?;

    let card_config = CardConfig {
        range: [oop_rs, ip_rs],
        flop,
        turn,
        river,
    };

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: config.pot,
        effective_stack: config.effective_stack,
        flop_bet_sizes: [sizes.clone(), sizes.clone()],
        turn_bet_sizes: [sizes.clone(), sizes.clone()],
        river_bet_sizes: [sizes.clone(), sizes],
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        ..Default::default()
    };

    let tree = ActionTree::new(tree_config).map_err(FullDepthError::ConfigError)?;
    let mut game =
        range_solver::PostFlopGame::with_config(card_config, tree)
            .map_err(FullDepthError::ConfigError)?;

    game.allocate_memory(true);

    let target = (config.target_exploitability * f64::from(config.pot)) as f32;
    range_solver::solve(&mut game, config.iterations, target, false);

    Ok(SolveResult { game })
}

/// Convenience: convert an `rs_poker` card to range-solver card ID.
///
/// This is useful when callers need to build range weight vectors using
/// `range_solver::card::card_pair_to_index`.
#[must_use]
pub fn rs_poker_card_to_id(card: Card) -> u8 {
    card_to_rs_id(card)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::Value;
    use range_solver::card::card_pair_to_index;

    /// Verify the card conversion for a few known cards.
    #[test]
    fn card_conversion_smoke() {
        // Ace of spades: rank=12, suit=spade(3) => 4*12+3 = 51
        let as_card = Card::new(Value::Ace, Suit::Spade);
        assert_eq!(card_to_rs_id(as_card), 51);

        // Two of clubs: rank=0, suit=club(0) => 0
        let tc = Card::new(Value::Two, Suit::Club);
        assert_eq!(card_to_rs_id(tc), 0);

        // King of hearts: rank=11, suit=heart(2) => 4*11+2 = 46
        let kh = Card::new(Value::King, Suit::Heart);
        assert_eq!(card_to_rs_id(kh), 46);

        // Queen of diamonds: rank=10, suit=diamond(1) => 4*10+1 = 41
        let qd = Card::new(Value::Queen, Suit::Diamond);
        assert_eq!(card_to_rs_id(qd), 41);
    }

    /// Verify all 52 card conversions produce unique values in [0, 52).
    #[test]
    fn card_conversion_exhaustive() {
        let mut seen = [false; 52];
        for &v in &Value::values() {
            for &s in &Suit::suits() {
                let card = Card::new(v, s);
                let id = card_to_rs_id(card) as usize;
                assert!(id < 52, "card id out of range: {id}");
                assert!(!seen[id], "duplicate card id: {id}");
                seen[id] = true;
            }
        }
        assert!(seen.iter().all(|&x| x), "not all card ids covered");
    }

    /// River spot: hero (OOP) has AA overpair, villain (IP) has 77 underpair.
    ///
    /// Board: Ks Qh Jd Tc 2d
    ///
    /// In equilibrium hero should almost never fold and should bet at high
    /// frequency since AA dominates 77 on this board.
    #[test]
    fn river_nuts_vs_bluff_catcher() {
        let board = vec![
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
            Card::new(Value::Ten, Suit::Club),
            Card::new(Value::Two, Suit::Diamond),
        ];

        // Hero: AhAc (overpair)
        let hero_c1 = Card::new(Value::Ace, Suit::Heart);
        let hero_c2 = Card::new(Value::Ace, Suit::Club);
        let hero_id1 = card_to_rs_id(hero_c1);
        let hero_id2 = card_to_rs_id(hero_c2);

        // Villain: 7h7s (underpair)
        let villain_c1 = Card::new(Value::Seven, Suit::Heart);
        let villain_c2 = Card::new(Value::Seven, Suit::Spade);
        let villain_id1 = card_to_rs_id(villain_c1);
        let villain_id2 = card_to_rs_id(villain_c2);

        // Build 1326-element weight vectors
        let mut oop_range = vec![0.0f64; 1326];
        let mut ip_range = vec![0.0f64; 1326];
        oop_range[card_pair_to_index(hero_id1, hero_id2)] = 1.0;
        ip_range[card_pair_to_index(villain_id1, villain_id2)] = 1.0;

        let config = FullDepthConfig {
            board,
            pot: 100,
            effective_stack: 100,
            bet_sizes: vec![1.0],
            iterations: 200,
            target_exploitability: 0.01,
        };

        let result = solve_full_depth(&config, &oop_range, &ip_range).unwrap();

        let actions = result.actions();
        assert!(
            result.num_actions() >= 2,
            "expected at least check + bet, got: {actions:?}"
        );

        // Find hero's hand index (OOP = player 0)
        let hero_idx = result
            .hand_index(0, hero_id1, hero_id2)
            .expect("hero hand should be in private cards");

        let strategy = result.strategy_for_hand(0, hero_idx);
        assert_eq!(strategy.len(), result.num_actions());

        // Strategy should sum to ~1
        let sum: f64 = strategy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "strategy should sum to 1.0, got {sum}"
        );

        // Find fold action index (if any)
        let fold_idx = actions.iter().position(|a| a.contains("Fold"));

        // Hero should never fold the overpair
        if let Some(fi) = fold_idx {
            assert!(
                strategy[fi] < 0.05,
                "hero should not fold AA, but fold freq = {}",
                strategy[fi]
            );
        }

        // Hero should bet at some frequency
        let bet_idx = actions
            .iter()
            .position(|a| a.contains("Bet") || a.contains("AllIn"));
        if let Some(bi) = bet_idx {
            assert!(
                strategy[bi] > 0.3,
                "hero should bet AA at meaningful frequency, got {}",
                strategy[bi]
            );
        }
    }
}
