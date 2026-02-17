//! Limit Hold'em state encoder for SD-CFR.
//!
//! Converts `LimitHoldemState` into `InfoSetFeatures` using suit-isomorphic
//! card canonicalization and fixed-size bet encoding.

use poker_solver_core::game::{Action, LimitHoldemState, Player};

use crate::card_features::{self, BetAction, InfoSetFeatures};
use crate::traverse::StateEncoder;

/// Fixed pot-fraction sentinel for limit hold'em bets and raises.
///
/// Since LHE has exactly one bet size per street, we use a constant value.
/// The neural network learns to interpret this in context.
const LHE_BET_POT_FRACTION: f64 = 1.0;

/// Encodes Limit Hold'em states into neural network features.
///
/// Stateless: LHE has fixed bet sizes so no configuration is needed.
#[derive(Debug, Clone)]
pub struct LheEncoder;

impl LheEncoder {
    /// Create a new Limit Hold'em encoder.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for LheEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StateEncoder<LimitHoldemState> for LheEncoder {
    fn encode(&self, state: &LimitHoldemState, player: Player) -> InfoSetFeatures {
        let hole = player_holding(state, player);
        let board = state.visible_board();
        let cards = card_features::canonicalize(hole, board);
        let bet_actions = build_lhe_bet_actions(&state.history);
        let bets = card_features::encode_bets(&bet_actions);
        InfoSetFeatures { cards, bets }
    }
}

/// Extract hole cards for the given player.
fn player_holding(state: &LimitHoldemState, player: Player) -> [rs_poker::core::Card; 2] {
    match player {
        Player::Player1 => state.p1_holding,
        Player::Player2 => state.p2_holding,
    }
}

/// Convert LHE action history into `BetAction` pairs.
///
/// - Fold/Check/Call: pot_frac = 0.0
/// - Bet(0)/Raise(0): pot_frac = `LHE_BET_POT_FRACTION`
fn build_lhe_bet_actions(
    history: &[(poker_solver_core::abstraction::Street, Action)],
) -> Vec<BetAction> {
    history
        .iter()
        .map(|&(_street, action)| {
            let pot_frac = lhe_action_pot_fraction(action);
            (action, pot_frac)
        })
        .collect()
}

/// Map a single LHE action to its pot-fraction encoding.
fn lhe_action_pot_fraction(action: Action) -> f64 {
    match action {
        Action::Fold | Action::Check | Action::Call => 0.0,
        Action::Bet(_) | Action::Raise(_) => LHE_BET_POT_FRACTION,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::abstraction::Street;
    use poker_solver_core::game::{Game, LimitHoldem, LimitHoldemConfig};

    fn default_game() -> LimitHoldem {
        LimitHoldem::new(LimitHoldemConfig::default(), 10, 42)
    }

    fn first_state(game: &LimitHoldem) -> LimitHoldemState {
        game.initial_states().into_iter().next().unwrap()
    }

    // -----------------------------------------------------------------------
    // 1. Encoder produces valid features at preflop root
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_produces_valid_features_at_preflop() {
        let game = default_game();
        let state = first_state(&game);
        let encoder = LheEncoder::new();

        let features = encoder.encode(&state, Player::Player1);

        // Hole cards should be valid (>= 0)
        assert!(
            features.cards[0] >= 0,
            "hole card 0 should be >= 0, got {}",
            features.cards[0]
        );
        assert!(
            features.cards[1] >= 0,
            "hole card 1 should be >= 0, got {}",
            features.cards[1]
        );
        // No board cards at preflop
        for i in 2..7 {
            assert_eq!(
                features.cards[i], -1,
                "card slot {i} should be -1 at preflop, got {}",
                features.cards[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 2. Different players get different hole cards
    // -----------------------------------------------------------------------

    #[test]
    fn different_players_get_different_cards() {
        let game = default_game();
        let state = first_state(&game);
        let encoder = LheEncoder::new();

        let p1_features = encoder.encode(&state, Player::Player1);
        let p2_features = encoder.encode(&state, Player::Player2);

        assert_ne!(
            &p1_features.cards[0..2],
            &p2_features.cards[0..2],
            "P1 and P2 should have different hole cards"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Board cards appear after reaching flop
    // -----------------------------------------------------------------------

    #[test]
    fn board_cards_appear_on_flop() {
        let game = default_game();
        let state = first_state(&game);
        let encoder = LheEncoder::new();

        // Navigate to flop: SB calls, BB checks
        let after_call = game.next_state(&state, Action::Call);
        let flop_state = game.next_state(&after_call, Action::Check);

        assert_eq!(flop_state.street, Street::Flop);
        assert_eq!(flop_state.board_len, 3);

        let features = encoder.encode(&flop_state, Player::Player1);

        for i in 2..5 {
            assert!(
                features.cards[i] >= 0,
                "flop card slot {i} should be >= 0, got {}",
                features.cards[i]
            );
        }
        // Turn and river still absent
        assert_eq!(features.cards[5], -1, "turn should be absent on flop");
        assert_eq!(features.cards[6], -1, "river should be absent on flop");
    }

    // -----------------------------------------------------------------------
    // 4. LHE action pot fractions are correct
    // -----------------------------------------------------------------------

    #[test]
    fn lhe_action_pot_fraction_values() {
        assert!(
            (lhe_action_pot_fraction(Action::Fold) - 0.0).abs() < f64::EPSILON,
            "Fold should be 0.0"
        );
        assert!(
            (lhe_action_pot_fraction(Action::Check) - 0.0).abs() < f64::EPSILON,
            "Check should be 0.0"
        );
        assert!(
            (lhe_action_pot_fraction(Action::Call) - 0.0).abs() < f64::EPSILON,
            "Call should be 0.0"
        );
        assert!(
            (lhe_action_pot_fraction(Action::Bet(0)) - LHE_BET_POT_FRACTION).abs() < f64::EPSILON,
            "Bet(0) should be {LHE_BET_POT_FRACTION}"
        );
        assert!(
            (lhe_action_pot_fraction(Action::Raise(0)) - LHE_BET_POT_FRACTION).abs() < f64::EPSILON,
            "Raise(0) should be {LHE_BET_POT_FRACTION}"
        );
    }

    // -----------------------------------------------------------------------
    // 5. Build bet actions from history
    // -----------------------------------------------------------------------

    #[test]
    fn build_lhe_bet_actions_from_history() {
        let history: [(Street, Action); 4] = [
            (Street::Preflop, Action::Call),
            (Street::Preflop, Action::Check),
            (Street::Flop, Action::Bet(0)),
            (Street::Flop, Action::Raise(0)),
        ];

        let result = build_lhe_bet_actions(&history);

        assert_eq!(result.len(), 4);
        assert!((result[0].1 - 0.0).abs() < f64::EPSILON, "Call = 0.0");
        assert!((result[1].1 - 0.0).abs() < f64::EPSILON, "Check = 0.0");
        assert!(
            (result[2].1 - LHE_BET_POT_FRACTION).abs() < f64::EPSILON,
            "Bet(0) = {LHE_BET_POT_FRACTION}"
        );
        assert!(
            (result[3].1 - LHE_BET_POT_FRACTION).abs() < f64::EPSILON,
            "Raise(0) = {LHE_BET_POT_FRACTION}"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Empty history produces all-zero bets
    // -----------------------------------------------------------------------

    #[test]
    fn empty_history_produces_zero_bets() {
        let game = default_game();
        let state = first_state(&game);
        let encoder = LheEncoder::new();

        let features = encoder.encode(&state, Player::Player1);

        // At preflop root, no actions have been taken yet
        assert!(
            features.bets.iter().all(|&v| v == 0.0),
            "preflop root should have all-zero bet features"
        );
    }

    // -----------------------------------------------------------------------
    // 7. Bet features record actions taken
    // -----------------------------------------------------------------------

    #[test]
    fn bet_features_record_actions() {
        let game = default_game();
        let state = first_state(&game);
        let encoder = LheEncoder::new();

        // SB raises
        let after_raise = game.next_state(&state, Action::Raise(0));
        let features = encoder.encode(&after_raise, Player::Player2);

        // Round 0, slot 0 should be filled (occurred=1.0)
        assert_eq!(features.bets[0], 1.0, "first action should be recorded");
        assert!(
            (features.bets[1] - LHE_BET_POT_FRACTION as f32).abs() < f32::EPSILON,
            "raise pot fraction should be {LHE_BET_POT_FRACTION}"
        );
    }

    // -----------------------------------------------------------------------
    // 8. Default impl works
    // -----------------------------------------------------------------------

    #[test]
    fn default_encoder_is_usable() {
        let encoder = LheEncoder::default();
        let game = default_game();
        let state = first_state(&game);

        let features = encoder.encode(&state, Player::Player1);
        assert!(features.cards[0] >= 0);
    }

    // -----------------------------------------------------------------------
    // 9. Encoding is deterministic
    // -----------------------------------------------------------------------

    #[test]
    fn encoding_is_deterministic() {
        let game = default_game();
        let state = first_state(&game);
        let encoder = LheEncoder::new();

        let f1 = encoder.encode(&state, Player::Player1);
        let f2 = encoder.encode(&state, Player::Player1);

        assert_eq!(f1.cards, f2.cards, "cards should be identical");
        assert_eq!(f1.bets, f2.bets, "bets should be identical");
    }
}
