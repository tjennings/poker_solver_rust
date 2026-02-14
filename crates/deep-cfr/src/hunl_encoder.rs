//! HUNL state encoder for SD-CFR.
//!
//! Converts `PostflopState` into `InfoSetFeatures` using suit-isomorphic
//! card canonicalization and pot-fraction bet encoding.

use poker_solver_core::game::{ALL_IN, Action, Player, PostflopState};

use crate::card_features::{self, BetAction, InfoSetFeatures};
use crate::traverse::StateEncoder;

/// Encodes HUNL postflop states into neural network features.
///
/// Stores the bet sizes from `PostflopConfig` to resolve bet action
/// indices to pot fractions for the bet feature encoder.
#[derive(Debug, Clone)]
pub struct HunlStateEncoder {
    bet_sizes: Vec<f32>,
}

impl HunlStateEncoder {
    /// Create a new encoder with the given bet sizes (pot fractions).
    pub fn new(bet_sizes: Vec<f32>) -> Self {
        Self { bet_sizes }
    }
}

impl StateEncoder<PostflopState> for HunlStateEncoder {
    fn encode(&self, state: &PostflopState, player: Player) -> InfoSetFeatures {
        let hole = match player {
            Player::Player1 => state.p1_holding,
            Player::Player2 => state.p2_holding,
        };
        let cards = card_features::canonicalize(hole, &state.board);
        let bet_actions = build_bet_actions(&self.bet_sizes, &state.history);
        let bets = card_features::encode_bets(&bet_actions);
        InfoSetFeatures { cards, bets }
    }
}

/// Convert the `PostflopState` action history into `BetAction` pairs.
///
/// Maps each action to a pot fraction:
/// - Fold/Check/Call: 0.0 (sentinel; the network learns meaning from context)
/// - Bet(idx)/Raise(idx): `bet_sizes[idx]` as pot fraction
/// - All-in (`idx == ALL_IN`): 10.0 sentinel
fn build_bet_actions(
    bet_sizes: &[f32],
    history: &[(poker_solver_core::abstraction::Street, Action)],
) -> Vec<BetAction> {
    history
        .iter()
        .map(|&(_street, action)| {
            let pot_frac = action_to_pot_fraction(bet_sizes, action);
            (action, pot_frac)
        })
        .collect()
}

/// Convert a single action to its pot fraction value.
fn action_to_pot_fraction(bet_sizes: &[f32], action: Action) -> f64 {
    match action {
        Action::Fold | Action::Check | Action::Call => 0.0,
        Action::Bet(idx) | Action::Raise(idx) if idx == ALL_IN => 10.0,
        Action::Bet(idx) | Action::Raise(idx) => {
            debug_assert!(
                (idx as usize) < bet_sizes.len(),
                "bet index {idx} out of range for {} sizes",
                bet_sizes.len()
            );
            bet_sizes
                .get(idx as usize)
                .map(|&s| f64::from(s))
                .unwrap_or(1.0) // fallback for release; debug_assert catches in dev
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::abstraction::Street;
    use poker_solver_core::game::{Game, HunlPostflop, PostflopConfig};

    /// Build a simple game and get one initial state with a full board.
    fn make_game_and_state(bet_sizes: Vec<f32>) -> (HunlPostflop, PostflopState) {
        let config = PostflopConfig {
            stack_depth: 25,
            bet_sizes,
            max_raises_per_street: 3,
        };
        let game = HunlPostflop::new(config, None, 1);
        let states = game.initial_states();
        let state = states.into_iter().next().unwrap();
        (game, state)
    }

    // -----------------------------------------------------------------------
    // 1. Encoder produces valid features at root (preflop)
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_produces_valid_features_at_root() {
        let bet_sizes = vec![0.5, 1.0];
        let (_, state) = make_game_and_state(bet_sizes.clone());
        let encoder = HunlStateEncoder::new(bet_sizes);

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
        // Board cards should be absent at preflop (board is empty)
        for i in 2..7 {
            assert_eq!(
                features.cards[i], -1,
                "card slot {i} should be -1 at preflop, got {}",
                features.cards[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 2. Different players get different cards
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_different_players_get_different_cards() {
        let bet_sizes = vec![0.5, 1.0];
        let (_, state) = make_game_and_state(bet_sizes.clone());
        let encoder = HunlStateEncoder::new(bet_sizes);

        let features_p1 = encoder.encode(&state, Player::Player1);
        let features_p2 = encoder.encode(&state, Player::Player2);

        // Hole cards (indices 0..2) should differ between players
        let p1_hole = &features_p1.cards[0..2];
        let p2_hole = &features_p2.cards[0..2];
        assert_ne!(
            p1_hole, p2_hole,
            "P1 and P2 should have different hole cards: P1={p1_hole:?}, P2={p2_hole:?}"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Board cards appear after reaching the flop
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_board_cards_appear_after_flop() {
        let bet_sizes = vec![0.5, 1.0];
        let config = PostflopConfig {
            stack_depth: 25,
            bet_sizes: bet_sizes.clone(),
            max_raises_per_street: 3,
        };
        let game = HunlPostflop::new(config, None, 1);
        let states = game.initial_states();
        let state = states.into_iter().next().unwrap();

        // Navigate to flop: SB calls (limps), BB checks
        let after_call = game.next_state(&state, Action::Call);
        let after_check = game.next_state(&after_call, Action::Check);

        assert_eq!(
            after_check.street,
            Street::Flop,
            "should be on the flop after SB call + BB check"
        );

        let encoder = HunlStateEncoder::new(bet_sizes);
        let features = encoder.encode(&after_check, Player::Player1);

        // Flop cards (indices 2..5) should be present (>= 0)
        for i in 2..5 {
            assert!(
                features.cards[i] >= 0,
                "flop card slot {i} should be >= 0 after dealing flop, got {}",
                features.cards[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 4. action_to_pot_fraction maps correctly
    // -----------------------------------------------------------------------

    #[test]
    fn action_to_pot_fraction_maps_correctly() {
        let bet_sizes = vec![0.5, 1.0];

        assert!(
            (action_to_pot_fraction(&bet_sizes, Action::Fold) - 0.0).abs() < f64::EPSILON,
            "Fold should map to 0.0"
        );
        assert!(
            (action_to_pot_fraction(&bet_sizes, Action::Check) - 0.0).abs() < f64::EPSILON,
            "Check should map to 0.0"
        );
        assert!(
            (action_to_pot_fraction(&bet_sizes, Action::Call) - 0.0).abs() < f64::EPSILON,
            "Call should map to 0.0"
        );
        assert!(
            (action_to_pot_fraction(&bet_sizes, Action::Bet(0)) - 0.5).abs() < f64::EPSILON,
            "Bet(0) should map to 0.5"
        );
        assert!(
            (action_to_pot_fraction(&bet_sizes, Action::Bet(1)) - 1.0).abs() < f64::EPSILON,
            "Bet(1) should map to 1.0"
        );
        assert!(
            (action_to_pot_fraction(&bet_sizes, Action::Bet(ALL_IN)) - 10.0).abs() < f64::EPSILON,
            "Bet(ALL_IN) should map to 10.0"
        );
        assert!(
            (action_to_pot_fraction(&bet_sizes, Action::Raise(ALL_IN)) - 10.0).abs() < f64::EPSILON,
            "Raise(ALL_IN) should map to 10.0"
        );
    }

    // -----------------------------------------------------------------------
    // 5. build_bet_actions from history
    // -----------------------------------------------------------------------

    #[test]
    fn build_bet_actions_from_history() {
        let bet_sizes = vec![0.5, 1.0];

        let history: [(Street, Action); 3] = [
            (Street::Preflop, Action::Call),
            (Street::Preflop, Action::Check),
            (Street::Flop, Action::Bet(0)),
        ];

        let result = build_bet_actions(&bet_sizes, &history);

        assert_eq!(result.len(), 3, "should have 3 bet actions");

        // Call -> 0.0
        assert!(
            (result[0].1 - 0.0).abs() < f64::EPSILON,
            "Call pot fraction should be 0.0, got {}",
            result[0].1
        );
        // Check -> 0.0
        assert!(
            (result[1].1 - 0.0).abs() < f64::EPSILON,
            "Check pot fraction should be 0.0, got {}",
            result[1].1
        );
        // Bet(0) -> 0.5
        assert!(
            (result[2].1 - 0.5).abs() < f64::EPSILON,
            "Bet(0) pot fraction should be 0.5, got {}",
            result[2].1
        );
    }
}
