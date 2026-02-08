use super::{Action, Actions, Game, Player};

/// Kuhn Poker - a simplified poker game with 3 cards (J, Q, K) and 2 players.
///
/// Each player antes 1 chip, receives one card, then:
/// - Player 1 can check or bet 1
/// - If check: Player 2 can check (showdown) or bet 1
///   - If bet: Player 1 can fold or call
/// - If bet: Player 2 can fold or call (showdown)
#[derive(Debug, Clone)]
pub struct KuhnPoker;

/// Cards in Kuhn Poker (Jack < Queen < King)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Card {
    Jack = 0,
    Queen = 1,
    King = 2,
}

impl Card {
    const ALL: [Card; 3] = [Card::Jack, Card::Queen, Card::King];

    fn beats(self, other: Card) -> bool {
        (self as u8) > (other as u8)
    }
}

/// State of a Kuhn Poker game
#[derive(Debug, Clone)]
pub struct KuhnState {
    /// Card held by Player 1
    p1_card: Card,
    /// Card held by Player 2
    p2_card: Card,
    /// Action history
    history: Vec<KuhnAction>,
    /// Current pot size
    pot: u32,
    /// Amount each player has contributed
    p1_contrib: u32,
    p2_contrib: u32,
}

/// Actions specific to Kuhn Poker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KuhnAction {
    Check,
    Bet,
    Fold,
    Call,
}

impl KuhnPoker {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for KuhnPoker {
    fn default() -> Self {
        Self::new()
    }
}

impl Game for KuhnPoker {
    type State = KuhnState;

    fn initial_states(&self) -> Vec<Self::State> {
        let mut states = Vec::with_capacity(6);

        // All possible deals: 3 cards, 2 players = 3 * 2 = 6 deals
        for &p1_card in &Card::ALL {
            for &p2_card in &Card::ALL {
                if p1_card != p2_card {
                    states.push(KuhnState {
                        p1_card,
                        p2_card,
                        history: Vec::new(),
                        pot: 2, // Both players ante 1
                        p1_contrib: 1,
                        p2_contrib: 1,
                    });
                }
            }
        }

        states
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        matches!(
            state.history.as_slice(),
            [KuhnAction::Check, KuhnAction::Check]
                | [
                    KuhnAction::Check,
                    KuhnAction::Bet,
                    KuhnAction::Fold | KuhnAction::Call
                ]
                | [KuhnAction::Bet, KuhnAction::Fold | KuhnAction::Call]
        )
    }

    fn player(&self, state: &Self::State) -> Player {
        if state.history.len() == 1 {
            Player::Player2
        } else {
            Player::Player1
        }
    }

    fn actions(&self, state: &Self::State) -> Actions {
        let mut actions = Actions::new();
        match state.history.as_slice() {
            // No bet yet: can check or bet
            [] | [KuhnAction::Check] => {
                actions.push(Action::Check);
                actions.push(Action::Bet(1));
            }
            // Facing a bet: can fold or call
            [KuhnAction::Bet] | [KuhnAction::Check, KuhnAction::Bet] => {
                actions.push(Action::Fold);
                actions.push(Action::Call);
            }
            _ => {}
        }
        actions
    }

    fn next_state(&self, state: &Self::State, action: Action) -> Self::State {
        let mut new_state = state.clone();

        let kuhn_action = match action {
            Action::Check => KuhnAction::Check,
            Action::Bet(_) => KuhnAction::Bet,
            Action::Fold => KuhnAction::Fold,
            Action::Call => KuhnAction::Call,
            Action::Raise(_) => unreachable!("Raise not valid in Kuhn Poker"),
        };

        // Update contributions and pot
        match (self.player(state), kuhn_action) {
            (Player::Player1, KuhnAction::Bet) => {
                new_state.p1_contrib += 1;
                new_state.pot += 1;
            }
            (Player::Player2, KuhnAction::Bet) => {
                new_state.p2_contrib += 1;
                new_state.pot += 1;
            }
            (Player::Player1, KuhnAction::Call) => {
                let to_call = new_state.p2_contrib - new_state.p1_contrib;
                new_state.p1_contrib += to_call;
                new_state.pot += to_call;
            }
            (Player::Player2, KuhnAction::Call) => {
                let to_call = new_state.p1_contrib - new_state.p2_contrib;
                new_state.p2_contrib += to_call;
                new_state.pot += to_call;
            }
            _ => {}
        }

        new_state.history.push(kuhn_action);
        new_state
    }

    fn utility(&self, state: &Self::State, player: Player) -> f64 {
        debug_assert!(self.is_terminal(state));

        let winner = match state.history.as_slice() {
            // Folds: folder loses
            [KuhnAction::Bet, KuhnAction::Fold] => Some(Player::Player1),
            [KuhnAction::Check, KuhnAction::Bet, KuhnAction::Fold] => Some(Player::Player2),
            // Showdowns: higher card wins
            [KuhnAction::Check, KuhnAction::Check]
            | [KuhnAction::Bet, KuhnAction::Call]
            | [KuhnAction::Check, KuhnAction::Bet, KuhnAction::Call] => {
                if state.p1_card.beats(state.p2_card) {
                    Some(Player::Player1)
                } else {
                    Some(Player::Player2)
                }
            }
            _ => None,
        };

        match (winner, player) {
            (Some(w), p) if w == p => f64::from(state.pot) - f64::from(contribution(state, p)),
            (Some(_), p) => -(f64::from(contribution(state, p))),
            (None, _) => 0.0,
        }
    }

    fn info_set_key(&self, state: &Self::State) -> String {
        let card = match self.player(state) {
            Player::Player1 => state.p1_card,
            Player::Player2 => state.p2_card,
        };

        let card_char = match card {
            Card::Jack => 'J',
            Card::Queen => 'Q',
            Card::King => 'K',
        };

        let history_str: String = state
            .history
            .iter()
            .map(|a| match a {
                KuhnAction::Check => 'c',
                KuhnAction::Bet => 'b',
                KuhnAction::Fold => 'f',
                KuhnAction::Call => 'k',
            })
            .collect();

        format!("{card_char}{history_str}")
    }
}

fn contribution(state: &KuhnState, player: Player) -> u32 {
    match player {
        Player::Player1 => state.p1_contrib,
        Player::Player2 => state.p2_contrib,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn initial_states_returns_six_deals() {
        let game = KuhnPoker::new();
        let states = game.initial_states();

        assert_eq!(states.len(), 6);

        // Verify all deals are unique
        for (i, s1) in states.iter().enumerate() {
            for s2 in states.iter().skip(i + 1) {
                assert!(s1.p1_card != s2.p1_card || s1.p2_card != s2.p2_card);
            }
        }

        // Verify no player has same card
        for state in &states {
            assert_ne!(state.p1_card, state.p2_card);
        }
    }

    #[timed_test]
    fn initial_pot_is_two() {
        let game = KuhnPoker::new();
        for state in game.initial_states() {
            assert_eq!(state.pot, 2);
            assert_eq!(state.p1_contrib, 1);
            assert_eq!(state.p2_contrib, 1);
        }
    }

    #[timed_test]
    fn terminal_after_check_check() {
        let game = KuhnPoker::new();
        let mut state = game.initial_states().remove(0);

        assert!(!game.is_terminal(&state));

        state = game.next_state(&state, Action::Check);
        assert!(!game.is_terminal(&state));

        state = game.next_state(&state, Action::Check);
        assert!(game.is_terminal(&state));
    }

    #[timed_test]
    fn terminal_after_bet_fold() {
        let game = KuhnPoker::new();
        let mut state = game.initial_states().remove(0);

        state = game.next_state(&state, Action::Bet(1));
        assert!(!game.is_terminal(&state));

        state = game.next_state(&state, Action::Fold);
        assert!(game.is_terminal(&state));
    }

    #[timed_test]
    fn terminal_after_bet_call() {
        let game = KuhnPoker::new();
        let mut state = game.initial_states().remove(0);

        state = game.next_state(&state, Action::Bet(1));
        state = game.next_state(&state, Action::Call);
        assert!(game.is_terminal(&state));
    }

    #[timed_test]
    fn terminal_after_check_bet_fold() {
        let game = KuhnPoker::new();
        let mut state = game.initial_states().remove(0);

        state = game.next_state(&state, Action::Check);
        state = game.next_state(&state, Action::Bet(1));
        state = game.next_state(&state, Action::Fold);
        assert!(game.is_terminal(&state));
    }

    #[timed_test]
    fn terminal_after_check_bet_call() {
        let game = KuhnPoker::new();
        let mut state = game.initial_states().remove(0);

        state = game.next_state(&state, Action::Check);
        state = game.next_state(&state, Action::Bet(1));
        state = game.next_state(&state, Action::Call);
        assert!(game.is_terminal(&state));
    }

    #[timed_test]
    fn bet_fold_gives_pot_to_bettor() {
        let game = KuhnPoker::new();
        // Find a deal where P1 has Jack (weakest)
        let mut state = game
            .initial_states()
            .into_iter()
            .find(|s| s.p1_card == Card::Jack)
            .unwrap();

        state = game.next_state(&state, Action::Bet(1));
        state = game.next_state(&state, Action::Fold);

        // P1 bet 1 (total contrib 2), P2 folded (contrib 1)
        // P1 wins pot of 3, profit = 3 - 2 = 1
        assert_eq!(game.utility(&state, Player::Player1), 1.0);
        assert_eq!(game.utility(&state, Player::Player2), -1.0);
    }

    #[timed_test]
    fn check_bet_fold_gives_pot_to_bettor() {
        let game = KuhnPoker::new();
        let mut state = game.initial_states().remove(0);

        state = game.next_state(&state, Action::Check);
        state = game.next_state(&state, Action::Bet(1));
        state = game.next_state(&state, Action::Fold);

        // P2 bet 1 (total contrib 2), P1 folded (contrib 1)
        // P2 wins pot of 3, profit = 3 - 2 = 1
        assert_eq!(game.utility(&state, Player::Player2), 1.0);
        assert_eq!(game.utility(&state, Player::Player1), -1.0);
    }

    #[timed_test]
    fn showdown_higher_card_wins() {
        let game = KuhnPoker::new();
        // King vs Jack: King wins
        let mut state = game
            .initial_states()
            .into_iter()
            .find(|s| s.p1_card == Card::King && s.p2_card == Card::Jack)
            .unwrap();

        state = game.next_state(&state, Action::Check);
        state = game.next_state(&state, Action::Check);

        // Check-check showdown, pot is 2, each contributed 1
        // P1 (King) wins, profit = 2 - 1 = 1
        assert_eq!(game.utility(&state, Player::Player1), 1.0);
        assert_eq!(game.utility(&state, Player::Player2), -1.0);
    }

    #[timed_test]
    fn bet_call_showdown_higher_card_wins() {
        let game = KuhnPoker::new();
        // King vs Queen: King wins
        let mut state = game
            .initial_states()
            .into_iter()
            .find(|s| s.p1_card == Card::King && s.p2_card == Card::Queen)
            .unwrap();

        state = game.next_state(&state, Action::Bet(1));
        state = game.next_state(&state, Action::Call);

        // Bet-call showdown, pot is 4, each contributed 2
        // P1 (King) wins, profit = 4 - 2 = 2
        assert_eq!(game.utility(&state, Player::Player1), 2.0);
        assert_eq!(game.utility(&state, Player::Player2), -2.0);
    }

    #[timed_test]
    fn info_set_key_includes_card_and_history() {
        let game = KuhnPoker::new();
        let mut state = game
            .initial_states()
            .into_iter()
            .find(|s| s.p1_card == Card::King && s.p2_card == Card::Jack)
            .unwrap();

        // P1's info set at start: just their card
        assert_eq!(game.info_set_key(&state), "K");

        state = game.next_state(&state, Action::Check);
        // P2's info set: their card + history
        assert_eq!(game.info_set_key(&state), "Jc");

        state = game.next_state(&state, Action::Bet(1));
        // P1's info set: their card + history
        assert_eq!(game.info_set_key(&state), "Kcb");
    }

    #[timed_test]
    fn player_alternates_correctly() {
        let game = KuhnPoker::new();
        let mut state = game.initial_states().remove(0);

        assert_eq!(game.player(&state), Player::Player1);

        state = game.next_state(&state, Action::Check);
        assert_eq!(game.player(&state), Player::Player2);

        state = game.next_state(&state, Action::Bet(1));
        assert_eq!(game.player(&state), Player::Player1);
    }

    #[timed_test]
    fn actions_correct_at_each_decision() {
        let game = KuhnPoker::new();
        let mut state = game.initial_states().remove(0);

        // P1 opens
        assert_eq!(
            game.actions(&state).as_slice(),
            &[Action::Check, Action::Bet(1)]
        );

        state = game.next_state(&state, Action::Check);
        // P2 after check
        assert_eq!(
            game.actions(&state).as_slice(),
            &[Action::Check, Action::Bet(1)]
        );

        state = game.next_state(&state, Action::Bet(1));
        // P1 facing bet
        assert_eq!(
            game.actions(&state).as_slice(),
            &[Action::Fold, Action::Call]
        );
    }

    #[timed_test]
    fn pot_updates_correctly() {
        let game = KuhnPoker::new();
        let mut state = game.initial_states().remove(0);

        assert_eq!(state.pot, 2);

        state = game.next_state(&state, Action::Bet(1));
        assert_eq!(state.pot, 3);
        assert_eq!(state.p1_contrib, 2);

        state = game.next_state(&state, Action::Call);
        assert_eq!(state.pot, 4);
        assert_eq!(state.p2_contrib, 2);
    }
}
