//! Heads-Up No-Limit Texas Hold'em Preflop game implementation.
//!
//! Models the preflop betting round in a heads-up cash game with configurable
//! bet sizing and stack depths.

use crate::config::Config;
use crate::equity::equity;
use crate::hands::CanonicalHand;

use super::{Action, Game, Player};

/// Heads-Up No-Limit Preflop game.
///
/// Models preflop play between two players (SB and BB) with configurable
/// stack depth and bet sizes.
#[derive(Debug, Clone)]
pub struct HunlPreflop {
    /// Game configuration (bet sizes, max bets)
    config: Config,
    /// Stack depth in big blinds
    stack_depth: u32,
}

impl HunlPreflop {
    /// Create a new HUNL Preflop game with the given configuration and stack depth.
    #[must_use]
    pub fn new(config: Config, stack_depth: u32) -> Self {
        Self {
            config,
            stack_depth,
        }
    }

    /// Create with default configuration at the given stack depth.
    #[must_use]
    pub fn with_stack(stack_depth: u32) -> Self {
        Self::new(Config::default(), stack_depth)
    }

    /// Get the stack depth for this game.
    #[must_use]
    pub fn stack_depth(&self) -> u32 {
        self.stack_depth
    }

    /// Get the configuration for this game.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }
}

/// State of a HUNL Preflop hand.
#[derive(Debug, Clone)]
pub struct HunlState {
    /// SB's canonical hand (Player1)
    sb_hand: CanonicalHand,
    /// BB's canonical hand (Player2)
    bb_hand: CanonicalHand,
    /// SB's remaining stack (in BB)
    sb_stack: f64,
    /// BB's remaining stack (in BB)
    bb_stack: f64,
    /// Current pot (total chips in the middle)
    pot: f64,
    /// Current bet to call (amount BB needs to call if it's their turn)
    to_call: f64,
    /// Who acts next (None if terminal)
    to_act: Option<Player>,
    /// Number of bets made this round
    num_bets: u8,
    /// Betting history for info set key
    history: Vec<ActionRecord>,
    /// Terminal state type
    terminal: Option<TerminalType>,
}

/// Record of an action taken for history tracking.
#[derive(Debug, Clone, Copy)]
enum ActionRecord {
    Fold,
    Check,
    Call,
    Raise(u32), // Raise to amount in BB (as integer for consistent keys)
}

/// Type of terminal state.
#[derive(Debug, Clone, Copy)]
enum TerminalType {
    /// Player folded, opponent wins pot
    Fold(Player),
    /// Showdown - hands are compared
    Showdown,
}

impl HunlState {
    /// Create the initial state for a preflop hand.
    fn new(sb_hand: CanonicalHand, bb_hand: CanonicalHand, stack_depth: f64) -> Self {
        // SB posts 0.5BB, BB posts 1BB
        // SB acts first preflop
        Self {
            sb_hand,
            bb_hand,
            sb_stack: stack_depth - 0.5, // SB posted
            bb_stack: stack_depth - 1.0, // BB posted
            pot: 1.5,
            to_call: 0.5,                  // SB needs to call 0.5 more to match BB
            to_act: Some(Player::Player1), // SB acts first
            num_bets: 1,                   // BB's post counts as first bet
            history: Vec::new(),
            terminal: None,
        }
    }

    /// Get the current player's stack.
    fn current_stack(&self) -> f64 {
        match self.to_act {
            Some(Player::Player1) => self.sb_stack,
            Some(Player::Player2) => self.bb_stack,
            None => 0.0,
        }
    }

    /// Get the opponent's stack.
    fn opponent_stack(&self) -> f64 {
        match self.to_act {
            Some(Player::Player1) => self.bb_stack,
            Some(Player::Player2) => self.sb_stack,
            None => 0.0,
        }
    }
}

impl Game for HunlPreflop {
    type State = HunlState;

    fn initial_states(&self) -> Vec<Self::State> {
        use crate::hands::all_hands;

        let stack = f64::from(self.stack_depth);
        let mut states = Vec::new();

        // Generate all hand combinations
        // For preflop, we iterate over all 169*169 = 28561 hand matchups
        // But we skip same-hand matchups (can't both hold AA)
        for sb_hand in all_hands() {
            for bb_hand in all_hands() {
                // Skip if hands share cards (simplified check for canonical hands)
                // In reality we'd check actual card overlap, but for canonical
                // hands we allow the same hand type since they could be different suits
                states.push(HunlState::new(sb_hand, bb_hand, stack));
            }
        }

        states
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        state.terminal.is_some()
    }

    fn player(&self, state: &Self::State) -> Player {
        state.to_act.unwrap_or(Player::Player1)
    }

    fn actions(&self, state: &Self::State) -> Vec<Action> {
        if state.terminal.is_some() {
            return vec![];
        }

        let mut actions = Vec::new();
        let stack = state.current_stack();
        let to_call = state.to_call;

        // Can always fold if facing a bet
        if to_call > 0.001 {
            actions.push(Action::Fold);
        }

        // Check if no bet to call
        if to_call < 0.001 {
            actions.push(Action::Check);
        } else if stack >= to_call {
            // Call if we have enough chips
            actions.push(Action::Call);
        }

        // Get legal raise sizes from config
        let current_bet = state.pot - state.sb_stack - state.bb_stack;
        let raise_sizes = self.config.get_legal_raise_sizes(
            current_bet / 2.0, // Approximate current bet level
            stack,
            state.num_bets,
        );

        for size in raise_sizes {
            // Convert to absolute raise amount (total bet size)
            // Size is in BB (0-1000 range typically), so size * 100 fits in u32
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let raise_amount = (size * 100.0).round() as u32; // Store as cents for precision
            if to_call < 0.001 {
                actions.push(Action::Bet(raise_amount));
            } else {
                actions.push(Action::Raise(raise_amount));
            }
        }

        actions
    }

    fn next_state(&self, state: &Self::State, action: Action) -> Self::State {
        let mut new_state = state.clone();
        let is_sb = state.to_act == Some(Player::Player1);

        match action {
            Action::Fold => {
                let folder = state.to_act.unwrap_or(Player::Player1);
                new_state.terminal = Some(TerminalType::Fold(folder));
                new_state.to_act = None;
                new_state.history.push(ActionRecord::Fold);
            }

            Action::Check => {
                new_state.history.push(ActionRecord::Check);
                // If BB checks after SB limped, go to showdown
                if !is_sb && state.to_call < 0.001 {
                    new_state.terminal = Some(TerminalType::Showdown);
                    new_state.to_act = None;
                } else {
                    // Switch to other player
                    new_state.to_act = Some(state.to_act.unwrap().opponent());
                    new_state.to_call = 0.0;
                }
            }

            Action::Call => {
                let call_amount = state.to_call.min(state.current_stack());
                if is_sb {
                    new_state.sb_stack -= call_amount;
                } else {
                    new_state.bb_stack -= call_amount;
                }
                new_state.pot += call_amount;
                new_state.to_call = 0.0;
                new_state.history.push(ActionRecord::Call);

                // Check if this is SB's first action (limping)
                // In that case, BB gets option to raise
                if is_sb && state.num_bets == 1 {
                    // SB limped, BB gets option
                    new_state.to_act = Some(Player::Player2);
                } else {
                    // Call after a raise closes the action - go to showdown
                    new_state.terminal = Some(TerminalType::Showdown);
                    new_state.to_act = None;
                }
            }

            Action::Bet(amount) | Action::Raise(amount) => {
                let raise_amount = f64::from(amount) / 100.0; // Convert from cents
                let actual_raise = raise_amount.min(state.current_stack());

                if is_sb {
                    new_state.sb_stack -= actual_raise;
                } else {
                    new_state.bb_stack -= actual_raise;
                }
                new_state.pot += actual_raise;
                new_state.to_call = actual_raise - state.to_call;
                new_state.num_bets += 1;
                new_state.to_act = Some(state.to_act.unwrap().opponent());
                new_state.history.push(ActionRecord::Raise(amount));

                // Check if opponent is all-in (can only call or fold)
                if new_state.opponent_stack() < 0.001 {
                    // Opponent already all-in, immediate showdown
                    new_state.terminal = Some(TerminalType::Showdown);
                    new_state.to_act = None;
                }
            }
        }

        new_state
    }

    fn utility(&self, state: &Self::State, player: Player) -> f64 {
        let Some(terminal) = state.terminal else {
            return 0.0;
        };

        match terminal {
            TerminalType::Fold(folder) => {
                // Folder loses, opponent wins current pot
                let stack = f64::from(self.stack_depth);
                let sb_invested = stack - state.sb_stack;
                let bb_invested = stack - state.bb_stack;

                if folder == Player::Player1 {
                    // SB folded
                    if player == Player::Player1 {
                        -sb_invested // SB loses their investment
                    } else {
                        sb_invested // BB wins SB's investment
                    }
                } else {
                    // BB folded
                    if player == Player::Player2 {
                        -bb_invested // BB loses their investment
                    } else {
                        bb_invested // SB wins BB's investment
                    }
                }
            }

            TerminalType::Showdown => {
                // Calculate expected value using preflop equity
                let sb_equity = equity(state.sb_hand, state.bb_hand);

                let stack = f64::from(self.stack_depth);
                let sb_invested = stack - state.sb_stack;
                let bb_invested = stack - state.bb_stack;
                let pot = sb_invested + bb_invested;

                // EV = (equity * pot) - amount_invested
                // For SB: sb_equity * pot - sb_invested
                // For BB: (1 - sb_equity) * pot - bb_invested
                let sb_ev = sb_equity * pot - sb_invested;
                let bb_ev = (1.0 - sb_equity) * pot - bb_invested;

                match player {
                    Player::Player1 => sb_ev,
                    Player::Player2 => bb_ev,
                }
            }
        }
    }

    fn info_set_key(&self, state: &Self::State) -> String {
        let (hand, position) = match state.to_act {
            Some(Player::Player1) => (state.sb_hand, "SB"),
            Some(Player::Player2) => (state.bb_hand, "BB"),
            None => (state.sb_hand, "?"), // Shouldn't happen for info set lookup
        };

        let history_str: String = state
            .history
            .iter()
            .map(|a| match a {
                ActionRecord::Fold => "f".to_string(),
                ActionRecord::Check => "x".to_string(),
                ActionRecord::Call => "c".to_string(),
                ActionRecord::Raise(amt) => format!("r{amt}"),
            })
            .collect();

        format!("{position}:{hand}:{history_str}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_game() -> HunlPreflop {
        HunlPreflop::with_stack(100)
    }

    #[test]
    fn initial_states_not_empty() {
        let game = create_game();
        let states = game.initial_states();
        assert!(!states.is_empty());
        // 169 * 169 = 28561 matchups
        assert_eq!(states.len(), 169 * 169);
    }

    #[test]
    fn initial_state_correct_stacks() {
        let game = create_game();
        let states = game.initial_states();
        let state = &states[0];

        // SB posted 0.5, BB posted 1.0
        assert!((state.sb_stack - 99.5).abs() < 0.001);
        assert!((state.bb_stack - 99.0).abs() < 0.001);
        assert!((state.pot - 1.5).abs() < 0.001);
    }

    #[test]
    fn sb_acts_first() {
        let game = create_game();
        let states = game.initial_states();
        let state = &states[0];

        assert_eq!(game.player(state), Player::Player1);
    }

    #[test]
    fn sb_can_fold_call_or_raise() {
        let game = create_game();
        let states = game.initial_states();
        let state = &states[0];

        let actions = game.actions(state);
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::Call));
        // Should have some raise options
        assert!(actions.iter().any(|a| matches!(a, Action::Raise(_))));
    }

    #[test]
    fn fold_is_terminal() {
        let game = create_game();
        let states = game.initial_states();
        let state = &states[0];

        let folded = game.next_state(state, Action::Fold);
        assert!(game.is_terminal(&folded));
    }

    #[test]
    fn fold_gives_pot_to_opponent() {
        let game = create_game();
        let states = game.initial_states();
        let state = &states[0];

        let folded = game.next_state(state, Action::Fold);

        // SB folded, BB wins SB's 0.5 posted
        assert!(game.utility(&folded, Player::Player2) > 0.0);
        assert!(game.utility(&folded, Player::Player1) < 0.0);
    }

    #[test]
    fn call_then_check_is_showdown() {
        let game = create_game();
        let states = game.initial_states();
        let state = &states[0];

        // SB calls (limps)
        let after_call = game.next_state(state, Action::Call);
        assert!(!game.is_terminal(&after_call));
        assert_eq!(game.player(&after_call), Player::Player2);

        // BB checks
        let actions = game.actions(&after_call);
        assert!(actions.contains(&Action::Check));

        let after_check = game.next_state(&after_call, Action::Check);
        assert!(game.is_terminal(&after_check));
    }

    #[test]
    fn raise_switches_action() {
        let game = create_game();
        let states = game.initial_states();
        let state = &states[0];

        // Find a raise action
        let actions = game.actions(state);
        let raise = actions
            .iter()
            .find(|a| matches!(a, Action::Raise(_)))
            .unwrap();

        let after_raise = game.next_state(state, *raise);
        assert!(!game.is_terminal(&after_raise));
        assert_eq!(game.player(&after_raise), Player::Player2);
    }

    #[test]
    fn info_set_includes_hand_and_history() {
        let game = create_game();
        let states = game.initial_states();
        let state = &states[0];

        let info_set = game.info_set_key(state);
        assert!(info_set.starts_with("SB:"));
        assert!(info_set.contains(':')); // Has hand component

        // After an action, history should be included
        let actions = game.actions(state);
        let raise = actions
            .iter()
            .find(|a| matches!(a, Action::Raise(_)))
            .unwrap();
        let after_raise = game.next_state(state, *raise);

        let info_set2 = game.info_set_key(&after_raise);
        assert!(info_set2.starts_with("BB:"));
        assert!(info_set2.contains(":r")); // Contains raise history
    }

    #[test]
    fn call_closes_action() {
        let game = create_game();
        let states = game.initial_states();
        let state = &states[0];

        // SB raises
        let actions = game.actions(state);
        let raise = actions
            .iter()
            .find(|a| matches!(a, Action::Raise(_)))
            .unwrap();
        let after_raise = game.next_state(state, *raise);

        // BB calls
        let after_call = game.next_state(&after_raise, Action::Call);
        assert!(game.is_terminal(&after_call));
    }

    #[test]
    fn utilities_are_zero_sum() {
        let game = create_game();
        let states = game.initial_states();
        let state = &states[0];

        // SB folds
        let folded = game.next_state(state, Action::Fold);
        let u1 = game.utility(&folded, Player::Player1);
        let u2 = game.utility(&folded, Player::Player2);
        assert!((u1 + u2).abs() < 0.001, "Utilities should be zero-sum");

        // SB calls, BB checks (showdown)
        let called = game.next_state(state, Action::Call);
        let checked = game.next_state(&called, Action::Check);
        let u1 = game.utility(&checked, Player::Player1);
        let u2 = game.utility(&checked, Player::Player2);
        assert!(
            (u1 + u2).abs() < 0.001,
            "Showdown utilities should be zero-sum"
        );
    }
}
