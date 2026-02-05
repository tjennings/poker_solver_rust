mod hunl_postflop;
mod hunl_preflop;
mod kuhn;

pub use hunl_postflop::{HunlPostflop, PostflopConfig, PostflopState, TerminalType};
pub use hunl_preflop::HunlPreflop;
pub use kuhn::KuhnPoker;

/// Player in a two-player game
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    Player1,
    Player2,
}

impl Player {
    /// Returns the opponent of this player
    #[must_use]
    pub const fn opponent(self) -> Self {
        match self {
            Self::Player1 => Self::Player2,
            Self::Player2 => Self::Player1,
        }
    }
}

/// Actions available in poker games
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Fold,
    Check,
    Call,
    Bet(u32),
    Raise(u32),
}

/// Trait defining a two-player zero-sum game
pub trait Game: Send + Sync {
    /// The state type for this game
    type State: Clone + Send;

    /// Returns all possible initial states (e.g., all card deals)
    fn initial_states(&self) -> Vec<Self::State>;

    /// Returns true if the state is terminal (game over)
    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Returns the player to act at this state (undefined for terminal states)
    fn player(&self, state: &Self::State) -> Player;

    /// Returns available actions at this state
    fn actions(&self, state: &Self::State) -> Vec<Action>;

    /// Returns the next state after taking an action
    fn next_state(&self, state: &Self::State, action: Action) -> Self::State;

    /// Returns the utility for the given player at a terminal state
    fn utility(&self, state: &Self::State, player: Player) -> f64;

    /// Returns the information set key for the current player
    fn info_set_key(&self, state: &Self::State) -> String;
}
