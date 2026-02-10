mod hunl_postflop;
mod hunl_preflop;
mod kuhn;

use arrayvec::ArrayVec;

pub use hunl_postflop::{AbstractionMode, HunlPostflop, PostflopConfig, PostflopState, TerminalType};
pub use hunl_preflop::HunlPreflop;
pub use kuhn::KuhnPoker;

/// Maximum number of actions at any decision point.
///
/// Fold + Check/Call + up to 6 bet/raise sizes (4 pot fractions + all-in + min-raise).
pub const MAX_ACTIONS: usize = 8;

/// Sentinel value for all-in bets/raises.
///
/// Used as the index in `Bet(ALL_IN)` / `Raise(ALL_IN)` to represent
/// going all-in, as distinct from any config bet-size index.
pub const ALL_IN: u32 = u32::MAX;

/// Stack-allocated action list returned by [`Game::actions`].
pub type Actions = ArrayVec<Action, MAX_ACTIONS>;

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

/// Actions available in poker games.
///
/// The `u32` payload in `Bet` and `Raise` is **game-specific**:
/// - **`HunlPostflop`**: index into `config.bet_sizes`, or [`ALL_IN`] for all-in.
/// - **`HunlPreflop`**: absolute amount in cents.
/// - **`KuhnPoker`**: unused (always 0).
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
    type State: Clone + Send + Sync;

    /// Returns all possible initial states (e.g., all card deals)
    fn initial_states(&self) -> Vec<Self::State>;

    /// Returns true if the state is terminal (game over)
    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Returns the player to act at this state (undefined for terminal states)
    fn player(&self, state: &Self::State) -> Player;

    /// Returns available actions at this state
    fn actions(&self, state: &Self::State) -> Actions;

    /// Returns the next state after taking an action
    fn next_state(&self, state: &Self::State, action: Action) -> Self::State;

    /// Returns the utility for the given player at a terminal state
    fn utility(&self, state: &Self::State, player: Player) -> f64;

    /// Returns the numeric information set key for the current player.
    ///
    /// Encoded as a u64 via [`InfoKey`](crate::info_key::InfoKey) for
    /// zero-allocation hashing and comparison.
    fn info_set_key(&self, state: &Self::State) -> u64;
}
