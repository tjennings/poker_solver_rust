mod config;

use arrayvec::ArrayVec;

pub use config::{AbstractionMode, PostflopConfig};

/// Maximum number of actions at any decision point.
pub const MAX_ACTIONS: usize = 8;

/// Sentinel value for all-in bets/raises.
pub const ALL_IN: u32 = u32::MAX;

/// Stack-allocated action list.
pub type Actions = ArrayVec<Action, MAX_ACTIONS>;

/// Player in a two-player game
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    Player1,
    Player2,
}

impl Player {
    #[must_use]
    pub const fn opponent(self) -> Self {
        match self {
            Self::Player1 => Self::Player2,
            Self::Player2 => Self::Player1,
        }
    }
}

/// Actions available in poker games.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Fold,
    Check,
    Call,
    Bet(u32),
    Raise(u32),
}
