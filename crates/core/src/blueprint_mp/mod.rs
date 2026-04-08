pub mod types;

/// Maximum number of players supported in multiplayer blueprints.
pub const MAX_PLAYERS: usize = 8;

pub use types::{Bucket, Chips, PlayerSet, Seat, Street};
