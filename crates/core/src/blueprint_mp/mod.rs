pub mod config;
pub mod game_tree;
pub mod info_key;
pub mod terminal;
pub mod types;

/// Maximum number of players supported in multiplayer blueprints.
pub const MAX_PLAYERS: usize = 8;

pub use config::*;
pub use info_key::InfoKey128;
pub use types::{Bucket, Chips, Deal, DealWithBuckets, PlayerSet, Seat, Street};
