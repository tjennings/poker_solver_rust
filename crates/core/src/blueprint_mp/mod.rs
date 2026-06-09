pub mod config;
pub mod exploitability;
pub mod game_tree;
pub mod info_key;
pub mod lazy_mccfr;
pub mod mccfr;
pub mod mmap_buffer;
pub mod sparse_storage;
pub mod storage;
pub mod terminal;
pub mod trainer;
pub mod training_runtime_adapter;
pub mod types;

/// Maximum number of players supported in multiplayer blueprints.
pub const MAX_PLAYERS: usize = 8;

pub use config::*;
pub use info_key::InfoKey128;
pub use types::{
    Bucket, Chips, Deal, DealWithBuckets, PlayerSet, Seat, Street, parse_position, position_label,
};
