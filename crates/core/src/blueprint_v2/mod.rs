pub mod bucket_file;
pub mod cluster_pipeline;
pub mod clustering;
pub mod config;
pub mod game_tree;
pub mod storage;

use serde::{Deserialize, Serialize};

/// A poker street — shared across all blueprint_v2 modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Street {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3,
}

impl Street {
    /// Parse from a raw byte (used in binary deserialization).
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Preflop),
            1 => Some(Self::Flop),
            2 => Some(Self::Turn),
            3 => Some(Self::River),
            _ => None,
        }
    }

    /// Returns the next street, or `None` if already on the river.
    #[must_use]
    pub fn next(self) -> Option<Self> {
        match self {
            Self::Preflop => Some(Self::Flop),
            Self::Flop => Some(Self::Turn),
            Self::Turn => Some(Self::River),
            Self::River => None,
        }
    }
}
