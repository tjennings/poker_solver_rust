pub mod bucket_file;
pub mod continuation;
pub mod centroid_file;
pub mod per_flop_bucket_file;
pub mod bundle;
pub mod cbv;
pub mod cbv_compute;
pub mod cluster_diagnostics;
pub mod cluster_pipeline;
pub mod clustering;
pub mod config;
pub mod equity_cache;
pub mod exploitable_spots;
pub mod full_depth_solver;
pub mod game_tree;
pub mod mccfr;

pub mod storage;
pub mod leaf_evaluator;
pub mod subgame;
pub mod subgame_cfr;
pub mod trainer;

// Re-export subgame types at module level for convenience.
pub use leaf_evaluator::LeafEvaluator;
pub use subgame::SubgameConfig;
pub use subgame_cfr::{SubgameHands, SubgameStrategy, compute_combo_equities, build_boundary_mapping};

use serde::{Deserialize, Serialize};

/// A poker street — shared across all `blueprint_v2` modules.
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
    #[must_use]
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
