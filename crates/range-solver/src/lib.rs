pub mod card;
pub mod range;
pub mod bet_size;
pub mod action_tree;
pub mod interface;
pub mod mutex_like;
pub mod game;
pub mod solver;

pub(crate) mod isomorphism;
pub(crate) mod sliceop;
pub(crate) mod utility;

mod hand_table;
pub(crate) mod hand;

/// Returns the sorted hand evaluation lookup table (4824 entries).
///
/// Each entry is an i32 encoding a unique 7-card hand equivalence class.
/// Used by `Hand::evaluate()` via binary search to map internal hand
/// representations to u16 strength indices.
pub fn hand_table_data() -> &'static [i32] {
    &hand_table::HAND_TABLE
}

// Convenience re-exports
pub use game::PostFlopGame;
pub use solver::{solve, solve_step, compute_exploitability, compute_average, finalize};
pub use card::CardConfig;
pub use action_tree::{Action, ActionTree, TreeConfig, BoardState, PLAYER_DEPTH_BOUNDARY_FLAG};
