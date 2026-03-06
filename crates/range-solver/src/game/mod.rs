mod node;

use crate::action_tree::*;
use crate::card::*;
use crate::mutex_like::*;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// StrengthItem
// ---------------------------------------------------------------------------

/// Hand-strength entry: pairs a strength rank with a hand index.
///
/// Used in sorted order (ascending strength) to evaluate showdown equity.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct StrengthItem {
    pub(crate) strength: u16,
    pub(crate) index: u16,
}

// ---------------------------------------------------------------------------
// SwapList
// ---------------------------------------------------------------------------

/// Per-player list of hand-index pairs that must be swapped when applying
/// a suit isomorphism.
pub(crate) type SwapList = [Vec<(u16, u16)>; 2];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum State {
    ConfigError = 0,
    #[default]
    Uninitialized = 1,
    TreeBuilt = 2,
    MemoryAllocated = 3,
    Solved = 4,
}

// ---------------------------------------------------------------------------
// PostFlopNode
// ---------------------------------------------------------------------------

/// A node in the postflop game tree.
///
/// Nodes are stored contiguously in `PostFlopGame::node_arena` as
/// `Vec<MutexLike<PostFlopNode>>`. The `children_offset` / `num_children`
/// fields encode child positions as offsets from `self`, enabling
/// pointer-arithmetic traversal without separate allocations.
///
/// The three `storage*` raw pointers point into the game-level byte vectors
/// (`storage1`, `storage2`, `storage_ip`) and are reinterpreted as `f32`,
/// `u16`, or `i16` slices depending on the phase of the solve and whether
/// compression is enabled.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PostFlopNode {
    pub(crate) prev_action: Action,
    pub(crate) player: u8,
    pub(crate) turn: Card,
    pub(crate) river: Card,
    pub(crate) is_locked: bool,
    pub(crate) amount: i32,
    pub(crate) children_offset: u32,
    pub(crate) num_children: u16,
    pub(crate) num_elements_ip: u16,
    pub(crate) num_elements: u32,
    pub(crate) scale1: f32,
    pub(crate) scale2: f32,
    pub(crate) scale3: f32,
    pub(crate) storage1: *mut u8, // strategy (or chance cfvalues)
    pub(crate) storage2: *mut u8, // regrets (or cfvalues)
    pub(crate) storage3: *mut u8, // IP cfvalues
}

// SAFETY: The raw pointers in PostFlopNode point into `PostFlopGame`'s storage
// vectors. The game's solve loop guarantees that concurrent access is
// coordinated via the tree structure (parent before children, disjoint
// subtrees in parallel). No two threads mutate the same node's storage
// simultaneously.
unsafe impl Send for PostFlopNode {}
unsafe impl Sync for PostFlopNode {}

// ---------------------------------------------------------------------------
// PostFlopGame
// ---------------------------------------------------------------------------

/// The top-level struct for a postflop game.
///
/// Owns the full game tree (`node_arena`), the global storage buffers, hand
/// information, isomorphism tables, and interpreter state.
#[derive(Default)]
pub struct PostFlopGame {
    // -- state --
    pub(crate) state: State,

    // -- postflop game configuration --
    pub(crate) card_config: CardConfig,
    pub(crate) tree_config: TreeConfig,
    pub(crate) added_lines: Vec<Vec<Action>>,
    pub(crate) removed_lines: Vec<Vec<Action>>,
    pub(crate) action_root: Box<MutexLike<ActionTreeNode>>,

    // -- computed from configuration --
    pub(crate) num_combinations: f64,
    pub(crate) initial_weights: [Vec<f32>; 2],
    pub(crate) private_cards: [Vec<(Card, Card)>; 2],
    pub(crate) same_hand_index: [Vec<u16>; 2],

    // -- indices in `private_cards` that do not conflict with the board --
    pub(crate) valid_indices_flop: [Vec<u16>; 2],
    pub(crate) valid_indices_turn: Vec<[Vec<u16>; 2]>,
    pub(crate) valid_indices_river: Vec<[Vec<u16>; 2]>,

    // -- hand strength: indices stored in ascending strength order --
    pub(crate) hand_strength: Vec<[Vec<StrengthItem>; 2]>,

    // -- isomorphism information --
    pub(crate) isomorphism_ref_turn: Vec<u8>,
    pub(crate) isomorphism_card_turn: Vec<Card>,
    pub(crate) isomorphism_swap_turn: [SwapList; 4],
    pub(crate) isomorphism_ref_river: Vec<Vec<u8>>,
    pub(crate) isomorphism_card_river: [Vec<Card>; 4],
    pub(crate) isomorphism_swap_river: [[SwapList; 4]; 4],

    // -- bunching effect (not yet implemented) --
    pub(crate) bunching_num_dead_cards: usize,
    pub(crate) bunching_num_combinations: f64,
    pub(crate) bunching_arena: Vec<f32>,
    pub(crate) bunching_strength: Vec<[Vec<u16>; 2]>,
    pub(crate) bunching_num_flop: [Vec<usize>; 2],
    pub(crate) bunching_num_turn: [Vec<Vec<usize>>; 2],
    pub(crate) bunching_num_river: [Vec<Vec<usize>>; 2],
    pub(crate) bunching_coef_flop: [Vec<usize>; 2],
    pub(crate) bunching_coef_turn: [Vec<Vec<usize>>; 2],

    // -- storage mode --
    pub(crate) storage_mode: BoardState,
    pub(crate) target_storage_mode: BoardState,
    pub(crate) num_nodes: [u64; 3],
    pub(crate) is_compression_enabled: bool,
    pub(crate) num_storage: u64,
    pub(crate) num_storage_ip: u64,
    pub(crate) num_storage_chance: u64,
    pub(crate) misc_memory_usage: u64,

    // -- global storage --
    // `storage*` are the backing buffers referenced by `PostFlopNode::storage*`.
    pub(crate) node_arena: Vec<MutexLike<PostFlopNode>>,
    pub(crate) storage1: Vec<u8>,
    pub(crate) storage2: Vec<u8>,
    pub(crate) storage_ip: Vec<u8>,
    pub(crate) storage_chance: Vec<u8>,
    pub(crate) locking_strategy: BTreeMap<usize, Vec<f32>>,

    // -- result interpreter state --
    pub(crate) action_history: Vec<usize>,
    pub(crate) node_history: Vec<usize>,
    pub(crate) is_normalized_weight_cached: bool,
    pub(crate) turn: Card,
    pub(crate) river: Card,
    pub(crate) turn_swapped_suit: Option<(u8, u8)>,
    pub(crate) turn_swap: Option<u8>,
    pub(crate) river_swap: Option<(u8, u8)>,
    pub(crate) total_bet_amount: [i32; 2],
    pub(crate) weights: [Vec<f32>; 2],
    pub(crate) normalized_weights: [Vec<f32>; 2],
    pub(crate) cfvalues_cache: [Vec<f32>; 2],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_default_state() {
        let game = PostFlopGame::default();
        assert_eq!(game.state, State::Uninitialized);
        assert!(game.node_arena.is_empty());
        assert!(game.storage1.is_empty());
        assert!(game.storage2.is_empty());
        assert!(game.storage_ip.is_empty());
        assert!(game.storage_chance.is_empty());
        assert!(game.private_cards[0].is_empty());
        assert!(game.private_cards[1].is_empty());
        assert_eq!(game.num_combinations, 0.0);
        assert!(!game.is_compression_enabled);
        assert_eq!(game.bunching_num_dead_cards, 0);
    }

    #[test]
    fn game_default_interpreter_state() {
        let game = PostFlopGame::default();
        assert!(game.action_history.is_empty());
        assert!(game.node_history.is_empty());
        assert!(!game.is_normalized_weight_cached);
        // Default u8 is 0; `turn` and `river` are set to meaningful values
        // by `apply_history()` after tree construction, not by Default.
        assert_eq!(game.turn, 0);
        assert_eq!(game.river, 0);
        assert!(game.turn_swapped_suit.is_none());
        assert!(game.turn_swap.is_none());
        assert!(game.river_swap.is_none());
        assert_eq!(game.total_bet_amount, [0, 0]);
    }

    #[test]
    fn state_ordering() {
        assert!(State::ConfigError < State::Uninitialized);
        assert!(State::Uninitialized < State::TreeBuilt);
        assert!(State::TreeBuilt < State::MemoryAllocated);
        assert!(State::MemoryAllocated < State::Solved);
    }

    #[test]
    fn strength_item_ordering() {
        let a = StrengthItem {
            strength: 10,
            index: 5,
        };
        let b = StrengthItem {
            strength: 20,
            index: 3,
        };
        assert!(a < b);
    }
}
