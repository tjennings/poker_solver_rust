mod evaluation;
mod interpreter;
mod node;
mod query;

use crate::action_tree::*;
use crate::card::*;
use crate::mutex_like::*;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Callback for computing boundary CFVs on demand during solving.
///
/// When the solver reaches a depth boundary with no cached CFVs, it calls
/// `compute_cfvs` to get per-hand values. The implementation (e.g. rollout
/// evaluator) receives the boundary's pot, remaining stack, the opponent's
/// reach at the boundary, and the player being evaluated.
pub trait BoundaryEvaluator: Send + Sync {
    /// Number of continuation strategies at each boundary. Default: 1 (legacy).
    fn num_continuations(&self) -> usize { 1 }

    /// Compute per-hand CFVs at a depth boundary for a specific continuation.
    ///
    /// - `player`: the traverser (0=OOP, 1=IP)
    /// - `pot`: pot size at the boundary (in half-BB chips)
    /// - `remaining_stack`: chips behind at the boundary
    /// - `opponent_reach`: opponent's reach probabilities (per private hand, game ordering)
    /// - `num_hands`: number of private hands for `player`
    /// - `continuation_index`: which of the K continuation strategies to evaluate
    ///   (0 = unbiased/blueprint, 1..K-1 = biased variants)
    ///
    /// Returns `Vec<f32>` with one CFV per private hand, in pot-normalised units
    /// (1.0 = win one half-pot).
    fn compute_cfvs(
        &self,
        player: usize,
        pot: i32,
        remaining_stack: f64,
        opponent_reach: &[f32],
        num_hands: usize,
        continuation_index: usize,
    ) -> Vec<f32>;

    /// Amortized: returns CFVs for both players in one pass.
    /// Default impl calls compute_cfvs twice (backwards compat).
    /// Implementations with internal amortization should override.
    fn compute_cfvs_both(
        &self,
        pot: i32,
        remaining_stack: f64,
        oop_reach: &[f32],
        ip_reach: &[f32],
        num_oop: usize,
        num_ip: usize,
        continuation_index: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let oop = self.compute_cfvs(0, pot, remaining_stack, ip_reach, num_oop, continuation_index);
        let ip = self.compute_cfvs(1, pot, remaining_stack, oop_reach, num_ip, continuation_index);
        (oop, ip)
    }
}

// ---------------------------------------------------------------------------
// StrengthItem
// ---------------------------------------------------------------------------

/// Hand-strength entry: pairs a strength rank with a hand index.
///
/// Used in sorted order (ascending strength) to evaluate showdown equity.
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct StrengthItem {
    pub strength: u16,
    pub index: u16,
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

    // -- depth boundary data --
    /// Per-boundary leaf CFVs, indexed by boundary ordinal.
    /// Each inner `Vec<f32>` has one f32 per private hand for the evaluation
    /// player. Set externally before solving via [`set_boundary_cfvs`].
    pub(crate) boundary_cfvs: Vec<std::sync::Mutex<Vec<f32>>>,
    /// Maps node arena index to boundary ordinal. `u32::MAX` means not a
    /// boundary. Built during tree construction.
    pub(crate) node_to_boundary: Vec<u32>,
    /// Per-boundary opponent reach probabilities, captured during solve_step.
    /// Layout: `boundary_reach[ordinal * 2 + player]` → `Vec<f32>` with one
    /// entry per private hand of that player. Updated each solve iteration.
    /// Uses `Mutex` for interior mutability (written during `evaluate` which takes `&self`).
    pub boundary_reach: Vec<std::sync::Mutex<Vec<f32>>>,
    /// Optional callback for lazy boundary CFV computation.
    /// When set, boundaries with empty CFVs call this instead of returning zero.
    pub boundary_evaluator: Option<Arc<dyn BoundaryEvaluator>>,

    // -- multi-continuation boundary data --
    /// Number of continuation strategies per boundary. 1 = legacy, 4 = multi-valued.
    pub(crate) num_continuations: usize,
    /// Per-boundary continuation regrets for opponent's choice among K continuations.
    /// One `Mutex<Vec<f32>>` of length K per boundary ordinal.
    pub(crate) boundary_cont_regrets: Vec<std::sync::Mutex<Vec<f32>>>,
    /// Per-boundary continuation cumulative strategy sums for averaging.
    /// One `Mutex<Vec<f32>>` of length K per boundary ordinal.
    pub(crate) boundary_cont_strategy: Vec<std::sync::Mutex<Vec<f32>>>,
    /// DCFR discount parameters for boundary continuation regrets/strategy.
    /// Stored as f32 bit patterns in AtomicU32 for interior mutability.
    pub(crate) boundary_discount_alpha: std::sync::atomic::AtomicU32,
    pub(crate) boundary_discount_beta: std::sync::atomic::AtomicU32,
    pub(crate) boundary_discount_gamma: std::sync::atomic::AtomicU32,

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

impl PostFlopGame {
    /// Returns the number of nodes in the tree arena.
    pub fn num_nodes(&self) -> usize {
        self.node_arena.len()
    }

    /// Locks and returns a reference to the node at arena `index`.
    pub fn node_at(&self, index: usize) -> crate::mutex_like::MutexGuardLike<'_, PostFlopNode> {
        self.node_arena[index].lock()
    }

    /// Returns the same-hand index mapping for `player`.
    pub fn same_hand_index(&self, player: usize) -> &[u16] {
        &self.same_hand_index[player]
    }

    /// Returns the hand strength ordering for all boards.
    pub fn hand_strength(&self) -> &Vec<[Vec<StrengthItem>; 2]> {
        &self.hand_strength
    }

    /// Returns valid hand indices for the flop board.
    pub fn valid_indices_flop(&self) -> &[Vec<u16>; 2] {
        &self.valid_indices_flop
    }

    /// Returns valid hand indices indexed by turn card.
    pub fn valid_indices_turn(&self) -> &Vec<[Vec<u16>; 2]> {
        &self.valid_indices_turn
    }

    /// Returns valid hand indices indexed by card_pair_to_index(turn, river).
    pub fn valid_indices_river(&self) -> &Vec<[Vec<u16>; 2]> {
        &self.valid_indices_river
    }

    /// Returns the number of valid card combinations.
    pub fn num_combinations_f64(&self) -> f64 {
        self.num_combinations
    }
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

    #[test]
    fn storage_buffers_returns_all_four() {
        let game = PostFlopGame::default();
        let (s1, s2, s_ip, s_ch) = game.storage_buffers();
        assert!(s1.is_empty());
        assert!(s2.is_empty());
        assert!(s_ip.is_empty());
        assert!(s_ch.is_empty());
    }

    fn make_river_game() -> PostFlopGame {
        use crate::bet_size::BetSizeOptions;
        use crate::card::{card_from_str, flop_from_str};

        let oop_range: crate::range::Range = "AA,KK".parse().unwrap();
        let ip_range: crate::range::Range = "QQ,JJ".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        PostFlopGame::with_config(card_config, tree).unwrap()
    }

    #[test]
    fn game_accessor_num_nodes() {
        let game = make_river_game();
        assert!(game.num_nodes() > 0);
    }

    #[test]
    fn game_accessor_node_at_and_index() {
        let game = make_river_game();
        let root = game.node_at(0);
        assert_eq!(game.node_index(&root), 0);
    }

    #[test]
    fn game_accessor_same_hand_index() {
        let game = make_river_game();
        let shi = game.same_hand_index(0);
        assert!(!shi.is_empty());
        let shi_ip = game.same_hand_index(1);
        assert!(!shi_ip.is_empty());
    }

    #[test]
    fn game_accessor_hand_strength() {
        let game = make_river_game();
        let hs = game.hand_strength();
        assert!(!hs.is_empty());
    }

    #[test]
    fn game_accessor_valid_indices_river() {
        let game = make_river_game();
        // River game has river indices populated
        let river = game.valid_indices_river();
        assert!(!river.is_empty());
    }

    #[test]
    fn game_accessor_valid_indices_return_types() {
        let game = make_river_game();
        // Just verify accessors return without panic
        let _flop = game.valid_indices_flop();
        let _turn = game.valid_indices_turn();
        let _river = game.valid_indices_river();
    }

    #[test]
    fn game_accessor_num_combinations_f64() {
        let game = make_river_game();
        assert!(game.num_combinations_f64() > 0.0);
    }

    #[test]
    fn strength_item_public_fields() {
        let item = StrengthItem {
            strength: 42,
            index: 7,
        };
        assert_eq!(item.strength, 42);
        assert_eq!(item.index, 7);
    }

    #[test]
    fn boundary_evaluator_default_compute_cfvs_both_delegates_to_single_side() {
        // Mock evaluator that returns [player as f32; num_hands]
        struct MockEval;
        impl BoundaryEvaluator for MockEval {
            fn compute_cfvs(
                &self, player: usize, _pot: i32, _rs: f64,
                _opp: &[f32], num_hands: usize, _ci: usize,
            ) -> Vec<f32> {
                vec![player as f32; num_hands]
            }
        }
        let e = MockEval;
        let (oop, ip) = e.compute_cfvs_both(100, 50.0, &vec![0.5; 5], &vec![0.3; 7], 5, 7, 0);
        assert_eq!(oop, vec![0.0; 5]);
        assert_eq!(ip, vec![1.0; 7]);
    }
}
