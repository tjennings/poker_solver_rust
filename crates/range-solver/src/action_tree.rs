use std::fmt;

use crate::bet_size::*;
use crate::card::*;
use crate::mutex_like::*;

// ---------------------------------------------------------------------------
// Player constants
// ---------------------------------------------------------------------------

pub(crate) const PLAYER_OOP: u8 = 0;
pub(crate) const PLAYER_IP: u8 = 1;
pub(crate) const PLAYER_CHANCE: u8 = 2; // only used with `PLAYER_CHANCE_FLAG`
pub(crate) const PLAYER_MASK: u8 = 3;
pub(crate) const PLAYER_CHANCE_FLAG: u8 = 4; // chance_player = PLAYER_CHANCE_FLAG | prev_player
pub(crate) const PLAYER_TERMINAL_FLAG: u8 = 8;
pub(crate) const PLAYER_FOLD_FLAG: u8 = 24; // TERMINAL_FLAG(8) | 16

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// Available actions of the postflop game.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Action {
    /// Default / sentinel value.
    #[default]
    None,

    /// Fold action.
    Fold,

    /// Check action.
    Check,

    /// Call action.
    Call,

    /// Bet action with a specified amount.
    Bet(i32),

    /// Raise action with a specified amount.
    Raise(i32),

    /// All-in action with a specified amount.
    AllIn(i32),

    /// Chance action with a card ID, i.e., the dealing of a turn or river card.
    Chance(Card),
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::None => write!(f, "None"),
            Action::Fold => write!(f, "Fold"),
            Action::Check => write!(f, "Check"),
            Action::Call => write!(f, "Call"),
            Action::Bet(amount) => write!(f, "Bet({amount})"),
            Action::Raise(amount) => write!(f, "Raise({amount})"),
            Action::AllIn(amount) => write!(f, "AllIn({amount})"),
            Action::Chance(card) => {
                if let Ok(s) = card_to_string(*card) {
                    write!(f, "Chance({s})")
                } else {
                    write!(f, "Chance({card})")
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BoardState
// ---------------------------------------------------------------------------

/// An enum representing the board state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum BoardState {
    #[default]
    Flop = 0,
    Turn = 1,
    River = 2,
}

impl fmt::Display for BoardState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoardState::Flop => write!(f, "Flop"),
            BoardState::Turn => write!(f, "Turn"),
            BoardState::River => write!(f, "River"),
        }
    }
}

// ---------------------------------------------------------------------------
// TreeConfig
// ---------------------------------------------------------------------------

/// A struct containing the game tree configuration.
///
/// # Examples
/// ```
/// use range_solver::action_tree::*;
/// use range_solver::bet_size::*;
///
/// let bet_sizes = BetSizeOptions::try_from(("60%, e, a", "2.5x")).unwrap();
/// let donk_sizes = DonkSizeOptions::try_from("50%").unwrap();
///
/// let tree_config = TreeConfig {
///     initial_state: BoardState::Turn,
///     starting_pot: 200,
///     effective_stack: 900,
///     rake_rate: 0.05,
///     rake_cap: 30.0,
///     flop_bet_sizes: Default::default(),
///     turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     turn_donk_sizes: None,
///     river_donk_sizes: Some(donk_sizes),
///     add_allin_threshold: 1.5,
///     force_allin_threshold: 0.15,
///     merging_threshold: 0.1,
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub struct TreeConfig {
    /// Initial state of the game tree (flop, turn, or river).
    pub initial_state: BoardState,

    /// Starting pot size. Must be greater than `0`.
    pub starting_pot: i32,

    /// Initial effective stack. Must be greater than `0`.
    pub effective_stack: i32,

    /// Rake rate. Must be between `0.0` and `1.0`, inclusive.
    pub rake_rate: f64,

    /// Rake cap. Must be non-negative.
    pub rake_cap: f64,

    /// Bet size options of each player for the flop.
    pub flop_bet_sizes: [BetSizeOptions; 2],

    /// Bet size options of each player for the turn.
    pub turn_bet_sizes: [BetSizeOptions; 2],

    /// Bet size options of each player for the river.
    pub river_bet_sizes: [BetSizeOptions; 2],

    /// Donk size options for the turn (set `None` to use default sizes).
    pub turn_donk_sizes: Option<DonkSizeOptions>,

    /// Donk size options for the river (set `None` to use default sizes).
    pub river_donk_sizes: Option<DonkSizeOptions>,

    /// Add all-in action if the ratio of maximum bet size to the pot is below or equal to this
    /// value (set `0.0` to disable).
    pub add_allin_threshold: f64,

    /// Force all-in action if the SPR (stack/pot) after the opponent's call is below or equal to
    /// this value (set `0.0` to disable).
    ///
    /// Personal recommendation: between `0.1` and `0.2`
    pub force_allin_threshold: f64,

    /// Merge bet actions if there are bet actions with "close" values (set `0.0` to disable).
    ///
    /// Algorithm: The same as PioSOLVER. That is, select the highest bet size (= X% of the pot)
    /// and remove all bet actions with a value (= Y% of the pot) satisfying the following
    /// inequality:
    ///   (100 + X) / (100 + Y) < 1.0 + threshold.
    /// Continue this process with the next highest bet size.
    ///
    /// Personal recommendation: around `0.1`
    pub merging_threshold: f64,
}

// ---------------------------------------------------------------------------
// ActionTreeNode (crate-internal)
// ---------------------------------------------------------------------------

/// An internal node of the action tree.
#[derive(Default)]
pub(crate) struct ActionTreeNode {
    pub(crate) player: u8,
    pub(crate) board_state: BoardState,
    pub(crate) amount: i32,
    pub(crate) actions: Vec<Action>,
    pub(crate) children: Vec<MutexLike<ActionTreeNode>>,
}

impl ActionTreeNode {
    /// Returns `true` if this node is a terminal node (fold or showdown).
    #[inline]
    pub(crate) fn is_terminal(&self) -> bool {
        self.player & PLAYER_TERMINAL_FLAG != 0
    }

    /// Returns `true` if this node is a chance node (card deal).
    #[inline]
    pub(crate) fn is_chance(&self) -> bool {
        self.player & PLAYER_CHANCE_FLAG != 0
    }
}

// ---------------------------------------------------------------------------
// ActionTree
// ---------------------------------------------------------------------------

/// A struct representing an abstract game tree.
///
/// An [`ActionTree`] does not distinguish between possible chance events (i.e., the dealing of
/// turn and river cards) and treats them as the same action.
#[derive(Default)]
pub struct ActionTree {
    pub(crate) config: TreeConfig,
    pub(crate) added_lines: Vec<Vec<Action>>,
    pub(crate) removed_lines: Vec<Vec<Action>>,
    pub(crate) root: Box<MutexLike<ActionTreeNode>>,
    history: Vec<Action>,
}

impl ActionTree {
    /// Obtains the configuration of the game tree.
    #[inline]
    pub fn config(&self) -> &TreeConfig {
        &self.config
    }

    /// Obtains the list of added lines.
    #[inline]
    pub fn added_lines(&self) -> &[Vec<Action>] {
        &self.added_lines
    }

    /// Obtains the list of removed lines.
    #[inline]
    pub fn removed_lines(&self) -> &[Vec<Action>] {
        &self.removed_lines
    }

    /// Obtains the current action history.
    #[inline]
    pub fn history(&self) -> &[Action] {
        &self.history
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Action Display --

    #[test]
    fn action_display_simple() {
        assert_eq!(Action::None.to_string(), "None");
        assert_eq!(Action::Fold.to_string(), "Fold");
        assert_eq!(Action::Check.to_string(), "Check");
        assert_eq!(Action::Call.to_string(), "Call");
    }

    #[test]
    fn action_display_with_amount() {
        assert_eq!(Action::Bet(100).to_string(), "Bet(100)");
        assert_eq!(Action::Raise(250).to_string(), "Raise(250)");
        assert_eq!(Action::AllIn(1000).to_string(), "AllIn(1000)");
    }

    #[test]
    fn action_display_chance() {
        // Card 0 = 2c
        assert_eq!(Action::Chance(0).to_string(), "Chance(2c)");
        // Card 51 = As
        assert_eq!(Action::Chance(51).to_string(), "Chance(As)");
    }

    #[test]
    fn action_display_chance_invalid_card() {
        // Card 255 is not valid => falls back to numeric
        assert_eq!(Action::Chance(255).to_string(), "Chance(255)");
    }

    // -- Action ordering --

    #[test]
    fn action_ordering() {
        // Derived Ord follows variant declaration order, then inner value
        assert!(Action::None < Action::Fold);
        assert!(Action::Fold < Action::Check);
        assert!(Action::Check < Action::Call);
        assert!(Action::Call < Action::Bet(1));
        assert!(Action::Bet(1) < Action::Bet(2));
        assert!(Action::Bet(100) < Action::Raise(1));
        assert!(Action::Raise(100) < Action::AllIn(1));
        assert!(Action::AllIn(100) < Action::Chance(0));
    }

    // -- Action equality and hashing --

    #[test]
    fn action_eq_and_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(Action::Bet(100));
        set.insert(Action::Bet(100));
        set.insert(Action::Raise(100));
        assert_eq!(set.len(), 2);
    }

    // -- Action default --

    #[test]
    fn action_default_is_none() {
        assert_eq!(Action::default(), Action::None);
    }

    // -- BoardState --

    #[test]
    fn board_state_display() {
        assert_eq!(BoardState::Flop.to_string(), "Flop");
        assert_eq!(BoardState::Turn.to_string(), "Turn");
        assert_eq!(BoardState::River.to_string(), "River");
    }

    #[test]
    fn board_state_ordering() {
        assert!(BoardState::Flop < BoardState::Turn);
        assert!(BoardState::Turn < BoardState::River);
    }

    #[test]
    fn board_state_default_is_flop() {
        assert_eq!(BoardState::default(), BoardState::Flop);
    }

    #[test]
    fn board_state_repr() {
        assert_eq!(BoardState::Flop as u8, 0);
        assert_eq!(BoardState::Turn as u8, 1);
        assert_eq!(BoardState::River as u8, 2);
    }

    // -- Player constants --

    #[test]
    fn player_constants() {
        assert_eq!(PLAYER_OOP, 0);
        assert_eq!(PLAYER_IP, 1);
        assert_eq!(PLAYER_CHANCE, 2);
        assert_eq!(PLAYER_MASK, 3);
        assert_eq!(PLAYER_CHANCE_FLAG, 4);
        assert_eq!(PLAYER_TERMINAL_FLAG, 8);
        assert_eq!(PLAYER_FOLD_FLAG, 24);
        // FOLD_FLAG includes TERMINAL_FLAG
        assert_ne!(PLAYER_FOLD_FLAG & PLAYER_TERMINAL_FLAG, 0);
    }

    // -- TreeConfig construction --

    #[test]
    fn tree_config_default() {
        let config = TreeConfig::default();
        assert_eq!(config.initial_state, BoardState::Flop);
        assert_eq!(config.starting_pot, 0);
        assert_eq!(config.effective_stack, 0);
        assert_eq!(config.rake_rate, 0.0);
        assert_eq!(config.rake_cap, 0.0);
        assert_eq!(config.add_allin_threshold, 0.0);
        assert_eq!(config.force_allin_threshold, 0.0);
        assert_eq!(config.merging_threshold, 0.0);
        assert!(config.turn_donk_sizes.is_none());
        assert!(config.river_donk_sizes.is_none());
    }

    #[test]
    fn tree_config_custom() {
        let bet_sizes = BetSizeOptions::try_from(("60%, e, a", "2.5x")).unwrap();
        let donk_sizes = DonkSizeOptions::try_from("50%").unwrap();

        let config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 200,
            effective_stack: 900,
            rake_rate: 0.05,
            rake_cap: 30.0,
            flop_bet_sizes: Default::default(),
            turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            turn_donk_sizes: None,
            river_donk_sizes: Some(donk_sizes),
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.1,
        };

        assert_eq!(config.initial_state, BoardState::Turn);
        assert_eq!(config.starting_pot, 200);
        assert_eq!(config.effective_stack, 900);
        assert!((config.rake_rate - 0.05).abs() < f64::EPSILON);
        assert!((config.rake_cap - 30.0).abs() < f64::EPSILON);
        assert!(config.river_donk_sizes.is_some());
    }

    // -- ActionTreeNode --

    #[test]
    fn action_tree_node_default() {
        let node = ActionTreeNode::default();
        assert_eq!(node.player, 0);
        assert_eq!(node.board_state, BoardState::Flop);
        assert_eq!(node.amount, 0);
        assert!(node.actions.is_empty());
        assert!(node.children.is_empty());
    }

    #[test]
    fn action_tree_node_terminal_flag() {
        let mut node = ActionTreeNode::default();
        assert!(!node.is_terminal());
        assert!(!node.is_chance());

        node.player = PLAYER_TERMINAL_FLAG;
        assert!(node.is_terminal());
        assert!(!node.is_chance());
    }

    #[test]
    fn action_tree_node_chance_flag() {
        let mut node = ActionTreeNode::default();
        node.player = PLAYER_CHANCE_FLAG | PLAYER_OOP;
        assert!(!node.is_terminal());
        assert!(node.is_chance());
    }

    #[test]
    fn action_tree_node_fold_is_terminal() {
        let mut node = ActionTreeNode::default();
        node.player = PLAYER_FOLD_FLAG | PLAYER_OOP;
        // FOLD_FLAG includes TERMINAL_FLAG, so is_terminal should be true
        assert!(node.is_terminal());
    }

    // -- ActionTree --

    #[test]
    fn action_tree_default_accessors() {
        let tree = ActionTree::default();
        assert_eq!(tree.config().starting_pot, 0);
        assert!(tree.added_lines().is_empty());
        assert!(tree.removed_lines().is_empty());
        assert!(tree.history().is_empty());
    }
}
