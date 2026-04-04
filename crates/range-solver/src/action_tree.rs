use std::fmt;

use crate::bet_size::*;
use crate::card::*;
use crate::mutex_like::*;

// ---------------------------------------------------------------------------
// Player constants
// ---------------------------------------------------------------------------

pub const PLAYER_OOP: u8 = 0;
pub const PLAYER_IP: u8 = 1;
pub const PLAYER_CHANCE: u8 = 2; // only used with `PLAYER_CHANCE_FLAG`
pub const PLAYER_MASK: u8 = 3;
pub const PLAYER_CHANCE_FLAG: u8 = 4; // chance_player = PLAYER_CHANCE_FLAG | prev_player
pub const PLAYER_TERMINAL_FLAG: u8 = 8;
pub const PLAYER_FOLD_FLAG: u8 = 24; // TERMINAL_FLAG(8) | 16
pub const PLAYER_DEPTH_BOUNDARY_FLAG: u8 = 40; // TERMINAL_FLAG(8) | 32

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
///     depth_limit: None,
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

    /// Optional depth limit: maximum number of street transitions allowed.
    ///
    /// Street transitions are counted each time the tree would deal a new card
    /// (flop->turn = 1, turn->river = 2). When the limit is reached, a depth
    /// boundary terminal is emitted instead of a chance node. The boundary
    /// node's counterfactual values must be supplied externally before solving
    /// (e.g., from a neural network).
    ///
    /// `None` means no limit (default).
    pub depth_limit: Option<u8>,
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
// BuildTreeInfo (private helper)
// ---------------------------------------------------------------------------

struct BuildTreeInfo {
    prev_action: Action,
    num_bets: i32,
    allin_flag: bool,
    oop_call_flag: bool,
    stack: [i32; 2],
    prev_amount: i32,
    /// Number of street transitions that have occurred so far.
    street_transitions: u8,
}

impl BuildTreeInfo {
    #[inline]
    fn new(stack: i32) -> Self {
        Self {
            prev_action: Action::None,
            num_bets: 0,
            allin_flag: false,
            oop_call_flag: false,
            stack: [stack, stack],
            prev_amount: 0,
            street_transitions: 0,
        }
    }

    #[inline]
    fn create_next(&self, player: u8, action: Action) -> Self {
        let mut num_bets = self.num_bets;
        let mut allin_flag = self.allin_flag;
        let mut oop_call_flag = self.oop_call_flag;
        let mut stack = self.stack;
        let mut prev_amount = self.prev_amount;
        let mut street_transitions = self.street_transitions;

        match action {
            Action::Check => {
                oop_call_flag = false;
            }
            Action::Call => {
                num_bets = 0;
                oop_call_flag = player == PLAYER_OOP;
                stack[player as usize] = stack[player as usize ^ 1];
                prev_amount = 0;
            }
            Action::Bet(amount) | Action::Raise(amount) | Action::AllIn(amount) => {
                let to_call = stack[player as usize] - stack[player as usize ^ 1];
                num_bets += 1;
                allin_flag = matches!(action, Action::AllIn(_));
                stack[player as usize] -= amount - prev_amount + to_call;
                prev_amount = amount;
            }
            Action::Chance(_) => {
                street_transitions += 1;
            }
            _ => {}
        }

        BuildTreeInfo {
            prev_action: action,
            num_bets,
            allin_flag,
            oop_call_flag,
            stack,
            prev_amount,
            street_transitions,
        }
    }
}

// ---------------------------------------------------------------------------
// EjectedActionTree
// ---------------------------------------------------------------------------

pub(crate) type EjectedActionTree = (
    TreeConfig,
    Vec<Vec<Action>>,
    Vec<Vec<Action>>,
    Box<MutexLike<ActionTreeNode>>,
);

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
    /// Creates a new [`ActionTree`] with the specified configuration.
    #[inline]
    pub fn new(config: TreeConfig) -> Result<Self, String> {
        Self::check_config(&config)?;
        let mut ret = Self {
            config,
            ..Default::default()
        };
        ret.build_tree();
        Ok(ret)
    }

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

    /// Returns a list of all terminal nodes that should not be.
    #[inline]
    pub fn invalid_terminals(&self) -> Vec<Vec<Action>> {
        let mut ret = Vec::new();
        let mut line = Vec::new();
        Self::invalid_terminals_recursive(&self.root.lock(), &mut ret, &mut line);
        ret
    }

    /// Adds a given line to the action tree.
    ///
    /// - `line` except the last action must exist in the current tree.
    /// - The last action of the `line` must not exist in the current tree.
    /// - Except for the case that the `line` is in the removed lines, the last action of the
    ///   `line` must be a bet action (including raise and all-in action).
    /// - Chance actions (i.e., dealing turn and river cards) must be omitted from the `line`.
    #[inline]
    pub fn add_line(&mut self, line: &[Action]) -> Result<(), String> {
        let removed_index = self.removed_lines.iter().position(|x| x == line);
        let is_replaced = self.add_line_recursive(
            &mut self.root.lock(),
            line,
            removed_index.is_some(),
            BuildTreeInfo::new(self.config.effective_stack),
        )?;
        if let Some(index) = removed_index {
            self.removed_lines.remove(index);
        } else {
            let mut line = line.to_vec();
            if is_replaced {
                if let Some(&Action::Bet(amount) | &Action::Raise(amount)) = line.last() {
                    *line.last_mut().unwrap() = Action::AllIn(amount);
                }
            }
            self.added_lines.push(line);
        }
        Ok(())
    }

    /// Removes a given line from the action tree.
    ///
    /// - `line` must exist in the current tree.
    /// - Chance actions (i.e., dealing turn and river cards) must be omitted from the `line`.
    /// - If the current node is removed by this method, the current node is moved to the nearest
    ///   ancestor node that is not removed.
    #[inline]
    pub fn remove_line(&mut self, line: &[Action]) -> Result<(), String> {
        Self::remove_line_recursive(&mut self.root.lock(), line)?;
        let was_added = self.added_lines.iter().any(|l| l == line);
        self.added_lines.retain(|l| !l.starts_with(line));
        self.removed_lines.retain(|l| !l.starts_with(line));
        if !was_added {
            self.removed_lines.push(line.to_vec());
        }
        if self.history.starts_with(line) {
            self.history.truncate(line.len() - 1);
        }
        Ok(())
    }

    /// Moves back to the root node.
    #[inline]
    pub fn back_to_root(&mut self) {
        self.history.clear();
    }

    /// Obtains the current action history.
    #[inline]
    pub fn history(&self) -> &[Action] {
        &self.history
    }

    /// Applies the given action history from the root node.
    #[inline]
    pub fn apply_history(&mut self, history: &[Action]) -> Result<(), String> {
        self.back_to_root();
        for &action in history {
            self.play(action)?;
        }
        Ok(())
    }

    /// Returns whether the current node is a terminal node.
    #[inline]
    pub fn is_terminal_node(&self) -> bool {
        self.current_node_skip_chance().is_terminal()
    }

    /// Returns whether the current node is a chance node.
    #[inline]
    pub fn is_chance_node(&self) -> bool {
        self.current_node().is_chance() && !self.is_terminal_node()
    }

    /// Returns the available actions for the current node.
    ///
    /// If the current node is a chance node, returns possible actions after the chance event.
    #[inline]
    pub fn available_actions(&self) -> &[Action] {
        &self.current_node_skip_chance().actions
    }

    /// Plays the given action. Returns `Ok(())` if the action is valid.
    ///
    /// The `action` must be one of the possible actions at the current node.
    /// If the current node is a chance node, the chance action is automatically played before
    /// playing the given action.
    #[inline]
    pub fn play(&mut self, action: Action) -> Result<(), String> {
        let node = self.current_node_skip_chance();
        if !node.actions.contains(&action) {
            return Err(format!("Action `{action:?}` is not available"));
        }

        self.history.push(action);
        Ok(())
    }

    /// Undoes the last action. Returns `Ok(())` if the action is successfully undone.
    #[inline]
    pub fn undo(&mut self) -> Result<(), String> {
        if self.history.is_empty() {
            return Err("No action to undo".to_string());
        }

        self.history.pop();
        Ok(())
    }

    /// Adds a given action to the current node.
    ///
    /// Internally, this method calls [`add_line`] with the current action history and the given
    /// action. See [`add_line`] for details.
    ///
    /// [`add_line`]: #method.add_line
    #[inline]
    pub fn add_action(&mut self, action: Action) -> Result<(), String> {
        let mut action_line = self.history.clone();
        action_line.push(action);
        self.add_line(&action_line)
    }

    /// Removes a given action from the current node.
    ///
    /// Internally, this method calls [`remove_line`] with the current action history and the
    /// given action. See [`remove_line`] for details.
    ///
    /// [`remove_line`]: #method.remove_line
    #[inline]
    pub fn remove_action(&mut self, action: Action) -> Result<(), String> {
        let mut action_line = self.history.clone();
        action_line.push(action);
        self.remove_line(&action_line)
    }

    /// Removes the current node.
    ///
    /// Internally, this method calls [`remove_line`] with the current action history. See
    /// [`remove_line`] for details.
    ///
    /// [`remove_line`]: #method.remove_line
    #[inline]
    pub fn remove_current_node(&mut self) -> Result<(), String> {
        let history = self.history.clone();
        self.remove_line(&history)
    }

    /// Returns the total bet amount of each player (OOP, IP).
    #[inline]
    pub fn total_bet_amount(&self) -> [i32; 2] {
        let info = BuildTreeInfo::new(self.config.effective_stack);
        self.total_bet_amount_recursive(&self.root.lock(), &self.history, info)
    }

    /// Ejects the fields.
    #[inline]
    pub(crate) fn eject(self) -> EjectedActionTree {
        (self.config, self.added_lines, self.removed_lines, self.root)
    }

    /// Returns a reference to the current node.
    #[inline]
    fn current_node(&self) -> &ActionTreeNode {
        // SAFETY: We hold `&self` so no concurrent mutation occurs. The raw pointer traversal
        // is needed because `MutexGuardLike` temporaries would be dropped before the final
        // dereference if we used safe code. The tree structure is invariant while `&self` is held.
        unsafe {
            let mut node = &*self.root.lock() as *const ActionTreeNode;
            for action in &self.history {
                while (*node).is_chance() {
                    node = &*(&(*node).children)[0].lock();
                }
                let index = (*node).actions.iter().position(|x| x == action).unwrap();
                node = &*(&(*node).children)[index].lock();
            }
            &*node
        }
    }

    /// Returns a reference to the current node, skipping past any leading chance nodes.
    #[inline]
    fn current_node_skip_chance(&self) -> &ActionTreeNode {
        // SAFETY: Same invariant as `current_node` — no concurrent mutation while `&self` held.
        unsafe {
            let mut node = self.current_node() as *const ActionTreeNode;
            while (*node).is_chance() {
                node = &*(&(*node).children)[0].lock();
            }
            &*node
        }
    }

    /// Checks the configuration for validity.
    #[inline]
    fn check_config(config: &TreeConfig) -> Result<(), String> {
        if config.starting_pot <= 0 {
            return Err(format!(
                "Starting pot must be positive: {}",
                config.starting_pot
            ));
        }

        if config.effective_stack <= 0 {
            return Err(format!(
                "Effective stack must be positive: {}",
                config.effective_stack
            ));
        }

        if config.rake_rate < 0.0 {
            return Err(format!(
                "Rake rate must be non-negative: {}",
                config.rake_rate
            ));
        }

        if config.rake_rate > 1.0 {
            return Err(format!(
                "Rake rate must be less than or equal to 1.0: {}",
                config.rake_rate
            ));
        }

        if config.rake_cap < 0.0 {
            return Err(format!(
                "Rake cap must be non-negative: {}",
                config.rake_cap
            ));
        }

        if config.add_allin_threshold < 0.0 {
            return Err(format!(
                "Add all-in threshold must be non-negative: {}",
                config.add_allin_threshold
            ));
        }

        if config.force_allin_threshold < 0.0 {
            return Err(format!(
                "Force all-in threshold must be non-negative: {}",
                config.force_allin_threshold
            ));
        }

        if config.merging_threshold < 0.0 {
            return Err(format!(
                "Merging threshold must be non-negative: {}",
                config.merging_threshold
            ));
        }

        Ok(())
    }

    /// Builds the action tree from scratch.
    #[inline]
    fn build_tree(&mut self) {
        let mut root = self.root.lock();
        *root = ActionTreeNode::default();
        root.board_state = self.config.initial_state;
        self.build_tree_recursive(&mut root, BuildTreeInfo::new(self.config.effective_stack));
    }

    /// Recursively builds the action tree.
    fn build_tree_recursive(&self, node: &mut ActionTreeNode, info: BuildTreeInfo) {
        if node.is_terminal() {
            // do nothing
        } else if node.is_chance() {
            // Check if this street transition would exceed the depth limit.
            // info.street_transitions counts transitions that have already happened.
            // This chance node represents a new transition, so block it if
            // street_transitions >= depth_limit.
            let depth_limit_hit = self.config.depth_limit.is_some_and(|limit| {
                info.street_transitions >= limit
            });

            if depth_limit_hit {
                // Convert this chance node to a depth boundary terminal.
                node.player = PLAYER_DEPTH_BOUNDARY_FLAG;
                node.actions.clear();
                node.children.clear();
                return;
            }

            let next_state = match node.board_state {
                BoardState::Flop => BoardState::Turn,
                BoardState::Turn => BoardState::River,
                BoardState::River => unreachable!(),
            };

            // For the all-in + flop case, the child is another chance node. Check
            // whether the *next* transition (turn->river) would also exceed the depth
            // limit. After processing this chance, street_transitions will be
            // info.street_transitions + 1.
            let second_transition_blocked = self.config.depth_limit.is_some_and(|limit| {
                info.street_transitions + 1 >= limit
            });

            let next_player = match (info.allin_flag, node.board_state) {
                (false, _) => PLAYER_OOP,
                (true, BoardState::Flop) if second_transition_blocked => {
                    PLAYER_DEPTH_BOUNDARY_FLAG
                }
                (true, BoardState::Flop) => PLAYER_CHANCE_FLAG | PLAYER_CHANCE,
                (true, _) => PLAYER_TERMINAL_FLAG,
            };

            node.actions.push(Action::Chance(0));
            node.children.push(MutexLike::new(ActionTreeNode {
                player: next_player,
                board_state: next_state,
                amount: node.amount,
                ..Default::default()
            }));

            self.build_tree_recursive(
                &mut node.children[0].lock(),
                info.create_next(0, Action::Chance(0)),
            );
        } else {
            self.push_actions(node, &info);
            for (action, child) in node.actions.iter().zip(node.children.iter()) {
                self.build_tree_recursive(
                    &mut child.lock(),
                    info.create_next(node.player, *action),
                );
            }
        }
    }

    /// Pushes all possible actions to the given node.
    fn push_actions(&self, node: &mut ActionTreeNode, info: &BuildTreeInfo) {
        let player = node.player;
        let opponent = node.player ^ 1;

        let player_stack = info.stack[player as usize];
        let opponent_stack = info.stack[opponent as usize];
        let prev_amount = info.prev_amount;
        let to_call = player_stack - opponent_stack;

        let pot = self.config.starting_pot + 2 * (node.amount + to_call);
        let max_amount = opponent_stack + prev_amount;
        let min_amount = (prev_amount + to_call).clamp(1, max_amount);

        let spr_after_call = opponent_stack as f64 / pot as f64;
        let compute_geometric = |num_streets: i32, max_ratio: f64| {
            let ratio =
                ((2.0 * spr_after_call + 1.0).powf(1.0 / num_streets as f64) - 1.0) / 2.0;
            (pot as f64 * ratio.min(max_ratio)).round() as i32
        };

        let (bet_options, donk_options, num_remaining_streets) = match node.board_state {
            BoardState::Flop => (&self.config.flop_bet_sizes, &None, 3),
            BoardState::Turn => (&self.config.turn_bet_sizes, &self.config.turn_donk_sizes, 2),
            BoardState::River => (
                &self.config.river_bet_sizes,
                &self.config.river_donk_sizes,
                1,
            ),
        };

        let mut actions = Vec::new();

        if donk_options.is_some()
            && matches!(info.prev_action, Action::Chance(_))
            && info.oop_call_flag
        {
            // check
            actions.push(Action::Check);

            // donk bet
            for &donk_size in &donk_options.as_ref().unwrap().donk {
                match donk_size {
                    BetSize::PotRelative(ratio) => {
                        let amount = (pot as f64 * ratio).round() as i32;
                        actions.push(Action::Bet(amount));
                    }
                    BetSize::PrevBetRelative(_) => panic!("Unexpected `PrevBetRelative`"),
                    BetSize::Additive(adder, _) => actions.push(Action::Bet(adder)),
                    BetSize::Geometric(num_streets, max_ratio) => {
                        let num_streets = match num_streets {
                            0 => num_remaining_streets,
                            _ => num_streets,
                        };
                        let amount = compute_geometric(num_streets, max_ratio);
                        actions.push(Action::Bet(amount));
                    }
                    BetSize::AllIn => actions.push(Action::AllIn(max_amount)),
                }
            }

            // all-in
            if max_amount <= (pot as f64 * self.config.add_allin_threshold).round() as i32 {
                actions.push(Action::AllIn(max_amount));
            }
        } else if matches!(
            info.prev_action,
            Action::None | Action::Check | Action::Chance(_)
        ) {
            // check
            actions.push(Action::Check);

            // bet
            for &bet_size in &bet_options[player as usize].bet {
                match bet_size {
                    BetSize::PotRelative(ratio) => {
                        let amount = (pot as f64 * ratio).round() as i32;
                        actions.push(Action::Bet(amount));
                    }
                    BetSize::PrevBetRelative(_) => panic!("Unexpected `PrevBetRelative`"),
                    BetSize::Additive(adder, _) => actions.push(Action::Bet(adder)),
                    BetSize::Geometric(num_streets, max_ratio) => {
                        let num_streets = match num_streets {
                            0 => num_remaining_streets,
                            _ => num_streets,
                        };
                        let amount = compute_geometric(num_streets, max_ratio);
                        actions.push(Action::Bet(amount));
                    }
                    BetSize::AllIn => actions.push(Action::AllIn(max_amount)),
                }
            }

            // all-in
            if max_amount <= (pot as f64 * self.config.add_allin_threshold).round() as i32 {
                actions.push(Action::AllIn(max_amount));
            }
        } else {
            // fold
            actions.push(Action::Fold);

            // call
            actions.push(Action::Call);

            if !info.allin_flag {
                // raise
                for &bet_size in &bet_options[player as usize].raise {
                    match bet_size {
                        BetSize::PotRelative(ratio) => {
                            let amount = prev_amount + (pot as f64 * ratio).round() as i32;
                            actions.push(Action::Raise(amount));
                        }
                        BetSize::PrevBetRelative(ratio) => {
                            let amount = (prev_amount as f64 * ratio).round() as i32;
                            actions.push(Action::Raise(amount));
                        }
                        BetSize::Additive(adder, raise_cap) => {
                            if raise_cap == 0 || info.num_bets <= raise_cap {
                                actions.push(Action::Raise(prev_amount + adder));
                            }
                        }
                        BetSize::Geometric(num_streets, max_ratio) => {
                            let num_streets = match num_streets {
                                0 => i32::max(num_remaining_streets - info.num_bets + 1, 1),
                                _ => i32::max(num_streets - info.num_bets + 1, 1),
                            };
                            let amount = compute_geometric(num_streets, max_ratio);
                            actions.push(Action::Raise(prev_amount + amount));
                        }
                        BetSize::AllIn => actions.push(Action::AllIn(max_amount)),
                    }
                }

                // all-in
                let allin_threshold = pot as f64 * self.config.add_allin_threshold;
                if max_amount <= prev_amount + allin_threshold.round() as i32 {
                    actions.push(Action::AllIn(max_amount));
                }
            }
        }

        let is_above_threshold = |amount: i32| {
            let new_amount_diff = amount - prev_amount;
            let new_pot = pot + 2 * new_amount_diff;
            let threshold = (new_pot as f64 * self.config.force_allin_threshold).round() as i32;
            max_amount <= amount + threshold
        };

        // clamp bet amounts
        for action in actions.iter_mut() {
            match *action {
                Action::Bet(amount) => {
                    let clamped = amount.clamp(min_amount, max_amount);
                    if is_above_threshold(clamped) {
                        *action = Action::AllIn(max_amount);
                    } else if clamped != amount {
                        *action = Action::Bet(clamped);
                    }
                }
                Action::Raise(amount) => {
                    let clamped = amount.clamp(min_amount, max_amount);
                    if is_above_threshold(clamped) {
                        *action = Action::AllIn(max_amount);
                    } else if clamped != amount {
                        *action = Action::Raise(clamped);
                    }
                }
                _ => {}
            }
        }

        // remove duplicates
        actions.sort_unstable();
        actions.dedup();

        // merge bet actions with close amounts
        actions = merge_bet_actions(actions, pot, prev_amount, self.config.merging_threshold);

        let depth_limit_reached = self.config.depth_limit.is_some_and(|limit| {
            info.street_transitions >= limit
        });

        let player_after_call = match node.board_state {
            BoardState::River => PLAYER_TERMINAL_FLAG,
            _ if depth_limit_reached => PLAYER_DEPTH_BOUNDARY_FLAG,
            _ => PLAYER_CHANCE_FLAG | player,
        };

        let player_after_check = match player {
            PLAYER_OOP => opponent,
            _ => player_after_call,
        };

        // push actions
        for action in actions {
            let mut amount = node.amount;
            let next_player = match action {
                Action::Fold => PLAYER_FOLD_FLAG | player,
                Action::Check => player_after_check,
                Action::Call => {
                    amount += to_call;
                    player_after_call
                }
                Action::Bet(_) | Action::Raise(_) | Action::AllIn(_) => {
                    amount += to_call;
                    opponent
                }
                _ => panic!("Unexpected action: {action:?}"),
            };

            node.actions.push(action);
            node.children.push(MutexLike::new(ActionTreeNode {
                player: next_player,
                board_state: node.board_state,
                amount,
                ..Default::default()
            }));
        }

        node.actions.shrink_to_fit();
        node.children.shrink_to_fit();
    }

    /// Recursive function to enumerate all invalid terminal nodes.
    fn invalid_terminals_recursive(
        node: &ActionTreeNode,
        result: &mut Vec<Vec<Action>>,
        line: &mut Vec<Action>,
    ) {
        if node.is_terminal() {
            // do nothing
        } else if node.children.is_empty() {
            result.push(line.clone());
        } else if node.is_chance() {
            Self::invalid_terminals_recursive(&node.children[0].lock(), result, line);
        } else {
            for (&action, child) in node.actions.iter().zip(node.children.iter()) {
                line.push(action);
                Self::invalid_terminals_recursive(&child.lock(), result, line);
                line.pop();
            }
        }
    }

    /// Recursive function to add a given line to the tree.
    fn add_line_recursive(
        &self,
        node: &mut ActionTreeNode,
        line: &[Action],
        was_removed: bool,
        info: BuildTreeInfo,
    ) -> Result<bool, String> {
        if line.is_empty() {
            return Err("Empty line".to_string());
        }

        if node.is_terminal() {
            return Err("Unexpected terminal node".to_string());
        }

        if node.is_chance() {
            return self.add_line_recursive(
                &mut node.children[0].lock(),
                line,
                was_removed,
                info.create_next(0, Action::Chance(0)),
            );
        }

        let action = line[0];
        let search_result = node.actions.binary_search(&action);

        let player = node.player;
        let opponent = node.player ^ 1;

        if line.len() > 1 {
            if search_result.is_err() {
                return Err(format!("Action does not exist: {action:?}"));
            }

            return self.add_line_recursive(
                &mut node.children[search_result.unwrap()].lock(),
                &line[1..],
                was_removed,
                info.create_next(player, action),
            );
        }

        if search_result.is_ok() {
            return Err(format!("Action already exists: {action:?}"));
        }

        let is_bet_action =
            matches!(action, Action::Bet(_) | Action::Raise(_) | Action::AllIn(_));
        if info.allin_flag && is_bet_action {
            return Err(format!("Bet action after all-in: {action:?}"));
        }

        let player_stack = info.stack[player as usize];
        let opponent_stack = info.stack[opponent as usize];
        let prev_amount = info.prev_amount;
        let to_call = player_stack - opponent_stack;

        let max_amount = opponent_stack + prev_amount;
        let min_amount = (prev_amount + to_call).clamp(1, max_amount);

        let mut is_replaced = false;
        let action = match action {
            Action::Bet(amount) | Action::Raise(amount) if amount == max_amount => {
                is_replaced = true;
                Action::AllIn(amount)
            }
            _ => action,
        };

        let is_valid_bet = match action {
            Action::Bet(amount) if amount >= min_amount && amount < max_amount => {
                matches!(
                    info.prev_action,
                    Action::None | Action::Check | Action::Chance(_)
                )
            }
            Action::Raise(amount) if amount >= min_amount && amount < max_amount => {
                matches!(info.prev_action, Action::Bet(_) | Action::Raise(_))
            }
            Action::AllIn(amount) => amount == max_amount,
            _ => false,
        };

        if !was_removed && !is_valid_bet {
            match action {
                Action::Bet(amount) | Action::Raise(amount) => {
                    return Err(format!(
                        "Invalid bet amount: {amount} (min: {min_amount}, max: {max_amount})"
                    ));
                }
                Action::AllIn(amount) => {
                    return Err(format!(
                        "Invalid all-in amount: {amount} (expected: {max_amount})"
                    ));
                }
                _ => {
                    return Err(format!("Invalid action: {action:?}"));
                }
            };
        }

        let depth_limit_reached = self.config.depth_limit.is_some_and(|limit| {
            info.street_transitions >= limit
        });

        let player_after_call = match node.board_state {
            BoardState::River => PLAYER_TERMINAL_FLAG,
            _ if depth_limit_reached => PLAYER_DEPTH_BOUNDARY_FLAG,
            _ => PLAYER_CHANCE_FLAG | player,
        };

        let player_after_check = match player {
            PLAYER_OOP => opponent,
            _ => player_after_call,
        };

        let mut amount = node.amount;
        let next_player = match action {
            Action::Fold => PLAYER_FOLD_FLAG | player,
            Action::Check => player_after_check,
            Action::Call => {
                amount += to_call;
                player_after_call
            }
            Action::Bet(_) | Action::Raise(_) | Action::AllIn(_) => {
                amount += to_call;
                opponent
            }
            _ => panic!("Unexpected action: {action:?}"),
        };

        let index = search_result.unwrap_err();
        node.actions.insert(index, action);
        node.children.insert(
            index,
            MutexLike::new(ActionTreeNode {
                player: next_player,
                board_state: node.board_state,
                amount,
                ..Default::default()
            }),
        );

        node.actions.shrink_to_fit();
        node.children.shrink_to_fit();

        self.build_tree_recursive(
            &mut node.children[index].lock(),
            info.create_next(player, action),
        );

        Ok(is_replaced)
    }

    /// Recursive function to remove a given line from the tree.
    fn remove_line_recursive(node: &mut ActionTreeNode, line: &[Action]) -> Result<(), String> {
        if line.is_empty() {
            return Err("Empty line".to_string());
        }

        if node.is_terminal() {
            return Err("Unexpected terminal node".to_string());
        }

        if node.is_chance() {
            return Self::remove_line_recursive(&mut node.children[0].lock(), line);
        }

        let action = line[0];
        let search_result = node.actions.binary_search(&action);
        if search_result.is_err() {
            return Err(format!("Action does not exist: {action:?}"));
        }

        if line.len() > 1 {
            return Self::remove_line_recursive(
                &mut node.children[search_result.unwrap()].lock(),
                &line[1..],
            );
        }

        let index = search_result.unwrap();
        node.actions.remove(index);
        node.children.remove(index);

        node.actions.shrink_to_fit();
        node.children.shrink_to_fit();

        Ok(())
    }

    /// Recursive function to compute total bet amount for each player.
    fn total_bet_amount_recursive(
        &self,
        node: &ActionTreeNode,
        line: &[Action],
        info: BuildTreeInfo,
    ) -> [i32; 2] {
        if line.is_empty() || node.is_terminal() {
            let stack = self.config.effective_stack;
            return [stack - info.stack[0], stack - info.stack[1]];
        }

        if node.is_chance() {
            return self.total_bet_amount_recursive(&node.children[0].lock(), line, info);
        }

        let action = line[0];
        let search_result = node.actions.binary_search(&action);
        if search_result.is_err() {
            panic!("Action does not exist: {action:?}");
        }

        let index = search_result.unwrap();
        let next_info = info.create_next(node.player, action);
        self.total_bet_amount_recursive(&node.children[index].lock(), &line[1..], next_info)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Returns the number of action nodes for [flop, turn, river].
///
/// `initial_state` determines the starting street offset so that nodes are
/// placed in the correct slot even when depth-limiting removes later streets.
pub(crate) fn count_num_action_nodes(
    node: &ActionTreeNode,
    initial_state: BoardState,
) -> [u64; 3] {
    let street_offset = initial_state as usize;
    let mut ret = [0, 0, 0];
    count_num_action_nodes_recursive(node, street_offset, &mut ret);
    ret
}

fn count_num_action_nodes_recursive(node: &ActionTreeNode, street: usize, count: &mut [u64; 3]) {
    count[street] += 1;
    if node.is_terminal() {
        // do nothing
    } else if node.is_chance() {
        count_num_action_nodes_recursive(&node.children[0].lock(), street + 1, count);
    } else {
        for child in &node.children {
            count_num_action_nodes_recursive(&child.lock(), street, count);
        }
    }
}

fn merge_bet_actions(actions: Vec<Action>, pot: i32, offset: i32, param: f64) -> Vec<Action> {
    const EPS: f64 = 1e-12;

    let get_amount = |action: Action| match action {
        Action::Bet(amount) | Action::Raise(amount) | Action::AllIn(amount) => amount,
        _ => -1,
    };

    let mut cur_amount = i32::MAX;
    let mut ret = Vec::new();

    for &action in actions.iter().rev() {
        let amount = get_amount(action);
        if amount > 0 {
            let ratio = (amount - offset) as f64 / pot as f64;
            let cur_ratio = (cur_amount - offset) as f64 / pot as f64;
            let threshold_ratio = (cur_ratio - param) / (1.0 + param);
            if ratio < threshold_ratio * (1.0 - EPS) {
                ret.push(action);
                cur_amount = amount;
            }
        } else {
            ret.push(action);
        }
    }

    ret.reverse();
    ret
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
            depth_limit: None,
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

    // -- ActionTree accessors --

    #[test]
    fn action_tree_default_accessors() {
        let tree = ActionTree::default();
        assert_eq!(tree.config().starting_pot, 0);
        assert!(tree.added_lines().is_empty());
        assert!(tree.removed_lines().is_empty());
        assert!(tree.history().is_empty());
    }

    // -- Config validation --

    #[test]
    fn check_config_rejects_zero_pot() {
        let config = TreeConfig {
            starting_pot: 0,
            effective_stack: 100,
            ..Default::default()
        };
        assert!(ActionTree::new(config).is_err());
    }

    #[test]
    fn check_config_rejects_negative_pot() {
        let config = TreeConfig {
            starting_pot: -10,
            effective_stack: 100,
            ..Default::default()
        };
        assert!(ActionTree::new(config).is_err());
    }

    #[test]
    fn check_config_rejects_zero_stack() {
        let config = TreeConfig {
            starting_pot: 100,
            effective_stack: 0,
            ..Default::default()
        };
        assert!(ActionTree::new(config).is_err());
    }

    #[test]
    fn check_config_rejects_negative_rake_rate() {
        let config = TreeConfig {
            starting_pot: 100,
            effective_stack: 100,
            rake_rate: -0.1,
            ..Default::default()
        };
        assert!(ActionTree::new(config).is_err());
    }

    #[test]
    fn check_config_rejects_rake_rate_above_one() {
        let config = TreeConfig {
            starting_pot: 100,
            effective_stack: 100,
            rake_rate: 1.01,
            ..Default::default()
        };
        assert!(ActionTree::new(config).is_err());
    }

    #[test]
    fn check_config_rejects_negative_rake_cap() {
        let config = TreeConfig {
            starting_pot: 100,
            effective_stack: 100,
            rake_cap: -1.0,
            ..Default::default()
        };
        assert!(ActionTree::new(config).is_err());
    }

    #[test]
    fn check_config_rejects_negative_allin_threshold() {
        let config = TreeConfig {
            starting_pot: 100,
            effective_stack: 100,
            add_allin_threshold: -0.1,
            ..Default::default()
        };
        assert!(ActionTree::new(config).is_err());
    }

    #[test]
    fn check_config_rejects_negative_force_allin_threshold() {
        let config = TreeConfig {
            starting_pot: 100,
            effective_stack: 100,
            force_allin_threshold: -0.1,
            ..Default::default()
        };
        assert!(ActionTree::new(config).is_err());
    }

    #[test]
    fn check_config_rejects_negative_merging_threshold() {
        let config = TreeConfig {
            starting_pot: 100,
            effective_stack: 100,
            merging_threshold: -0.1,
            ..Default::default()
        };
        assert!(ActionTree::new(config).is_err());
    }

    // -- Simple river tree --

    fn river_config(bet_str: &str, raise_str: &str, pot: i32, stack: i32) -> TreeConfig {
        let sizes = BetSizeOptions::try_from((bet_str, raise_str)).unwrap();
        TreeConfig {
            initial_state: BoardState::River,
            starting_pot: pot,
            effective_stack: stack,
            flop_bet_sizes: Default::default(),
            turn_bet_sizes: Default::default(),
            river_bet_sizes: [sizes.clone(), sizes],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            ..Default::default()
        }
    }

    #[test]
    fn test_simple_river_tree() {
        let config = river_config("50%", "60%", 100, 100);
        let tree = ActionTree::new(config).unwrap();
        let actions = tree.available_actions();

        // OOP to act first on river. Actions: Check, Bet(50), AllIn(100)
        // 50% of 100 pot = 50. Since add_allin_threshold = 1.5, and max_amount(100) <=
        // pot(100) * 1.5 = 150, all-in is added.
        assert!(actions.contains(&Action::Check));
        assert!(actions.contains(&Action::Bet(50)));
        assert!(actions.contains(&Action::AllIn(100)));
        assert_eq!(actions.len(), 3);
    }

    #[test]
    fn test_river_tree_bet_clamped_to_allin() {
        // If the bet would be close to all-in, the force_allin_threshold kicks in.
        // pot=100, stack=60, bet=50% => 50 chips. After bet of 50: new_pot = 100+2*50=200,
        // threshold = 200*0.15 = 30. max_amount(60) <= 50+30 = 80? Yes. So it becomes all-in.
        let config = river_config("50%", "60%", 100, 60);
        let tree = ActionTree::new(config).unwrap();
        let actions = tree.available_actions();

        // Check and AllIn(60) — the 50% bet gets forced to all-in
        assert!(actions.contains(&Action::Check));
        assert!(actions.contains(&Action::AllIn(60)));
        // No raw bet action since it was forced to all-in
        assert!(!actions.contains(&Action::Bet(50)));
    }

    #[test]
    fn test_river_check_check_is_terminal() {
        let config = river_config("50%", "60%", 100, 100);
        let mut tree = ActionTree::new(config).unwrap();

        // OOP checks
        tree.play(Action::Check).unwrap();
        // IP should also have check as option
        let actions = tree.available_actions();
        assert!(actions.contains(&Action::Check));

        // IP checks => terminal (river showdown)
        tree.play(Action::Check).unwrap();
        assert!(tree.is_terminal_node());
    }

    #[test]
    fn test_river_bet_fold_is_terminal() {
        let config = river_config("50%", "60%", 100, 100);
        let mut tree = ActionTree::new(config).unwrap();

        // OOP bets 50
        tree.play(Action::Bet(50)).unwrap();
        // IP should have Fold, Call, and possibly raise/all-in
        let actions = tree.available_actions();
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::Call));

        // IP folds => terminal
        tree.play(Action::Fold).unwrap();
        assert!(tree.is_terminal_node());
    }

    #[test]
    fn test_river_bet_call_is_terminal() {
        let config = river_config("50%", "60%", 100, 100);
        let mut tree = ActionTree::new(config).unwrap();

        tree.play(Action::Bet(50)).unwrap();
        tree.play(Action::Call).unwrap();
        assert!(tree.is_terminal_node());
    }

    // -- Navigation --

    #[test]
    fn test_play_and_undo() {
        let config = river_config("50%", "60%", 100, 100);
        let mut tree = ActionTree::new(config).unwrap();

        assert!(tree.history().is_empty());
        tree.play(Action::Check).unwrap();
        assert_eq!(tree.history(), &[Action::Check]);
        tree.undo().unwrap();
        assert!(tree.history().is_empty());
    }

    #[test]
    fn test_play_invalid_action() {
        let config = river_config("50%", "60%", 100, 100);
        let mut tree = ActionTree::new(config).unwrap();

        // Fold is not available at root (OOP to act, no bet to fold to)
        assert!(tree.play(Action::Fold).is_err());
    }

    #[test]
    fn test_undo_at_root_is_err() {
        let config = river_config("50%", "60%", 100, 100);
        let mut tree = ActionTree::new(config).unwrap();

        assert!(tree.undo().is_err());
    }

    #[test]
    fn test_back_to_root() {
        let config = river_config("50%", "60%", 100, 100);
        let mut tree = ActionTree::new(config).unwrap();

        tree.play(Action::Check).unwrap();
        tree.play(Action::Check).unwrap();
        tree.back_to_root();
        assert!(tree.history().is_empty());
    }

    #[test]
    fn test_apply_history() {
        let config = river_config("50%", "60%", 100, 100);
        let mut tree = ActionTree::new(config).unwrap();

        tree.apply_history(&[Action::Check, Action::Check]).unwrap();
        assert_eq!(tree.history(), &[Action::Check, Action::Check]);
        assert!(tree.is_terminal_node());
    }

    #[test]
    fn test_apply_history_invalid() {
        let config = river_config("50%", "60%", 100, 100);
        let mut tree = ActionTree::new(config).unwrap();

        assert!(tree.apply_history(&[Action::Fold]).is_err());
    }

    // -- Total bet amount --

    #[test]
    fn test_total_bet_amount_at_root() {
        let config = river_config("50%", "60%", 100, 100);
        let mut tree = ActionTree::new(config).unwrap();

        assert_eq!(tree.total_bet_amount(), [0, 0]);

        tree.play(Action::Bet(50)).unwrap();
        assert_eq!(tree.total_bet_amount(), [50, 0]);

        tree.play(Action::Call).unwrap();
        assert_eq!(tree.total_bet_amount(), [50, 50]);
    }

    // -- Flop tree with chance nodes --

    #[test]
    fn test_flop_tree_has_chance_nodes() {
        let config = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: 100,
            effective_stack: 200,
            flop_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            turn_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            river_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            ..Default::default()
        };
        let mut tree = ActionTree::new(config).unwrap();

        // OOP checks, IP checks => should hit chance node for turn
        tree.play(Action::Check).unwrap();
        tree.play(Action::Check).unwrap();
        // After flop check-check, next is chance node for the turn
        assert!(tree.is_chance_node());
        // available_actions should skip chance and show turn actions
        let actions = tree.available_actions();
        assert!(actions.contains(&Action::Check));
    }

    #[test]
    fn test_flop_through_all_streets() {
        let config = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: 100,
            effective_stack: 1000,
            flop_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            turn_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            river_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            ..Default::default()
        };
        let mut tree = ActionTree::new(config).unwrap();

        // Flop: check-check => chance (turn)
        tree.play(Action::Check).unwrap();
        tree.play(Action::Check).unwrap();
        assert!(tree.is_chance_node());
        assert!(!tree.is_terminal_node());

        // Turn: check-check => chance (river)
        tree.play(Action::Check).unwrap();
        tree.play(Action::Check).unwrap();
        assert!(tree.is_chance_node());

        // River: check-check => terminal
        tree.play(Action::Check).unwrap();
        tree.play(Action::Check).unwrap();
        assert!(tree.is_terminal_node());
    }

    // -- Raise actions --

    #[test]
    fn test_river_raise_actions() {
        // pot=100, stack=200, bet 50%, raise 60%
        let config = river_config("50%", "60%", 100, 200);
        let mut tree = ActionTree::new(config).unwrap();

        // OOP bets 50 (50% of pot=100)
        tree.play(Action::Bet(50)).unwrap();

        // IP facing bet of 50. pot_after_call = 100 + 2*50 = 200.
        // Raise 60% means prev_amount(50) + (200 * 0.60) = 50 + 120 = 170.
        // max_amount = opponent_stack(150) + prev_amount(50) = 200.
        // min_amount = (50 + 50).clamp(1, 200) = 100.
        // is_above_threshold(170): new_pot=200+2*(170-50)=440, threshold=440*0.15=66.
        //   200 <= 170+66=236? yes. So 170 => AllIn(200).
        // So IP actions should be: Fold, Call, AllIn(200).
        let actions = tree.available_actions();
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::Call));
        assert!(actions.contains(&Action::AllIn(200)));
    }

    #[test]
    fn test_river_raise_survives_threshold() {
        // Make a scenario where raise does NOT get forced to all-in.
        // pot=100, stack=500, bet 33%, raise 60%
        let config = river_config("33%", "60%", 100, 500);
        let mut tree = ActionTree::new(config).unwrap();

        // OOP bets 33 (33% of pot=100)
        tree.play(Action::Bet(33)).unwrap();

        // IP facing bet of 33. pot_after_call = 100 + 2*33 = 166.
        // Raise 60%: prev_amount(33) + round(166*0.60) = 33 + 100 = 133.
        // max_amount = 500 - 33 + 33 = 500.
        // min_amount = (33 + 33).clamp(1, 500) = 66.
        // is_above_threshold(133): new_pot = 166 + 2*(133-33) = 366.
        //   threshold = round(366*0.15) = 55.
        //   500 <= 133+55 = 188? No. So raise survives.
        let actions = tree.available_actions();
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::Call));
        assert!(actions.contains(&Action::Raise(133)));
    }

    // -- Geometric sizing --

    #[test]
    fn test_geometric_bet_sizing() {
        // River, pot=100, stack=100, geometric bet sizing with 1 street
        let sizes = BetSizeOptions::try_from(("e", "")).unwrap();
        let config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: [sizes.clone(), sizes],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            ..Default::default()
        };
        let tree = ActionTree::new(config).unwrap();
        let actions = tree.available_actions();

        // River with "e" = 1 street geometric.
        // spr_after_call = 100/100 = 1.0
        // ratio = ((2*1+1)^(1/1) - 1)/2 = (3-1)/2 = 1.0
        // amount = round(100 * 1.0) = 100
        // 100 == max_amount(100), so it becomes AllIn(100)
        assert!(actions.contains(&Action::Check));
        assert!(actions.contains(&Action::AllIn(100)));
    }

    // -- Donk sizing --

    #[test]
    fn test_turn_donk_bet() {
        let flop_sizes = BetSizeOptions::try_from(("50%", "")).unwrap();
        let turn_sizes = BetSizeOptions::try_from(("50%", "")).unwrap();
        let donk_sizes = DonkSizeOptions::try_from("75%").unwrap();
        let config = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: 100,
            effective_stack: 1000,
            flop_bet_sizes: [flop_sizes.clone(), flop_sizes],
            turn_bet_sizes: [turn_sizes.clone(), turn_sizes],
            river_bet_sizes: Default::default(),
            turn_donk_sizes: Some(donk_sizes),
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            ..Default::default()
        };
        let mut tree = ActionTree::new(config).unwrap();

        // Flop: IP bets, OOP calls => sets oop_call_flag
        tree.play(Action::Check).unwrap();
        let actions = tree.available_actions();
        assert!(actions.contains(&Action::Bet(50))); // IP can bet
        tree.back_to_root();

        // To trigger donk: OOP checks, IP bets, OOP calls => chance (turn)
        // Then on turn, OOP has donk bet option
        tree.play(Action::Check).unwrap();
        tree.play(Action::Bet(50)).unwrap();
        tree.play(Action::Call).unwrap();
        // Now at chance node for turn
        assert!(tree.is_chance_node());
        // Turn actions should include donk bet
        let actions = tree.available_actions();
        // pot = 100 + 2*50 = 200 (starting pot + flop action)
        // donk 75% = round(200 * 0.75) = 150
        assert!(actions.contains(&Action::Check));
        assert!(actions.contains(&Action::Bet(150)));
    }

    // -- Merging threshold --

    #[test]
    fn test_merging_close_bet_sizes() {
        // pot=100, stack=300. Two close bet sizes: 50% and 55%.
        // Without merging both would appear. With merging (0.12), they should merge.
        let sizes = BetSizeOptions::try_from(("50%, 55%", "")).unwrap();
        let config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 300,
            river_bet_sizes: [sizes.clone(), sizes],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            ..Default::default()
        };
        let tree = ActionTree::new(config).unwrap();
        let actions = tree.available_actions();

        // 50% = 50, 55% = 55. After sort+dedup: Bet(50), Bet(55).
        // Merging: iterate from highest. cur_amount=MAX. 55: ratio=55/100=0.55,
        //   cur_ratio=(MAX-0)/100 ~ INF, threshold_ratio ~ INF. 0.55 < INF. Keep 55.
        //   cur_amount=55.
        // 50: ratio=50/100=0.50, cur_ratio=55/100=0.55.
        //   threshold_ratio=(0.55-0.12)/(1+0.12)=0.43/1.12=0.384.
        //   0.50 < 0.384? No. So 50 is DROPPED.
        // Only Bet(55) remains.
        assert!(actions.contains(&Action::Check));
        assert!(actions.contains(&Action::Bet(55)));
        assert!(!actions.contains(&Action::Bet(50)));
    }

    #[test]
    fn test_merging_distant_bet_sizes() {
        // pot=100, stack=300. Two distant bet sizes: 33% and 100%.
        // With merging (0.12), both should survive.
        let sizes = BetSizeOptions::try_from(("33%, 100%", "")).unwrap();
        let config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 300,
            river_bet_sizes: [sizes.clone(), sizes],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            ..Default::default()
        };
        let tree = ActionTree::new(config).unwrap();
        let actions = tree.available_actions();

        assert!(actions.contains(&Action::Bet(33)));
        assert!(actions.contains(&Action::Bet(100)));
    }

    // -- Add and remove lines --

    #[test]
    fn test_add_line() {
        let config = river_config("50%", "", 100, 200);
        let mut tree = ActionTree::new(config).unwrap();

        // Add a new bet size (Bet(75)) at the root
        tree.add_line(&[Action::Bet(75)]).unwrap();
        let actions = tree.available_actions();
        assert!(actions.contains(&Action::Bet(75)));
        assert_eq!(tree.added_lines().len(), 1);
    }

    #[test]
    fn test_add_line_duplicate_fails() {
        let config = river_config("50%", "", 100, 200);
        let mut tree = ActionTree::new(config).unwrap();

        // Bet(50) already exists
        assert!(tree.add_line(&[Action::Bet(50)]).is_err());
    }

    #[test]
    fn test_remove_line() {
        let config = river_config("50%", "", 100, 200);
        let mut tree = ActionTree::new(config).unwrap();

        // Remove bet 50
        tree.remove_line(&[Action::Bet(50)]).unwrap();
        let actions = tree.available_actions();
        assert!(!actions.contains(&Action::Bet(50)));
        assert_eq!(tree.removed_lines().len(), 1);
    }

    #[test]
    fn test_remove_and_readd_line() {
        let config = river_config("50%", "", 100, 200);
        let mut tree = ActionTree::new(config).unwrap();

        tree.remove_line(&[Action::Bet(50)]).unwrap();
        assert_eq!(tree.removed_lines().len(), 1);

        // Re-add the same line
        tree.add_line(&[Action::Bet(50)]).unwrap();
        assert!(tree.removed_lines().is_empty());
        assert!(tree.added_lines().is_empty()); // removed from removed, not added to added
    }

    #[test]
    fn test_add_action_and_remove_action() {
        let config = river_config("50%", "", 100, 200);
        let mut tree = ActionTree::new(config).unwrap();

        tree.add_action(Action::Bet(75)).unwrap();
        assert!(tree.available_actions().contains(&Action::Bet(75)));

        tree.remove_action(Action::Bet(75)).unwrap();
        assert!(!tree.available_actions().contains(&Action::Bet(75)));
    }

    // -- Invalid terminals --

    #[test]
    fn test_no_invalid_terminals() {
        let config = river_config("50%", "60%", 100, 100);
        let tree = ActionTree::new(config).unwrap();
        assert!(tree.invalid_terminals().is_empty());
    }

    // -- count_num_action_nodes --

    #[test]
    fn test_count_action_nodes_river() {
        let config = river_config("50%", "60%", 100, 200);
        let tree = ActionTree::new(config).unwrap();
        let counts = count_num_action_nodes(&tree.root.lock(), tree.config.initial_state);
        // River-only tree: counts[2] should be non-zero
        assert!(counts[2] > 0);
        assert_eq!(counts[0], 0);
        assert_eq!(counts[1], 0);
    }

    #[test]
    fn test_count_action_nodes_flop() {
        let config = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: 100,
            effective_stack: 200,
            flop_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            turn_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            river_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            ..Default::default()
        };
        let tree = ActionTree::new(config).unwrap();
        let counts = count_num_action_nodes(&tree.root.lock(), tree.config.initial_state);
        // Flop tree should have nodes on all three streets
        assert!(counts[0] > 0);
        assert!(counts[1] > 0);
        assert!(counts[2] > 0);
    }

    // -- Additive bet sizes --

    #[test]
    fn test_additive_bet_size() {
        let sizes = BetSizeOptions::try_from(("100c", "")).unwrap();
        let config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 500,
            river_bet_sizes: [sizes.clone(), sizes],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            ..Default::default()
        };
        let tree = ActionTree::new(config).unwrap();
        let actions = tree.available_actions();
        assert!(actions.contains(&Action::Bet(100)));
    }

    // -- PrevBetRelative raise sizes --

    #[test]
    fn test_prev_bet_relative_raise() {
        // pot=100, stack=500, bet=50%, raise=3x
        let sizes = BetSizeOptions::try_from(("50%", "3x")).unwrap();
        let config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 500,
            river_bet_sizes: [sizes.clone(), sizes],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            ..Default::default()
        };
        let mut tree = ActionTree::new(config).unwrap();

        // OOP bets 50
        tree.play(Action::Bet(50)).unwrap();

        // IP raise = round(50 * 3.0) = 150
        // max_amount = 450 + 50 = 500
        // is_above_threshold(150): new_pot = 200 + 2*(150-50) = 400.
        //   threshold = round(400*0.15) = 60. 500 <= 150+60 = 210? No. Survives.
        let actions = tree.available_actions();
        assert!(actions.contains(&Action::Raise(150)));
    }

    // -- Allin-only when stack is small --

    #[test]
    fn test_tiny_stack_allin_only() {
        // pot=100, stack=5 => very small stack relative to pot.
        // Any bet forces all-in due to force_allin_threshold.
        let sizes = BetSizeOptions::try_from(("50%", "")).unwrap();
        let config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 5,
            river_bet_sizes: [sizes.clone(), sizes],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            ..Default::default()
        };
        let tree = ActionTree::new(config).unwrap();
        let actions = tree.available_actions();

        // Only Check and AllIn(5) should remain
        assert!(actions.contains(&Action::Check));
        assert!(actions.contains(&Action::AllIn(5)));
        assert_eq!(actions.len(), 2);
    }

    // -- Eject --

    #[test]
    fn test_eject() {
        let config = river_config("50%", "", 100, 100);
        let tree = ActionTree::new(config).unwrap();
        let (cfg, added, removed, _root) = tree.eject();
        assert_eq!(cfg.starting_pot, 100);
        assert!(added.is_empty());
        assert!(removed.is_empty());
    }

    // -- BuildTreeInfo --

    #[test]
    fn test_build_tree_info_new() {
        let info = BuildTreeInfo::new(500);
        assert_eq!(info.stack, [500, 500]);
        assert_eq!(info.prev_amount, 0);
        assert_eq!(info.num_bets, 0);
        assert!(!info.allin_flag);
        assert!(!info.oop_call_flag);
        assert!(matches!(info.prev_action, Action::None));
    }

    #[test]
    fn test_build_tree_info_create_next_bet() {
        let info = BuildTreeInfo::new(500);
        let next = info.create_next(PLAYER_OOP, Action::Bet(100));
        assert_eq!(next.num_bets, 1);
        assert!(!next.allin_flag);
        assert_eq!(next.prev_amount, 100);
        assert_eq!(next.stack[0], 400); // OOP spent 100
        assert_eq!(next.stack[1], 500); // IP unchanged
    }

    #[test]
    fn test_build_tree_info_create_next_call() {
        let info = BuildTreeInfo::new(500);
        let after_bet = info.create_next(PLAYER_OOP, Action::Bet(100));
        let after_call = after_bet.create_next(PLAYER_IP, Action::Call);
        assert_eq!(after_call.num_bets, 0);
        assert_eq!(after_call.prev_amount, 0);
        assert_eq!(after_call.stack[0], 400);
        assert_eq!(after_call.stack[1], 400); // IP matched OOP
        assert!(after_call.oop_call_flag == false); // IP called, not OOP
    }

    #[test]
    fn test_build_tree_info_create_next_allin() {
        let info = BuildTreeInfo::new(100);
        let next = info.create_next(PLAYER_OOP, Action::AllIn(100));
        assert!(next.allin_flag);
        assert_eq!(next.num_bets, 1);
        assert_eq!(next.stack[0], 0);
    }

    #[test]
    fn test_build_tree_info_street_transitions() {
        let info = BuildTreeInfo::new(500);
        assert_eq!(info.street_transitions, 0);

        // Chance action increments street_transitions
        let after_chance = info.create_next(0, Action::Chance(0));
        assert_eq!(after_chance.street_transitions, 1);

        // Non-chance actions don't increment
        let after_check = info.create_next(PLAYER_OOP, Action::Check);
        assert_eq!(after_check.street_transitions, 0);
    }

    // -- Depth limit --

    #[test]
    fn depth_limit_none_does_not_affect_tree() {
        // A turn tree with no depth_limit should have chance nodes.
        let config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            turn_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            river_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            depth_limit: None,
            ..Default::default()
        };
        let mut tree = ActionTree::new(config).unwrap();

        // Check-check leads to chance (river deal)
        tree.play(Action::Check).unwrap();
        tree.play(Action::Check).unwrap();
        assert!(tree.is_chance_node());
    }

    #[test]
    fn depth_limit_zero_blocks_all_transitions() {
        // A turn tree with depth_limit=0 should have no chance nodes.
        // Check-check should lead to a terminal (depth boundary).
        let config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            turn_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            river_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            depth_limit: Some(0),
            ..Default::default()
        };
        let mut tree = ActionTree::new(config).unwrap();

        // Root: OOP to act on the turn
        assert!(!tree.is_terminal_node());
        assert!(!tree.is_chance_node());

        // Check-check should be terminal (depth boundary)
        tree.play(Action::Check).unwrap();
        tree.play(Action::Check).unwrap();
        assert!(tree.is_terminal_node());
        assert!(!tree.is_chance_node());
    }

    #[test]
    fn depth_limit_zero_bet_call_is_terminal() {
        let config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            turn_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            river_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            depth_limit: Some(0),
            ..Default::default()
        };
        let mut tree = ActionTree::new(config).unwrap();

        // OOP bets, IP calls => should be terminal (depth boundary)
        tree.play(Action::Bet(50)).unwrap();
        tree.play(Action::Call).unwrap();
        assert!(tree.is_terminal_node());
    }

    #[test]
    fn depth_limit_one_allows_one_transition() {
        // Flop tree with depth_limit=1: flop->turn allowed, turn->river blocked.
        let config = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: 100,
            effective_stack: 1000,
            flop_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            turn_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            river_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.12,
            depth_limit: Some(1),
            ..Default::default()
        };
        let mut tree = ActionTree::new(config).unwrap();

        // Flop: check-check => chance (turn) — first transition allowed
        tree.play(Action::Check).unwrap();
        tree.play(Action::Check).unwrap();
        assert!(tree.is_chance_node());

        // Turn: check-check => should be terminal (depth boundary), not chance
        tree.play(Action::Check).unwrap();
        tree.play(Action::Check).unwrap();
        assert!(tree.is_terminal_node());
        assert!(!tree.is_chance_node());
    }

    #[test]
    fn depth_boundary_flag_constant() {
        // PLAYER_DEPTH_BOUNDARY_FLAG includes PLAYER_TERMINAL_FLAG
        assert_ne!(PLAYER_DEPTH_BOUNDARY_FLAG & PLAYER_TERMINAL_FLAG, 0);
        // But not PLAYER_FOLD_FLAG or PLAYER_CHANCE_FLAG
        assert_eq!(PLAYER_DEPTH_BOUNDARY_FLAG & PLAYER_CHANCE_FLAG, 0);
        // Different from PLAYER_FOLD_FLAG
        assert_ne!(PLAYER_DEPTH_BOUNDARY_FLAG, PLAYER_FOLD_FLAG);
    }
}
