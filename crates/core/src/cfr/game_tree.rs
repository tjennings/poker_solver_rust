//! Materialized game tree for sequence-form CFR.
//!
//! Flattens the recursive game tree into a dense array of nodes for
//! cache-friendly, allocation-free traversal. The tree structure is
//! **deal-independent**: it captures all possible action sequences,
//! with deal-specific data (info set keys, terminal utilities)
//! computed at traversal time.
//!
//! # Architecture
//!
//! ```text
//! GameTree { nodes: Vec<TreeNode>, levels: Vec<Vec<u32>> }
//!     │
//!     ├── Decision nodes: player, street, spr/depth buckets, action codes
//!     └── Terminal nodes: fold(player) or showdown, pot, stacks
//! ```

use arrayvec::ArrayVec;

use crate::game::{Game, Player, Action, MAX_ACTIONS};
use crate::info_key::{self, InfoKey};

/// A node in the materialized game tree.
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Index of parent node (`u32::MAX` for root).
    pub parent: u32,
    /// Action that led here from parent (undefined for root).
    pub action_from_parent: Action,
    /// Indices of child nodes.
    pub children: ArrayVec<u32, MAX_ACTIONS>,
    /// Tree depth (root = 0).
    pub depth: u16,
    /// Node payload (decision or terminal data).
    pub node_type: NodeType,
}

/// Type-specific data stored at each tree node.
#[derive(Debug, Clone)]
pub enum NodeType {
    /// A decision node where a player must choose an action.
    Decision {
        player: Player,
        street: u8,
        pot: u32,
        stacks: [u32; 2],
        spr_bucket: u32,
        /// Encoded action codes for actions on the current street
        /// leading to this node (used for info set key computation).
        street_action_codes: ArrayVec<u8, 6>,
    },
    /// A terminal node where the game is over.
    Terminal {
        /// `true` if a player folded, `false` for showdown.
        is_fold: bool,
        /// Which player folded (only meaningful when `is_fold` is true).
        fold_player: Player,
        /// Pot size at this terminal.
        pot: u32,
        /// Player stacks at this terminal `[P1, P2]`.
        stacks: [u32; 2],
        /// Starting stack (for utility computation).
        starting_stack: u32,
        /// Pre-computed P1 utility when P1 wins the showdown (for generic trees).
        /// Only valid for fold terminals or when showdown is pre-computed.
        utility_p1_wins: f64,
        /// Pre-computed P1 utility when P2 wins the showdown.
        utility_p1_loses: f64,
    },
}

/// A materialized game tree with dense node storage.
#[derive(Debug, Clone)]
pub struct GameTree {
    /// All nodes in DFS order.
    pub nodes: Vec<TreeNode>,
    /// Nodes grouped by depth level (for level-by-level traversal).
    pub levels: Vec<Vec<u32>>,
    /// Tree statistics.
    pub stats: TreeStats,
}

/// Statistics about a materialized game tree.
#[derive(Debug, Default, Clone)]
pub struct TreeStats {
    pub total_nodes: u32,
    pub decision_nodes: u32,
    pub terminal_nodes: u32,
    pub fold_terminals: u32,
    pub showdown_terminals: u32,
    pub max_depth: u16,
    pub p1_decision_nodes: u32,
    pub p2_decision_nodes: u32,
}

impl std::fmt::Display for TreeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tree Statistics:")?;
        writeln!(f, "  Total nodes:      {}", self.total_nodes)?;
        writeln!(f, "  Decision nodes:   {} (P1: {}, P2: {})",
            self.decision_nodes, self.p1_decision_nodes, self.p2_decision_nodes)?;
        writeln!(f, "  Terminal nodes:   {} (fold: {}, showdown: {})",
            self.terminal_nodes, self.fold_terminals, self.showdown_terminals)?;
        writeln!(f, "  Max depth:        {}", self.max_depth)?;
        let bytes = self.total_nodes as usize * std::mem::size_of::<TreeNode>();
        #[allow(clippy::cast_precision_loss)]
        let mb = bytes as f64 / 1_048_576.0;
        writeln!(f, "  Memory (nodes):   {mb:.1} MB")
    }
}

/// Materialize the game tree by DFS traversal from a root state.
///
/// The resulting tree captures the full action sequence structure.
/// Terminal utilities and info set keys are computed per-deal at
/// traversal time, not stored in the tree.
pub fn materialize<G: Game>(game: &G, root_state: &G::State) -> GameTree {
    let mut nodes: Vec<TreeNode> = Vec::new();
    let mut levels: Vec<Vec<u32>> = Vec::new();
    let mut stats = TreeStats::default();

    // DFS stack: (state, node_index, depth)
    // We first create a placeholder node, then fill in children during DFS
    build_recursive(game, root_state, u32::MAX, Action::Check, 0,
        &mut nodes, &mut levels, &mut stats);

    #[allow(clippy::cast_possible_truncation)]
    { stats.total_nodes = nodes.len() as u32; }

    GameTree { nodes, levels, stats }
}

#[allow(clippy::too_many_arguments)]
fn build_recursive<G: Game>(
    game: &G,
    state: &G::State,
    parent: u32,
    action_from_parent: Action,
    depth: u16,
    nodes: &mut Vec<TreeNode>,
    levels: &mut Vec<Vec<u32>>,
    tree_stats: &mut TreeStats,
) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let node_idx = nodes.len() as u32;

    // Ensure levels vec is large enough
    while levels.len() <= depth as usize {
        levels.push(Vec::new());
    }
    levels[depth as usize].push(node_idx);

    if depth > tree_stats.max_depth {
        tree_stats.max_depth = depth;
    }

    if game.is_terminal(state) {
        let is_fold = action_from_parent == Action::Fold;
        let fold_player = if is_fold { game.player(state) } else { Player::Player1 };
        let utility_p1 = game.utility(state, Player::Player1);
        let node = make_terminal_node_generic(
            is_fold, fold_player, parent, action_from_parent, depth, utility_p1,
        );
        update_terminal_stats(tree_stats, &node.node_type);
        nodes.push(node);
        return node_idx;
    }

    // Decision node — push placeholder, then build children
    let node = make_decision_node_generic(game, state, parent, action_from_parent, depth);
    update_decision_stats(tree_stats, &node.node_type);
    nodes.push(node);

    let actions = game.actions(state);
    let mut child_indices = ArrayVec::<u32, MAX_ACTIONS>::new();

    for &action in &actions {
        let child_state = game.next_state(state, action);
        let child_idx = build_recursive(
            game, &child_state, node_idx, action, depth + 1,
            nodes, levels, tree_stats,
        );
        child_indices.push(child_idx);
    }

    nodes[node_idx as usize].children = child_indices;
    node_idx
}

fn make_terminal_node_generic(
    is_fold: bool,
    fold_player: Player,
    parent: u32,
    action_from_parent: Action,
    depth: u16,
    utility_p1: f64,
) -> TreeNode {
    TreeNode {
        parent,
        action_from_parent,
        children: ArrayVec::new(),
        depth,
        node_type: NodeType::Terminal {
            is_fold,
            fold_player,
            pot: 0,
            stacks: [0; 2],
            starting_stack: 0,
            utility_p1_wins: utility_p1,
            utility_p1_loses: utility_p1,
        },
    }
}

fn make_decision_node_generic<G: Game>(
    game: &G,
    state: &G::State,
    parent: u32,
    action_from_parent: Action,
    depth: u16,
) -> TreeNode {
    let player = game.player(state);

    // Extract position data from the game's info set key
    let raw_key = game.info_set_key(state);
    let key = InfoKey::from_raw(raw_key);

    // Reconstruct action codes from the key's lower 24 bits
    let mut street_action_codes = ArrayVec::<u8, 6>::new();
    let action_bits = key.actions_bits();
    for i in 0..6 {
        let code = ((action_bits >> (20 - i * 4)) & 0xF) as u8;
        if code == 0 {
            break;
        }
        street_action_codes.push(code);
    }

    TreeNode {
        parent,
        action_from_parent,
        children: ArrayVec::new(),
        depth,
        node_type: NodeType::Decision {
            player,
            street: key.street(),
            pot: 0,
            stacks: [0; 2],
            spr_bucket: key.spr_bucket(),
            street_action_codes,
        },
    }
}

fn update_terminal_stats(stats: &mut TreeStats, node_type: &NodeType) {
    stats.terminal_nodes += 1;
    if let NodeType::Terminal { is_fold, .. } = node_type {
        if *is_fold {
            stats.fold_terminals += 1;
        } else {
            stats.showdown_terminals += 1;
        }
    }
}

fn update_decision_stats(stats: &mut TreeStats, node_type: &NodeType) {
    stats.decision_nodes += 1;
    if let NodeType::Decision { player, .. } = node_type {
        match player {
            Player::Player1 => stats.p1_decision_nodes += 1,
            Player::Player2 => stats.p2_decision_nodes += 1,
        }
    }
}

/// Materialize the game tree for `HunlPostflop`, storing rich position data
/// at each node for efficient sequence-form CFR.
///
/// This specialized builder extracts pot, stacks, street, SPR/depth buckets,
/// and action codes from the `PostflopState` at each node.
#[must_use]
pub fn materialize_postflop(
    game: &crate::game::HunlPostflop,
    root_state: &crate::game::PostflopState,
) -> GameTree {
    let mut nodes: Vec<TreeNode> = Vec::new();
    let mut levels: Vec<Vec<u32>> = Vec::new();
    let mut stats = TreeStats::default();

    build_postflop_recursive(
        game, root_state, u32::MAX, Action::Check, 0,
        &mut nodes, &mut levels, &mut stats,
    );

    #[allow(clippy::cast_possible_truncation)]
    { stats.total_nodes = nodes.len() as u32; }

    GameTree { nodes, levels, stats }
}

#[allow(clippy::too_many_arguments)]
fn build_postflop_recursive(
    game: &crate::game::HunlPostflop,
    state: &crate::game::PostflopState,
    parent: u32,
    action_from_parent: Action,
    depth: u16,
    nodes: &mut Vec<TreeNode>,
    levels: &mut Vec<Vec<u32>>,
    tree_stats: &mut TreeStats,
) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let node_idx = nodes.len() as u32;

    while levels.len() <= depth as usize {
        levels.push(Vec::new());
    }
    levels[depth as usize].push(node_idx);

    if depth > tree_stats.max_depth {
        tree_stats.max_depth = depth;
    }

    if game.is_terminal(state) {
        let node = make_postflop_terminal(state, parent, action_from_parent, depth, game);
        update_terminal_stats(tree_stats, &node.node_type);
        nodes.push(node);
        return node_idx;
    }

    // Decision node
    let node = make_postflop_decision(state, parent, action_from_parent, depth, game);
    update_decision_stats(tree_stats, &node.node_type);
    nodes.push(node);

    let actions = game.actions(state);
    let mut child_indices = ArrayVec::<u32, MAX_ACTIONS>::new();

    for &action in &actions {
        let child_state = game.next_state(state, action);
        let child_idx = build_postflop_recursive(
            game, &child_state, node_idx, action, depth + 1,
            nodes, levels, tree_stats,
        );
        child_indices.push(child_idx);
    }

    nodes[node_idx as usize].children = child_indices;
    node_idx
}

fn make_postflop_terminal(
    state: &crate::game::PostflopState,
    parent: u32,
    action_from_parent: Action,
    depth: u16,
    game: &crate::game::HunlPostflop,
) -> TreeNode {
    use crate::game::TerminalType;

    let (is_fold, fold_player) = match state.terminal {
        Some(TerminalType::Fold(p)) => (true, p),
        _ => (false, Player::Player1),
    };

    let starting_stack = game.config().stack_depth * 2;

    TreeNode {
        parent,
        action_from_parent,
        children: ArrayVec::new(),
        depth,
        node_type: NodeType::Terminal {
            is_fold,
            fold_player,
            pot: state.pot,
            stacks: state.stacks,
            starting_stack,
            utility_p1_wins: 0.0,
            utility_p1_loses: 0.0,
        },
    }
}

fn make_postflop_decision(
    state: &crate::game::PostflopState,
    parent: u32,
    action_from_parent: Action,
    depth: u16,
    _game: &crate::game::HunlPostflop,
) -> TreeNode {
    use crate::abstraction::Street;

    let player = state.active_player();
    let street_code = match state.street {
        Street::Preflop => 0u8,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    };

    let eff_stack = state.stacks[0].min(state.stacks[1]);
    let spr = info_key::spr_bucket(state.pot, eff_stack);

    // Encode current-street actions
    let mut street_action_codes = ArrayVec::<u8, 6>::new();
    for (street, a) in &state.history {
        if *street == state.street && !street_action_codes.is_full() {
            street_action_codes.push(info_key::encode_action(*a));
        }
    }

    TreeNode {
        parent,
        action_from_parent,
        children: ArrayVec::new(),
        depth,
        node_type: NodeType::Decision {
            player,
            street: street_code,
            pot: state.pot,
            stacks: state.stacks,
            spr_bucket: spr,
            street_action_codes,
        },
    }
}

impl GameTree {
    /// Compute an info set key at a decision node for a given hand.
    ///
    /// Combines the deal-independent position data stored in the tree node
    /// with deal-specific hand bits.
    #[must_use]
    pub fn info_set_key(&self, node_idx: u32, hand_bits: u32) -> u64 {
        let node = &self.nodes[node_idx as usize];
        match &node.node_type {
            NodeType::Decision { street, spr_bucket, street_action_codes, .. } => {
                InfoKey::new(hand_bits, *street, *spr_bucket, street_action_codes)
                    .as_u64()
            }
            NodeType::Terminal { .. } => 0,
        }
    }

    /// Compute the P1 utility at a terminal node.
    ///
    /// For fold terminals, utility is determined by who folded and their investment.
    /// For showdown terminals, `p1_equity` determines the expected payoff:
    /// - 1.0 = P1 wins, 0.0 = P2 wins, 0.5 = tie/split.
    /// - Fractional values interpolate between win and lose utilities.
    ///
    /// For postflop trees (with `starting_stack > 0`), utility is computed from
    /// pot/stacks in BB (internal units / 2).
    /// For generic trees (with `starting_stack == 0`), pre-computed utilities are used.
    #[must_use]
    pub fn terminal_utility_p1(&self, node_idx: u32, p1_equity: f64) -> f64 {
        let node = &self.nodes[node_idx as usize];
        match &node.node_type {
            NodeType::Terminal {
                is_fold, fold_player, stacks, starting_stack,
                utility_p1_wins, utility_p1_loses, ..
            } => {
                // Generic tree: use pre-computed utilities
                if *starting_stack == 0 {
                    return p1_equity * utility_p1_wins
                        + (1.0 - p1_equity) * utility_p1_loses;
                }

                // Postflop tree: compute from pot/stacks
                let p1_invested = starting_stack - stacks[0];
                let p2_invested = starting_stack - stacks[1];
                let to_bb = |chips: u32| f64::from(chips) / 2.0;

                if *is_fold {
                    if *fold_player == Player::Player1 {
                        -to_bb(p1_invested)
                    } else {
                        to_bb(p2_invested)
                    }
                } else {
                    let win_util = to_bb(p1_invested + p2_invested) - to_bb(p1_invested);
                    let lose_util = -to_bb(p1_invested);
                    p1_equity * win_util + (1.0 - p1_equity) * lose_util
                }
            }
            NodeType::Decision { .. } => 0.0,
        }
    }

    /// Count unique "position keys" — info set keys ignoring hand bits.
    ///
    /// This gives the number of distinct action-history positions,
    /// which bounds the number of sequences per info set.
    #[must_use]
    pub fn unique_position_keys(&self) -> usize {
        use std::collections::HashSet;

        let mut seen = HashSet::new();
        for node in &self.nodes {
            if let NodeType::Decision { street, spr_bucket, street_action_codes, .. } = &node.node_type {
                let key = InfoKey::new(0, *street, *spr_bucket, street_action_codes)
                    .as_u64();
                seen.insert(key);
            }
        }
        seen.len()
    }

    /// Estimate info set count for a given number of hand classes.
    ///
    /// Each position key can appear with each hand class, giving an
    /// upper bound of `position_keys * hand_classes`. The actual count
    /// is lower because not all (position, hand) combinations are reachable.
    #[must_use]
    pub fn estimated_info_sets(&self, hand_classes: u32) -> u64 {
        let positions = self.unique_position_keys() as u64;
        // Each position has P1 and P2 nodes, but unique_position_keys
        // already counts each player's view separately.
        positions * u64::from(hand_classes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::KuhnPoker;
    use test_macros::timed_test;

    #[timed_test]
    fn kuhn_tree_has_correct_node_count() {
        let game = KuhnPoker::new();
        let states = game.initial_states();
        let tree = materialize(&game, &states[0]);

        // Kuhn poker tree from any single deal:
        //   Root (P1): Check, Bet
        //     Check → P2: Check, Bet
        //       Check → terminal (showdown)
        //       Bet → P1: Fold, Call
        //         Fold → terminal
        //         Call → terminal (showdown)
        //     Bet → P2: Fold, Call
        //       Fold → terminal
        //       Call → terminal (showdown)
        // Total: 5 decision + 5 terminal = 10 nodes (wrong, let me re-count)
        //
        // Actually:
        //   Root(0): P1 decides [Check, Bet]
        //     Check(1): P2 decides [Check, Bet]
        //       Check(2): terminal (showdown)
        //       Bet(3): P1 decides [Fold, Call]
        //         Fold(4): terminal
        //         Call(5): terminal (showdown)
        //     Bet(6): P2 decides [Fold, Call]
        //       Fold(7): terminal
        //       Call(8): terminal (showdown)
        // = 3 decision + 5 terminal = ... wait
        //   decision: Root(0), After-check(1), After-check-bet(3) = 3 decision
        //   Wait: After bet(6) is also a decision (P2 decides fold/call)
        //   decision: Root(0), After-check(1), After-check-bet(3), After-bet(6) = 4 decision
        //   terminal: check-check(2), check-bet-fold(4), check-bet-call(5), bet-fold(7), bet-call(8) = 5 terminal
        // Total: 4 + 5 = 9 nodes
        //
        // But wait, node 3 "P1 decides" after check-bet: is P1 the player?
        // In Kuhn: history len 2 → P1's turn. History: [Check, Bet] → P1 faces bet.
        // So yes, 4 decision nodes.

        assert_eq!(tree.stats.total_nodes, 9, "Kuhn tree should have 9 nodes");
        assert_eq!(tree.stats.decision_nodes, 4, "Kuhn tree should have 4 decision nodes");
        assert_eq!(tree.stats.terminal_nodes, 5, "Kuhn tree should have 5 terminal nodes");
    }

    #[timed_test]
    fn kuhn_tree_max_depth() {
        let game = KuhnPoker::new();
        let states = game.initial_states();
        let tree = materialize(&game, &states[0]);

        // Deepest path: Root → Check → Bet → Fold/Call = depth 3
        assert_eq!(tree.stats.max_depth, 3);
    }

    #[timed_test]
    fn kuhn_tree_fold_and_showdown_counts() {
        let game = KuhnPoker::new();
        let states = game.initial_states();
        let tree = materialize(&game, &states[0]);

        // Fold terminals: check-bet-fold, bet-fold = 2
        assert_eq!(tree.stats.fold_terminals, 2);
        // Showdown terminals: check-check, check-bet-call, bet-call = 3
        assert_eq!(tree.stats.showdown_terminals, 3);
    }

    #[timed_test]
    fn kuhn_tree_root_has_two_children() {
        let game = KuhnPoker::new();
        let states = game.initial_states();
        let tree = materialize(&game, &states[0]);

        assert_eq!(tree.nodes[0].children.len(), 2, "Root should have 2 children (Check, Bet)");
    }

    #[timed_test]
    fn kuhn_tree_levels_correct() {
        let game = KuhnPoker::new();
        let states = game.initial_states();
        let tree = materialize(&game, &states[0]);

        // Level 0: Root (P1 decides)
        // Level 1: After-Check (P2), After-Bet (P2)
        // Level 2: Check-Check(T), Check-Bet(P1), Bet-Fold(T), Bet-Call(T)
        // Level 3: Check-Bet-Fold(T), Check-Bet-Call(T)
        assert_eq!(tree.levels.len(), 4, "Kuhn tree has 4 depth levels (0-3)");
        assert_eq!(tree.levels[0].len(), 1, "Level 0 has root only");
        assert_eq!(tree.levels[1].len(), 2, "Level 1 has 2 nodes");
        assert_eq!(tree.levels[2].len(), 4, "Level 2 has 4 nodes");
        assert_eq!(tree.levels[3].len(), 2, "Level 3 has 2 nodes");
    }

    #[timed_test]
    fn kuhn_tree_parent_child_consistency() {
        let game = KuhnPoker::new();
        let states = game.initial_states();
        let tree = materialize(&game, &states[0]);

        for (idx, node) in tree.nodes.iter().enumerate() {
            for &child_idx in &node.children {
                let child = &tree.nodes[child_idx as usize];
                assert_eq!(
                    child.parent, idx as u32,
                    "Child {} should point back to parent {}",
                    child_idx, idx
                );
            }
        }
    }

    #[timed_test]
    fn kuhn_tree_terminal_nodes_have_no_children() {
        let game = KuhnPoker::new();
        let states = game.initial_states();
        let tree = materialize(&game, &states[0]);

        for node in &tree.nodes {
            if matches!(node.node_type, NodeType::Terminal { .. }) {
                assert!(
                    node.children.is_empty(),
                    "Terminal nodes should have no children"
                );
            }
        }
    }

    #[timed_test]
    fn kuhn_tree_same_structure_different_deals() {
        let game = KuhnPoker::new();
        let states = game.initial_states();

        // All deals should produce identical tree structures
        let tree0 = materialize(&game, &states[0]);
        let tree1 = materialize(&game, &states[1]);

        assert_eq!(tree0.stats.total_nodes, tree1.stats.total_nodes);
        assert_eq!(tree0.stats.decision_nodes, tree1.stats.decision_nodes);
        assert_eq!(tree0.stats.terminal_nodes, tree1.stats.terminal_nodes);
        assert_eq!(tree0.stats.max_depth, tree1.stats.max_depth);
    }

    #[timed_test]
    fn kuhn_unique_position_keys() {
        let game = KuhnPoker::new();
        let states = game.initial_states();
        let tree = materialize(&game, &states[0]);

        // In Kuhn, positions (ignoring hand) are:
        // Root (empty history), after-check, after-check-bet, after-bet
        // = 4 unique position keys
        let positions = tree.unique_position_keys();
        assert_eq!(positions, 4);
    }

    #[timed_test]
    fn postflop_tree_basic() {
        use crate::game::{HunlPostflop, PostflopConfig};

        let config = PostflopConfig {
            stack_depth: 10,
            bet_sizes: vec![1.0],
            max_raises_per_street: 2,
        };
        let game = HunlPostflop::new(config, None, 1);
        let deals = game.initial_states();
        let tree = materialize_postflop(&game, &deals[0]);

        assert!(tree.stats.total_nodes > 0, "Tree should have nodes");
        assert!(tree.stats.terminal_nodes > 0, "Tree should have terminals");
        assert!(tree.stats.decision_nodes > 0, "Tree should have decisions");

        // Verify parent-child consistency
        for (idx, node) in tree.nodes.iter().enumerate() {
            for &child_idx in &node.children {
                let child = &tree.nodes[child_idx as usize];
                assert_eq!(child.parent, idx as u32);
            }
        }
    }

    #[timed_test]
    fn postflop_tree_stores_position_data() {
        use crate::game::{HunlPostflop, PostflopConfig};

        let config = PostflopConfig {
            stack_depth: 25,
            bet_sizes: vec![0.5, 1.0],
            max_raises_per_street: 2,
        };
        let game = HunlPostflop::new(config, None, 1);
        let deals = game.initial_states();
        let tree = materialize_postflop(&game, &deals[0]);

        // Root should be P1 decision at preflop
        let root = &tree.nodes[0];
        match &root.node_type {
            NodeType::Decision { player, street, pot, .. } => {
                assert_eq!(*player, Player::Player1);
                assert_eq!(*street, 0); // preflop
                assert_eq!(*pot, 3);    // SB(1) + BB(2)
            }
            _ => panic!("Root should be a decision node"),
        }
    }

    #[timed_test]
    fn postflop_terminal_utility_fold() {
        use crate::game::{HunlPostflop, PostflopConfig};

        let config = PostflopConfig {
            stack_depth: 25,
            bet_sizes: vec![1.0],
            max_raises_per_street: 2,
        };
        let game = HunlPostflop::new(config, None, 1);
        let deals = game.initial_states();
        let tree = materialize_postflop(&game, &deals[0]);

        // Find a fold terminal (SB folds preflop)
        // First child of root should be Fold if to_call > 0
        let root_children = &tree.nodes[0].children;
        // Find the fold child
        let fold_child_idx = root_children.iter()
            .find(|&&idx| tree.nodes[idx as usize].action_from_parent == Action::Fold);

        if let Some(&fold_idx) = fold_child_idx {
            let utility = tree.terminal_utility_p1(fold_idx, 0.5);
            // SB folds preflop: loses 0.5 BB (invested 1 internal unit = 0.5 BB)
            assert!(
                (utility - (-0.5)).abs() < 0.01,
                "SB fold should lose 0.5 BB, got {utility}"
            );
        }
    }

    #[timed_test]
    fn info_set_key_from_tree_matches_game() {
        use crate::game::{HunlPostflop, PostflopConfig, AbstractionMode};

        let config = PostflopConfig {
            stack_depth: 25,
            bet_sizes: vec![1.0],
            max_raises_per_street: 2,
        };
        let game = HunlPostflop::new(config, Some(AbstractionMode::HandClassV2 { strength_bits: 0, equity_bits: 0 }), 1);
        let deals = game.initial_states();
        let tree = materialize_postflop(&game, &deals[0]);

        // The root node's info set key should match what the game computes
        // (for the same hand bits).
        // At preflop, hand_bits = canonical_hand_index
        let state = &deals[0];
        let game_key = game.info_set_key(state);

        // Extract hand_bits from the game key
        let game_info = InfoKey::from_raw(game_key);
        let hand_bits = game_info.hand_bits();

        let tree_key = tree.info_set_key(0, hand_bits);
        assert_eq!(tree_key, game_key, "Tree info set key should match game");
    }
}
