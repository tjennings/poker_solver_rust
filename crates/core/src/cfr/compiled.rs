//! Game tree compilation to tensor representation.
//!
//! This module converts a `Game` into a tensor-based representation that can be
//! processed efficiently on GPU. The game tree is flattened into arrays indexed
//! by node ID.

// Allow casts for tensor index conversions (i32/i64/usize interop)
#![allow(
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::bool_to_int_with_if
)]

use std::collections::HashMap;

use burn::prelude::*;

use crate::game::{Game, Player};

/// A compiled game tree represented as tensors.
///
/// All game tree structure is encoded in tensors for efficient GPU processing.
/// Nodes are indexed by a flat node ID.
#[derive(Debug)]
pub struct CompiledGame<B: Backend> {
    /// Which player acts at each node (0 = Player1, 1 = Player2)
    pub node_player: Tensor<B, 1, Int>,
    /// Info set index for each node (-1 for terminals)
    pub node_info_set: Tensor<B, 1, Int>,
    /// Number of actions available at each node
    pub node_num_actions: Tensor<B, 1, Int>,
    /// Child node index for each (node, action) pair, -1 if invalid
    pub action_child: Tensor<B, 2, Int>,
    /// Mask for valid actions at each node
    pub action_mask: Tensor<B, 2, Bool>,
    /// Whether each node is terminal
    pub terminal_mask: Tensor<B, 1, Bool>,
    /// Utility at terminal nodes for each player [num_nodes, 2]
    pub terminal_utils: Tensor<B, 2>,
    /// Initial state reach probabilities (for chance nodes / deals)
    pub initial_reach: Tensor<B, 1>,
    /// Root node indices (one per initial state)
    pub root_nodes: Tensor<B, 1, Int>,

    /// Mapping from info set key to index
    pub info_set_to_idx: HashMap<String, usize>,
    /// Number of info sets
    pub num_info_sets: usize,
    /// Maximum number of actions at any node
    pub max_actions: usize,
    /// Total number of nodes
    pub num_nodes: usize,
}

/// Intermediate representation during tree building.
struct TreeBuilder {
    /// Node player (0 or 1)
    node_player: Vec<i32>,
    /// Info set index per node
    node_info_set: Vec<i32>,
    /// Number of actions per node
    node_num_actions: Vec<i32>,
    /// Child indices [node][action]
    action_child: Vec<Vec<i32>>,
    /// Terminal flag per node
    terminal: Vec<bool>,
    /// Terminal utilities [node][player]
    terminal_utils: Vec<[f32; 2]>,

    /// Info set key to index mapping
    info_set_to_idx: HashMap<String, usize>,
    /// Maximum actions seen
    max_actions: usize,
}

impl TreeBuilder {
    fn new() -> Self {
        Self {
            node_player: Vec::new(),
            node_info_set: Vec::new(),
            node_num_actions: Vec::new(),
            action_child: Vec::new(),
            terminal: Vec::new(),
            terminal_utils: Vec::new(),
            info_set_to_idx: HashMap::new(),
            max_actions: 0,
        }
    }

    fn add_node(
        &mut self,
        player: i32,
        info_set_idx: i32,
        num_actions: usize,
        is_terminal: bool,
        utils: [f32; 2],
    ) -> usize {
        let node_id = self.node_player.len();
        self.node_player.push(player);
        self.node_info_set.push(info_set_idx);
        self.node_num_actions.push(num_actions as i32);
        self.action_child.push(vec![-1; num_actions.max(1)]);
        self.terminal.push(is_terminal);
        self.terminal_utils.push(utils);
        self.max_actions = self.max_actions.max(num_actions);
        node_id
    }

    fn set_child(&mut self, parent: usize, action_idx: usize, child: usize) {
        if action_idx < self.action_child[parent].len() {
            self.action_child[parent][action_idx] = child as i32;
        }
    }

    fn get_or_create_info_set(&mut self, key: &str) -> usize {
        let next_idx = self.info_set_to_idx.len();
        *self
            .info_set_to_idx
            .entry(key.to_string())
            .or_insert(next_idx)
    }
}

/// Compile a game into tensor representation.
///
/// This traverses the entire game tree and converts it to tensors.
#[allow(clippy::cast_precision_loss)]
pub fn compile<G, B>(game: &G, device: &B::Device) -> CompiledGame<B>
where
    G: Game,
    B: Backend,
{
    let mut builder = TreeBuilder::new();
    let initial_states = game.initial_states();
    let num_initial = initial_states.len();
    let mut root_nodes = Vec::with_capacity(num_initial);

    // Build tree for each initial state
    for state in &initial_states {
        let root = build_tree_recursive(game, state, &mut builder);
        root_nodes.push(root as i32);
    }

    let num_nodes = builder.node_player.len();
    let max_actions = builder.max_actions.max(1);

    // Pad action_child to uniform width
    for children in &mut builder.action_child {
        children.resize(max_actions, -1);
    }

    // Convert to tensors
    let node_player = Tensor::<B, 1, Int>::from_ints(builder.node_player.as_slice(), device);

    let node_info_set = Tensor::<B, 1, Int>::from_ints(builder.node_info_set.as_slice(), device);

    let node_num_actions =
        Tensor::<B, 1, Int>::from_ints(builder.node_num_actions.as_slice(), device);

    // Flatten action_child to [num_nodes, max_actions]
    let action_child_flat: Vec<i32> = builder
        .action_child
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    let action_child = Tensor::<B, 1, Int>::from_ints(action_child_flat.as_slice(), device)
        .reshape([num_nodes, max_actions]);

    // Build action mask: create int tensor and compare to get bool
    let action_mask_int: Vec<i32> = builder
        .action_child
        .iter()
        .flat_map(|v| v.iter().map(|&c| if c >= 0 { 1i32 } else { 0i32 }))
        .collect();
    let action_mask = Tensor::<B, 1, Int>::from_ints(action_mask_int.as_slice(), device)
        .reshape([num_nodes, max_actions])
        .equal_elem(1);

    // Terminal mask
    let terminal_int: Vec<i32> = builder
        .terminal
        .iter()
        .map(|&b| if b { 1i32 } else { 0i32 })
        .collect();
    let terminal_mask =
        Tensor::<B, 1, Int>::from_ints(terminal_int.as_slice(), device).equal_elem(1);

    // Terminal utilities [num_nodes, 2]
    let terminal_utils_flat: Vec<f32> = builder
        .terminal_utils
        .iter()
        .flat_map(|u| u.iter().copied())
        .collect();
    let terminal_utils =
        Tensor::<B, 1>::from_floats(terminal_utils_flat.as_slice(), device).reshape([num_nodes, 2]);

    // Initial reach: uniform over initial states
    let initial_reach_val = 1.0 / num_initial as f32;
    let initial_reach =
        Tensor::<B, 1>::from_floats(vec![initial_reach_val; num_initial].as_slice(), device);

    let root_nodes_tensor = Tensor::<B, 1, Int>::from_ints(root_nodes.as_slice(), device);

    let num_info_sets = builder.info_set_to_idx.len();

    CompiledGame {
        node_player,
        node_info_set,
        node_num_actions,
        action_child,
        action_mask,
        terminal_mask,
        terminal_utils,
        initial_reach,
        root_nodes: root_nodes_tensor,
        info_set_to_idx: builder.info_set_to_idx,
        num_info_sets,
        max_actions,
        num_nodes,
    }
}

/// Recursively build the game tree.
fn build_tree_recursive<G: Game>(game: &G, state: &G::State, builder: &mut TreeBuilder) -> usize {
    if game.is_terminal(state) {
        // Terminal node
        let u1 = game.utility(state, Player::Player1) as f32;
        let u2 = game.utility(state, Player::Player2) as f32;
        return builder.add_node(0, -1, 0, true, [u1, u2]);
    }

    let player = game.player(state);
    let player_idx = match player {
        Player::Player1 => 0,
        Player::Player2 => 1,
    };

    let info_set_key = game.info_set_key(state);
    let info_set_idx = builder.get_or_create_info_set(&info_set_key) as i32;

    let actions = game.actions(state);
    let num_actions = actions.len();

    // Create this node first (to get its ID)
    let node_id = builder.add_node(player_idx, info_set_idx, num_actions, false, [0.0, 0.0]);

    // Recursively build children
    for (action_idx, action) in actions.iter().enumerate() {
        let next_state = game.next_state(state, *action);
        let child_id = build_tree_recursive(game, &next_state, builder);
        builder.set_child(node_id, action_idx, child_id);
    }

    node_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::KuhnPoker;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn compile_kuhn_poker() {
        let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);

        // Kuhn Poker has 6 initial states (deals)
        assert_eq!(compiled.root_nodes.dims()[0], 6);

        // Should have 12 info sets (3 cards × 4 decision points)
        // J, Q, K at root
        // Jc, Qc, Kc after check
        // Jb, Qb, Kb after bet
        // Jcb, Qcb, Kcb after check-bet
        assert_eq!(compiled.num_info_sets, 12);

        // Max 2 actions (check/bet or fold/call)
        assert_eq!(compiled.max_actions, 2);

        // Verify node count is reasonable
        // Each deal creates a subtree, some nodes are shared via info sets
        assert!(
            compiled.num_nodes > 30,
            "Expected more than 30 nodes, got {}",
            compiled.num_nodes
        );
    }

    #[test]
    fn info_sets_mapped_correctly() {
        let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);

        // Check that expected info sets exist
        assert!(compiled.info_set_to_idx.contains_key("K"));
        assert!(compiled.info_set_to_idx.contains_key("Q"));
        assert!(compiled.info_set_to_idx.contains_key("J"));
        assert!(compiled.info_set_to_idx.contains_key("Kc"));
        assert!(compiled.info_set_to_idx.contains_key("Kb"));
        assert!(compiled.info_set_to_idx.contains_key("Kcb"));
    }

    #[test]
    fn terminal_utilities_set() {
        let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);

        // Get terminal mask as data
        let terminal_data: Vec<bool> = compiled.terminal_mask.to_data().to_vec().unwrap();
        let num_terminals = terminal_data.iter().filter(|&&t| t).count();

        // Should have terminal nodes
        assert!(num_terminals > 0, "Expected terminal nodes");

        // Terminal utilities should be non-zero for terminal nodes
        let utils_data: Vec<f32> = compiled.terminal_utils.to_data().to_vec().unwrap();

        // At least some utilities should be non-zero
        let has_nonzero = utils_data.iter().any(|&u| u.abs() > 0.0);
        assert!(has_nonzero, "Expected some non-zero utilities");
    }

    #[test]
    fn action_mask_valid() {
        let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);

        let mask_data: Vec<bool> = compiled
            .action_mask
            .clone()
            .reshape([compiled.num_nodes * compiled.max_actions])
            .to_data()
            .to_vec()
            .unwrap();
        // Use i64 since that's what burn uses internally for Int tensors
        let num_actions_data: Vec<i64> = compiled.node_num_actions.to_data().to_vec().unwrap();

        // For each non-terminal node, verify action mask matches num_actions
        for (node_idx, &num_actions) in num_actions_data.iter().enumerate() {
            if num_actions > 0 {
                let start = node_idx * compiled.max_actions;
                let valid_count = (0..compiled.max_actions)
                    .filter(|&a| mask_data[start + a])
                    .count();
                assert_eq!(
                    valid_count, num_actions as usize,
                    "Node {} has {} actions but mask shows {}",
                    node_idx, num_actions, valid_count
                );
            }
        }
    }
}
