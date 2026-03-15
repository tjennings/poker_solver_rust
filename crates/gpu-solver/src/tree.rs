use range_solver::interface::Game;
use range_solver::{Action, PostFlopGame};

/// Node type discriminant for GPU.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeType {
    DecisionOop = 0,
    DecisionIp = 1,
    TerminalFold = 2,
    TerminalShowdown = 3,
}

/// Flat level-order game tree for GPU upload.
/// All arrays indexed by node_id (BFS order).
#[derive(Debug)]
pub struct FlatTree {
    /// Per-node type tag.
    pub node_types: Vec<NodeType>,
    /// Per-node pot size (total chips in pot).
    pub pots: Vec<f32>,
    /// CSR-style child offsets: children of node `i` are
    /// `children[child_offsets[i]..child_offsets[i+1]]`.
    pub child_offsets: Vec<u32>,
    /// Flat child-node-id array (indexed via `child_offsets`).
    pub children: Vec<u32>,
    /// Parent node id for each node (`u32::MAX` for root).
    pub parent_nodes: Vec<u32>,
    /// Index of the action that led to this node from its parent.
    pub parent_actions: Vec<u32>,
    /// Level boundaries in the BFS order: nodes in level `l` are
    /// `level_starts[l]..level_starts[l+1]`.
    pub level_starts: Vec<u32>,
    /// Information-set id for each decision node (`u32::MAX` for terminals).
    pub infoset_ids: Vec<u32>,
    /// Number of actions at each information set (indexed by infoset id).
    pub infoset_num_actions: Vec<u32>,
    /// Total number of distinct information sets.
    pub num_infosets: usize,
    /// Indices into `node_types` that are terminal nodes.
    pub terminal_indices: Vec<u32>,
    /// For showdown terminals, an id into `equity_tables`.
    pub showdown_equity_ids: Vec<u32>,
    /// Precomputed showdown payoff constants for showdown terminal nodes.
    /// Each inner vec contains `[amount_win, amount_lose]` where amounts
    /// are per-combination payoffs (i.e. divided by num_combinations).
    pub equity_tables: Vec<Vec<f32>>,
    /// Precomputed fold payoff constants for fold terminal nodes.
    /// Each inner vec contains `[amount_win, amount_lose, folded_player]`.
    pub fold_payoffs: Vec<Vec<f32>>,
    /// Number of canonical hand combos (max of OOP and IP).
    pub num_hands: usize,
    /// Number of OOP hands.
    pub num_hands_oop: usize,
    /// Number of IP hands.
    pub num_hands_ip: usize,
    /// Hand strength for each OOP combo (higher = stronger).
    /// Length = `num_hands_oop`. Combos blocked by the board have strength 0.
    pub hand_strengths_oop: Vec<u32>,
    /// Hand strength for each IP combo (higher = stronger).
    /// Length = `num_hands_ip`. Combos blocked by the board have strength 0.
    pub hand_strengths_ip: Vec<u32>,
    /// Initial reach probabilities for OOP (range weights after card removal).
    /// Length = `num_hands` (padded with zeros if `num_hands_oop < num_hands`).
    pub initial_reach_oop: Vec<f32>,
    /// Initial reach probabilities for IP (range weights after card removal).
    /// Length = `num_hands` (padded with zeros if `num_hands_ip < num_hands`).
    pub initial_reach_ip: Vec<f32>,
}

impl FlatTree {
    /// Total number of nodes in the tree.
    pub fn num_nodes(&self) -> usize {
        self.node_types.len()
    }

    /// Number of BFS levels.
    pub fn num_levels(&self) -> usize {
        self.level_starts.len().saturating_sub(1)
    }

    /// Maximum number of actions across all infosets.
    pub fn max_actions(&self) -> usize {
        self.infoset_num_actions
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as usize
    }

    /// Number of children for a given node.
    pub fn num_children(&self, node: usize) -> usize {
        (self.child_offsets[node + 1] - self.child_offsets[node]) as usize
    }

    /// Number of nodes in a given BFS level.
    pub fn level_node_count(&self, level: usize) -> usize {
        (self.level_starts[level + 1] - self.level_starts[level]) as usize
    }

    /// The player to act at a decision node (0=OOP, 1=IP).
    /// Returns `u8::MAX` for terminal nodes.
    pub fn player(&self, node: usize) -> u8 {
        match self.node_types[node] {
            NodeType::DecisionOop => 0,
            NodeType::DecisionIp => 1,
            _ => u8::MAX,
        }
    }

    /// Whether the node is a terminal (fold or showdown).
    pub fn is_terminal(&self, node: usize) -> bool {
        matches!(
            self.node_types[node],
            NodeType::TerminalFold | NodeType::TerminalShowdown
        )
    }

    // ------------------------------------------------------------------
    // Builder: from a range-solver PostFlopGame
    // ------------------------------------------------------------------

    /// Build a `FlatTree` from an allocated `PostFlopGame`.
    ///
    /// The game must have been initialized with `with_config` and had memory
    /// allocated via `allocate_memory`. Only river (single-board) games are
    /// supported in Phase 1; chance nodes cause a panic.
    ///
    /// The builder performs a BFS traversal using the interpreter API
    /// (`apply_history`, `available_actions`, `is_terminal_node`, etc.)
    /// to extract pot sizes and action information.
    pub fn from_postflop_game(game: &mut PostFlopGame) -> Self {
        let num_hands_oop = game.num_private_hands(0);
        let num_hands_ip = game.num_private_hands(1);
        let num_hands = num_hands_oop.max(num_hands_ip);
        let starting_pot = game.tree_config().starting_pot;
        let num_combinations = game.num_combinations();
        let rake_rate = game.tree_config().rake_rate;
        let rake_cap = game.tree_config().rake_cap;

        // BFS entry: history path from root + metadata.
        struct BfsEntry {
            history: Vec<usize>,
            parent_flat_id: u32,
            action_from_parent: u32,
            /// Whether this node was reached by a Fold action.
            reached_by_fold: bool,
        }

        let mut queue: Vec<BfsEntry> = Vec::new();

        // Output arrays
        let mut node_types: Vec<NodeType> = Vec::new();
        let mut pots: Vec<f32> = Vec::new();
        let mut parent_nodes: Vec<u32> = Vec::new();
        let mut parent_actions: Vec<u32> = Vec::new();
        let mut level_starts: Vec<u32> = Vec::new();
        let mut infoset_ids: Vec<u32> = Vec::new();
        let mut terminal_indices: Vec<u32> = Vec::new();
        let mut node_num_actions: Vec<usize> = Vec::new();

        // Infoset tracking
        let mut infoset_num_actions_vec: Vec<u32> = Vec::new();
        let mut next_infoset_id: u32 = 0;

        // Terminal payoff tracking
        let mut showdown_equity_ids_vec: Vec<u32> = Vec::new();
        let mut equity_tables_vec: Vec<Vec<f32>> = Vec::new();
        let mut fold_payoffs_vec: Vec<Vec<f32>> = Vec::new();

        // Seed BFS with root
        queue.push(BfsEntry {
            history: Vec::new(),
            parent_flat_id: u32::MAX,
            action_from_parent: u32::MAX,
            reached_by_fold: false,
        });

        let mut head = 0usize;
        level_starts.push(0);

        while head < queue.len() {
            // Process one BFS level: all nodes from head..queue.len()
            let level_end = queue.len();

            while head < level_end {
                let flat_id = head as u32;
                head += 1;

                // Navigate to this node. We borrow entry data before mutating game.
                let history = queue[flat_id as usize].history.clone();
                let parent_id = queue[flat_id as usize].parent_flat_id;
                let action_idx = queue[flat_id as usize].action_from_parent;
                let reached_by_fold = queue[flat_id as usize].reached_by_fold;

                game.apply_history(&history);

                let is_terminal = game.is_terminal_node();
                let is_chance = game.is_chance_node();

                if is_chance {
                    panic!(
                        "Chance nodes not supported in Phase 1 (river games only). \
                         Got chance node at history {history:?}",
                    );
                }

                // Compute pot
                let bet_amounts = game.total_bet_amount();
                let pot = (starting_pot + bet_amounts[0] + bet_amounts[1]) as f32;

                parent_nodes.push(parent_id);
                parent_actions.push(action_idx);
                pots.push(pot);

                if is_terminal {
                    if reached_by_fold {
                        node_types.push(NodeType::TerminalFold);

                        // Determine which player folded. The fold action was
                        // taken by the player who was acting at the parent node.
                        // Navigate to parent to find out.
                        let parent_history = &history[..history.len() - 1];
                        game.apply_history(parent_history);
                        let folded_player = game.current_player();

                        let pot_f64 = pot as f64;
                        let half_pot = 0.5 * pot_f64;
                        let amount_win = half_pot / num_combinations;
                        let amount_lose = -half_pot / num_combinations;

                        fold_payoffs_vec.push(vec![
                            amount_win as f32,
                            amount_lose as f32,
                            folded_player as f32,
                        ]);
                        showdown_equity_ids_vec.push(u32::MAX);
                    } else {
                        node_types.push(NodeType::TerminalShowdown);

                        let pot_f64 = pot as f64;
                        let half_pot = 0.5 * pot_f64;
                        let rake = (pot_f64 * rake_rate).min(rake_cap);
                        let amount_win = (half_pot - rake) / num_combinations;
                        let amount_lose = -half_pot / num_combinations;

                        let eq_id = equity_tables_vec.len() as u32;
                        showdown_equity_ids_vec.push(eq_id);
                        equity_tables_vec.push(vec![amount_win as f32, amount_lose as f32]);
                        fold_payoffs_vec.push(Vec::new());
                    }

                    terminal_indices.push(flat_id);
                    infoset_ids.push(u32::MAX);
                    node_num_actions.push(0);
                } else {
                    // Decision node
                    let player = game.current_player();
                    let actions = game.available_actions();
                    let n_actions = actions.len();

                    match player {
                        0 => node_types.push(NodeType::DecisionOop),
                        1 => node_types.push(NodeType::DecisionIp),
                        _ => panic!("Unexpected player {player}"),
                    }

                    let iset_id = next_infoset_id;
                    next_infoset_id += 1;
                    infoset_ids.push(iset_id);
                    infoset_num_actions_vec.push(n_actions as u32);
                    node_num_actions.push(n_actions);

                    // Enqueue children
                    for (ai, action) in actions.iter().enumerate() {
                        let mut child_history = history.clone();
                        child_history.push(ai);
                        queue.push(BfsEntry {
                            history: child_history,
                            parent_flat_id: flat_id,
                            action_from_parent: ai as u32,
                            reached_by_fold: *action == Action::Fold,
                        });
                    }
                }
            }

            // Record level boundary
            if queue.len() > level_end {
                level_starts.push(level_end as u32);
            }
        }
        // Final level boundary
        let total = node_types.len() as u32;
        if *level_starts.last().unwrap() != total {
            level_starts.push(total);
        }

        // Build CSR child structure.
        // In BFS order, the children of node i are contiguous and appear
        // right after all prior nodes' children. We know node_num_actions[i].
        let num_total_nodes = node_types.len();
        let mut child_offsets = Vec::with_capacity(num_total_nodes + 1);
        let mut children_vec: Vec<u32> = Vec::new();
        let mut offset = 0u32;
        let mut next_child_flat_id = 1u32;

        for i in 0..num_total_nodes {
            child_offsets.push(offset);
            let n = node_num_actions[i];
            for j in 0..n {
                children_vec.push(next_child_flat_id + j as u32);
            }
            offset += n as u32;
            next_child_flat_id += n as u32;
        }
        child_offsets.push(offset);

        // Compute per-combo hand strengths for showdown evaluation.
        let card_cfg = game.card_config();
        let board = [
            card_cfg.flop[0],
            card_cfg.flop[1],
            card_cfg.flop[2],
            card_cfg.turn,
            card_cfg.river,
        ];

        let hand_strengths_oop: Vec<u32> = game
            .private_cards(0)
            .iter()
            .map(|&hole| {
                range_solver::card::evaluate_hand_strength(&board, hole) as u32
            })
            .collect();

        let hand_strengths_ip: Vec<u32> = game
            .private_cards(1)
            .iter()
            .map(|&hole| {
                range_solver::card::evaluate_hand_strength(&board, hole) as u32
            })
            .collect();

        // Extract initial reach weights (range weights after card removal).
        let weights_oop = game.initial_weights(0);
        let weights_ip = game.initial_weights(1);
        let mut initial_reach_oop = vec![0.0f32; num_hands];
        let mut initial_reach_ip = vec![0.0f32; num_hands];
        for (i, &w) in weights_oop.iter().enumerate() {
            if i < num_hands {
                initial_reach_oop[i] = w;
            }
        }
        for (i, &w) in weights_ip.iter().enumerate() {
            if i < num_hands {
                initial_reach_ip[i] = w;
            }
        }

        // Reset game to root
        game.back_to_root();

        FlatTree {
            node_types,
            pots,
            child_offsets,
            children: children_vec,
            parent_nodes,
            parent_actions,
            level_starts,
            infoset_ids,
            infoset_num_actions: infoset_num_actions_vec,
            num_infosets: next_infoset_id as usize,
            terminal_indices,
            showdown_equity_ids: showdown_equity_ids_vec,
            equity_tables: equity_tables_vec,
            fold_payoffs: fold_payoffs_vec,
            num_hands,
            num_hands_oop,
            num_hands_ip,
            hand_strengths_oop,
            hand_strengths_ip,
            initial_reach_oop,
            initial_reach_ip,
        }
    }

    /// Build a tiny test tree for unit tests.
    ///
    /// Tree structure:
    /// ```text
    ///        0 (OOP, pot=100)
    ///       / \
    ///      1   2
    ///   fold  (IP, pot=100)
    ///         / \
    ///        3   4
    ///     show  (OOP, pot=150)
    ///           / \
    ///          5   6
    ///       fold  show
    /// ```
    pub fn new_test_tree() -> Self {
        let node_types = vec![
            NodeType::DecisionOop,      // 0: root (OOP to act)
            NodeType::TerminalFold,     // 1: fold after root
            NodeType::DecisionIp,       // 2: IP decision
            NodeType::TerminalShowdown, // 3: check-check showdown
            NodeType::DecisionOop,      // 4: OOP facing bet
            NodeType::TerminalFold,     // 5: fold to bet
            NodeType::TerminalShowdown, // 6: call showdown
        ];
        let pots = vec![100.0, 100.0, 100.0, 100.0, 150.0, 150.0, 200.0];
        let child_offsets = vec![0, 2, 2, 4, 4, 6, 6, 6];
        let children = vec![1, 2, 3, 4, 5, 6];
        let parent_nodes = vec![u32::MAX, 0, 0, 2, 2, 4, 4];
        let parent_actions = vec![u32::MAX, 0, 1, 0, 1, 0, 1];
        let level_starts = vec![0, 1, 3, 5, 7];
        let infoset_ids = vec![0, u32::MAX, 1, u32::MAX, 2, u32::MAX, u32::MAX];
        let infoset_num_actions = vec![2, 2, 2];
        let terminal_indices = vec![1, 3, 5, 6];

        FlatTree {
            node_types,
            pots,
            child_offsets,
            children,
            parent_nodes,
            parent_actions,
            level_starts,
            infoset_ids,
            infoset_num_actions,
            num_infosets: 3,
            terminal_indices,
            showdown_equity_ids: vec![],
            equity_tables: vec![],
            fold_payoffs: vec![],
            num_hands: 2,
            num_hands_oop: 2,
            num_hands_ip: 2,
            // Hand 0 is stronger than hand 1 for test purposes.
            hand_strengths_oop: vec![100, 50],
            hand_strengths_ip: vec![100, 50],
            initial_reach_oop: vec![1.0, 1.0],
            initial_reach_ip: vec![1.0, 1.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_tree_structure() {
        let tree = FlatTree::new_test_tree();

        // Basic counts
        assert_eq!(tree.num_nodes(), 7);
        assert_eq!(tree.num_levels(), 4);
        assert_eq!(tree.num_infosets, 3);

        // Children of root
        assert_eq!(tree.num_children(0), 2);
        assert_eq!(tree.children[0], 1);
        assert_eq!(tree.children[1], 2);

        // Level sizes
        assert_eq!(tree.level_node_count(0), 1);
        assert_eq!(tree.level_node_count(1), 2);
        assert_eq!(tree.level_node_count(2), 2);
        assert_eq!(tree.level_node_count(3), 2);

        // Player at each decision node
        assert_eq!(tree.player(0), 0);
        assert_eq!(tree.player(2), 1);
        assert_eq!(tree.player(4), 0);

        // Terminal checks
        assert!(!tree.is_terminal(0));
        assert!(tree.is_terminal(1));
        assert!(!tree.is_terminal(2));
        assert!(tree.is_terminal(3));
        assert!(!tree.is_terminal(4));
        assert!(tree.is_terminal(5));
        assert!(tree.is_terminal(6));

        // Terminal indices
        assert_eq!(tree.terminal_indices, vec![1, 3, 5, 6]);

        // Parent structure
        assert_eq!(tree.parent_nodes[0], u32::MAX);
        assert_eq!(tree.parent_nodes[1], 0);
        assert_eq!(tree.parent_nodes[2], 0);
        assert_eq!(tree.parent_nodes[3], 2);
        assert_eq!(tree.parent_nodes[4], 2);

        // Infoset assignments
        assert_eq!(tree.infoset_ids[0], 0);
        assert_eq!(tree.infoset_ids[1], u32::MAX);
        assert_eq!(tree.infoset_ids[2], 1);
    }
}
