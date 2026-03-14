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
    /// Precomputed hand-vs-hand equity matrices for showdown nodes.
    /// Each inner vec has length `num_hands * num_hands`.
    pub equity_tables: Vec<Vec<f32>>,
    /// Precomputed per-hand fold payoffs for fold terminal nodes.
    /// Each inner vec has length `num_hands`.
    pub fold_payoffs: Vec<Vec<f32>>,
    /// Number of canonical hand combos per player.
    pub num_hands: usize,
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
        // CSR offsets for children (length = num_nodes + 1)
        let child_offsets = vec![0, 2, 2, 4, 4, 6, 6, 6];
        // Flat children array
        let children = vec![1, 2, 3, 4, 5, 6];
        let parent_nodes = vec![u32::MAX, 0, 0, 2, 2, 4, 4];
        let parent_actions = vec![u32::MAX, 0, 1, 0, 1, 0, 1];
        // 4 BFS levels: [0], [1,2], [3,4], [5,6]
        let level_starts = vec![0, 1, 3, 5, 7];
        // Infoset ids: decision nodes get sequential ids, terminals get MAX
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
        assert_eq!(tree.level_node_count(0), 1); // root only
        assert_eq!(tree.level_node_count(1), 2); // fold + IP decision
        assert_eq!(tree.level_node_count(2), 2); // showdown + OOP decision
        assert_eq!(tree.level_node_count(3), 2); // fold + showdown

        // Player at each decision node
        assert_eq!(tree.player(0), 0); // OOP
        assert_eq!(tree.player(2), 1); // IP
        assert_eq!(tree.player(4), 0); // OOP

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
