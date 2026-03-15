use range_solver::interface::Game;
use range_solver::{Action, BoardState, PostFlopGame};

/// Node type discriminant for GPU.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeType {
    DecisionOop = 0,
    DecisionIp = 1,
    TerminalFold = 2,
    TerminalShowdown = 3,
    DepthBoundary = 4,
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
    /// Card pairs for OOP hands: (card1, card2) for each hand index.
    pub cards_oop: Vec<(u8, u8)>,
    /// Card pairs for IP hands: (card1, card2) for each hand index.
    pub cards_ip: Vec<(u8, u8)>,
    /// Valid matchup matrix when OOP is the traverser.
    /// `valid_matchups_oop[oop_hand * num_hands + ip_hand] = 1.0` if no card conflict, `0.0` if blocked.
    /// Size: `num_hands * num_hands`.
    pub valid_matchups_oop: Vec<f32>,
    /// Valid matchup matrix when IP is the traverser.
    /// `valid_matchups_ip[ip_hand * num_hands + oop_hand] = 1.0` if no card conflict, `0.0` if blocked.
    /// Size: `num_hands * num_hands`.
    pub valid_matchups_ip: Vec<f32>,
    /// For each OOP hand index, the IP hand index holding the same two cards,
    /// or `u32::MAX` if no such hand exists. Used for inclusion-exclusion
    /// correction in the O(n) fold evaluation kernel.
    /// Length = `num_hands` (padded with `u32::MAX`).
    pub same_hand_index_oop: Vec<u32>,
    /// For each IP hand index, the OOP hand index holding the same two cards,
    /// or `u32::MAX` if no such hand exists.
    /// Length = `num_hands` (padded with `u32::MAX`).
    pub same_hand_index_ip: Vec<u32>,
    /// Node IDs of depth-boundary nodes (terminals that need neural leaf evaluation).
    pub boundary_indices: Vec<u32>,
    /// Pot size at each depth-boundary node.
    pub boundary_pots: Vec<f32>,
    /// Effective stack remaining at each depth-boundary node.
    pub boundary_stacks: Vec<f32>,
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

    /// Whether the node is a terminal (fold, showdown, or depth boundary).
    pub fn is_terminal(&self, node: usize) -> bool {
        matches!(
            self.node_types[node],
            NodeType::TerminalFold | NodeType::TerminalShowdown | NodeType::DepthBoundary
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
        let effective_stack = game.tree_config().effective_stack;
        let num_combinations = game.num_combinations();
        let rake_rate = game.tree_config().rake_rate;
        let rake_cap = game.tree_config().rake_cap;
        let has_depth_limit = game.tree_config().depth_limit.is_some();
        let initial_state = game.tree_config().initial_state.clone();

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

        // Depth boundary tracking
        let mut boundary_indices: Vec<u32> = Vec::new();
        let mut boundary_pots: Vec<f32> = Vec::new();
        let mut boundary_stacks: Vec<f32> = Vec::new();

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
                         Got chance node at history {history:?}, \
                         is_terminal={is_terminal}",
                    );
                }

                // Compute pot using matched amounts only.
                // The CPU evaluator uses `pot = starting_pot + 2 * node.amount`
                // where `node.amount` is the per-player matched bet. This equals
                // `starting_pot + 2 * min(bet[0], bet[1])`, because unmatched
                // bets (e.g., a bet followed by a fold) are excluded from the pot
                // calculation used for payoff evaluation.
                let bet_amounts = game.total_bet_amount();
                let matched = bet_amounts[0].min(bet_amounts[1]);
                let pot = (starting_pot + 2 * matched) as f32;

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
                    } else if has_depth_limit && initial_state != BoardState::River {
                        // Non-fold terminal in a depth-limited game (not starting
                        // at the river) is a depth boundary — the tree stops before
                        // the next street and leaf values come from a neural net.
                        node_types.push(NodeType::DepthBoundary);

                        // Compute effective stack remaining at this boundary.
                        let bet_amounts_boundary = game.total_bet_amount();
                        let max_bet = bet_amounts_boundary[0].max(bet_amounts_boundary[1]);
                        let stack_remaining = (effective_stack - max_bet) as f32;

                        boundary_indices.push(flat_id);
                        boundary_pots.push(pot);
                        boundary_stacks.push(stack_remaining);

                        // Fill placeholder payoff entries so terminal_indices
                        // stays aligned with fold_payoffs / showdown_equity_ids.
                        showdown_equity_ids_vec.push(u32::MAX);
                        fold_payoffs_vec.push(Vec::new());
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
        // Only needed when there are showdown terminals (not for depth-limited turn games).
        let card_cfg = game.card_config();
        let has_showdowns = node_types.iter().any(|t| *t == NodeType::TerminalShowdown);
        let (hand_strengths_oop, hand_strengths_ip) = if has_showdowns {
            let board = [
                card_cfg.flop[0],
                card_cfg.flop[1],
                card_cfg.flop[2],
                card_cfg.turn,
                card_cfg.river,
            ];

            let strengths_oop: Vec<u32> = game
                .private_cards(0)
                .iter()
                .map(|&hole| {
                    range_solver::card::evaluate_hand_strength(&board, hole) as u32
                })
                .collect();

            let strengths_ip: Vec<u32> = game
                .private_cards(1)
                .iter()
                .map(|&hole| {
                    range_solver::card::evaluate_hand_strength(&board, hole) as u32
                })
                .collect();

            (strengths_oop, strengths_ip)
        } else {
            // No showdowns: hand strengths not needed, use empty vecs
            (vec![0u32; num_hands_oop], vec![0u32; num_hands_ip])
        };

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

        // Extract card pairs for each player's hands.
        let cards_oop: Vec<(u8, u8)> = game
            .private_cards(0)
            .iter()
            .map(|&(c1, c2)| (c1, c2))
            .collect();
        let cards_ip: Vec<(u8, u8)> = game
            .private_cards(1)
            .iter()
            .map(|&(c1, c2)| (c1, c2))
            .collect();

        // Precompute valid matchup matrices for card blocking.
        // When OOP is traverser, opponents are IP hands.
        let mut valid_matchups_oop = vec![0.0f32; num_hands * num_hands];
        for oop_h in 0..num_hands_oop {
            let (oop_c1, oop_c2) = cards_oop[oop_h];
            for ip_h in 0..num_hands_ip {
                let (ip_c1, ip_c2) = cards_ip[ip_h];
                let conflicts = oop_c1 == ip_c1
                    || oop_c1 == ip_c2
                    || oop_c2 == ip_c1
                    || oop_c2 == ip_c2;
                if !conflicts {
                    valid_matchups_oop[oop_h * num_hands + ip_h] = 1.0;
                }
            }
        }

        // When IP is traverser, opponents are OOP hands.
        let mut valid_matchups_ip = vec![0.0f32; num_hands * num_hands];
        for ip_h in 0..num_hands_ip {
            let (ip_c1, ip_c2) = cards_ip[ip_h];
            for oop_h in 0..num_hands_oop {
                let (oop_c1, oop_c2) = cards_oop[oop_h];
                let conflicts = ip_c1 == oop_c1
                    || ip_c1 == oop_c2
                    || ip_c2 == oop_c1
                    || ip_c2 == oop_c2;
                if !conflicts {
                    valid_matchups_ip[ip_h * num_hands + oop_h] = 1.0;
                }
            }
        }

        // Compute same_hand_index: for each hand of one player, find the
        // opponent hand holding the same two cards (if any).
        let mut same_hand_index_oop = vec![u32::MAX; num_hands];
        for (oop_h, &(oop_c1, oop_c2)) in cards_oop.iter().enumerate() {
            for (ip_h, &(ip_c1, ip_c2)) in cards_ip.iter().enumerate() {
                if oop_c1 == ip_c1 && oop_c2 == ip_c2 {
                    same_hand_index_oop[oop_h] = ip_h as u32;
                    break;
                }
            }
        }
        let mut same_hand_index_ip = vec![u32::MAX; num_hands];
        for (ip_h, &(ip_c1, ip_c2)) in cards_ip.iter().enumerate() {
            for (oop_h, &(oop_c1, oop_c2)) in cards_oop.iter().enumerate() {
                if ip_c1 == oop_c1 && ip_c2 == oop_c2 {
                    same_hand_index_ip[ip_h] = oop_h as u32;
                    break;
                }
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
            cards_oop,
            cards_ip,
            valid_matchups_oop,
            valid_matchups_ip,
            same_hand_index_oop,
            same_hand_index_ip,
            boundary_indices,
            boundary_pots,
            boundary_stacks,
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
            // Test cards: no conflicts between any hands
            cards_oop: vec![(0, 1), (2, 3)],
            cards_ip: vec![(4, 5), (6, 7)],
            // All matchups valid (no card conflicts in test tree)
            valid_matchups_oop: vec![1.0, 1.0, 1.0, 1.0],
            valid_matchups_ip: vec![1.0, 1.0, 1.0, 1.0],
            // No same-hand overlaps in test tree
            same_hand_index_oop: vec![u32::MAX, u32::MAX],
            same_hand_index_ip: vec![u32::MAX, u32::MAX],
            // No depth boundaries in test tree
            boundary_indices: vec![],
            boundary_pots: vec![],
            boundary_stacks: vec![],
        }
    }
}

/// Build a turn `PostFlopGame` with `depth_limit: Some(0)` so that the tree
/// has fold terminals and depth-boundary leaves (no chance nodes, no showdown).
///
/// `depth_limit: Some(0)` blocks ALL street transitions, so check-check or
/// bet-call on the turn produces a depth boundary instead of dealing a river.
///
/// Uses uniform ranges (all 1326 combos at weight 1.0) and the provided
/// bet-size configuration for both players.
pub fn build_turn_game(
    flop: [u8; 3],
    turn: u8,
    pot: i32,
    stack: i32,
    bet_sizes: &range_solver::bet_size::BetSizeOptions,
) -> PostFlopGame {
    use range_solver::{ActionTree, CardConfig, TreeConfig};

    let card_config = CardConfig {
        range: [range_solver::range::Range::ones(), range_solver::range::Range::ones()],
        flop,
        turn,
        river: range_solver::card::NOT_DEALT,
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: pot,
        effective_stack: stack,
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        depth_limit: Some(0),
        ..Default::default()
    };

    let tree = ActionTree::new(tree_config).expect("Failed to build turn action tree");
    let mut game = PostFlopGame::with_config(card_config, tree)
        .expect("Failed to build turn PostFlopGame");
    game.allocate_memory(false);
    game
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

    #[test]
    fn test_turn_tree_with_depth_boundaries() {
        use range_solver::bet_size::BetSizeOptions;

        // Build a turn game with depth_limit=1.
        // Board: Qs Jh 2c 8d => flop=[46,37,8], turn=29
        let flop = range_solver::card::flop_from_str("Qs Jh 2c").unwrap();
        let turn = range_solver::card::card_from_str("8d").unwrap();
        let bet_sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();

        let mut game = build_turn_game(flop, turn, 100, 100, &bet_sizes);
        let flat = FlatTree::from_postflop_game(&mut game);

        // Basic sanity
        assert!(flat.num_nodes() > 0, "tree should have nodes");
        assert!(flat.num_levels() > 0, "tree should have levels");
        assert!(flat.num_infosets > 0, "tree should have infosets");

        // Count terminal types
        let num_fold = flat.node_types.iter().filter(|t| **t == NodeType::TerminalFold).count();
        let num_showdown = flat.node_types.iter().filter(|t| **t == NodeType::TerminalShowdown).count();
        let num_boundary = flat.node_types.iter().filter(|t| **t == NodeType::DepthBoundary).count();

        // With depth_limit=1 on a turn game:
        // - NO showdown terminals (can't reach river)
        // - NO chance nodes (depth boundary replaces them)
        // - SOME fold terminals
        // - SOME depth boundaries
        assert!(num_fold > 0, "should have fold terminals, got 0");
        assert_eq!(num_showdown, 0, "should have no showdown terminals, got {num_showdown}");
        assert!(num_boundary > 0, "should have depth boundaries, got 0");

        // boundary_indices should be populated and match count
        assert_eq!(flat.boundary_indices.len(), num_boundary);
        assert_eq!(flat.boundary_pots.len(), num_boundary);
        assert_eq!(flat.boundary_stacks.len(), num_boundary);

        // Verify all boundary indices point to DepthBoundary nodes
        for &idx in &flat.boundary_indices {
            assert_eq!(
                flat.node_types[idx as usize],
                NodeType::DepthBoundary,
                "boundary_indices[{idx}] should be DepthBoundary"
            );
        }

        // Boundary pots should be >= starting pot
        for &pot in &flat.boundary_pots {
            assert!(pot >= 100.0, "boundary pot {pot} should be >= starting pot 100");
        }

        // Boundary stacks should be > 0 and <= effective stack
        for &stack in &flat.boundary_stacks {
            assert!(stack >= 0.0, "boundary stack {stack} should be >= 0");
            assert!(stack <= 100.0, "boundary stack {stack} should be <= effective stack 100");
        }

        // DepthBoundary should be recognized as terminal
        for &idx in &flat.boundary_indices {
            assert!(flat.is_terminal(idx as usize), "depth boundary should be terminal");
        }

        // Root should be a decision node
        assert!(
            flat.node_types[0] == NodeType::DecisionOop || flat.node_types[0] == NodeType::DecisionIp,
            "root should be a decision node"
        );

        // Both players should have hands (some blocked by board cards)
        assert!(flat.num_hands > 0, "should have some hands");
        assert!(flat.num_hands <= 1326, "at most 1326 hands");
        // With a 4-card board, expect C(48,2) = 1128 valid combos
        assert_eq!(flat.num_hands, 1128);
    }
}
