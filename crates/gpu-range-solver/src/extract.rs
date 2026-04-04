//! Extracts PostFlopGame tree topology into flat arrays for GPU tensor creation.

use range_solver::action_tree::{PLAYER_FOLD_FLAG, PLAYER_MASK};
use range_solver::card::card_pair_to_index;
use range_solver::interface::GameNode;
use range_solver::PostFlopGame;
use std::collections::VecDeque;

/// Classification of a node in the game tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// Fold terminal -- one player folded.
    Fold { folded_player: usize },
    /// Showdown terminal -- both players reached showdown.
    Showdown,
    /// Chance node -- deals a card (turn or river).
    Chance,
    /// Decision node -- a player acts.
    Player { player: usize },
}

/// Flat topology extracted from a `PostFlopGame` tree.
#[derive(Debug)]
pub struct TreeTopology {
    pub num_nodes: usize,
    pub num_edges: usize,

    // Per-node data (indexed by node_id 0..num_nodes)
    pub node_type: Vec<NodeType>,
    pub node_depth: Vec<usize>,
    pub node_arena_index: Vec<usize>,
    pub node_amount: Vec<i32>,
    pub node_turn: Vec<u8>,
    pub node_river: Vec<u8>,
    pub node_num_actions: Vec<usize>,

    // Per-edge data (indexed by edge_id 0..num_edges)
    pub edge_parent: Vec<usize>,
    pub edge_child: Vec<usize>,
    pub edge_action_index: Vec<usize>,

    // Level groupings
    pub max_depth: usize,
    pub level_nodes: Vec<Vec<usize>>,
    pub level_edges: Vec<Vec<usize>>,

    // Classified node lists
    pub fold_nodes: Vec<usize>,
    pub showdown_nodes: Vec<usize>,
    pub chance_nodes: Vec<usize>,
    pub player_nodes: [Vec<usize>; 2],
}

/// Extracts flat topology from a `PostFlopGame` via BFS.
pub fn extract_topology(game: &PostFlopGame) -> TreeTopology {
    let arena_len = game.num_nodes();
    let mut arena_to_node: Vec<Option<usize>> = vec![None; arena_len];

    let mut node_type = Vec::new();
    let mut node_depth = Vec::new();
    let mut node_arena_index = Vec::new();
    let mut node_amount = Vec::new();
    let mut node_turn = Vec::new();
    let mut node_river = Vec::new();
    let mut node_num_actions = Vec::new();

    let mut edge_parent = Vec::new();
    let mut edge_child = Vec::new();
    let mut edge_action_index = Vec::new();

    let mut fold_nodes = Vec::new();
    let mut showdown_nodes = Vec::new();
    let mut chance_nodes = Vec::new();
    let mut player_nodes: [Vec<usize>; 2] = [Vec::new(), Vec::new()];

    // BFS queue: (arena_index, depth)
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    queue.push_back((0, 0));

    let mut max_depth: usize = 0;

    while let Some((arena_idx, depth)) = queue.pop_front() {
        if arena_to_node[arena_idx].is_some() {
            continue;
        }

        let node_id = node_type.len();
        arena_to_node[arena_idx] = Some(node_id);

        let (ntype, n_actions, amount, turn, river) = {
            let node = game.node_at(arena_idx);
            let flags = node.player_flags();
            let is_terminal = node.is_terminal();
            let is_chance = node.is_chance();
            let is_fold = flags & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG;

            let ntype = if is_fold {
                let folded = (flags & PLAYER_MASK) as usize;
                NodeType::Fold { folded_player: folded }
            } else if is_terminal {
                NodeType::Showdown
            } else if is_chance {
                NodeType::Chance
            } else {
                let p = (flags & PLAYER_MASK) as usize;
                NodeType::Player { player: p }
            };

            let n_actions = if is_terminal { 0 } else { node.num_actions() };
            (ntype, n_actions, node.bet_amount(), node.turn_card(), node.river_card())
        };

        node_type.push(ntype);
        node_depth.push(depth);
        node_arena_index.push(arena_idx);
        node_amount.push(amount);
        node_turn.push(turn);
        node_river.push(river);
        node_num_actions.push(n_actions);
        max_depth = max_depth.max(depth);

        match ntype {
            NodeType::Fold { .. } => fold_nodes.push(node_id),
            NodeType::Showdown => showdown_nodes.push(node_id),
            NodeType::Chance => chance_nodes.push(node_id),
            NodeType::Player { player } => player_nodes[player].push(node_id),
        }

        // Enqueue children and create edge placeholders
        if n_actions > 0 {
            let children = game.child_indices(arena_idx);
            for (action_idx, &child_arena_idx) in children.iter().enumerate() {
                edge_parent.push(node_id);
                edge_child.push(0); // placeholder
                edge_action_index.push(action_idx);
                queue.push_back((child_arena_idx, depth + 1));
            }
        }
    }

    // Fix edge_child: map arena indices to node_ids
    let mut edge_idx = 0;
    for node_id in 0..node_type.len() {
        let n_actions = node_num_actions[node_id];
        if n_actions > 0 {
            let arena_idx = node_arena_index[node_id];
            let children = game.child_indices(arena_idx);
            for &child_arena in &children {
                edge_child[edge_idx] = arena_to_node[child_arena]
                    .expect("child must have been visited");
                edge_idx += 1;
            }
        }
    }

    let num_nodes = node_type.len();
    let num_edges = edge_parent.len();

    // Build level groupings
    let mut level_nodes = vec![Vec::new(); max_depth + 1];
    for (id, &d) in node_depth.iter().enumerate() {
        level_nodes[d].push(id);
    }

    let mut level_edges = vec![Vec::new(); max_depth + 1];
    for (eid, &child) in edge_child.iter().enumerate() {
        level_edges[node_depth[child]].push(eid);
    }

    TreeTopology {
        num_nodes,
        num_edges,
        node_type,
        node_depth,
        node_arena_index,
        node_amount,
        node_turn,
        node_river,
        node_num_actions,
        edge_parent,
        edge_child,
        edge_action_index,
        max_depth,
        level_nodes,
        level_edges,
        fold_nodes,
        showdown_nodes,
        chance_nodes,
        player_nodes,
    }
}

/// Pre-computed fold terminal data.
#[derive(Debug, Clone)]
pub struct FoldData {
    pub folded_player: usize,
    pub amount_win: f64,
    pub amount_lose: f64,
}

/// Pre-computed showdown outcome matrix for one terminal node.
#[derive(Debug, Clone)]
pub struct ShowdownData {
    pub num_player_hands: [usize; 2],
    /// From OOP perspective: `[h_oop * num_ip + h_ip]` = +1 (OOP wins), -1 (IP wins), 0 (tie/blocked).
    pub outcome_matrix_p0: Vec<f64>,
    pub amount_win: f64,
    pub amount_tie: f64,
    pub amount_lose: f64,
}

/// All terminal evaluation data needed by the GPU solver.
#[derive(Debug)]
pub struct TerminalData {
    /// Fold data for each fold node (same order as `topo.fold_nodes`).
    pub fold_payoffs: Vec<FoldData>,
    /// Showdown data for each showdown node (same order as `topo.showdown_nodes`).
    pub showdown_outcomes: Vec<ShowdownData>,
    /// Per-player: hand index to (card1, card2).
    pub hand_cards: [Vec<(u8, u8)>; 2],
    /// Per-player: `same_hand_index[i]` = opponent index or `u16::MAX`.
    pub same_hand_index: [Vec<u16>; 2],
    /// Number of valid card combinations (for payoff normalization).
    pub num_combinations: f64,
}

/// Extracts terminal evaluation data from the game.
pub fn extract_terminal_data(game: &PostFlopGame, topo: &TreeTopology) -> TerminalData {
    let tree_config = game.tree_config();
    let num_combinations = game.num_combinations_f64();

    let hand_cards: [Vec<(u8, u8)>; 2] = [
        game.private_cards(0).to_vec(),
        game.private_cards(1).to_vec(),
    ];
    let same_hand_index: [Vec<u16>; 2] = [
        game.same_hand_index(0).to_vec(),
        game.same_hand_index(1).to_vec(),
    ];

    // Fold payoffs
    let fold_payoffs: Vec<FoldData> = topo
        .fold_nodes
        .iter()
        .map(|&node_id| {
            let amount = topo.node_amount[node_id];
            let pot = (tree_config.starting_pot + 2 * amount) as f64;
            let half_pot = 0.5 * pot;
            let rake = (pot * tree_config.rake_rate).min(tree_config.rake_cap);
            let folded_player = match topo.node_type[node_id] {
                NodeType::Fold { folded_player } => folded_player,
                _ => unreachable!(),
            };
            FoldData {
                folded_player,
                amount_win: (half_pot - rake) / num_combinations,
                amount_lose: -half_pot / num_combinations,
            }
        })
        .collect();

    // Showdown outcomes
    let hand_strength = game.hand_strength();
    let oop_cards = game.private_cards(0);
    let ip_cards = game.private_cards(1);
    let num_oop = oop_cards.len();
    let num_ip = ip_cards.len();
    let flop = game.card_config().flop;

    let showdown_outcomes: Vec<ShowdownData> = topo
        .showdown_nodes
        .iter()
        .map(|&node_id| {
            let amount = topo.node_amount[node_id];
            let pot = (tree_config.starting_pot + 2 * amount) as f64;
            let half_pot = 0.5 * pot;
            let rake = (pot * tree_config.rake_rate).min(tree_config.rake_cap);

            let turn = topo.node_turn[node_id];
            let river = topo.node_river[node_id];
            let pair_idx = card_pair_to_index(turn, river);

            let strengths = &hand_strength[pair_idx];

            // Build hand-index-to-strength lookup
            let mut oop_strength = vec![0u16; num_oop];
            let mut ip_strength = vec![0u16; num_ip];
            for item in &strengths[0] {
                if (item.index as usize) < num_oop {
                    oop_strength[item.index as usize] = item.strength;
                }
            }
            for item in &strengths[1] {
                if (item.index as usize) < num_ip {
                    ip_strength[item.index as usize] = item.strength;
                }
            }

            // Board card mask
            let board_mask: u64 = (1u64 << flop[0])
                | (1u64 << flop[1])
                | (1u64 << flop[2])
                | (1u64 << turn)
                | (1u64 << river);

            // Build outcome matrix (OOP perspective)
            let mut outcome = vec![0.0f64; num_oop * num_ip];
            for h_oop in 0..num_oop {
                let (c1, c2) = oop_cards[h_oop];
                let oop_mask = (1u64 << c1) | (1u64 << c2);
                if oop_mask & board_mask != 0 {
                    continue;
                }
                let s_oop = oop_strength[h_oop];
                if s_oop == 0 {
                    continue;
                }

                for h_ip in 0..num_ip {
                    let (c3, c4) = ip_cards[h_ip];
                    let ip_mask = (1u64 << c3) | (1u64 << c4);
                    if ip_mask & board_mask != 0 {
                        continue;
                    }
                    if oop_mask & ip_mask != 0 {
                        continue;
                    }
                    let s_ip = ip_strength[h_ip];
                    if s_ip == 0 {
                        continue;
                    }

                    outcome[h_oop * num_ip + h_ip] = if s_oop > s_ip {
                        1.0
                    } else if s_oop < s_ip {
                        -1.0
                    } else {
                        0.0
                    };
                }
            }

            ShowdownData {
                num_player_hands: [num_oop, num_ip],
                outcome_matrix_p0: outcome,
                amount_win: (half_pot - rake) / num_combinations,
                amount_tie: -0.5 * rake / num_combinations,
                amount_lose: -half_pot / num_combinations,
            }
        })
        .collect();

    TerminalData {
        fold_payoffs,
        showdown_outcomes,
        hand_cards,
        same_hand_index,
        num_combinations,
    }
}

/// Result of decomposing a turn tree at chance nodes.
#[derive(Debug)]
pub struct ChanceDecomposition {
    /// Turn subtree: root down to chance nodes (chance nodes become terminal leaves).
    pub turn_topo: TreeTopology,
    /// River subtree template: topology below one chance child (same for all runouts).
    pub river_topo: TreeTopology,
    /// Node IDs in turn_topo that are leaves (where chance nodes were).
    pub turn_leaf_node_ids: Vec<usize>,
    /// For each chance node (in turn_topo order), the list of child node IDs in the
    /// original full topology. Each child corresponds to one river card runout.
    pub chance_children: Vec<Vec<usize>>,
    /// River card dealt for each runout (same order as chance_children[0]).
    pub river_cards: Vec<u8>,
}

/// Decompose a tree at chance nodes for batched river solving.
///
/// Requires that all chance nodes have the same number of children (same river cards)
/// and that all river subtrees have identical topology (same bet structure).
pub fn decompose_at_chance(topo: &TreeTopology) -> ChanceDecomposition {
    assert!(
        !topo.chance_nodes.is_empty(),
        "decompose_at_chance requires chance nodes"
    );

    // Collect the children of the first chance node to define the river subtree template.
    let first_chance = topo.chance_nodes[0];
    let first_chance_child_edges: Vec<usize> = (0..topo.num_edges)
        .filter(|&e| topo.edge_parent[e] == first_chance)
        .collect();
    let num_runouts = first_chance_child_edges.len();
    assert!(num_runouts > 0, "chance node must have children");

    // River cards: read from the first chance node's children
    let first_child = topo.edge_child[first_chance_child_edges[0]];
    let river_cards: Vec<u8> = first_chance_child_edges
        .iter()
        .map(|&e| topo.node_river[topo.edge_child[e]])
        .collect();

    // --- Build river subtree template from the first chance child ---
    // BFS from first_child, remapping node IDs starting at 0
    let mut river_old_to_new: Vec<Option<usize>> = vec![None; topo.num_nodes];
    let mut river_queue: VecDeque<usize> = VecDeque::new();
    river_queue.push_back(first_child);

    let mut r_node_type = Vec::new();
    let mut r_node_depth = Vec::new();
    let mut r_node_arena_index = Vec::new();
    let mut r_node_amount = Vec::new();
    let mut r_node_turn = Vec::new();
    let mut r_node_river = Vec::new();
    let mut r_node_num_actions = Vec::new();
    let mut r_edge_parent = Vec::new();
    let mut r_edge_child = Vec::new();
    let mut r_edge_action_index = Vec::new();
    let mut r_fold_nodes = Vec::new();
    let mut r_showdown_nodes = Vec::new();
    let mut r_chance_nodes = Vec::new();
    let mut r_player_nodes: [Vec<usize>; 2] = [Vec::new(), Vec::new()];

    let base_depth = topo.node_depth[first_child];

    while let Some(old_id) = river_queue.pop_front() {
        if river_old_to_new[old_id].is_some() {
            continue;
        }
        let new_id = r_node_type.len();
        river_old_to_new[old_id] = Some(new_id);

        let nt = topo.node_type[old_id];
        r_node_type.push(nt);
        r_node_depth.push(topo.node_depth[old_id] - base_depth);
        r_node_arena_index.push(topo.node_arena_index[old_id]);
        r_node_amount.push(topo.node_amount[old_id]);
        r_node_turn.push(topo.node_turn[old_id]);
        r_node_river.push(topo.node_river[old_id]);
        r_node_num_actions.push(topo.node_num_actions[old_id]);

        match nt {
            NodeType::Fold { .. } => r_fold_nodes.push(new_id),
            NodeType::Showdown => r_showdown_nodes.push(new_id),
            NodeType::Chance => r_chance_nodes.push(new_id),
            NodeType::Player { player } => r_player_nodes[player].push(new_id),
        }

        // Enqueue children
        for e in 0..topo.num_edges {
            if topo.edge_parent[e] == old_id {
                r_edge_parent.push(new_id);
                r_edge_child.push(0); // placeholder
                r_edge_action_index.push(topo.edge_action_index[e]);
                river_queue.push_back(topo.edge_child[e]);
            }
        }
    }

    // Fix river edge_child
    let mut ei = 0;
    for new_id in 0..r_node_type.len() {
        let n_actions = r_node_num_actions[new_id];
        if n_actions > 0 {
            // Find the original old_id for this new_id
            let old_id = river_old_to_new
                .iter()
                .position(|x| *x == Some(new_id))
                .unwrap();
            for e in 0..topo.num_edges {
                if topo.edge_parent[e] == old_id {
                    r_edge_child[ei] = river_old_to_new[topo.edge_child[e]]
                        .expect("river child must be mapped");
                    ei += 1;
                }
            }
        }
    }

    let r_num_nodes = r_node_type.len();
    let r_num_edges = r_edge_parent.len();
    let r_max_depth = r_node_depth.iter().copied().max().unwrap_or(0);

    let mut r_level_nodes = vec![Vec::new(); r_max_depth + 1];
    for (id, &d) in r_node_depth.iter().enumerate() {
        r_level_nodes[d].push(id);
    }
    let mut r_level_edges = vec![Vec::new(); r_max_depth + 1];
    for (eid, &child) in r_edge_child.iter().enumerate() {
        r_level_edges[r_node_depth[child]].push(eid);
    }

    let river_topo = TreeTopology {
        num_nodes: r_num_nodes,
        num_edges: r_num_edges,
        node_type: r_node_type,
        node_depth: r_node_depth,
        node_arena_index: r_node_arena_index,
        node_amount: r_node_amount,
        node_turn: r_node_turn,
        node_river: r_node_river,
        node_num_actions: r_node_num_actions,
        edge_parent: r_edge_parent,
        edge_child: r_edge_child,
        edge_action_index: r_edge_action_index,
        max_depth: r_max_depth,
        level_nodes: r_level_nodes,
        level_edges: r_level_edges,
        fold_nodes: r_fold_nodes,
        showdown_nodes: r_showdown_nodes,
        chance_nodes: r_chance_nodes,
        player_nodes: r_player_nodes,
    };

    // --- Build turn subtree: root to chance nodes, chance nodes become leaves ---
    // Collect all nodes above chance (inclusive). Chance nodes become showdown-like leaves.
    let mut chance_set: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for &cn in &topo.chance_nodes {
        chance_set.insert(cn);
    }

    let mut turn_old_to_new: Vec<Option<usize>> = vec![None; topo.num_nodes];
    let mut turn_queue: VecDeque<usize> = VecDeque::new();
    turn_queue.push_back(0);

    let mut t_node_type = Vec::new();
    let mut t_node_depth = Vec::new();
    let mut t_node_arena_index = Vec::new();
    let mut t_node_amount = Vec::new();
    let mut t_node_turn = Vec::new();
    let mut t_node_river = Vec::new();
    let mut t_node_num_actions = Vec::new();
    let mut t_edge_parent = Vec::new();
    let mut t_edge_child = Vec::new();
    let mut t_edge_action_index = Vec::new();
    let mut t_fold_nodes = Vec::new();
    let mut t_showdown_nodes = Vec::new();
    let mut t_chance_nodes_out = Vec::new();
    let mut t_player_nodes: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    let mut turn_leaf_node_ids = Vec::new();

    while let Some(old_id) = turn_queue.pop_front() {
        if turn_old_to_new[old_id].is_some() {
            continue;
        }
        let new_id = t_node_type.len();
        turn_old_to_new[old_id] = Some(new_id);

        let is_chance = chance_set.contains(&old_id);

        if is_chance {
            // Chance node becomes a showdown-like leaf (CFVs will be injected externally)
            t_node_type.push(NodeType::Showdown);
            t_node_depth.push(topo.node_depth[old_id]);
            t_node_arena_index.push(topo.node_arena_index[old_id]);
            t_node_amount.push(topo.node_amount[old_id]);
            t_node_turn.push(topo.node_turn[old_id]);
            t_node_river.push(topo.node_river[old_id]);
            t_node_num_actions.push(0); // leaf: no actions
            t_showdown_nodes.push(new_id);
            turn_leaf_node_ids.push(new_id);
        } else {
            let nt = topo.node_type[old_id];
            let n_actions = topo.node_num_actions[old_id];
            t_node_type.push(nt);
            t_node_depth.push(topo.node_depth[old_id]);
            t_node_arena_index.push(topo.node_arena_index[old_id]);
            t_node_amount.push(topo.node_amount[old_id]);
            t_node_turn.push(topo.node_turn[old_id]);
            t_node_river.push(topo.node_river[old_id]);
            t_node_num_actions.push(n_actions);

            match nt {
                NodeType::Fold { .. } => t_fold_nodes.push(new_id),
                NodeType::Showdown => t_showdown_nodes.push(new_id),
                NodeType::Chance => t_chance_nodes_out.push(new_id),
                NodeType::Player { player } => t_player_nodes[player].push(new_id),
            }

            // Enqueue children (but don't descend past chance nodes)
            for e in 0..topo.num_edges {
                if topo.edge_parent[e] == old_id {
                    let child = topo.edge_child[e];
                    t_edge_parent.push(new_id);
                    t_edge_child.push(0); // placeholder
                    t_edge_action_index.push(topo.edge_action_index[e]);
                    turn_queue.push_back(child);
                }
            }
        }
    }

    // Fix turn edge_child
    let mut tei = 0;
    for new_id in 0..t_node_type.len() {
        let n_actions = t_node_num_actions[new_id];
        if n_actions > 0 {
            let old_id = turn_old_to_new
                .iter()
                .position(|x| *x == Some(new_id))
                .unwrap();
            for e in 0..topo.num_edges {
                if topo.edge_parent[e] == old_id {
                    t_edge_child[tei] = turn_old_to_new[topo.edge_child[e]]
                        .expect("turn child must be mapped");
                    tei += 1;
                }
            }
        }
    }

    let t_num_nodes = t_node_type.len();
    let t_num_edges = t_edge_parent.len();
    let t_max_depth = t_node_depth.iter().copied().max().unwrap_or(0);

    let mut t_level_nodes = vec![Vec::new(); t_max_depth + 1];
    for (id, &d) in t_node_depth.iter().enumerate() {
        t_level_nodes[d].push(id);
    }
    let mut t_level_edges = vec![Vec::new(); t_max_depth + 1];
    for (eid, &child) in t_edge_child.iter().enumerate() {
        t_level_edges[t_node_depth[child]].push(eid);
    }

    let turn_topo = TreeTopology {
        num_nodes: t_num_nodes,
        num_edges: t_num_edges,
        node_type: t_node_type,
        node_depth: t_node_depth,
        node_arena_index: t_node_arena_index,
        node_amount: t_node_amount,
        node_turn: t_node_turn,
        node_river: t_node_river,
        node_num_actions: t_node_num_actions,
        edge_parent: t_edge_parent,
        edge_child: t_edge_child,
        edge_action_index: t_edge_action_index,
        max_depth: t_max_depth,
        level_nodes: t_level_nodes,
        level_edges: t_level_edges,
        fold_nodes: t_fold_nodes,
        showdown_nodes: t_showdown_nodes,
        chance_nodes: t_chance_nodes_out,
        player_nodes: t_player_nodes,
    };

    // Collect chance children from full topology
    let mut chance_children = Vec::new();
    for &cn in &topo.chance_nodes {
        let children: Vec<usize> = (0..topo.num_edges)
            .filter(|&e| topo.edge_parent[e] == cn)
            .map(|e| topo.edge_child[e])
            .collect();
        chance_children.push(children);
    }

    ChanceDecomposition {
        turn_topo,
        river_topo,
        turn_leaf_node_ids,
        chance_children,
        river_cards,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{card_from_str, flop_from_str, CardConfig};
    use range_solver::PostFlopGame;

    fn make_river_game() -> PostFlopGame {
        let oop_range = "AA,KK,QQ,AKs".parse().unwrap();
        let ip_range = "QQ-JJ,AQs,AJs".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let sizes = BetSizeOptions::try_from(("100%", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);
        game
    }

    #[test]
    fn extract_river_topology_has_nodes_and_edges() {
        let game = make_river_game();
        let topo = extract_topology(&game);

        assert!(topo.num_nodes > 0, "must have at least one node");
        assert!(topo.num_edges > 0, "must have at least one edge");
        assert!(topo.num_nodes > 1, "must have more than just root");
    }

    #[test]
    fn extract_river_topology_edges_have_valid_indices() {
        let game = make_river_game();
        let topo = extract_topology(&game);

        for e in 0..topo.num_edges {
            assert!(
                topo.edge_parent[e] < topo.num_nodes,
                "edge {e} parent {} out of range (num_nodes={})",
                topo.edge_parent[e],
                topo.num_nodes
            );
            assert!(
                topo.edge_child[e] < topo.num_nodes,
                "edge {e} child {} out of range (num_nodes={})",
                topo.edge_child[e],
                topo.num_nodes
            );
        }
    }

    #[test]
    fn extract_river_topology_level_grouping_covers_all_nodes() {
        let game = make_river_game();
        let topo = extract_topology(&game);

        let total_in_levels: usize = topo.level_nodes.iter().map(|l| l.len()).sum();
        assert_eq!(total_in_levels, topo.num_nodes);
    }

    #[test]
    fn extract_river_topology_root_at_depth_zero() {
        let game = make_river_game();
        let topo = extract_topology(&game);

        assert!(
            topo.level_nodes[0].contains(&0),
            "root (node 0) must be at depth 0"
        );
    }

    #[test]
    fn extract_river_topology_no_chance_nodes() {
        let game = make_river_game();
        let topo = extract_topology(&game);

        assert!(
            topo.chance_nodes.is_empty(),
            "river game should have no chance nodes"
        );
    }

    #[test]
    fn extract_river_topology_fold_and_showdown_nodes_exist() {
        let game = make_river_game();
        let topo = extract_topology(&game);

        assert!(!topo.fold_nodes.is_empty(), "should have fold nodes");
        assert!(!topo.showdown_nodes.is_empty(), "should have showdown nodes");
    }

    #[test]
    fn extract_river_topology_classified_nodes_cover_all() {
        let game = make_river_game();
        let topo = extract_topology(&game);

        let classified_count = topo.fold_nodes.len()
            + topo.showdown_nodes.len()
            + topo.chance_nodes.len()
            + topo.player_nodes[0].len()
            + topo.player_nodes[1].len();
        assert_eq!(
            classified_count, topo.num_nodes,
            "classified node counts must sum to total"
        );
    }

    #[test]
    fn extract_river_topology_terminal_nodes_have_zero_actions() {
        let game = make_river_game();
        let topo = extract_topology(&game);

        for &node_id in &topo.fold_nodes {
            assert_eq!(
                topo.node_num_actions[node_id], 0,
                "fold node {node_id} should have 0 actions"
            );
        }
        for &node_id in &topo.showdown_nodes {
            assert_eq!(
                topo.node_num_actions[node_id], 0,
                "showdown node {node_id} should have 0 actions"
            );
        }
    }

    #[test]
    fn extract_river_topology_player_nodes_have_positive_actions() {
        let game = make_river_game();
        let topo = extract_topology(&game);

        for p in 0..2 {
            for &node_id in &topo.player_nodes[p] {
                assert!(
                    topo.node_num_actions[node_id] > 0,
                    "player node {node_id} (player {p}) should have >0 actions"
                );
            }
        }
    }

    #[test]
    fn extract_river_topology_edge_count_matches_actions() {
        let game = make_river_game();
        let topo = extract_topology(&game);

        let total_actions: usize = topo.node_num_actions.iter().sum();
        assert_eq!(
            topo.num_edges, total_actions,
            "total edges should equal sum of node action counts"
        );
    }

    #[test]
    fn extract_river_topology_level_edges_cover_all_edges() {
        let game = make_river_game();
        let topo = extract_topology(&game);

        let total_in_level_edges: usize = topo.level_edges.iter().map(|l| l.len()).sum();
        assert_eq!(
            total_in_level_edges, topo.num_edges,
            "level_edges must cover all edges"
        );
    }

    // -- Terminal data tests --

    #[test]
    fn extract_terminal_data_fold_count_matches() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);

        assert_eq!(
            term.fold_payoffs.len(),
            topo.fold_nodes.len(),
            "fold_payoffs count must match fold_nodes count"
        );
    }

    #[test]
    fn extract_terminal_data_showdown_count_matches() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);

        assert_eq!(
            term.showdown_outcomes.len(),
            topo.showdown_nodes.len(),
            "showdown_outcomes count must match showdown_nodes count"
        );
    }

    #[test]
    fn extract_terminal_data_fold_amounts_have_correct_signs() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);

        for (i, fp) in term.fold_payoffs.iter().enumerate() {
            assert!(
                fp.amount_win > 0.0,
                "fold_payoffs[{i}].amount_win should be positive, got {}",
                fp.amount_win
            );
            assert!(
                fp.amount_lose < 0.0,
                "fold_payoffs[{i}].amount_lose should be negative, got {}",
                fp.amount_lose
            );
        }
    }

    #[test]
    fn extract_terminal_data_fold_win_greater_than_lose_abs() {
        // With no rake, |win| == |lose|. With rake, |win| < |lose| but win > 0
        // and lose < 0. In both cases win + lose <= 0.
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);

        for (i, fp) in term.fold_payoffs.iter().enumerate() {
            // No rake => amount_win == -amount_lose
            // With rake => amount_win < -amount_lose
            assert!(
                fp.amount_win <= -fp.amount_lose + 1e-10,
                "fold_payoffs[{i}]: win ({}) should be <= -lose ({})",
                fp.amount_win,
                -fp.amount_lose
            );
        }
    }

    #[test]
    fn extract_terminal_data_hand_cards_match_game() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);

        let num_oop = game.private_cards(0).len();
        let num_ip = game.private_cards(1).len();

        assert_eq!(term.hand_cards[0].len(), num_oop);
        assert_eq!(term.hand_cards[1].len(), num_ip);
    }

    #[test]
    fn extract_terminal_data_same_hand_index_match_game() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);

        assert_eq!(
            term.same_hand_index[0].len(),
            game.same_hand_index(0).len()
        );
        assert_eq!(
            term.same_hand_index[1].len(),
            game.same_hand_index(1).len()
        );
    }

    #[test]
    fn extract_terminal_data_showdown_outcome_matrix_size() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);

        let num_oop = game.private_cards(0).len();
        let num_ip = game.private_cards(1).len();

        for (i, sd) in term.showdown_outcomes.iter().enumerate() {
            assert_eq!(sd.num_player_hands[0], num_oop);
            assert_eq!(sd.num_player_hands[1], num_ip);
            assert_eq!(
                sd.outcome_matrix_p0.len(),
                num_oop * num_ip,
                "showdown_outcomes[{i}] matrix size mismatch"
            );
        }
    }

    #[test]
    fn extract_terminal_data_showdown_amounts_have_correct_signs() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);

        for (i, sd) in term.showdown_outcomes.iter().enumerate() {
            assert!(
                sd.amount_win > 0.0,
                "showdown_outcomes[{i}].amount_win should be positive"
            );
            assert!(
                sd.amount_lose < 0.0,
                "showdown_outcomes[{i}].amount_lose should be negative"
            );
        }
    }

    #[test]
    fn extract_terminal_data_showdown_outcome_values_in_range() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);

        for (i, sd) in term.showdown_outcomes.iter().enumerate() {
            for (j, &val) in sd.outcome_matrix_p0.iter().enumerate() {
                assert!(
                    val == -1.0 || val == 0.0 || val == 1.0,
                    "showdown_outcomes[{i}].outcome_matrix_p0[{j}] = {val}, expected -1, 0, or 1"
                );
            }
        }
    }

    #[test]
    fn extract_terminal_data_showdown_has_nonzero_outcomes() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);

        // At least one showdown should have some non-zero outcomes
        let has_nonzero = term.showdown_outcomes.iter().any(|sd| {
            sd.outcome_matrix_p0.iter().any(|&v| v != 0.0)
        });
        assert!(has_nonzero, "at least one showdown should have non-zero outcomes");
    }

    #[test]
    fn extract_terminal_data_num_combinations_positive() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);

        assert!(
            term.num_combinations > 0.0,
            "num_combinations should be positive"
        );
    }

    fn make_turn_game() -> PostFlopGame {
        let oop_range = "AA".parse().unwrap();
        let ip_range = "KK".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: range_solver::card::NOT_DEALT,
        };
        let sizes = BetSizeOptions::try_from(("100%", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 100,
            turn_bet_sizes: [sizes.clone(), sizes.clone()],
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);
        game
    }

    #[test]
    fn turn_topology_has_chance_nodes() {
        let game = make_turn_game();
        let topo = extract_topology(&game);
        assert!(
            !topo.chance_nodes.is_empty(),
            "turn game should have chance nodes"
        );
    }

    #[test]
    fn decompose_at_chance_splits_turn_tree() {
        let game = make_turn_game();
        let topo = extract_topology(&game);

        let decomp = decompose_at_chance(&topo);

        // Turn subtree: root to chance nodes (chance nodes become leaves)
        assert!(decomp.turn_topo.num_nodes > 0, "turn subtree must have nodes");
        assert!(
            decomp.turn_topo.chance_nodes.is_empty(),
            "turn subtree should have no chance nodes (they become terminals)"
        );

        // River subtree: below chance (same topology for all runouts)
        assert!(decomp.river_topo.num_nodes > 0, "river subtree must have nodes");
        assert!(
            decomp.river_topo.chance_nodes.is_empty(),
            "river subtree should have no chance nodes"
        );

        // Chance edges mapping
        assert!(
            !decomp.chance_children.is_empty(),
            "must have chance children (river cards)"
        );
    }

    #[test]
    fn decompose_river_subtree_has_terminals() {
        let game = make_turn_game();
        let topo = extract_topology(&game);

        let decomp = decompose_at_chance(&topo);

        assert!(
            !decomp.river_topo.fold_nodes.is_empty() || !decomp.river_topo.showdown_nodes.is_empty(),
            "river subtree must have terminal nodes"
        );
    }

    #[test]
    fn decompose_turn_subtree_chance_nodes_become_leaves() {
        let game = make_turn_game();
        let topo = extract_topology(&game);

        let decomp = decompose_at_chance(&topo);

        // The turn subtree's leaf node IDs should correspond to old chance node positions
        assert!(
            !decomp.turn_leaf_node_ids.is_empty(),
            "turn subtree must have leaf nodes where chance nodes were"
        );

        // Turn leaves should have 0 actions (they are terminal in the turn subtree)
        for &leaf_id in &decomp.turn_leaf_node_ids {
            assert_eq!(
                decomp.turn_topo.node_num_actions[leaf_id], 0,
                "turn leaf node {} should have 0 actions",
                leaf_id
            );
        }
    }
}
