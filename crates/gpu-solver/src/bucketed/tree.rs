//! BucketedTree: a flat game tree with bucket-level terminal data.
//!
//! Reuses the same BFS walk topology as `FlatTree::from_postflop_game()`,
//! but instead of per-hand card data, stores per-terminal bucket equity
//! tables and fold half-pots.
//!
//! The key insight: in the bucketed solver, the "hand" dimension is replaced
//! by "bucket" dimension. Showdown payoffs are computed via pre-built bucket
//! equity tables, and fold payoffs use a simple half-pot scalar (no card
//! blocking in bucket space).

use range_solver::{Action, BoardState, PostFlopGame};

use crate::tree::NodeType;
use poker_solver_core::blueprint_v2::bucket_file::BucketFile;

use super::equity::{compute_bucket_equity_table, BucketedBoardCache};

/// A flat game tree with bucket-level terminal data for GPU solving.
///
/// Topology arrays are identical to `FlatTree` (same BFS walk). The
/// difference is in terminal data:
/// - Showdown: stored as `num_buckets x num_buckets` equity tables
/// - Fold: stored as half-pot scalars and folded-player indicators
/// - No per-hand cards, card blocking, or hand strengths
#[derive(Debug)]
pub struct BucketedTree {
    // ---- Topology (same as FlatTree) ----

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
    /// Indices into `node_types` that are terminal nodes (fold + showdown + boundary).
    pub terminal_indices: Vec<u32>,

    // ---- Depth boundary data ----

    /// Node IDs of depth-boundary nodes (terminals that need neural leaf evaluation).
    pub boundary_indices: Vec<u32>,
    /// Pot size at each depth-boundary node.
    pub boundary_pots: Vec<f32>,
    /// Effective stack remaining at each depth-boundary node.
    pub boundary_stacks: Vec<f32>,

    // ---- Bucketed data ----

    /// Number of buckets (e.g. 500 for river).
    pub num_buckets: usize,

    /// Per-showdown-terminal equity tables.
    /// `showdown_equity_tables[ordinal]` is a flat `[num_buckets * num_buckets]` f32 array
    /// where `E[i * num_buckets + j]` is the average payoff for bucket i vs bucket j.
    pub showdown_equity_tables: Vec<Vec<f32>>,
    /// Half-pot for each showdown terminal (pot / 2.0).
    pub showdown_half_pots: Vec<f32>,

    /// Half-pot for each fold terminal (pot / 2.0).
    pub fold_half_pots: Vec<f32>,
    /// Which player folded at each fold terminal (0=OOP, 1=IP).
    pub fold_players: Vec<u32>,

    /// Mapping from terminal_indices ordinal to showdown ordinal.
    /// `showdown_ordinals[i]` is the index into `showdown_equity_tables` for
    /// the i-th entry in `terminal_indices`, or `u32::MAX` if not a showdown.
    pub showdown_ordinals: Vec<u32>,
    /// Mapping from terminal_indices ordinal to fold ordinal.
    /// `fold_ordinals[i]` is the index into `fold_half_pots` for the i-th
    /// entry in `terminal_indices`, or `u32::MAX` if not a fold.
    pub fold_ordinals: Vec<u32>,
}

impl BucketedTree {
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

    /// Which player acts at this node (0=OOP, 1=IP, u8::MAX for terminals).
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

    /// Build a `BucketedTree` from an allocated `PostFlopGame` and a bucket file.
    ///
    /// The BFS walk is identical to `FlatTree::from_postflop_game()` for
    /// topology. Terminal handling differs:
    /// - Fold: stores `half_pot = pot / 2.0` and the folded player
    /// - Showdown: computes bucket equity table for the board
    /// - DepthBoundary: stores boundary pot/stack (for turn/flop, not river)
    ///
    /// For river games, all showdown terminals share the same board, so a
    /// single equity table is computed and shared.
    ///
    /// # Arguments
    /// - `game`: an allocated `PostFlopGame` (must have `allocate_memory` called)
    /// - `bucket_file`: loaded bucket file for the river street
    /// - `board_cache`: pre-built cache for board index lookups
    /// - `num_buckets`: number of buckets in the bucket file
    pub fn from_postflop_game(
        game: &mut PostFlopGame,
        bucket_file: &BucketFile,
        board_cache: &BucketedBoardCache,
        num_buckets: usize,
    ) -> Self {
        let starting_pot = game.tree_config().starting_pot;
        let effective_stack = game.tree_config().effective_stack;
        let has_depth_limit = game.tree_config().depth_limit.is_some();
        let initial_state = game.tree_config().initial_state.clone();

        // Extract board cards for equity table computation
        let card_cfg = game.card_config();
        let board_u8: [u8; 5] = [
            card_cfg.flop[0],
            card_cfg.flop[1],
            card_cfg.flop[2],
            card_cfg.turn,
            card_cfg.river,
        ];

        // Pre-compute the board index (only needed for river games with showdowns)
        let board_idx = board_cache.find_board_index(&board_u8);

        // For river games, compute the equity table once (all showdowns share the same board)
        let shared_equity_table: Option<Vec<f32>> =
            if initial_state == BoardState::River || !has_depth_limit {
                board_idx.map(|idx| {
                    compute_bucket_equity_table(&board_u8, bucket_file, idx, num_buckets)
                })
            } else {
                None
            };

        // BFS entry
        struct BfsEntry {
            history: Vec<usize>,
            parent_flat_id: u32,
            action_from_parent: u32,
            reached_by_fold: bool,
        }

        let mut queue: Vec<BfsEntry> = Vec::new();

        // Topology arrays
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

        // Bucketed terminal data
        let mut showdown_equity_tables: Vec<Vec<f32>> = Vec::new();
        let mut showdown_half_pots: Vec<f32> = Vec::new();
        let mut fold_half_pots: Vec<f32> = Vec::new();
        let mut fold_players: Vec<u32> = Vec::new();

        // Per-terminal ordinal mapping
        let mut showdown_ordinals: Vec<u32> = Vec::new();
        let mut fold_ordinals: Vec<u32> = Vec::new();

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
            let level_end = queue.len();

            while head < level_end {
                let flat_id = head as u32;
                head += 1;

                let history = queue[flat_id as usize].history.clone();
                let parent_id = queue[flat_id as usize].parent_flat_id;
                let action_idx = queue[flat_id as usize].action_from_parent;
                let reached_by_fold = queue[flat_id as usize].reached_by_fold;

                game.apply_history(&history);

                let is_terminal = game.is_terminal_node();
                let is_chance = game.is_chance_node();

                if is_chance {
                    panic!(
                        "Chance nodes not supported in BucketedTree (river games only). \
                         Got chance node at history {history:?}",
                    );
                }

                // Compute pot
                let bet_amounts = game.total_bet_amount();
                let matched = bet_amounts[0].min(bet_amounts[1]);
                let pot = (starting_pot + 2 * matched) as f32;

                parent_nodes.push(parent_id);
                parent_actions.push(action_idx);
                pots.push(pot);

                if is_terminal {
                    if reached_by_fold {
                        node_types.push(NodeType::TerminalFold);

                        // Determine which player folded
                        let parent_history = &history[..history.len() - 1];
                        game.apply_history(parent_history);
                        let folded_player = game.current_player();

                        let half_pot = pot / 2.0;
                        let fold_ord = fold_half_pots.len() as u32;
                        fold_half_pots.push(half_pot);
                        fold_players.push(folded_player as u32);

                        fold_ordinals.push(fold_ord);
                        showdown_ordinals.push(u32::MAX);
                    } else if has_depth_limit && initial_state != BoardState::River {
                        node_types.push(NodeType::DepthBoundary);

                        let bet_amounts_boundary = game.total_bet_amount();
                        let max_bet = bet_amounts_boundary[0].max(bet_amounts_boundary[1]);
                        let stack_remaining = (effective_stack - max_bet) as f32;

                        boundary_indices.push(flat_id);
                        boundary_pots.push(pot);
                        boundary_stacks.push(stack_remaining);

                        fold_ordinals.push(u32::MAX);
                        showdown_ordinals.push(u32::MAX);
                    } else {
                        node_types.push(NodeType::TerminalShowdown);

                        let half_pot = pot / 2.0;
                        let sd_ord = showdown_equity_tables.len() as u32;

                        // Use the shared equity table (river: all showdowns share one board)
                        if let Some(ref eq_table) = shared_equity_table {
                            showdown_equity_tables.push(eq_table.clone());
                        } else {
                            // Shouldn't happen for river games, but handle gracefully
                            showdown_equity_tables
                                .push(vec![0.0f32; num_buckets * num_buckets]);
                        }
                        showdown_half_pots.push(half_pot);

                        showdown_ordinals.push(sd_ord);
                        fold_ordinals.push(u32::MAX);
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

            if queue.len() > level_end {
                level_starts.push(level_end as u32);
            }
        }

        // Final level boundary
        let total = node_types.len() as u32;
        if *level_starts.last().unwrap() != total {
            level_starts.push(total);
        }

        // Build CSR child structure (identical to FlatTree)
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

        // Reset game to root
        game.back_to_root();

        BucketedTree {
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
            boundary_indices,
            boundary_pots,
            boundary_stacks,
            num_buckets,
            showdown_equity_tables,
            showdown_half_pots,
            fold_half_pots,
            fold_players,
            showdown_ordinals,
            fold_ordinals,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::FlatTree;
    use poker_solver_core::blueprint_v2::bucket_file::{BucketFile, BucketFileHeader};
    use poker_solver_core::blueprint_v2::cluster_pipeline::canonical_key;
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::{ActionTree, CardConfig, TreeConfig};
    use std::path::Path;

    /// Build a river PostFlopGame for testing.
    fn build_river_game(
        flop: [u8; 3],
        turn: u8,
        river: u8,
        pot: i32,
        stack: i32,
        bet_sizes: &BetSizeOptions,
    ) -> PostFlopGame {
        let card_config = CardConfig {
            range: [
                range_solver::range::Range::ones(),
                range_solver::range::Range::ones(),
            ],
            flop,
            turn,
            river,
        };

        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: pot,
            effective_stack: stack,
            river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            ..Default::default()
        };

        let tree = ActionTree::new(tree_config).expect("action tree");
        let mut game = PostFlopGame::with_config(card_config, tree).expect("postflop game");
        game.allocate_memory(false);
        game
    }

    /// Helper: build a synthetic bucket file for the given board.
    fn make_synthetic_bucket_file(board_u8: &[u8; 5], num_buckets: u16) -> BucketFile {
        use super::super::equity::u8_to_rs_card;
        let rs_cards: Vec<rs_poker::core::Card> =
            board_u8.iter().map(|&c| u8_to_rs_card(c)).collect();
        let packed = canonical_key(&rs_cards);

        let combos_per_board = 1326u16;
        let bucket_data: Vec<u16> = (0..combos_per_board).map(|i| i % num_buckets).collect();

        BucketFile {
            header: BucketFileHeader {
                street: poker_solver_core::blueprint_v2::Street::River,
                bucket_count: num_buckets,
                board_count: 1,
                combos_per_board,
                version: 2,
            },
            boards: vec![packed],
            buckets: bucket_data,
        }
    }

    /// Test that BucketedTree topology matches FlatTree for a river game.
    #[test]
    fn test_bucketed_tree_topology_matches_flat_tree() {
        let flop = range_solver::card::flop_from_str("Qs Jh 2c").unwrap();
        let turn = range_solver::card::card_from_str("8d").unwrap();
        let river = range_solver::card::card_from_str("3s").unwrap();
        let bet_sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();

        let board_u8: [u8; 5] = [flop[0], flop[1], flop[2], turn, river];
        let num_buckets = 50u16; // small for testing

        let bf = make_synthetic_bucket_file(&board_u8, num_buckets);
        let cache = BucketedBoardCache::new(&bf);

        // Build both trees from the same game
        let mut game = build_river_game(flop, turn, river, 100, 100, &bet_sizes);
        let flat = FlatTree::from_postflop_game(&mut game);

        let mut game2 = build_river_game(flop, turn, river, 100, 100, &bet_sizes);
        let bucketed = BucketedTree::from_postflop_game(
            &mut game2,
            &bf,
            &cache,
            num_buckets as usize,
        );

        // Topology must match exactly
        assert_eq!(
            bucketed.num_nodes(),
            flat.num_nodes(),
            "node count mismatch"
        );
        assert_eq!(
            bucketed.num_infosets, flat.num_infosets,
            "infoset count mismatch"
        );
        assert_eq!(
            bucketed.num_levels(),
            flat.num_levels(),
            "level count mismatch"
        );
        assert_eq!(
            bucketed.node_types.len(),
            flat.node_types.len(),
            "node_types length mismatch"
        );

        // Node types must match
        for i in 0..bucketed.num_nodes() {
            assert_eq!(
                bucketed.node_types[i], flat.node_types[i],
                "node_types[{i}] mismatch"
            );
        }

        // Infoset IDs must match
        assert_eq!(bucketed.infoset_ids, flat.infoset_ids);

        // Infoset action counts must match
        assert_eq!(bucketed.infoset_num_actions, flat.infoset_num_actions);

        // CSR child structure must match
        assert_eq!(bucketed.child_offsets, flat.child_offsets);
        assert_eq!(bucketed.children, flat.children);

        // Parent structure must match
        assert_eq!(bucketed.parent_nodes, flat.parent_nodes);
        assert_eq!(bucketed.parent_actions, flat.parent_actions);

        // Level starts must match
        assert_eq!(bucketed.level_starts, flat.level_starts);

        // Terminal indices must match
        assert_eq!(bucketed.terminal_indices, flat.terminal_indices);

        // Pots must match
        assert_eq!(bucketed.pots, flat.pots);
    }

    /// Test bucketed terminal data.
    #[test]
    fn test_bucketed_tree_terminal_data() {
        let flop = range_solver::card::flop_from_str("Qs Jh 2c").unwrap();
        let turn = range_solver::card::card_from_str("8d").unwrap();
        let river = range_solver::card::card_from_str("3s").unwrap();
        let bet_sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();

        let board_u8: [u8; 5] = [flop[0], flop[1], flop[2], turn, river];
        let num_buckets = 50u16;

        let bf = make_synthetic_bucket_file(&board_u8, num_buckets);
        let cache = BucketedBoardCache::new(&bf);

        let mut game = build_river_game(flop, turn, river, 100, 100, &bet_sizes);
        let bucketed = BucketedTree::from_postflop_game(
            &mut game,
            &bf,
            &cache,
            num_buckets as usize,
        );

        assert_eq!(bucketed.num_buckets, num_buckets as usize);

        // Count terminal types
        let num_fold = bucketed
            .node_types
            .iter()
            .filter(|&&t| t == NodeType::TerminalFold)
            .count();
        let num_showdown = bucketed
            .node_types
            .iter()
            .filter(|&&t| t == NodeType::TerminalShowdown)
            .count();

        assert!(num_fold > 0, "should have fold terminals");
        assert!(num_showdown > 0, "should have showdown terminals");

        // Fold data
        assert_eq!(bucketed.fold_half_pots.len(), num_fold);
        assert_eq!(bucketed.fold_players.len(), num_fold);

        for &hp in &bucketed.fold_half_pots {
            assert!(hp > 0.0, "fold half_pot should be positive, got {hp}");
        }
        for &fp in &bucketed.fold_players {
            assert!(fp <= 1, "fold_player should be 0 or 1, got {fp}");
        }

        // Showdown data
        assert_eq!(bucketed.showdown_equity_tables.len(), num_showdown);
        assert_eq!(bucketed.showdown_half_pots.len(), num_showdown);

        for eq_table in &bucketed.showdown_equity_tables {
            assert_eq!(
                eq_table.len(),
                (num_buckets as usize) * (num_buckets as usize),
                "equity table should be num_buckets x num_buckets"
            );
        }

        for &hp in &bucketed.showdown_half_pots {
            assert!(hp > 0.0, "showdown half_pot should be positive, got {hp}");
        }

        // Ordinal mappings
        assert_eq!(
            bucketed.showdown_ordinals.len(),
            bucketed.terminal_indices.len()
        );
        assert_eq!(
            bucketed.fold_ordinals.len(),
            bucketed.terminal_indices.len()
        );

        // No depth boundaries for river games
        assert_eq!(bucketed.boundary_indices.len(), 0);
    }

    /// Test with real 500-bucket file if it exists.
    #[test]
    fn test_bucketed_tree_real_bucket_file() {
        let path = Path::new("../../local_data/clusters_500bkt_v3/river.buckets");
        if !path.exists() {
            eprintln!(
                "Skipping test: river.buckets not found at {}",
                path.display()
            );
            return;
        }

        let bf = BucketFile::load(path).expect("Failed to load river.buckets");
        let num_buckets = bf.header.bucket_count as usize;
        assert_eq!(num_buckets, 500);

        let cache = BucketedBoardCache::new(&bf);

        // Pick a board that exists in the bucket file
        let first_packed = bf.boards[0];
        let board_cards_rs = first_packed.to_cards(5);
        let board_u8: Vec<u8> = board_cards_rs
            .iter()
            .map(|c| {
                poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id(*c)
            })
            .collect();

        let flop = [board_u8[0], board_u8[1], board_u8[2]];
        let turn = board_u8[3];
        let river = board_u8[4];
        let bet_sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();

        let mut game = build_river_game(flop, turn, river, 100, 100, &bet_sizes);
        let bucketed = BucketedTree::from_postflop_game(
            &mut game,
            &bf,
            &cache,
            num_buckets,
        );

        // Also build FlatTree for topology comparison
        let mut game2 = build_river_game(flop, turn, river, 100, 100, &bet_sizes);
        let flat = FlatTree::from_postflop_game(&mut game2);

        assert_eq!(bucketed.num_nodes(), flat.num_nodes());
        assert_eq!(bucketed.num_infosets, flat.num_infosets);
        assert_eq!(bucketed.num_buckets, 500);

        let num_showdown = bucketed.showdown_equity_tables.len();
        assert!(num_showdown > 0, "should have showdown terminals");

        // Check equity table dimensions
        for eq in &bucketed.showdown_equity_tables {
            assert_eq!(eq.len(), 500 * 500);
        }

        eprintln!(
            "BucketedTree: {} nodes, {} infosets, {} showdowns, {} folds, {} buckets",
            bucketed.num_nodes(),
            bucketed.num_infosets,
            bucketed.showdown_equity_tables.len(),
            bucketed.fold_half_pots.len(),
            bucketed.num_buckets,
        );
    }
}
