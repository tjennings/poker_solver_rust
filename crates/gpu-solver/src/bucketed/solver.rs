//! BucketedGpuSolver: Supremus DCFR+ solver operating on bucket abstractions.
//!
//! This solver replaces per-hand-combo data with per-bucket data, dramatically
//! reducing dimensionality (e.g. 1326 hands -> 500 buckets). It reuses the
//! existing GPU kernels for regret matching, forward pass, backward CFV, and
//! strategy extraction (all parameterized by `num_hands`, which here equals
//! `num_buckets`). The terminal evaluation uses new bucketed kernels that
//! operate on precomputed equity tables instead of per-hand card blocking.

#[cfg(feature = "cuda")]
use crate::gpu::{GpuContext, GpuError};
#[cfg(feature = "cuda")]
use crate::tree::NodeType;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
use super::tree::BucketedTree;

/// Precomputed per-level data for GPU forward pass kernels.
#[cfg(feature = "cuda")]
struct LevelData {
    /// Global node IDs at this level.
    nodes: CudaSlice<u32>,
    /// Parent node IDs (per node at this level).
    parent_nodes: CudaSlice<u32>,
    /// Action index from parent (per node at this level).
    parent_actions: CudaSlice<u32>,
    /// Infoset ID of parent (per node at this level).
    parent_infosets: CudaSlice<u32>,
    /// Player who acts at parent: 0=OOP, 1=IP (per node at this level).
    parent_players: CudaSlice<u32>,
    /// Number of nodes at this level.
    num_nodes: u32,
}

/// Precomputed per-level decision nodes for backward CFV pass.
#[cfg(feature = "cuda")]
struct BackwardLevelData {
    /// Decision node IDs at this level.
    decision_nodes: CudaSlice<u32>,
    /// Player who acts at each decision node (0=OOP, 1=IP).
    decision_players: CudaSlice<u32>,
    /// Number of decision nodes at this level.
    num_decision_nodes: u32,
}

/// Result of running the bucketed GPU solver.
#[cfg(feature = "cuda")]
pub struct BucketedSolveResult {
    /// Final averaged strategy: `[num_infosets * max_actions * num_buckets]`.
    pub strategy: Vec<f32>,
    /// Number of iterations completed.
    pub iterations: u32,
}

/// GPU Supremus DCFR+ solver operating on bucket abstractions.
///
/// Instead of per-hand-combo arrays, all state arrays use `num_buckets` as
/// the "hand" dimension. Terminal evaluations use precomputed bucket equity
/// tables (showdown) and simple reach sums (fold) with no card blocking.
///
/// Existing kernels (regret_match, forward_pass, backward_cfv, extract_strategy,
/// zero_buffer, set_root_reach) work unchanged with `num_hands = num_buckets`.
/// New bucketed kernels handle showdown and fold evaluation.
#[cfg(feature = "cuda")]
pub struct BucketedGpuSolver<'a> {
    gpu: &'a GpuContext,

    // Solver state on GPU
    regrets: CudaSlice<f32>,           // [num_infosets * max_actions * num_buckets]
    strategy_sum: CudaSlice<f32>,
    current_strategy: CudaSlice<f32>,
    reach_oop: CudaSlice<f32>,         // [num_nodes * num_buckets]
    reach_ip: CudaSlice<f32>,
    cfvalues: CudaSlice<f32>,

    // Tree topology on GPU (uploaded once)
    gpu_child_offsets: CudaSlice<u32>,
    gpu_children: CudaSlice<u32>,
    gpu_infoset_ids: CudaSlice<u32>,
    gpu_num_actions: CudaSlice<u32>,

    // Per-level data for forward pass
    level_data: Vec<LevelData>,

    // Per-level data for backward CFV pass
    backward_level_data: Vec<BackwardLevelData>,

    // Bucketed fold terminal data
    fold_terminal_nodes: CudaSlice<u32>,
    fold_half_pots: CudaSlice<f32>,    // [num_fold_terminals]
    fold_player: CudaSlice<u32>,
    num_fold_terminals: u32,

    // Bucketed showdown terminal data
    showdown_terminal_nodes: CudaSlice<u32>,
    showdown_equity_tables: CudaSlice<f32>, // [num_sd_terminals * num_buckets * num_buckets]
    showdown_half_pots: CudaSlice<f32>,     // [num_sd_terminals]
    num_showdown_terminals: u32,

    // Initial reach (on GPU, uploaded once)
    gpu_initial_reach_oop: CudaSlice<f32>,  // [num_buckets]
    gpu_initial_reach_ip: CudaSlice<f32>,

    // Decision nodes per player (for regret updates)
    decision_nodes_oop: CudaSlice<u32>,
    decision_nodes_ip: CudaSlice<u32>,
    num_oop_decisions: u32,
    num_ip_decisions: u32,

    // Dimensions
    num_buckets: u32,
    max_actions: u32,
    num_infosets: u32,
    num_nodes: u32,
    num_levels: usize,
}

#[cfg(feature = "cuda")]
impl<'a> BucketedGpuSolver<'a> {
    /// Create a new bucketed GPU solver, uploading all tree data to the GPU.
    ///
    /// # Arguments
    /// - `gpu`: GPU context for CUDA operations
    /// - `tree`: bucketed game tree with topology and terminal data
    /// - `initial_reach_oop`: initial reach probabilities for OOP player `[num_buckets]`
    /// - `initial_reach_ip`: initial reach probabilities for IP player `[num_buckets]`
    pub fn new(
        gpu: &'a GpuContext,
        tree: &BucketedTree,
        initial_reach_oop: &[f32],
        initial_reach_ip: &[f32],
    ) -> Result<Self, GpuError> {
        let num_buckets = tree.num_buckets as u32;
        let num_nodes = tree.num_nodes() as u32;
        let num_infosets = tree.num_infosets as u32;
        let max_actions = tree.max_actions() as u32;
        let num_levels = tree.num_levels();

        assert_eq!(initial_reach_oop.len(), num_buckets as usize);
        assert_eq!(initial_reach_ip.len(), num_buckets as usize);

        let strat_size = (num_infosets * max_actions * num_buckets) as usize;
        let reach_size = (num_nodes * num_buckets) as usize;

        // Allocate solver state (all zeros)
        let regrets = gpu.alloc_zeros::<f32>(strat_size)?;
        let strategy_sum = gpu.alloc_zeros::<f32>(strat_size)?;
        let current_strategy = gpu.alloc_zeros::<f32>(strat_size)?;
        let reach_oop = gpu.alloc_zeros::<f32>(reach_size)?;
        let reach_ip = gpu.alloc_zeros::<f32>(reach_size)?;
        let cfvalues = gpu.alloc_zeros::<f32>(reach_size)?;

        // Upload tree topology
        let gpu_child_offsets = gpu.upload(&tree.child_offsets)?;
        let gpu_children = gpu.upload(&tree.children)?;
        let gpu_infoset_ids = gpu.upload(&tree.infoset_ids)?;
        let gpu_num_actions = gpu.upload(&tree.infoset_num_actions)?;

        // Build per-level data for forward pass (same pattern as GpuSolver)
        let mut level_data = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            let start = tree.level_starts[level] as usize;
            let end = tree.level_starts[level + 1] as usize;

            let mut nodes_vec = Vec::with_capacity(end - start);
            let mut parent_nodes_vec = Vec::with_capacity(end - start);
            let mut parent_actions_vec = Vec::with_capacity(end - start);
            let mut parent_infosets_vec = Vec::with_capacity(end - start);
            let mut parent_players_vec = Vec::with_capacity(end - start);

            for node_id in start..end {
                nodes_vec.push(node_id as u32);
                let parent = tree.parent_nodes[node_id];
                parent_nodes_vec.push(parent);
                parent_actions_vec.push(tree.parent_actions[node_id]);
                if parent != u32::MAX {
                    parent_infosets_vec.push(tree.infoset_ids[parent as usize]);
                    parent_players_vec.push(tree.player(parent as usize) as u32);
                } else {
                    parent_infosets_vec.push(0);
                    parent_players_vec.push(0);
                }
            }

            let num_nodes_level = nodes_vec.len() as u32;
            level_data.push(LevelData {
                nodes: gpu.upload(&nodes_vec)?,
                parent_nodes: gpu.upload(&parent_nodes_vec)?,
                parent_actions: gpu.upload(&parent_actions_vec)?,
                parent_infosets: gpu.upload(&parent_infosets_vec)?,
                parent_players: gpu.upload(&parent_players_vec)?,
                num_nodes: num_nodes_level,
            });
        }

        // Build per-level decision node lists for backward CFV
        let mut backward_level_data = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            let start = tree.level_starts[level] as usize;
            let end = tree.level_starts[level + 1] as usize;

            let mut decision_nodes_vec = Vec::new();
            let mut decision_players_vec = Vec::new();
            for node_id in start..end {
                if !tree.is_terminal(node_id) {
                    decision_nodes_vec.push(node_id as u32);
                    decision_players_vec.push(tree.player(node_id) as u32);
                }
            }

            let num_decision = decision_nodes_vec.len() as u32;
            // Ensure non-empty for GPU buffer
            if decision_nodes_vec.is_empty() {
                decision_nodes_vec.push(0);
                decision_players_vec.push(0);
            }
            backward_level_data.push(BackwardLevelData {
                decision_nodes: gpu.upload(&decision_nodes_vec)?,
                decision_players: gpu.upload(&decision_players_vec)?,
                num_decision_nodes: num_decision,
            });
        }

        // Build bucketed fold terminal data
        let mut fold_nodes = Vec::new();
        let mut fold_hps = Vec::new();
        let mut fold_pl = Vec::new();

        for &node_id in &tree.terminal_indices {
            if tree.node_types[node_id as usize] == NodeType::TerminalFold {
                fold_nodes.push(node_id);
                // Find the fold ordinal for this terminal
                let term_pos = tree
                    .terminal_indices
                    .iter()
                    .position(|&n| n == node_id)
                    .unwrap();
                let fold_ord = tree.fold_ordinals[term_pos] as usize;
                fold_hps.push(tree.fold_half_pots[fold_ord]);
                fold_pl.push(tree.fold_players[fold_ord]);
            }
        }

        let num_fold_terminals = fold_nodes.len() as u32;
        // Ensure non-empty
        if fold_nodes.is_empty() {
            fold_nodes.push(0);
            fold_hps.push(0.0);
            fold_pl.push(0);
        }

        let fold_terminal_nodes = gpu.upload(&fold_nodes)?;
        let fold_half_pots = gpu.upload(&fold_hps)?;
        let fold_player = gpu.upload(&fold_pl)?;

        // Build bucketed showdown terminal data
        let mut sd_nodes = Vec::new();
        let mut sd_hps = Vec::new();
        let mut sd_equity_flat: Vec<f32> = Vec::new(); // flattened equity tables

        for &node_id in &tree.terminal_indices {
            if tree.node_types[node_id as usize] == NodeType::TerminalShowdown {
                sd_nodes.push(node_id);
                let term_pos = tree
                    .terminal_indices
                    .iter()
                    .position(|&n| n == node_id)
                    .unwrap();
                let sd_ord = tree.showdown_ordinals[term_pos] as usize;
                sd_hps.push(tree.showdown_half_pots[sd_ord]);
                sd_equity_flat.extend_from_slice(&tree.showdown_equity_tables[sd_ord]);
            }
        }

        let num_showdown_terminals = sd_nodes.len() as u32;
        // Ensure non-empty
        if sd_nodes.is_empty() {
            sd_nodes.push(0);
            sd_hps.push(0.0);
            sd_equity_flat
                .resize((num_buckets * num_buckets) as usize, 0.0);
        }

        let showdown_terminal_nodes = gpu.upload(&sd_nodes)?;
        let showdown_half_pots = gpu.upload(&sd_hps)?;
        let showdown_equity_tables = gpu.upload(&sd_equity_flat)?;

        // Build decision node lists per player (for regret updates)
        let mut oop_decisions = Vec::new();
        let mut ip_decisions = Vec::new();
        for (i, nt) in tree.node_types.iter().enumerate() {
            match nt {
                NodeType::DecisionOop => oop_decisions.push(i as u32),
                NodeType::DecisionIp => ip_decisions.push(i as u32),
                _ => {}
            }
        }

        let num_oop_decisions = oop_decisions.len() as u32;
        let num_ip_decisions = ip_decisions.len() as u32;

        // Ensure non-empty
        if oop_decisions.is_empty() {
            oop_decisions.push(0);
        }
        if ip_decisions.is_empty() {
            ip_decisions.push(0);
        }

        let decision_nodes_oop = gpu.upload(&oop_decisions)?;
        let decision_nodes_ip = gpu.upload(&ip_decisions)?;

        // Upload initial reach
        let gpu_initial_reach_oop = gpu.upload(initial_reach_oop)?;
        let gpu_initial_reach_ip = gpu.upload(initial_reach_ip)?;

        Ok(Self {
            gpu,
            regrets,
            strategy_sum,
            current_strategy,
            reach_oop,
            reach_ip,
            cfvalues,
            gpu_child_offsets,
            gpu_children,
            gpu_infoset_ids,
            gpu_num_actions,
            level_data,
            backward_level_data,
            fold_terminal_nodes,
            fold_half_pots,
            fold_player,
            num_fold_terminals,
            showdown_terminal_nodes,
            showdown_equity_tables,
            showdown_half_pots,
            num_showdown_terminals,
            gpu_initial_reach_oop,
            gpu_initial_reach_ip,
            decision_nodes_oop,
            decision_nodes_ip,
            num_oop_decisions,
            num_ip_decisions,
            num_buckets,
            max_actions,
            num_infosets,
            num_nodes,
            num_levels,
        })
    }

    /// Run the Supremus DCFR+ solver for `max_iterations` iterations.
    ///
    /// Uses alternating traverser updates with Supremus-style regret
    /// discounting and delayed strategy accumulation.
    ///
    /// # Arguments
    /// - `max_iterations`: number of DCFR+ iterations to run
    /// - `delay`: number of early iterations to skip for strategy accumulation (typically 100)
    pub fn solve(
        &mut self,
        max_iterations: u32,
        delay: u32,
    ) -> Result<BucketedSolveResult, GpuError> {
        for t in 1..=max_iterations {
            for traverser in 0..2u32 {
                // 1. Regret match -> current strategy (uses latest regrets)
                self.gpu.launch_regret_match(
                    &self.regrets,
                    &self.gpu_num_actions,
                    &mut self.current_strategy,
                    self.num_infosets,
                    self.max_actions,
                    self.num_buckets,
                )?;

                // 2. Initialize reach at root
                self.init_reach()?;

                // 3. Forward pass (top-down, level by level)
                for level in 1..self.num_levels {
                    let ld = &self.level_data[level];
                    if ld.num_nodes == 0 {
                        continue;
                    }
                    self.gpu.launch_forward_pass(
                        &mut self.reach_oop,
                        &mut self.reach_ip,
                        &self.current_strategy,
                        &ld.nodes,
                        &ld.parent_nodes,
                        &ld.parent_actions,
                        &ld.parent_infosets,
                        &ld.parent_players,
                        ld.num_nodes,
                        self.num_buckets,
                        self.max_actions,
                    )?;
                }

                // 4a. Zero out cfvalues
                let cfv_size = self.num_nodes * self.num_buckets;
                self.gpu.launch_zero_buffer(&mut self.cfvalues, cfv_size)?;

                // 4b. Terminal fold eval (bucketed kernel)
                if self.num_fold_terminals > 0 {
                    let opp_reach = if traverser == 0 {
                        &self.reach_ip
                    } else {
                        &self.reach_oop
                    };
                    self.gpu.launch_bucketed_fold_eval(
                        &mut self.cfvalues,
                        opp_reach,
                        &self.fold_terminal_nodes,
                        &self.fold_half_pots,
                        &self.fold_player,
                        traverser,
                        self.num_fold_terminals,
                        self.num_buckets, // total_hands = num_buckets for single-spot
                        self.num_buckets,
                    )?;
                }

                // 4c. Terminal showdown eval (bucketed kernel)
                if self.num_showdown_terminals > 0 {
                    let opp_reach = if traverser == 0 {
                        &self.reach_ip
                    } else {
                        &self.reach_oop
                    };
                    self.gpu.launch_bucketed_showdown_eval(
                        &mut self.cfvalues,
                        opp_reach,
                        &self.showdown_terminal_nodes,
                        &self.showdown_equity_tables,
                        &self.showdown_half_pots,
                        self.num_showdown_terminals,
                        self.num_buckets, // total_hands = num_buckets for single-spot
                        self.num_buckets,
                    )?;
                }

                // 4d. Backward CFV (bottom-up, level by level)
                for level in (0..self.num_levels).rev() {
                    let bld = &self.backward_level_data[level];
                    if bld.num_decision_nodes == 0 {
                        continue;
                    }
                    self.gpu.launch_backward_cfv(
                        &mut self.cfvalues,
                        &self.current_strategy,
                        &bld.decision_nodes,
                        &self.gpu_child_offsets,
                        &self.gpu_children,
                        &self.gpu_infoset_ids,
                        &bld.decision_players,
                        traverser,
                        bld.num_decision_nodes,
                        self.num_buckets,
                        self.max_actions,
                    )?;
                }

                // 4e. Update regrets (Supremus DCFR+)
                let (decision_nodes, num_decision) = if traverser == 0 {
                    (&self.decision_nodes_oop, self.num_oop_decisions)
                } else {
                    (&self.decision_nodes_ip, self.num_ip_decisions)
                };

                if num_decision > 0 {
                    self.gpu.launch_update_regrets_supremus(
                        &mut self.regrets,
                        &mut self.strategy_sum,
                        &self.current_strategy,
                        &self.cfvalues,
                        decision_nodes,
                        &self.gpu_child_offsets,
                        &self.gpu_children,
                        &self.gpu_infoset_ids,
                        &self.gpu_num_actions,
                        num_decision,
                        self.num_buckets,
                        self.max_actions,
                        t,
                        delay,
                    )?;
                }
            }
        }

        // Extract final strategy from strategy_sum
        self.gpu.launch_extract_strategy(
            &self.strategy_sum,
            &self.gpu_num_actions,
            &mut self.current_strategy,
            self.num_infosets,
            self.max_actions,
            self.num_buckets,
        )?;

        let strategy = self.gpu.download(&self.current_strategy)?;

        Ok(BucketedSolveResult {
            strategy,
            iterations: max_iterations,
        })
    }

    /// Initialize reach probabilities: zero everything, then set root reach.
    fn init_reach(&mut self) -> Result<(), GpuError> {
        let reach_size = self.num_nodes * self.num_buckets;
        self.gpu
            .launch_zero_buffer(&mut self.reach_oop, reach_size)?;
        self.gpu
            .launch_zero_buffer(&mut self.reach_ip, reach_size)?;
        self.gpu.launch_set_root_reach(
            &mut self.reach_oop,
            &self.gpu_initial_reach_oop,
            self.num_buckets,
        )?;
        self.gpu.launch_set_root_reach(
            &mut self.reach_ip,
            &self.gpu_initial_reach_ip,
            self.num_buckets,
        )?;
        Ok(())
    }

    /// Get the number of buckets this solver operates on.
    pub fn num_buckets(&self) -> u32 {
        self.num_buckets
    }

    /// Get the number of infosets in the tree.
    pub fn num_infosets(&self) -> u32 {
        self.num_infosets
    }

    /// Get the maximum number of actions across all infosets.
    pub fn max_actions(&self) -> u32 {
        self.max_actions
    }

    /// Get the number of nodes in the tree.
    pub fn num_nodes(&self) -> u32 {
        self.num_nodes
    }

    /// Download current regrets from GPU for inspection.
    pub fn download_regrets(&self) -> Result<Vec<f32>, GpuError> {
        self.gpu.download(&self.regrets)
    }

    /// Download current strategy from GPU for inspection.
    pub fn download_strategy(&self) -> Result<Vec<f32>, GpuError> {
        self.gpu.download(&self.current_strategy)
    }

    /// Download strategy sum from GPU for inspection.
    pub fn download_strategy_sum(&self) -> Result<Vec<f32>, GpuError> {
        self.gpu.download(&self.strategy_sum)
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::bucketed::equity::BucketedBoardCache;
    use poker_solver_core::blueprint_v2::bucket_file::{BucketFile, BucketFileHeader};
    use poker_solver_core::blueprint_v2::cluster_pipeline::canonical_key;
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::evaluate_hand_strength;
    use range_solver::{ActionTree, BoardState, CardConfig, PostFlopGame, TreeConfig};

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
        use crate::bucketed::equity::u8_to_rs_card;
        use range_solver::card::index_to_card_pair;

        let rs_cards: Vec<rs_poker::core::Card> =
            board_u8.iter().map(|&c| u8_to_rs_card(c)).collect();
        let packed = canonical_key(&rs_cards);

        // Assign buckets by hand strength ranking
        let combos_per_board = 1326u16;
        let mut combo_strengths: Vec<(u16, u16)> = Vec::new();
        for i in 0..combos_per_board {
            let (c1, c2) = index_to_card_pair(i as usize);
            if board_u8.contains(&c1) || board_u8.contains(&c2) {
                combo_strengths.push((i, 0));
            } else {
                let s = evaluate_hand_strength(board_u8, (c1, c2));
                combo_strengths.push((i, s));
            }
        }

        // Sort by strength and divide into buckets
        let mut sorted = combo_strengths.clone();
        sorted.sort_by_key(|&(_, s)| s);

        let mut bucket_data = vec![0u16; combos_per_board as usize];
        let bucket_size = combos_per_board as usize / num_buckets as usize;
        for (rank, &(combo_idx, _)) in sorted.iter().enumerate() {
            let bucket = ((rank / bucket_size.max(1)) as u16).min(num_buckets - 1);
            bucket_data[combo_idx as usize] = bucket;
        }

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

    /// Test: build a BucketedGpuSolver, run 100 iterations, verify strategy
    /// sums to ~1.0 per (infoset, bucket) and is non-trivial.
    #[test]
    fn test_bucketed_solver_basic() {
        let gpu = GpuContext::new(0).expect("CUDA device required");

        let flop = range_solver::card::flop_from_str("Qs Jh 2c").unwrap();
        let turn = range_solver::card::card_from_str("8d").unwrap();
        let river = range_solver::card::card_from_str("3s").unwrap();
        let bet_sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();

        let board_u8: [u8; 5] = [flop[0], flop[1], flop[2], turn, river];
        let num_buckets = 10u16; // small for testing

        let bf = make_synthetic_bucket_file(&board_u8, num_buckets);
        let cache = BucketedBoardCache::new(&bf);

        let mut game = build_river_game(flop, turn, river, 100, 100, &bet_sizes);
        let tree = BucketedTree::from_postflop_game(
            &mut game,
            &bf,
            &cache,
            num_buckets as usize,
        );

        // Uniform initial reach
        let initial_reach = vec![1.0f32; num_buckets as usize];

        let mut solver = BucketedGpuSolver::new(
            &gpu,
            &tree,
            &initial_reach,
            &initial_reach,
        )
        .expect("solver creation");

        let result = solver.solve(100, 50).expect("solver run");

        let strategy = &result.strategy;
        let nb = num_buckets as usize;
        let ma = tree.max_actions();
        let ni = tree.num_infosets;

        // Verify strategy sums to ~1.0 per (infoset, bucket)
        for iset in 0..ni {
            let n_actions = tree.infoset_num_actions[iset] as usize;
            for bucket in 0..nb {
                let mut sum = 0.0f32;
                for a in 0..n_actions {
                    let idx = (iset * ma + a) * nb + bucket;
                    sum += strategy[idx];
                }
                assert!(
                    (sum - 1.0).abs() < 0.01,
                    "Strategy sum for infoset {iset}, bucket {bucket} = {sum} (expected ~1.0)"
                );
            }
        }

        // Verify strategy is non-trivial (not all uniform)
        let mut has_non_uniform = false;
        for iset in 0..ni {
            let n_actions = tree.infoset_num_actions[iset] as usize;
            if n_actions <= 1 {
                continue;
            }
            let uniform = 1.0 / n_actions as f32;
            for bucket in 0..nb {
                for a in 0..n_actions {
                    let idx = (iset * ma + a) * nb + bucket;
                    if (strategy[idx] - uniform).abs() > 0.05 {
                        has_non_uniform = true;
                    }
                }
            }
        }
        assert!(
            has_non_uniform,
            "Strategy appears all-uniform after 100 iterations"
        );
    }

    /// Test: bucketed fold kernel correctness with known values.
    #[test]
    fn test_bucketed_fold_kernel() {
        let gpu = GpuContext::new(0).expect("CUDA device required");

        let num_buckets: u32 = 4;
        let num_nodes: u32 = 3; // root + 2 terminals
        let total_hands = num_buckets;

        // Set up: node 1 is a fold terminal, node 2 is a fold terminal
        // Opponent reach at node 1: [0.1, 0.2, 0.3, 0.4] (sum = 1.0)
        // Opponent reach at node 2: [0.5, 0.0, 0.5, 0.0] (sum = 1.0)
        let mut opp_reach_host = vec![0.0f32; (num_nodes * num_buckets) as usize];
        // Node 1
        opp_reach_host[1 * num_buckets as usize + 0] = 0.1;
        opp_reach_host[1 * num_buckets as usize + 1] = 0.2;
        opp_reach_host[1 * num_buckets as usize + 2] = 0.3;
        opp_reach_host[1 * num_buckets as usize + 3] = 0.4;
        // Node 2
        opp_reach_host[2 * num_buckets as usize + 0] = 0.5;
        opp_reach_host[2 * num_buckets as usize + 2] = 0.5;

        let opp_reach = gpu.upload(&opp_reach_host).unwrap();
        let mut cfvalues = gpu.alloc_zeros::<f32>((num_nodes * num_buckets) as usize).unwrap();

        let terminal_nodes = gpu.upload(&[1u32, 2u32]).unwrap();
        let half_pots = gpu.upload(&[50.0f32, 75.0f32]).unwrap();
        let fold_player = gpu.upload(&[0u32, 1u32]).unwrap(); // node 1: OOP folded, node 2: IP folded

        // Traverser = 0 (OOP): at node 1, OOP folded -> payoff = -hp; at node 2, IP folded -> payoff = +hp
        gpu.launch_bucketed_fold_eval(
            &mut cfvalues,
            &opp_reach,
            &terminal_nodes,
            &half_pots,
            &fold_player,
            0, // traverser = OOP
            2, // num_fold_terminals
            total_hands,
            num_buckets,
        )
        .unwrap();

        let cfv = gpu.download(&cfvalues).unwrap();

        // Node 1: OOP folded, traverser=OOP -> payoff = -50.0, opp_sum = 1.0
        // cfv = -50.0 * 1.0 = -50.0 for all buckets
        for b in 0..num_buckets as usize {
            assert!(
                (cfv[1 * num_buckets as usize + b] - (-50.0)).abs() < 1e-4,
                "Node 1, bucket {b}: expected -50.0, got {}",
                cfv[1 * num_buckets as usize + b]
            );
        }

        // Node 2: IP folded, traverser=OOP -> payoff = +75.0, opp_sum = 1.0
        // cfv = 75.0 * 1.0 = 75.0 for all buckets
        for b in 0..num_buckets as usize {
            assert!(
                (cfv[2 * num_buckets as usize + b] - 75.0).abs() < 1e-4,
                "Node 2, bucket {b}: expected 75.0, got {}",
                cfv[2 * num_buckets as usize + b]
            );
        }
    }

    /// Test: bucketed showdown kernel correctness with known equity matrix.
    #[test]
    fn test_bucketed_showdown_kernel() {
        let gpu = GpuContext::new(0).expect("CUDA device required");

        let num_buckets: u32 = 3;
        let num_nodes: u32 = 2; // root + 1 showdown terminal
        let total_hands = num_buckets;

        // Equity table for the single showdown terminal (3x3):
        // E[0][0]=0, E[0][1]=-0.5, E[0][2]=-1.0
        // E[1][0]=0.5, E[1][1]=0,   E[1][2]=-0.5
        // E[2][0]=1.0, E[2][1]=0.5, E[2][2]=0
        let equity = vec![
            0.0, -0.5, -1.0, // bucket 0 vs others
            0.5, 0.0, -0.5, // bucket 1 vs others
            1.0, 0.5, 0.0, // bucket 2 vs others
        ];

        // Opponent reach at node 1: [0.3, 0.3, 0.4]
        let mut opp_reach_host = vec![0.0f32; (num_nodes * num_buckets) as usize];
        opp_reach_host[1 * num_buckets as usize + 0] = 0.3;
        opp_reach_host[1 * num_buckets as usize + 1] = 0.3;
        opp_reach_host[1 * num_buckets as usize + 2] = 0.4;

        let opp_reach = gpu.upload(&opp_reach_host).unwrap();
        let mut cfvalues = gpu.alloc_zeros::<f32>((num_nodes * num_buckets) as usize).unwrap();

        let terminal_nodes = gpu.upload(&[1u32]).unwrap();
        let half_pots = gpu.upload(&[50.0f32]).unwrap();
        let equity_tables = gpu.upload(&equity).unwrap();

        gpu.launch_bucketed_showdown_eval(
            &mut cfvalues,
            &opp_reach,
            &terminal_nodes,
            &equity_tables,
            &half_pots,
            1, // num_sd_terminals
            total_hands,
            num_buckets,
        )
        .unwrap();

        let cfv = gpu.download(&cfvalues).unwrap();

        // Expected: cfv[bucket_i] = 50.0 * sum_j(E[i][j] * opp_reach[j])
        //
        // bucket 0: 50.0 * (0*0.3 + (-0.5)*0.3 + (-1.0)*0.4) = 50.0 * (-0.15 - 0.40) = 50.0 * (-0.55) = -27.5
        // bucket 1: 50.0 * (0.5*0.3 + 0*0.3 + (-0.5)*0.4) = 50.0 * (0.15 - 0.20) = 50.0 * (-0.05) = -2.5
        // bucket 2: 50.0 * (1.0*0.3 + 0.5*0.3 + 0*0.4) = 50.0 * (0.30 + 0.15) = 50.0 * 0.45 = 22.5

        let expected = [-27.5f32, -2.5, 22.5];
        for b in 0..num_buckets as usize {
            let actual = cfv[1 * num_buckets as usize + b];
            assert!(
                (actual - expected[b]).abs() < 1e-3,
                "Bucket {b}: expected {}, got {actual}",
                expected[b]
            );
        }
    }
}
