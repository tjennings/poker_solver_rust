// GPU DCFR+ solver
//
// Orchestrates the full DCFR+ iteration loop on GPU:
// 1. Upload tree structure and initial data
// 2. Run alternating-traverser iterations
// 3. Extract final strategy

#[cfg(feature = "cuda")]
use crate::gpu::{GpuContext, GpuError};
#[cfg(feature = "cuda")]
use crate::tree::{FlatTree, NodeType};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

/// Precomputed per-level data for GPU kernels.
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

/// Result of running the GPU solver.
#[cfg(feature = "cuda")]
pub struct SolveResult {
    /// Final averaged strategy: `[num_infosets * max_actions * num_hands]`.
    pub strategy: Vec<f32>,
    /// Number of iterations completed.
    pub iterations: u32,
}

/// Debug snapshot of a single iteration's intermediate state.
#[cfg(feature = "cuda")]
pub struct DebugIteration {
    /// Reach probabilities for OOP at every node: `[num_nodes * num_hands]`.
    pub reach_oop: Vec<f32>,
    /// Reach probabilities for IP at every node: `[num_nodes * num_hands]`.
    pub reach_ip: Vec<f32>,
    /// CFVs when OOP is traverser (after terminal eval + backward): `[num_nodes * num_hands]`.
    pub cfvalues_oop_traverser: Vec<f32>,
    /// CFVs when IP is traverser (after terminal eval + backward): `[num_nodes * num_hands]`.
    pub cfvalues_ip_traverser: Vec<f32>,
    /// Cumulative regrets after update: `[num_infosets * max_actions * num_hands]`.
    pub regrets: Vec<f32>,
    /// Current strategy (from regret matching): `[num_infosets * max_actions * num_hands]`.
    pub strategy: Vec<f32>,
    /// Cumulative strategy sum after update: `[num_infosets * max_actions * num_hands]`.
    pub strategy_sum: Vec<f32>,
}

/// GPU DCFR+ solver that holds all GPU buffers and tree metadata.
#[cfg(feature = "cuda")]
pub struct GpuSolver<'a> {
    gpu: &'a GpuContext,

    // Solver state on GPU
    regrets: CudaSlice<f32>,
    strategy_sum: CudaSlice<f32>,
    current_strategy: CudaSlice<f32>,
    reach_oop: CudaSlice<f32>,
    reach_ip: CudaSlice<f32>,
    cfvalues: CudaSlice<f32>,

    // Tree structure on GPU (uploaded once)
    gpu_child_offsets: CudaSlice<u32>,
    gpu_children: CudaSlice<u32>,
    gpu_infoset_ids: CudaSlice<u32>,
    gpu_num_actions: CudaSlice<u32>,

    // Per-level data for forward pass
    level_data: Vec<LevelData>,

    // Per-level data for backward CFV pass
    backward_level_data: Vec<BackwardLevelData>,

    // Terminal data
    fold_terminal_nodes: CudaSlice<u32>,
    fold_amount_win: CudaSlice<f32>,
    fold_amount_lose: CudaSlice<f32>,
    fold_player: CudaSlice<u32>,
    num_fold_terminals: u32,

    showdown_terminal_nodes: CudaSlice<u32>,
    showdown_amount_win: CudaSlice<f32>,
    showdown_amount_lose: CudaSlice<f32>,
    num_showdown_terminals: u32,

    gpu_hand_strengths_oop: CudaSlice<u32>,
    gpu_hand_strengths_ip: CudaSlice<u32>,

    // Card blocking matrices — no longer used in the solve loop (replaced by
    // hand_cards-based blocking), but kept for backward-compatible kernel tests.
    #[allow(dead_code)]
    gpu_valid_matchups_oop: CudaSlice<f32>,
    #[allow(dead_code)]
    gpu_valid_matchups_ip: CudaSlice<f32>,

    // Hand card arrays for O(n) fold eval and shared-memory showdown eval
    // Each is [num_hands * 2] containing (c1, c2) per hand as u32
    gpu_hand_cards_oop: CudaSlice<u32>,
    gpu_hand_cards_ip: CudaSlice<u32>,

    // Same-hand index for inclusion-exclusion correction
    // For each player's hand, the opponent's hand index holding same cards (or u32::MAX)
    gpu_same_hand_index_oop: CudaSlice<u32>,
    gpu_same_hand_index_ip: CudaSlice<u32>,

    // Working buffers for O(n) fold aggregates (reused each iteration)
    gpu_fold_total_opp_reach: CudaSlice<f32>,
    gpu_fold_per_card_reach: CudaSlice<f32>,

    // Decision node lists per player
    decision_nodes_oop: CudaSlice<u32>,
    decision_nodes_ip: CudaSlice<u32>,
    num_oop_decisions: u32,
    num_ip_decisions: u32,

    // Initial reach (on GPU, uploaded once)
    gpu_initial_reach_oop: CudaSlice<f32>,
    gpu_initial_reach_ip: CudaSlice<f32>,

    // Dimensions
    num_hands: u32,
    max_actions: u32,
    num_infosets: u32,
    num_nodes: u32,
    num_levels: usize,
}

#[cfg(feature = "cuda")]
impl<'a> GpuSolver<'a> {
    /// Create a new GPU solver, uploading all tree data to the GPU.
    pub fn new(gpu: &'a GpuContext, tree: &FlatTree) -> Result<Self, GpuError> {
        let num_hands = tree.num_hands as u32;
        let num_nodes = tree.num_nodes() as u32;
        let num_infosets = tree.num_infosets as u32;
        let max_actions = tree.max_actions() as u32;
        let num_levels = tree.num_levels();

        let strat_size = (num_infosets * max_actions * num_hands) as usize;
        let reach_size = (num_nodes * num_hands) as usize;

        // Allocate solver state
        let regrets = gpu.alloc_zeros::<f32>(strat_size)?;
        let strategy_sum = gpu.alloc_zeros::<f32>(strat_size)?;
        let current_strategy = gpu.alloc_zeros::<f32>(strat_size)?;
        let reach_oop = gpu.alloc_zeros::<f32>(reach_size)?;
        let reach_ip = gpu.alloc_zeros::<f32>(reach_size)?;
        let cfvalues = gpu.alloc_zeros::<f32>(reach_size)?;

        // Upload tree structure
        let gpu_child_offsets = gpu.upload(&tree.child_offsets)?;
        let gpu_children = gpu.upload(&tree.children)?;
        let gpu_infoset_ids = gpu.upload(&tree.infoset_ids)?;
        let gpu_num_actions = gpu.upload(&tree.infoset_num_actions)?;

        // Build per-level data for forward pass (levels 1..num_levels)
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
            // Ensure we have at least one element for GPU buffer
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

        // Build terminal data - separate fold and showdown terminals.
        // fold_payoffs and showdown_equity_ids are indexed by terminal ordinal
        // (i.e. position in terminal_indices), NOT by node_id.
        let mut fold_nodes = Vec::new();
        let mut fold_win = Vec::new();
        let mut fold_lose = Vec::new();
        let mut fold_pl = Vec::new();

        let mut showdown_nodes = Vec::new();
        let mut showdown_win = Vec::new();
        let mut showdown_lose = Vec::new();

        let mut fold_ordinal = 0usize;
        let mut showdown_ordinal = 0usize;

        for (term_idx, &node_id) in tree.terminal_indices.iter().enumerate() {
            match tree.node_types[node_id as usize] {
                NodeType::TerminalFold => {
                    fold_nodes.push(node_id);
                    let payoff = &tree.fold_payoffs[term_idx];
                    fold_win.push(payoff[0]);
                    fold_lose.push(payoff[1]);
                    fold_pl.push(payoff[2] as u32);
                    fold_ordinal += 1;
                }
                NodeType::TerminalShowdown => {
                    showdown_nodes.push(node_id);
                    let eq_id = tree.showdown_equity_ids[term_idx] as usize;
                    let eq = &tree.equity_tables[eq_id];
                    showdown_win.push(eq[0]);
                    showdown_lose.push(eq[1]);
                    showdown_ordinal += 1;
                }
                _ => {}
            }
        }

        // Ensure non-empty buffers for GPU
        if fold_nodes.is_empty() {
            fold_nodes.push(0);
            fold_win.push(0.0);
            fold_lose.push(0.0);
            fold_pl.push(0);
        }
        if showdown_nodes.is_empty() {
            showdown_nodes.push(0);
            showdown_win.push(0.0);
            showdown_lose.push(0.0);
        }

        let num_fold_terminals = fold_ordinal as u32;
        let num_showdown_terminals = showdown_ordinal as u32;

        let fold_terminal_nodes = gpu.upload(&fold_nodes)?;
        let fold_amount_win = gpu.upload(&fold_win)?;
        let fold_amount_lose = gpu.upload(&fold_lose)?;
        let fold_player = gpu.upload(&fold_pl)?;

        let showdown_terminal_nodes = gpu.upload(&showdown_nodes)?;
        let showdown_amount_win = gpu.upload(&showdown_win)?;
        let showdown_amount_lose = gpu.upload(&showdown_lose)?;

        // Pad hand strengths to num_hands
        let mut hs_oop = tree.hand_strengths_oop.clone();
        hs_oop.resize(num_hands as usize, 0);
        let mut hs_ip = tree.hand_strengths_ip.clone();
        hs_ip.resize(num_hands as usize, 0);

        let gpu_hand_strengths_oop = gpu.upload(&hs_oop)?;
        let gpu_hand_strengths_ip = gpu.upload(&hs_ip)?;

        // Upload card blocking matrices
        let gpu_valid_matchups_oop = gpu.upload(&tree.valid_matchups_oop)?;
        let gpu_valid_matchups_ip = gpu.upload(&tree.valid_matchups_ip)?;

        // Upload hand card arrays for O(n) fold eval and shared-memory showdown.
        // Layout: [num_hands * 2] as flat u32 array of (c1, c2) pairs.
        let mut hand_cards_oop_flat = vec![255u32; num_hands as usize * 2];
        for (i, &(c1, c2)) in tree.cards_oop.iter().enumerate() {
            hand_cards_oop_flat[i * 2] = c1 as u32;
            hand_cards_oop_flat[i * 2 + 1] = c2 as u32;
        }
        let mut hand_cards_ip_flat = vec![255u32; num_hands as usize * 2];
        for (i, &(c1, c2)) in tree.cards_ip.iter().enumerate() {
            hand_cards_ip_flat[i * 2] = c1 as u32;
            hand_cards_ip_flat[i * 2 + 1] = c2 as u32;
        }
        let gpu_hand_cards_oop = gpu.upload(&hand_cards_oop_flat)?;
        let gpu_hand_cards_ip = gpu.upload(&hand_cards_ip_flat)?;

        // Upload same-hand index arrays
        let gpu_same_hand_index_oop = gpu.upload(&tree.same_hand_index_oop)?;
        let gpu_same_hand_index_ip = gpu.upload(&tree.same_hand_index_ip)?;

        // Allocate working buffers for fold aggregates
        let fold_agg_size = fold_ordinal.max(1);
        let gpu_fold_total_opp_reach = gpu.alloc_zeros::<f32>(fold_agg_size)?;
        let gpu_fold_per_card_reach = gpu.alloc_zeros::<f32>(fold_agg_size * 52)?;

        // Build decision node lists per player
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
            fold_amount_win,
            fold_amount_lose,
            fold_player,
            num_fold_terminals,
            showdown_terminal_nodes,
            showdown_amount_win,
            showdown_amount_lose,
            num_showdown_terminals,
            gpu_hand_strengths_oop,
            gpu_hand_strengths_ip,
            gpu_valid_matchups_oop,
            gpu_valid_matchups_ip,
            gpu_hand_cards_oop,
            gpu_hand_cards_ip,
            gpu_same_hand_index_oop,
            gpu_same_hand_index_ip,
            gpu_fold_total_opp_reach,
            gpu_fold_per_card_reach,
            decision_nodes_oop,
            decision_nodes_ip,
            num_oop_decisions,
            num_ip_decisions,
            gpu_initial_reach_oop: gpu.upload(&tree.initial_reach_oop)?,
            gpu_initial_reach_ip: gpu.upload(&tree.initial_reach_ip)?,
            num_hands,
            max_actions,
            num_infosets,
            num_nodes,
            num_levels,
        })
    }

    /// Run the DCFR+ solver for up to `max_iterations` iterations.
    ///
    /// If `target_exploitability` is set, early termination could be
    /// implemented in a future version.
    pub fn solve(
        &mut self,
        max_iterations: u32,
        _target_exploitability: Option<f32>,
    ) -> Result<SolveResult, GpuError> {
        for t in 1..=max_iterations {
            // DCFR discount factors — match range-solver's DiscountParams
            // CPU solver uses 0-indexed iteration `t`:
            //   alpha_t = (t-1)^1.5 / ((t-1)^1.5 + 1)
            //   beta_t  = 0.5
            //   gamma_t = ((t - nearest_lower_power_of_4) / (t - nlp4 + 1))^3
            // Here `t` is 1-indexed, so `current_iteration = t - 1` (0-indexed)
            let current_iteration = t - 1; // 0-indexed to match CPU

            let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
            let pow_alpha = t_alpha * t_alpha.sqrt();
            let pos_discount = (pow_alpha / (pow_alpha + 1.0)) as f32;
            let neg_discount = 0.5f32;

            // gamma_t: power-of-4 based strategy weight
            let nearest_lower_power_of_4 = match current_iteration {
                0 => 0u32,
                x => 1u32 << ((x.leading_zeros() ^ 31) & !1),
            };
            let t_gamma =
                (current_iteration - nearest_lower_power_of_4) as f64;
            let strat_discount =
                ((t_gamma / (t_gamma + 1.0)).powi(3)) as f32;

            // Alternating traverser updates — CPU solver recomputes strategy
            // from regrets between traversers (traverser 1 sees traverser 0's
            // regret updates). We must do the same: regret-match + forward pass
            // inside the traverser loop.
            for traverser in 0..2u32 {
                // 1. Regret match -> current strategy (uses latest regrets)
                self.gpu.launch_regret_match(
                    &self.regrets,
                    &self.gpu_num_actions,
                    &mut self.current_strategy,
                    self.num_infosets,
                    self.max_actions,
                    self.num_hands,
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
                        self.num_hands,
                        self.max_actions,
                    )?;
                }

                // 4a. Zero out cfvalues (GPU-only, no allocation)
                let cfv_size = (self.num_nodes * self.num_hands) as u32;
                self.gpu.launch_zero_buffer(&mut self.cfvalues, cfv_size)?;

                // 4b. Terminal fold eval — O(n) via inclusion-exclusion
                if self.num_fold_terminals > 0 {
                    let opp_reach = if traverser == 0 {
                        &self.reach_ip
                    } else {
                        &self.reach_oop
                    };
                    // Opponent's hand cards (for aggregate computation)
                    let opp_hand_cards = if traverser == 0 {
                        &self.gpu_hand_cards_ip
                    } else {
                        &self.gpu_hand_cards_oop
                    };
                    // Traverser's hand cards (for blocking lookup)
                    let trav_hand_cards = if traverser == 0 {
                        &self.gpu_hand_cards_oop
                    } else {
                        &self.gpu_hand_cards_ip
                    };
                    let same_hand_index = if traverser == 0 {
                        &self.gpu_same_hand_index_oop
                    } else {
                        &self.gpu_same_hand_index_ip
                    };

                    // Phase A: precompute per-card reach aggregates
                    self.gpu.launch_precompute_fold_aggregates(
                        opp_reach,
                        &self.fold_terminal_nodes,
                        opp_hand_cards,
                        &mut self.gpu_fold_total_opp_reach,
                        &mut self.gpu_fold_per_card_reach,
                        self.num_fold_terminals,
                        self.num_hands,
                    )?;

                    // Phase B: O(1) per-hand CFV using aggregates
                    self.gpu.launch_fold_eval_from_aggregates(
                        &mut self.cfvalues,
                        opp_reach,
                        &self.fold_terminal_nodes,
                        &self.fold_amount_win,
                        &self.fold_amount_lose,
                        &self.fold_player,
                        &self.gpu_fold_total_opp_reach,
                        &self.gpu_fold_per_card_reach,
                        trav_hand_cards,
                        same_hand_index,
                        traverser,
                        self.num_fold_terminals,
                        self.num_hands,
                    )?;
                }

                // 4c. Terminal showdown eval — shared memory optimization
                if self.num_showdown_terminals > 0 {
                    let opp_reach = if traverser == 0 {
                        &self.reach_ip
                    } else {
                        &self.reach_oop
                    };
                    let (traverser_strengths, opponent_strengths) = if traverser == 0 {
                        (&self.gpu_hand_strengths_oop, &self.gpu_hand_strengths_ip)
                    } else {
                        (&self.gpu_hand_strengths_ip, &self.gpu_hand_strengths_oop)
                    };
                    let trav_hand_cards = if traverser == 0 {
                        &self.gpu_hand_cards_oop
                    } else {
                        &self.gpu_hand_cards_ip
                    };
                    let opp_hand_cards = if traverser == 0 {
                        &self.gpu_hand_cards_ip
                    } else {
                        &self.gpu_hand_cards_oop
                    };
                    self.gpu.launch_showdown_eval_fast(
                        &mut self.cfvalues,
                        opp_reach,
                        &self.showdown_terminal_nodes,
                        &self.showdown_amount_win,
                        &self.showdown_amount_lose,
                        traverser_strengths,
                        opponent_strengths,
                        self.num_showdown_terminals,
                        self.num_hands,
                        self.num_hands,  // hands_per_spot = num_hands for single-spot
                    )?;
                }

                // 4d. Backward CFV (bottom-up, skip last level if all terminal)
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
                        self.num_hands,
                        self.max_actions,
                    )?;
                }

                // 4e. Update regrets for this traverser's decision nodes
                let (decision_nodes, num_decision) = if traverser == 0 {
                    (&self.decision_nodes_oop, self.num_oop_decisions)
                } else {
                    (&self.decision_nodes_ip, self.num_ip_decisions)
                };

                if num_decision > 0 {
                    self.gpu.launch_update_regrets(
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
                        self.num_hands,
                        self.max_actions,
                        pos_discount,
                        neg_discount,
                        strat_discount,
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
            self.num_hands,
        )?;

        let strategy = self.gpu.download(&self.current_strategy)?;

        Ok(SolveResult {
            strategy,
            iterations: max_iterations,
        })
    }

    /// Run the DCFR+ solver using the persistent mega-kernel (single launch).
    ///
    /// Eliminates all kernel launch overhead by running the entire solve loop
    /// in a single CUDA kernel using grid-wide atomic barriers.
    pub fn solve_persistent(
        &mut self,
        tree: &FlatTree,
        max_iterations: u32,
        _target_exploitability: Option<f32>,
    ) -> Result<SolveResult, GpuError> {
        use crate::gpu::GpuSolverContext;

        // Build per-node arrays indexed by global node_id
        let num_nodes = self.num_nodes as usize;

        let mut parent_infosets_full = vec![0u32; num_nodes];
        let mut parent_players_full = vec![0u32; num_nodes];
        for node_id in 0..num_nodes {
            let parent = tree.parent_nodes[node_id];
            if parent != u32::MAX {
                parent_infosets_full[node_id] = tree.infoset_ids[parent as usize];
                parent_players_full[node_id] = tree.player(parent as usize) as u32;
            }
        }

        // Build node_types: 0=OOP, 1=IP, 2=terminal
        let mut node_types_arr = vec![0u32; num_nodes];
        for i in 0..num_nodes {
            if tree.infoset_ids[i] == u32::MAX {
                node_types_arr[i] = 2; // terminal
            } else {
                node_types_arr[i] = tree.player(i) as u32;
            }
        }

        // Upload persistent kernel data
        let gpu_level_offsets = self.gpu.upload(&tree.level_starts)?;
        let gpu_parent_nodes_full = self.gpu.upload(&tree.parent_nodes)?;
        let gpu_parent_actions_full = self.gpu.upload(&tree.parent_actions)?;
        let gpu_parent_infosets_full = self.gpu.upload(&parent_infosets_full)?;
        let gpu_parent_players_full = self.gpu.upload(&parent_players_full)?;
        let gpu_node_types = self.gpu.upload(&node_types_arr)?;

        // Showdown sorted indices and rank arrays for single-spot
        // For single spot, we need to build the sorted/rank arrays
        let hands_per_spot = self.num_hands as usize;
        let num_spots = 1usize;

        let hs_oop: Vec<u32> = self.gpu.download(&self.gpu_hand_strengths_oop)?;
        let hs_ip: Vec<u32> = self.gpu.download(&self.gpu_hand_strengths_ip)?;

        let sorted_ip = crate::batch::BatchGpuSolver::sort_hands_by_strength(
            &hs_ip, num_spots, hands_per_spot,
        );
        let sorted_oop = crate::batch::BatchGpuSolver::sort_hands_by_strength(
            &hs_oop, num_spots, hands_per_spot,
        );
        let rank_win_oop = crate::batch::BatchGpuSolver::compute_cross_rank(
            &hs_oop, &sorted_ip, &hs_ip, num_spots, hands_per_spot,
        );
        let rank_next_oop = crate::batch::BatchGpuSolver::compute_cross_rank_next(
            &hs_oop, &sorted_ip, &hs_ip, num_spots, hands_per_spot,
        );
        let rank_win_ip = crate::batch::BatchGpuSolver::compute_cross_rank(
            &hs_ip, &sorted_oop, &hs_oop, num_spots, hands_per_spot,
        );
        let rank_next_ip = crate::batch::BatchGpuSolver::compute_cross_rank_next(
            &hs_ip, &sorted_oop, &hs_oop, num_spots, hands_per_spot,
        );

        let gpu_sorted_ip = self.gpu.upload(&sorted_ip)?;
        let gpu_sorted_oop = self.gpu.upload(&sorted_oop)?;
        let gpu_rank_win_oop = self.gpu.upload(&rank_win_oop)?;
        let gpu_rank_next_oop = self.gpu.upload(&rank_next_oop)?;
        let gpu_rank_win_ip = self.gpu.upload(&rank_win_ip)?;
        let gpu_rank_next_ip = self.gpu.upload(&rank_next_ip)?;

        // Showdown working buffers
        let sd_buf_size = (self.num_showdown_terminals as usize * self.num_hands as usize).max(1);
        let sd_total_size = (self.num_showdown_terminals as usize).max(1);
        let mut gpu_sd_sorted_reach = self.gpu.alloc_zeros::<f32>(sd_buf_size)?;
        let mut gpu_sd_prefix_excl = self.gpu.alloc_zeros::<f32>(sd_buf_size)?;
        let mut gpu_sd_totals = self.gpu.alloc_zeros::<f32>(sd_total_size)?;

        // Build per-hand showdown payoffs (single spot: broadcast scalar)
        let num_sd = self.num_showdown_terminals as usize;
        let nh = self.num_hands as usize;

        let sd_win_host: Vec<f32> = self.gpu.download(&self.showdown_amount_win)?;
        let sd_lose_host: Vec<f32> = self.gpu.download(&self.showdown_amount_lose)?;

        // For single spot, amount_win/lose are scalar per terminal.
        // Persistent kernel expects [num_sd_terminals * num_hands] layout.
        let mut sd_win_per_hand = vec![0.0f32; num_sd * nh];
        let mut sd_lose_per_hand = vec![0.0f32; num_sd * nh];
        for t in 0..num_sd {
            for h in 0..nh {
                sd_win_per_hand[t * nh + h] = sd_win_host[t];
                sd_lose_per_hand[t * nh + h] = sd_lose_host[t];
            }
        }
        let gpu_sd_amount_win = self.gpu.upload(&sd_win_per_hand)?;
        let gpu_sd_amount_lose = self.gpu.upload(&sd_lose_per_hand)?;

        // Build per-hand fold payoffs (single spot: broadcast scalar)
        let num_fold = self.num_fold_terminals as usize;
        let fold_win_host: Vec<f32> = self.gpu.download(&self.fold_amount_win)?;
        let fold_lose_host: Vec<f32> = self.gpu.download(&self.fold_amount_lose)?;

        let mut fold_win_per_hand = vec![0.0f32; num_fold * nh];
        let mut fold_lose_per_hand = vec![0.0f32; num_fold * nh];
        for t in 0..num_fold {
            for h in 0..nh {
                fold_win_per_hand[t * nh + h] = fold_win_host[t];
                fold_lose_per_hand[t * nh + h] = fold_lose_host[t];
            }
        }
        let gpu_fold_amount_win = self.gpu.upload(&fold_win_per_hand)?;
        let gpu_fold_amount_lose = self.gpu.upload(&fold_lose_per_hand)?;

        // Barrier state
        let mut gpu_barrier_counter = self.gpu.alloc_zeros::<i32>(1)?;
        let mut gpu_barrier_sense = self.gpu.alloc_zeros::<i32>(1)?;

        // Build context
        let ctx = GpuSolverContext {
            regrets: self.gpu.get_device_ptr_mut(&mut self.regrets),
            strategy_sum: self.gpu.get_device_ptr_mut(&mut self.strategy_sum),
            strategy: self.gpu.get_device_ptr_mut(&mut self.current_strategy),
            reach_oop: self.gpu.get_device_ptr_mut(&mut self.reach_oop),
            reach_ip: self.gpu.get_device_ptr_mut(&mut self.reach_ip),
            cfvalues: self.gpu.get_device_ptr_mut(&mut self.cfvalues),

            child_offsets: self.gpu.get_device_ptr(&self.gpu_child_offsets),
            children: self.gpu.get_device_ptr(&self.gpu_children),
            infoset_ids: self.gpu.get_device_ptr(&self.gpu_infoset_ids),
            num_actions_arr: self.gpu.get_device_ptr(&self.gpu_num_actions),
            node_types: self.gpu.get_device_ptr(&gpu_node_types),

            level_offsets: self.gpu.get_device_ptr(&gpu_level_offsets),
            parent_nodes: self.gpu.get_device_ptr(&gpu_parent_nodes_full),
            parent_actions: self.gpu.get_device_ptr(&gpu_parent_actions_full),
            parent_infosets: self.gpu.get_device_ptr(&gpu_parent_infosets_full),
            parent_players: self.gpu.get_device_ptr(&gpu_parent_players_full),

            fold_terminal_nodes: self.gpu.get_device_ptr(&self.fold_terminal_nodes),
            fold_amount_win: self.gpu.get_device_ptr(&gpu_fold_amount_win),
            fold_amount_lose: self.gpu.get_device_ptr(&gpu_fold_amount_lose),
            fold_player: self.gpu.get_device_ptr(&self.fold_player),

            hand_cards_oop: self.gpu.get_device_ptr(&self.gpu_hand_cards_oop),
            hand_cards_ip: self.gpu.get_device_ptr(&self.gpu_hand_cards_ip),
            same_hand_index_oop: self.gpu.get_device_ptr(&self.gpu_same_hand_index_oop),
            same_hand_index_ip: self.gpu.get_device_ptr(&self.gpu_same_hand_index_ip),
            fold_total_opp_reach: self.gpu.get_device_ptr_mut(&mut self.gpu_fold_total_opp_reach),
            fold_per_card_reach: self.gpu.get_device_ptr_mut(&mut self.gpu_fold_per_card_reach),

            showdown_terminal_nodes: self.gpu.get_device_ptr(&self.showdown_terminal_nodes),
            showdown_amount_win: self.gpu.get_device_ptr(&gpu_sd_amount_win),
            showdown_amount_lose: self.gpu.get_device_ptr(&gpu_sd_amount_lose),

            sorted_opp_oop: self.gpu.get_device_ptr(&gpu_sorted_ip),
            sorted_opp_ip: self.gpu.get_device_ptr(&gpu_sorted_oop),
            rank_win_oop: self.gpu.get_device_ptr(&gpu_rank_win_oop),
            rank_next_oop: self.gpu.get_device_ptr(&gpu_rank_next_oop),
            rank_win_ip: self.gpu.get_device_ptr(&gpu_rank_win_ip),
            rank_next_ip: self.gpu.get_device_ptr(&gpu_rank_next_ip),
            sd_sorted_reach: self.gpu.get_device_ptr_mut(&mut gpu_sd_sorted_reach),
            sd_prefix_excl: self.gpu.get_device_ptr_mut(&mut gpu_sd_prefix_excl),
            sd_totals: self.gpu.get_device_ptr_mut(&mut gpu_sd_totals),

            decision_nodes_oop: self.gpu.get_device_ptr(&self.decision_nodes_oop),
            decision_nodes_ip: self.gpu.get_device_ptr(&self.decision_nodes_ip),

            initial_reach_oop: self.gpu.get_device_ptr(&self.gpu_initial_reach_oop),
            initial_reach_ip: self.gpu.get_device_ptr(&self.gpu_initial_reach_ip),

            num_hands: self.num_hands,
            max_actions: self.max_actions,
            num_infosets: self.num_infosets,
            num_nodes: self.num_nodes,
            num_levels: self.num_levels as u32,
            hands_per_spot: self.num_hands, // single spot: hps = num_hands
            num_spots: 1,
            num_fold_terminals: self.num_fold_terminals,
            num_showdown_terminals: self.num_showdown_terminals,
            num_oop_decisions: self.num_oop_decisions,
            num_ip_decisions: self.num_ip_decisions,
            max_iterations,

            barrier_counter: self.gpu.get_device_ptr_mut(&mut gpu_barrier_counter),
            barrier_sense: self.gpu.get_device_ptr_mut(&mut gpu_barrier_sense),
        };

        let gpu_ctx = self.gpu.upload(&[ctx])?;

        let block_size = 256u32;
        let num_blocks = 256u32;

        self.gpu.launch_dcfr_persistent(&gpu_ctx, num_blocks, block_size)?;

        // Synchronize
        self.gpu.stream.synchronize().map_err(GpuError::Driver)?;

        // Download strategy
        let strategy = self.gpu.download(&self.current_strategy)?;

        Ok(SolveResult {
            strategy,
            iterations: max_iterations,
        })
    }

    /// Run a single DCFR+ iteration and return all intermediate state
    /// for debugging. Used to compare GPU results against CPU step-by-step.
    pub fn debug_iteration(&mut self) -> Result<DebugIteration, GpuError> {
        let pos_discount = 0.0f32; // iteration 0: alpha_t = 0
        let neg_discount = 0.5f32;
        let strat_discount = 0.0f32; // iteration 0: gamma_t = 0

        // We'll capture the state after OOP traversal's reach setup for the snapshot.
        let mut cfvalues_oop = Vec::new();
        let mut cfvalues_ip = Vec::new();
        let mut saved_reach_oop = Vec::new();
        let mut saved_reach_ip = Vec::new();

        for traverser in 0..2u32 {
            // 1. Regret match
            self.gpu.launch_regret_match(
                &self.regrets,
                &self.gpu_num_actions,
                &mut self.current_strategy,
                self.num_infosets,
                self.max_actions,
                self.num_hands,
            )?;

            // 2. Init reach
            self.init_reach()?;

            // 3. Forward pass
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
                    self.num_hands,
                    self.max_actions,
                )?;
            }

            // Save reach after first traverser's forward pass
            if traverser == 0 {
                saved_reach_oop = self.gpu.download(&self.reach_oop)?;
                saved_reach_ip = self.gpu.download(&self.reach_ip)?;
            }

            // 4a. Zero cfvalues
            let cfv_size = (self.num_nodes * self.num_hands) as u32;
            self.gpu.launch_zero_buffer(&mut self.cfvalues, cfv_size)?;

            // 4b. Terminal fold eval — O(n) via inclusion-exclusion
            if self.num_fold_terminals > 0 {
                let opp_reach = if traverser == 0 {
                    &self.reach_ip
                } else {
                    &self.reach_oop
                };
                let opp_hand_cards = if traverser == 0 {
                    &self.gpu_hand_cards_ip
                } else {
                    &self.gpu_hand_cards_oop
                };
                let trav_hand_cards = if traverser == 0 {
                    &self.gpu_hand_cards_oop
                } else {
                    &self.gpu_hand_cards_ip
                };
                let same_hand_index = if traverser == 0 {
                    &self.gpu_same_hand_index_oop
                } else {
                    &self.gpu_same_hand_index_ip
                };

                self.gpu.launch_precompute_fold_aggregates(
                    opp_reach,
                    &self.fold_terminal_nodes,
                    opp_hand_cards,
                    &mut self.gpu_fold_total_opp_reach,
                    &mut self.gpu_fold_per_card_reach,
                    self.num_fold_terminals,
                    self.num_hands,
                )?;

                self.gpu.launch_fold_eval_from_aggregates(
                    &mut self.cfvalues,
                    opp_reach,
                    &self.fold_terminal_nodes,
                    &self.fold_amount_win,
                    &self.fold_amount_lose,
                    &self.fold_player,
                    &self.gpu_fold_total_opp_reach,
                    &self.gpu_fold_per_card_reach,
                    trav_hand_cards,
                    same_hand_index,
                    traverser,
                    self.num_fold_terminals,
                    self.num_hands,
                )?;
            }

            // 4c. Terminal showdown eval — shared memory optimization
            if self.num_showdown_terminals > 0 {
                let opp_reach = if traverser == 0 {
                    &self.reach_ip
                } else {
                    &self.reach_oop
                };
                let (traverser_strengths, opponent_strengths) = if traverser == 0 {
                    (&self.gpu_hand_strengths_oop, &self.gpu_hand_strengths_ip)
                } else {
                    (&self.gpu_hand_strengths_ip, &self.gpu_hand_strengths_oop)
                };
                let trav_hand_cards = if traverser == 0 {
                    &self.gpu_hand_cards_oop
                } else {
                    &self.gpu_hand_cards_ip
                };
                let opp_hand_cards = if traverser == 0 {
                    &self.gpu_hand_cards_ip
                } else {
                    &self.gpu_hand_cards_oop
                };
                self.gpu.launch_showdown_eval_fast(
                    &mut self.cfvalues,
                    opp_reach,
                    &self.showdown_terminal_nodes,
                    &self.showdown_amount_win,
                    &self.showdown_amount_lose,
                    traverser_strengths,
                    opponent_strengths,
                    self.num_showdown_terminals,
                    self.num_hands,
                    self.num_hands,  // hands_per_spot = num_hands for single-spot
                )?;
            }

            // 4d. Backward CFV
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
                    self.num_hands,
                    self.max_actions,
                )?;
            }

            // Save cfvalues
            let cfv_snapshot = self.gpu.download(&self.cfvalues)?;
            if traverser == 0 {
                cfvalues_oop = cfv_snapshot;
            } else {
                cfvalues_ip = cfv_snapshot;
            }

            // 4e. Update regrets
            let (decision_nodes, num_decision) = if traverser == 0 {
                (&self.decision_nodes_oop, self.num_oop_decisions)
            } else {
                (&self.decision_nodes_ip, self.num_ip_decisions)
            };

            if num_decision > 0 {
                self.gpu.launch_update_regrets(
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
                    self.num_hands,
                    self.max_actions,
                    pos_discount,
                    neg_discount,
                    strat_discount,
                )?;
            }
        }

        // Download final state
        let strategy = self.gpu.download(&self.current_strategy)?;
        let regrets = self.gpu.download(&self.regrets)?;
        let strategy_sum = self.gpu.download(&self.strategy_sum)?;

        Ok(DebugIteration {
            reach_oop: saved_reach_oop,
            reach_ip: saved_reach_ip,
            cfvalues_oop_traverser: cfvalues_oop,
            cfvalues_ip_traverser: cfvalues_ip,
            regrets,
            strategy,
            strategy_sum,
        })
    }

    /// Initialize reach probabilities at the root node.
    /// Fully GPU-resident: zero buffers then set root reach from pre-uploaded values.
    fn init_reach(&mut self) -> Result<(), GpuError> {
        let reach_size = (self.num_nodes * self.num_hands) as u32;
        self.gpu.launch_zero_buffer(&mut self.reach_oop, reach_size)?;
        self.gpu.launch_zero_buffer(&mut self.reach_ip, reach_size)?;
        self.gpu.launch_set_root_reach(&mut self.reach_oop, &self.gpu_initial_reach_oop, self.num_hands)?;
        self.gpu.launch_set_root_reach(&mut self.reach_ip, &self.gpu_initial_reach_ip, self.num_hands)?;
        Ok(())
    }

    /// Run additional DCFR+ iterations without resetting regrets or strategy_sum.
    ///
    /// `additional_iters`: number of new iterations to run.
    /// `already_done`: number of iterations already completed (used for discount
    ///   parameter computation so the discount schedule is continuous).
    ///
    /// This is the inner loop of `solve()` extracted for progressive resolving.
    pub fn solve_iterations(
        &mut self,
        additional_iters: u32,
        already_done: u32,
    ) -> Result<(), GpuError> {
        for t_offset in 1..=additional_iters {
            let t = already_done + t_offset; // 1-indexed global iteration
            let current_iteration = t - 1;   // 0-indexed

            let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
            let pow_alpha = t_alpha * t_alpha.sqrt();
            let pos_discount = (pow_alpha / (pow_alpha + 1.0)) as f32;
            let neg_discount = 0.5f32;

            let nearest_lower_power_of_4 = match current_iteration {
                0 => 0u32,
                x => 1u32 << ((x.leading_zeros() ^ 31) & !1),
            };
            let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;
            let strat_discount = ((t_gamma / (t_gamma + 1.0)).powi(3)) as f32;

            for traverser in 0..2u32 {
                self.gpu.launch_regret_match(
                    &self.regrets,
                    &self.gpu_num_actions,
                    &mut self.current_strategy,
                    self.num_infosets,
                    self.max_actions,
                    self.num_hands,
                )?;

                self.init_reach()?;

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
                        self.num_hands,
                        self.max_actions,
                    )?;
                }

                let cfv_size = (self.num_nodes * self.num_hands) as u32;
                self.gpu.launch_zero_buffer(&mut self.cfvalues, cfv_size)?;

                if self.num_fold_terminals > 0 {
                    let opp_reach = if traverser == 0 {
                        &self.reach_ip
                    } else {
                        &self.reach_oop
                    };
                    let opp_hand_cards = if traverser == 0 {
                        &self.gpu_hand_cards_ip
                    } else {
                        &self.gpu_hand_cards_oop
                    };
                    let trav_hand_cards = if traverser == 0 {
                        &self.gpu_hand_cards_oop
                    } else {
                        &self.gpu_hand_cards_ip
                    };
                    let same_hand_index = if traverser == 0 {
                        &self.gpu_same_hand_index_oop
                    } else {
                        &self.gpu_same_hand_index_ip
                    };

                    self.gpu.launch_precompute_fold_aggregates(
                        opp_reach,
                        &self.fold_terminal_nodes,
                        opp_hand_cards,
                        &mut self.gpu_fold_total_opp_reach,
                        &mut self.gpu_fold_per_card_reach,
                        self.num_fold_terminals,
                        self.num_hands,
                    )?;

                    self.gpu.launch_fold_eval_from_aggregates(
                        &mut self.cfvalues,
                        opp_reach,
                        &self.fold_terminal_nodes,
                        &self.fold_amount_win,
                        &self.fold_amount_lose,
                        &self.fold_player,
                        &self.gpu_fold_total_opp_reach,
                        &self.gpu_fold_per_card_reach,
                        trav_hand_cards,
                        same_hand_index,
                        traverser,
                        self.num_fold_terminals,
                        self.num_hands,
                    )?;
                }

                if self.num_showdown_terminals > 0 {
                    let opp_reach = if traverser == 0 {
                        &self.reach_ip
                    } else {
                        &self.reach_oop
                    };
                    let (traverser_strengths, opponent_strengths) = if traverser == 0 {
                        (&self.gpu_hand_strengths_oop, &self.gpu_hand_strengths_ip)
                    } else {
                        (&self.gpu_hand_strengths_ip, &self.gpu_hand_strengths_oop)
                    };
                    self.gpu.launch_showdown_eval_fast(
                        &mut self.cfvalues,
                        opp_reach,
                        &self.showdown_terminal_nodes,
                        &self.showdown_amount_win,
                        &self.showdown_amount_lose,
                        traverser_strengths,
                        opponent_strengths,
                        self.num_showdown_terminals,
                        self.num_hands,
                        self.num_hands,
                    )?;
                }

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
                        self.num_hands,
                        self.max_actions,
                    )?;
                }

                let (decision_nodes, num_decision) = if traverser == 0 {
                    (&self.decision_nodes_oop, self.num_oop_decisions)
                } else {
                    (&self.decision_nodes_ip, self.num_ip_decisions)
                };

                if num_decision > 0 {
                    self.gpu.launch_update_regrets(
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
                        self.num_hands,
                        self.max_actions,
                        pos_discount,
                        neg_discount,
                        strat_discount,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Extract the current averaged strategy from strategy_sum.
    ///
    /// Runs the extract_strategy kernel and downloads the result to host.
    /// Does not modify regrets or strategy_sum.
    pub fn extract_current_strategy(&mut self) -> Result<Vec<f32>, GpuError> {
        self.gpu.launch_extract_strategy(
            &self.strategy_sum,
            &self.gpu_num_actions,
            &mut self.current_strategy,
            self.num_infosets,
            self.max_actions,
            self.num_hands,
        )?;
        self.gpu.download(&self.current_strategy)
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::tree::FlatTree;
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{card_from_str, flop_from_str};
    use range_solver::range::Range;
    use range_solver::{CardConfig, PostFlopGame};

    fn make_river_game() -> PostFlopGame {
        let oop_range: Range = "AA,KK,QQ,AKs".parse().unwrap();
        let ip_range: Range = "QQ-JJ,AQs,AJs".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);
        game
    }

    #[test]
    fn test_gpu_solver_runs() {
        let mut game = make_river_game();
        let flat_tree = FlatTree::from_postflop_game(&mut game);
        let gpu = GpuContext::new(0).unwrap();

        let result = GpuSolver::new(&gpu, &flat_tree)
            .unwrap()
            .solve(100, None)
            .unwrap();

        assert_eq!(result.iterations, 100);
        // Strategy should have entries for all infosets
        assert!(!result.strategy.is_empty());
        assert_eq!(
            result.strategy.len(),
            flat_tree.num_infosets * flat_tree.max_actions() * flat_tree.num_hands
        );
    }

    #[test]
    fn test_gpu_solver_strategy_valid() {
        let mut game = make_river_game();
        let flat_tree = FlatTree::from_postflop_game(&mut game);
        let gpu = GpuContext::new(0).unwrap();

        let result = GpuSolver::new(&gpu, &flat_tree)
            .unwrap()
            .solve(200, None)
            .unwrap();

        let num_hands = flat_tree.num_hands;
        let max_actions = flat_tree.max_actions();

        // For each infoset, verify strategy sums to ~1.0 for each hand
        for iset in 0..flat_tree.num_infosets {
            let n_actions = flat_tree.infoset_num_actions[iset] as usize;
            for h in 0..num_hands {
                let mut total = 0.0f32;
                for a in 0..n_actions {
                    let idx = (iset * max_actions + a) * num_hands + h;
                    let prob = result.strategy[idx];
                    assert!(
                        prob >= -1e-5,
                        "negative probability {prob} at iset={iset} action={a} hand={h}"
                    );
                    total += prob;
                }
                assert!(
                    (total - 1.0).abs() < 1e-4,
                    "strategy doesn't sum to 1.0 for iset={iset} hand={h}: sum={total}"
                );
            }
        }
    }

    #[test]
    fn test_persistent_solver_runs() {
        let mut game = make_river_game();
        let flat_tree = FlatTree::from_postflop_game(&mut game);
        let gpu = GpuContext::new(0).unwrap();

        let result = GpuSolver::new(&gpu, &flat_tree)
            .unwrap()
            .solve_persistent(&flat_tree, 100, None)
            .unwrap();

        assert_eq!(result.iterations, 100);
        assert!(!result.strategy.is_empty());
        assert_eq!(
            result.strategy.len(),
            flat_tree.num_infosets * flat_tree.max_actions() * flat_tree.num_hands
        );
    }

    #[test]
    fn test_persistent_solver_strategy_valid() {
        let mut game = make_river_game();
        let flat_tree = FlatTree::from_postflop_game(&mut game);
        let gpu = GpuContext::new(0).unwrap();

        let result = GpuSolver::new(&gpu, &flat_tree)
            .unwrap()
            .solve_persistent(&flat_tree, 200, None)
            .unwrap();

        let num_hands = flat_tree.num_hands;
        let max_actions = flat_tree.max_actions();

        // For each infoset, verify strategy sums to ~1.0 for each hand
        for iset in 0..flat_tree.num_infosets {
            let n_actions = flat_tree.infoset_num_actions[iset] as usize;
            for h in 0..num_hands {
                let mut total = 0.0f32;
                for a in 0..n_actions {
                    let idx = (iset * max_actions + a) * num_hands + h;
                    let prob = result.strategy[idx];
                    assert!(
                        prob >= -1e-5,
                        "negative probability {prob} at iset={iset} action={a} hand={h}"
                    );
                    total += prob;
                }
                assert!(
                    (total - 1.0).abs() < 1e-4,
                    "strategy doesn't sum to 1.0 for iset={iset} hand={h}: sum={total}"
                );
            }
        }
    }

    #[test]
    fn test_persistent_solver_convergence() {
        // Test that the persistent solver produces reasonable strategies
        // at different iteration counts (convergence behavior).
        let mut game = make_river_game();
        let flat_tree = FlatTree::from_postflop_game(&mut game);
        let gpu = GpuContext::new(0).unwrap();

        // Run at low iterations
        let result_low = GpuSolver::new(&gpu, &flat_tree)
            .unwrap()
            .solve_persistent(&flat_tree, 50, None)
            .unwrap();

        // Run at higher iterations
        let result_high = GpuSolver::new(&gpu, &flat_tree)
            .unwrap()
            .solve_persistent(&flat_tree, 500, None)
            .unwrap();

        // Strategies should be valid at both
        let num_hands = flat_tree.num_hands;
        let max_actions = flat_tree.max_actions();

        for result in &[&result_low, &result_high] {
            for iset in 0..flat_tree.num_infosets {
                let n_actions = flat_tree.infoset_num_actions[iset] as usize;
                for h in 0..num_hands {
                    let mut total = 0.0f32;
                    for a in 0..n_actions {
                        let idx = (iset * max_actions + a) * num_hands + h;
                        total += result.strategy[idx];
                    }
                    assert!(
                        (total - 1.0).abs() < 1e-4,
                        "strategy doesn't sum to 1.0 for iset={iset} hand={h}: sum={total}"
                    );
                }
            }
        }

        // Higher iteration strategies should be at least somewhat different
        // from lower iteration (solver is actually running and updating)
        let max_diff = result_low.strategy.iter()
            .zip(result_high.strategy.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        eprintln!("50 vs 500 iteration strategy max diff = {max_diff}");
        // The strategies should have evolved (max_diff > 0 proves the solver iterates)
        assert!(
            max_diff > 0.001,
            "Strategies at 50 and 500 iterations are identical (solver may not be iterating)"
        );
    }
}
