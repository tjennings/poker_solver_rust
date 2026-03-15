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

    // Decision node lists per player
    decision_nodes_oop: CudaSlice<u32>,
    decision_nodes_ip: CudaSlice<u32>,
    num_oop_decisions: u32,
    num_ip_decisions: u32,

    // Initial reach (host, uploaded each iteration)
    initial_reach_oop: Vec<f32>,
    initial_reach_ip: Vec<f32>,

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
            for node_id in start..end {
                if !tree.is_terminal(node_id) {
                    decision_nodes_vec.push(node_id as u32);
                }
            }

            let num_decision = decision_nodes_vec.len() as u32;
            // Ensure we have at least one element for GPU buffer
            if decision_nodes_vec.is_empty() {
                decision_nodes_vec.push(0);
            }
            backward_level_data.push(BackwardLevelData {
                decision_nodes: gpu.upload(&decision_nodes_vec)?,
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
            decision_nodes_oop,
            decision_nodes_ip,
            num_oop_decisions,
            num_ip_decisions,
            initial_reach_oop: tree.initial_reach_oop.clone(),
            initial_reach_ip: tree.initial_reach_ip.clone(),
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
            // 1. Regret match -> current strategy
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

            // 3. Forward pass (top-down, level by level, skip level 0 = root)
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

            // 4. For each traverser, compute CFV and update regrets
            for traverser in 0..2u32 {
                // 4a. Zero out cfvalues
                self.cfvalues = self
                    .gpu
                    .alloc_zeros::<f32>((self.num_nodes * self.num_hands) as usize)?;

                // 4b. Terminal fold eval
                if self.num_fold_terminals > 0 {
                    let opp_reach = if traverser == 0 {
                        &self.reach_ip
                    } else {
                        &self.reach_oop
                    };
                    self.gpu.launch_terminal_fold_eval(
                        &mut self.cfvalues,
                        opp_reach,
                        &self.fold_terminal_nodes,
                        &self.fold_amount_win,
                        &self.fold_amount_lose,
                        &self.fold_player,
                        traverser,
                        self.num_fold_terminals,
                        self.num_hands,
                    )?;
                }

                // 4c. Terminal showdown eval
                if self.num_showdown_terminals > 0 {
                    let opp_reach = if traverser == 0 {
                        &self.reach_ip
                    } else {
                        &self.reach_oop
                    };
                    let hand_strengths = if traverser == 0 {
                        &self.gpu_hand_strengths_oop
                    } else {
                        &self.gpu_hand_strengths_ip
                    };
                    self.gpu.launch_terminal_showdown_eval(
                        &mut self.cfvalues,
                        opp_reach,
                        &self.showdown_terminal_nodes,
                        &self.showdown_amount_win,
                        &self.showdown_amount_lose,
                        hand_strengths,
                        self.num_showdown_terminals,
                        self.num_hands,
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
                    // DCFR+ discount factors
                    let alpha = 1.5f32;
                    let t_f = t as f32;
                    let t_alpha = t_f.powf(alpha);
                    let pos_discount = t_alpha / (t_alpha + 1.0);
                    let neg_discount = 0.5;
                    let strat_weight = (t_f - 100.0).max(0.0);

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
                        strat_weight,
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

    /// Initialize reach probabilities at the root node.
    fn init_reach(&mut self) -> Result<(), GpuError> {
        // Build full reach arrays: only root (node 0) gets initial values
        let reach_size = (self.num_nodes * self.num_hands) as usize;
        let mut reach_oop_host = vec![0.0f32; reach_size];
        let mut reach_ip_host = vec![0.0f32; reach_size];

        // Root is node 0
        for h in 0..self.num_hands as usize {
            reach_oop_host[h] = self.initial_reach_oop[h];
            reach_ip_host[h] = self.initial_reach_ip[h];
        }

        self.reach_oop = self.gpu.upload(&reach_oop_host)?;
        self.reach_ip = self.gpu.upload(&reach_ip_host)?;

        Ok(())
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
}
