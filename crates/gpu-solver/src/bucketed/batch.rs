//! BatchBucketedSolver: solves N river spots simultaneously on the GPU.
//!
//! All spots share the same tree topology (same bet sizes => same action tree)
//! but differ in:
//! - Initial reach (bucket-space, per spot)
//! - Equity tables (per spot per showdown terminal — different boards)
//! - Fold/showdown half_pots (per spot — different pot sizes)
//!
//! The solver treats `num_spots * num_buckets` as `total_hands` for all kernels.
//! The generic kernels (regret_match, forward_pass, backward_cfv, etc.) work
//! unchanged — they just see a larger hand dimension. The batch fold/showdown
//! kernels index into per-spot terminal data.
//!
//! After solving, CFVs are extracted from the root node for both players.

#[cfg(feature = "cuda")]
use crate::gpu::{GpuContext, GpuError};
#[cfg(feature = "cuda")]
use crate::tree::NodeType;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
use super::equity::{compute_bucket_equity_table, BucketedBoardCache};
#[cfg(feature = "cuda")]
use super::sampler::BucketedSituation;
#[cfg(feature = "cuda")]
use super::tree::BucketedTree;
#[cfg(feature = "cuda")]
use poker_solver_core::blueprint_v2::bucket_file::BucketFile;

/// Precomputed per-level data for GPU forward pass kernels.
#[cfg(feature = "cuda")]
struct LevelData {
    nodes: CudaSlice<u32>,
    parent_nodes: CudaSlice<u32>,
    parent_actions: CudaSlice<u32>,
    parent_infosets: CudaSlice<u32>,
    parent_players: CudaSlice<u32>,
    num_nodes: u32,
}

/// Precomputed per-level decision nodes for backward CFV pass.
#[cfg(feature = "cuda")]
struct BackwardLevelData {
    decision_nodes: CudaSlice<u32>,
    decision_players: CudaSlice<u32>,
    num_decision_nodes: u32,
}

/// Result from batch solving: root CFVs for both players across all spots.
#[cfg(feature = "cuda")]
pub struct BatchSolveResult {
    /// OOP CFVs: `[num_spots * num_buckets]` — root node CFVs for OOP.
    pub cfvs_oop: Vec<f32>,
    /// IP CFVs: `[num_spots * num_buckets]` — root node CFVs for IP.
    pub cfvs_ip: Vec<f32>,
    /// Number of iterations completed.
    pub iterations: u32,
}

/// GPU Supremus DCFR+ solver operating on multiple river spots simultaneously.
///
/// Each spot shares the same tree topology (action tree) but has different:
/// - Initial bucket-space reach (different ranges per spot)
/// - Equity tables (different boards per spot)
/// - Pot sizes (different pot/stack per spot)
///
/// The generic CFR kernels see `total_hands = num_spots * num_buckets` and
/// work unchanged. The terminal kernels (fold/showdown) use batch variants
/// that index per-spot terminal data.
#[cfg(feature = "cuda")]
pub struct BatchBucketedSolver<'a> {
    gpu: &'a GpuContext,

    // Solver state on GPU
    regrets: CudaSlice<f32>,
    strategy_sum: CudaSlice<f32>,
    current_strategy: CudaSlice<f32>,
    reach_oop: CudaSlice<f32>,
    reach_ip: CudaSlice<f32>,
    cfvalues: CudaSlice<f32>,

    // Tree topology on GPU (shared across all spots)
    gpu_child_offsets: CudaSlice<u32>,
    gpu_children: CudaSlice<u32>,
    gpu_infoset_ids: CudaSlice<u32>,
    gpu_num_actions: CudaSlice<u32>,

    // Per-level data for forward pass
    level_data: Vec<LevelData>,

    // Per-level data for backward CFV pass
    backward_level_data: Vec<BackwardLevelData>,

    // Batch fold terminal data
    fold_terminal_nodes: CudaSlice<u32>,
    fold_half_pots: CudaSlice<f32>,    // [num_fold_terminals * num_spots]
    fold_player: CudaSlice<u32>,       // [num_fold_terminals]
    num_fold_terminals: u32,

    // Batch showdown terminal data
    showdown_terminal_nodes: CudaSlice<u32>,
    showdown_equity_tables: CudaSlice<f32>, // [num_sd * num_spots * nb * nb]
    showdown_half_pots: CudaSlice<f32>,     // [num_sd * num_spots]
    num_showdown_terminals: u32,

    // Initial reach (concatenated for all spots)
    gpu_initial_reach_oop: CudaSlice<f32>,  // [num_spots * num_buckets]
    gpu_initial_reach_ip: CudaSlice<f32>,

    // Decision nodes per player (for regret updates)
    decision_nodes_oop: CudaSlice<u32>,
    decision_nodes_ip: CudaSlice<u32>,
    num_oop_decisions: u32,
    num_ip_decisions: u32,

    // Dimensions
    num_buckets: u32,
    num_spots: u32,
    total_hands: u32,  // = num_spots * num_buckets
    max_actions: u32,
    num_infosets: u32,
    num_nodes: u32,
    num_levels: usize,
}

#[cfg(feature = "cuda")]
impl<'a> BatchBucketedSolver<'a> {
    /// Create a batch solver from a reference tree and a set of situations.
    ///
    /// The reference tree provides the shared topology (action tree structure).
    /// Each situation provides per-spot data: reach, board (for equity), pot.
    ///
    /// # Arguments
    /// - `gpu`: GPU context
    /// - `ref_tree`: reference BucketedTree (topology only — terminal data ignored)
    /// - `situations`: batch of river situations in bucket space
    /// - `bucket_file`: loaded bucket file for equity table computation
    /// - `board_cache`: precomputed board-to-index mapping
    pub fn new(
        gpu: &'a GpuContext,
        ref_tree: &BucketedTree,
        situations: &[BucketedSituation],
        bucket_file: &BucketFile,
        _board_cache: &BucketedBoardCache,
    ) -> Result<Self, GpuError> {
        let num_spots = situations.len() as u32;
        let num_buckets = ref_tree.num_buckets as u32;
        let total_hands = num_spots * num_buckets;
        let num_nodes = ref_tree.num_nodes() as u32;
        let num_infosets = ref_tree.num_infosets as u32;
        let max_actions = ref_tree.max_actions() as u32;
        let num_levels = ref_tree.num_levels();

        assert!(num_spots > 0, "need at least one situation");

        let strat_size = (num_infosets * max_actions * total_hands) as usize;
        let reach_size = (num_nodes * total_hands) as usize;

        // Allocate solver state
        let regrets = gpu.alloc_zeros::<f32>(strat_size)?;
        let strategy_sum = gpu.alloc_zeros::<f32>(strat_size)?;
        let current_strategy = gpu.alloc_zeros::<f32>(strat_size)?;
        let reach_oop = gpu.alloc_zeros::<f32>(reach_size)?;
        let reach_ip = gpu.alloc_zeros::<f32>(reach_size)?;
        let cfvalues = gpu.alloc_zeros::<f32>(reach_size)?;

        // Upload tree topology (shared)
        let gpu_child_offsets = gpu.upload(&ref_tree.child_offsets)?;
        let gpu_children = gpu.upload(&ref_tree.children)?;
        let gpu_infoset_ids = gpu.upload(&ref_tree.infoset_ids)?;
        let gpu_num_actions = gpu.upload(&ref_tree.infoset_num_actions)?;

        // Build per-level data for forward pass
        let mut level_data = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            let start = ref_tree.level_starts[level] as usize;
            let end = ref_tree.level_starts[level + 1] as usize;

            let mut nodes_vec = Vec::with_capacity(end - start);
            let mut parent_nodes_vec = Vec::with_capacity(end - start);
            let mut parent_actions_vec = Vec::with_capacity(end - start);
            let mut parent_infosets_vec = Vec::with_capacity(end - start);
            let mut parent_players_vec = Vec::with_capacity(end - start);

            for node_id in start..end {
                nodes_vec.push(node_id as u32);
                let parent = ref_tree.parent_nodes[node_id];
                parent_nodes_vec.push(parent);
                parent_actions_vec.push(ref_tree.parent_actions[node_id]);
                if parent != u32::MAX {
                    parent_infosets_vec.push(ref_tree.infoset_ids[parent as usize]);
                    parent_players_vec.push(ref_tree.player(parent as usize) as u32);
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
            let start = ref_tree.level_starts[level] as usize;
            let end = ref_tree.level_starts[level + 1] as usize;

            let mut decision_nodes_vec = Vec::new();
            let mut decision_players_vec = Vec::new();
            for node_id in start..end {
                if !ref_tree.is_terminal(node_id) {
                    decision_nodes_vec.push(node_id as u32);
                    decision_players_vec.push(ref_tree.player(node_id) as u32);
                }
            }

            let num_decision = decision_nodes_vec.len() as u32;
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

        // Identify fold and showdown terminal nodes from the reference tree
        let mut fold_node_ids: Vec<u32> = Vec::new();
        let mut fold_ordinals_map: Vec<usize> = Vec::new(); // fold ordinal per fold terminal
        let mut sd_node_ids: Vec<u32> = Vec::new();
        let mut sd_ordinals_map: Vec<usize> = Vec::new();

        for &node_id in &ref_tree.terminal_indices {
            match ref_tree.node_types[node_id as usize] {
                NodeType::TerminalFold => {
                    let term_pos = ref_tree
                        .terminal_indices
                        .iter()
                        .position(|&n| n == node_id)
                        .unwrap();
                    fold_node_ids.push(node_id);
                    fold_ordinals_map.push(ref_tree.fold_ordinals[term_pos] as usize);
                }
                NodeType::TerminalShowdown => {
                    let term_pos = ref_tree
                        .terminal_indices
                        .iter()
                        .position(|&n| n == node_id)
                        .unwrap();
                    sd_node_ids.push(node_id);
                    sd_ordinals_map.push(ref_tree.showdown_ordinals[term_pos] as usize);
                }
                _ => {}
            }
        }

        let num_fold_terminals = fold_node_ids.len() as u32;
        let num_showdown_terminals = sd_node_ids.len() as u32;
        let ns = num_spots as usize;
        let nb = num_buckets as usize;

        // Build per-spot fold data: half_pots[term_idx * num_spots + spot]
        // The fold player is the same for all spots (topology shared).
        let mut fold_hps = vec![0.0f32; fold_node_ids.len() * ns];
        let mut fold_pl = Vec::with_capacity(fold_node_ids.len());

        for (fi, &_node_id) in fold_node_ids.iter().enumerate() {
            let fold_ord = fold_ordinals_map[fi];
            fold_pl.push(ref_tree.fold_players[fold_ord]);

            // Compute per-spot half_pot using each situation's pot
            // The ref_tree's fold_half_pots are based on the ref pot.
            // We need to scale by the ratio of each spot's pot to the ref pot.
            // Actually, the pot at this terminal is determined by the tree topology
            // and the starting pot. Since all spots share topology, the pot at each
            // node is: starting_pot + 2 * matched_bets.
            // The matched_bets depend only on the action sequence (shared).
            // So the pot ratio at a terminal is fixed by the action sequence.
            //
            // For the batch solver, each spot has its own starting pot, so:
            //   pot_at_terminal(spot) = sit.pot + 2 * matched_bets
            // where matched_bets is the same for all spots.
            //
            // The ref_tree stores absolute pots based on its starting pot.
            // We can compute the ratio: pot_factor = terminal_pot / starting_pot.
            // Then for each spot: half_pot = (sit.pot * pot_factor) / 2.
            //
            // Actually, the ref_tree pots track the absolute pot at each node.
            // For a tree starting with `starting_pot` and `effective_stack`:
            //   pot_at_node = starting_pot + 2 * total_bet_at_node
            // So the additional chips = pot_at_node - starting_pot = 2 * total_bet_at_node
            //
            // For a different starting pot P_s:
            //   new_pot_at_node = P_s + (pot_at_node - ref_starting_pot)
            //   new_half_pot = new_pot_at_node / 2
            //
            // But we don't store ref_starting_pot. Let's use the node's pot from
            // the ref_tree and compute the bet increment.
            // ref_tree.pots[node_id] = ref_starting_pot + 2*bets
            // For spot s with starting pot P_s:
            //   spot_pot = P_s + (ref_tree.pots[node_id] - ref_starting_pot)
            //   spot_half_pot = spot_pot / 2
            //
            // We need the ref_starting_pot. The root node pot IS the starting pot.
            let ref_pot = ref_tree.pots[0]; // root pot = starting pot
            let terminal_pot = ref_tree.pots[fold_node_ids[fi] as usize];
            let bet_increment = terminal_pot - ref_pot;

            for (si, sit) in situations.iter().enumerate() {
                let spot_pot = sit.pot as f32 + bet_increment;
                fold_hps[fi * ns + si] = spot_pot / 2.0;
            }
        }

        // Ensure non-empty
        if fold_node_ids.is_empty() {
            fold_node_ids.push(0);
            fold_hps.push(0.0);
            fold_pl.push(0);
        }

        let fold_terminal_nodes = gpu.upload(&fold_node_ids)?;
        let fold_half_pots = gpu.upload(&fold_hps)?;
        let fold_player = gpu.upload(&fold_pl)?;

        // Build per-spot showdown data:
        // equity_tables[(term_idx * num_spots + spot) * nb * nb + ...]
        // half_pots[term_idx * num_spots + spot]
        let mut sd_equity_flat: Vec<f32> = Vec::with_capacity(
            sd_node_ids.len() * ns * nb * nb,
        );
        let mut sd_hps = vec![0.0f32; sd_node_ids.len() * ns];

        for (si_term, &_node_id) in sd_node_ids.iter().enumerate() {
            let ref_pot = ref_tree.pots[0];
            let terminal_pot = ref_tree.pots[sd_node_ids[si_term] as usize];
            let bet_increment = terminal_pot - ref_pot;

            for (si, sit) in situations.iter().enumerate() {
                // Compute equity table for this spot's board
                let eq_table = compute_bucket_equity_table(
                    &sit.board,
                    bucket_file,
                    sit.board_idx,
                    nb,
                );
                sd_equity_flat.extend_from_slice(&eq_table);

                let spot_pot = sit.pot as f32 + bet_increment;
                sd_hps[si_term * ns + si] = spot_pot / 2.0;
            }
        }

        // Ensure non-empty
        if sd_node_ids.is_empty() {
            sd_node_ids.push(0);
            sd_hps.resize(ns, 0.0);
            sd_equity_flat.resize(ns * nb * nb, 0.0);
        }

        let showdown_terminal_nodes = gpu.upload(&sd_node_ids)?;
        let showdown_half_pots = gpu.upload(&sd_hps)?;
        let showdown_equity_tables = gpu.upload(&sd_equity_flat)?;

        // Build decision node lists per player (for regret updates)
        let mut oop_decisions = Vec::new();
        let mut ip_decisions = Vec::new();
        for (i, nt) in ref_tree.node_types.iter().enumerate() {
            match nt {
                NodeType::DecisionOop => oop_decisions.push(i as u32),
                NodeType::DecisionIp => ip_decisions.push(i as u32),
                _ => {}
            }
        }

        let num_oop_decisions = oop_decisions.len() as u32;
        let num_ip_decisions = ip_decisions.len() as u32;

        if oop_decisions.is_empty() {
            oop_decisions.push(0);
        }
        if ip_decisions.is_empty() {
            ip_decisions.push(0);
        }

        let decision_nodes_oop = gpu.upload(&oop_decisions)?;
        let decision_nodes_ip = gpu.upload(&ip_decisions)?;

        // Concatenate initial reach for all spots: [spot0_bucket0..spot0_bucketN, spot1_bucket0..., ...]
        let mut initial_oop = Vec::with_capacity((total_hands) as usize);
        let mut initial_ip = Vec::with_capacity((total_hands) as usize);
        for sit in situations {
            initial_oop.extend_from_slice(&sit.oop_reach);
            initial_ip.extend_from_slice(&sit.ip_reach);
        }

        let gpu_initial_reach_oop = gpu.upload(&initial_oop)?;
        let gpu_initial_reach_ip = gpu.upload(&initial_ip)?;

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
            num_spots,
            total_hands,
            max_actions,
            num_infosets,
            num_nodes,
            num_levels,
        })
    }

    /// Solve all spots and return root CFVs for both players.
    ///
    /// Runs Supremus DCFR+ for `max_iterations` with strategy accumulation
    /// starting after `delay` iterations.
    ///
    /// Returns root CFVs: `cfvs_oop[spot * num_buckets + bucket]` is the
    /// OOP counterfactual value for bucket `bucket` in spot `spot`.
    pub fn solve_with_cfvs(
        &mut self,
        max_iterations: u32,
        delay: u32,
    ) -> Result<BatchSolveResult, GpuError> {
        for t in 1..=max_iterations {
            for traverser in 0..2u32 {
                // 1. Regret match -> current strategy
                self.gpu.launch_regret_match(
                    &self.regrets,
                    &self.gpu_num_actions,
                    &mut self.current_strategy,
                    self.num_infosets,
                    self.max_actions,
                    self.total_hands,
                )?;

                // 2. Initialize reach at root
                self.init_reach()?;

                // 3. Forward pass (top-down)
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
                        self.total_hands,
                        self.max_actions,
                    )?;
                }

                // 4a. Zero out cfvalues
                let cfv_size = self.num_nodes * self.total_hands;
                self.gpu.launch_zero_buffer(&mut self.cfvalues, cfv_size)?;

                // 4b. Terminal fold eval (batch kernel)
                if self.num_fold_terminals > 0 {
                    let opp_reach = if traverser == 0 {
                        &self.reach_ip
                    } else {
                        &self.reach_oop
                    };
                    self.gpu.launch_bucketed_fold_eval_batch(
                        &mut self.cfvalues,
                        opp_reach,
                        &self.fold_terminal_nodes,
                        &self.fold_half_pots,
                        &self.fold_player,
                        traverser,
                        self.num_fold_terminals,
                        self.total_hands,
                        self.num_buckets,
                        self.num_spots,
                    )?;
                }

                // 4c. Terminal showdown eval (batch kernel)
                if self.num_showdown_terminals > 0 {
                    let opp_reach = if traverser == 0 {
                        &self.reach_ip
                    } else {
                        &self.reach_oop
                    };
                    self.gpu.launch_bucketed_showdown_eval_batch(
                        &mut self.cfvalues,
                        opp_reach,
                        &self.showdown_terminal_nodes,
                        &self.showdown_equity_tables,
                        &self.showdown_half_pots,
                        self.num_showdown_terminals,
                        self.total_hands,
                        self.num_buckets,
                        self.num_spots,
                    )?;
                }

                // 4d. Backward CFV (bottom-up)
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
                        self.total_hands,
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
                        self.total_hands,
                        self.max_actions,
                        t,
                        delay,
                    )?;
                }
            }
        }

        // After the final iteration, run one more traverser=0 and traverser=1
        // pass to get both players' CFVs at the root.
        // Actually, the cfvalues buffer after the last iteration contains
        // CFVs for the last traverser. We need both.
        //
        // Alternative: after the main loop, do two final forward+CFV passes
        // using the averaged strategy to extract root CFVs for both players.
        //
        // Simplest approach: run two final iterations where we extract CFVs.
        // The cfvalues after traverser=0 pass give OOP CFVs at root.
        // The cfvalues after traverser=1 pass give IP CFVs at root.
        //
        // For speed, we just do one last pair of forward+terminal+backward
        // passes with the final averaged strategy.

        // Extract averaged strategy
        self.gpu.launch_extract_strategy(
            &self.strategy_sum,
            &self.gpu_num_actions,
            &mut self.current_strategy,
            self.num_infosets,
            self.max_actions,
            self.total_hands,
        )?;

        // OOP CFVs: run forward + terminal + backward with traverser=0
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
                self.total_hands,
                self.max_actions,
            )?;
        }

        let cfv_size = self.num_nodes * self.total_hands;
        self.gpu.launch_zero_buffer(&mut self.cfvalues, cfv_size)?;

        if self.num_fold_terminals > 0 {
            self.gpu.launch_bucketed_fold_eval_batch(
                &mut self.cfvalues,
                &self.reach_ip, // opp of OOP
                &self.fold_terminal_nodes,
                &self.fold_half_pots,
                &self.fold_player,
                0, // traverser=OOP
                self.num_fold_terminals,
                self.total_hands,
                self.num_buckets,
                self.num_spots,
            )?;
        }
        if self.num_showdown_terminals > 0 {
            self.gpu.launch_bucketed_showdown_eval_batch(
                &mut self.cfvalues,
                &self.reach_ip,
                &self.showdown_terminal_nodes,
                &self.showdown_equity_tables,
                &self.showdown_half_pots,
                self.num_showdown_terminals,
                self.total_hands,
                self.num_buckets,
                self.num_spots,
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
                0, // traverser=OOP
                bld.num_decision_nodes,
                self.total_hands,
                self.max_actions,
            )?;
        }

        // Download OOP root CFVs (node 0, all hands)
        let all_cfv = self.gpu.download(&self.cfvalues)?;
        let cfvs_oop = all_cfv[0..(self.total_hands as usize)].to_vec();

        // IP CFVs: same but traverser=1
        self.gpu.launch_zero_buffer(&mut self.cfvalues, cfv_size)?;

        if self.num_fold_terminals > 0 {
            self.gpu.launch_bucketed_fold_eval_batch(
                &mut self.cfvalues,
                &self.reach_oop, // opp of IP
                &self.fold_terminal_nodes,
                &self.fold_half_pots,
                &self.fold_player,
                1, // traverser=IP
                self.num_fold_terminals,
                self.total_hands,
                self.num_buckets,
                self.num_spots,
            )?;
        }
        if self.num_showdown_terminals > 0 {
            self.gpu.launch_bucketed_showdown_eval_batch(
                &mut self.cfvalues,
                &self.reach_oop,
                &self.showdown_terminal_nodes,
                &self.showdown_equity_tables,
                &self.showdown_half_pots,
                self.num_showdown_terminals,
                self.total_hands,
                self.num_buckets,
                self.num_spots,
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
                1, // traverser=IP
                bld.num_decision_nodes,
                self.total_hands,
                self.max_actions,
            )?;
        }

        let all_cfv_ip = self.gpu.download(&self.cfvalues)?;
        let cfvs_ip = all_cfv_ip[0..(self.total_hands as usize)].to_vec();

        Ok(BatchSolveResult {
            cfvs_oop,
            cfvs_ip,
            iterations: max_iterations,
        })
    }

    /// Initialize reach probabilities: zero everything, set root reach from initial.
    fn init_reach(&mut self) -> Result<(), GpuError> {
        let reach_size = self.num_nodes * self.total_hands;
        self.gpu
            .launch_zero_buffer(&mut self.reach_oop, reach_size)?;
        self.gpu
            .launch_zero_buffer(&mut self.reach_ip, reach_size)?;
        self.gpu.launch_set_root_reach(
            &mut self.reach_oop,
            &self.gpu_initial_reach_oop,
            self.total_hands,
        )?;
        self.gpu.launch_set_root_reach(
            &mut self.reach_ip,
            &self.gpu_initial_reach_ip,
            self.total_hands,
        )?;
        Ok(())
    }

    /// Get the number of spots in this batch.
    pub fn num_spots(&self) -> u32 {
        self.num_spots
    }

    /// Get the number of buckets.
    pub fn num_buckets(&self) -> u32 {
        self.num_buckets
    }

    /// Get the total hands dimension (num_spots * num_buckets).
    pub fn total_hands(&self) -> u32 {
        self.total_hands
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::bucketed::equity::BucketedBoardCache;
    use crate::bucketed::sampler::BucketedSituation;
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

    /// Test: batch solver with 1 spot should produce results similar to single solver.
    #[test]
    fn test_batch_solver_single_spot() {
        let gpu = GpuContext::new(0).expect("CUDA device required");

        let flop = range_solver::card::flop_from_str("Qs Jh 2c").unwrap();
        let turn = range_solver::card::card_from_str("8d").unwrap();
        let river = range_solver::card::card_from_str("3s").unwrap();
        let bet_sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();

        let board_u8: [u8; 5] = [flop[0], flop[1], flop[2], turn, river];
        let num_buckets = 10u16;

        let bf = make_synthetic_bucket_file(&board_u8, num_buckets);
        let cache = BucketedBoardCache::new(&bf);

        let mut game = build_river_game(flop, turn, river, 100, 100, &bet_sizes);
        let ref_tree = BucketedTree::from_postflop_game(
            &mut game,
            &bf,
            &cache,
            num_buckets as usize,
        );

        let initial_reach = vec![1.0f32; num_buckets as usize];
        let situation = BucketedSituation {
            board: board_u8,
            board_idx: cache.find_board_index(&board_u8).unwrap(),
            pot: 100,
            effective_stack: 100,
            oop_reach: initial_reach.clone(),
            ip_reach: initial_reach.clone(),
        };

        let mut batch_solver = BatchBucketedSolver::new(
            &gpu,
            &ref_tree,
            &[situation],
            &bf,
            &cache,
        )
        .expect("batch solver creation");

        let result = batch_solver.solve_with_cfvs(200, 50).expect("batch solve");

        let nb = num_buckets as usize;

        // CFVs should be approximately anti-symmetric: oop + ip ~ 0
        for b in 0..nb {
            let sum = result.cfvs_oop[b] + result.cfvs_ip[b];
            assert!(
                sum.abs() < 5.0,
                "CFV sum at bucket {b}: OOP={}, IP={}, sum={sum} (expected ~0)",
                result.cfvs_oop[b],
                result.cfvs_ip[b],
            );
        }

        // At least some CFVs should be non-zero
        let oop_nonzero = result.cfvs_oop.iter().filter(|&&v| v.abs() > 0.01).count();
        assert!(
            oop_nonzero > 0,
            "OOP CFVs all near zero — solver may not be working"
        );
    }

    /// Test: batch solver with multiple identical spots should produce identical CFVs.
    #[test]
    fn test_batch_solver_identical_spots() {
        let gpu = GpuContext::new(0).expect("CUDA device required");

        let flop = range_solver::card::flop_from_str("Qs Jh 2c").unwrap();
        let turn = range_solver::card::card_from_str("8d").unwrap();
        let river = range_solver::card::card_from_str("3s").unwrap();
        let bet_sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();

        let board_u8: [u8; 5] = [flop[0], flop[1], flop[2], turn, river];
        let num_buckets = 10u16;

        let bf = make_synthetic_bucket_file(&board_u8, num_buckets);
        let cache = BucketedBoardCache::new(&bf);

        let mut game = build_river_game(flop, turn, river, 100, 100, &bet_sizes);
        let ref_tree = BucketedTree::from_postflop_game(
            &mut game,
            &bf,
            &cache,
            num_buckets as usize,
        );

        let initial_reach = vec![1.0f32; num_buckets as usize];
        let nb = num_buckets as usize;

        // Create 3 identical situations
        let situations: Vec<BucketedSituation> = (0..3)
            .map(|_| BucketedSituation {
                board: board_u8,
                board_idx: cache.find_board_index(&board_u8).unwrap(),
                pot: 100,
                effective_stack: 100,
                oop_reach: initial_reach.clone(),
                ip_reach: initial_reach.clone(),
            })
            .collect();

        let mut batch_solver = BatchBucketedSolver::new(
            &gpu,
            &ref_tree,
            &situations,
            &bf,
            &cache,
        )
        .expect("batch solver creation");

        let result = batch_solver.solve_with_cfvs(200, 50).expect("batch solve");

        // All 3 spots should have identical CFVs
        for spot in 1..3 {
            for b in 0..nb {
                let v0 = result.cfvs_oop[b];
                let vs = result.cfvs_oop[spot * nb + b];
                assert!(
                    (v0 - vs).abs() < 0.5,
                    "OOP CFV mismatch spot 0 vs {spot}, bucket {b}: {v0} vs {vs}"
                );
            }
        }
    }
}
