//! Persistent GPU batch solver for solving multiple subgames with the same topology.
//!
//! Reuses CUDA context, compiled kernel, and topology uploads across batches.
//! Each call to `solve_batch` only uploads per-subgame data (weights, showdown outcomes).
#![allow(clippy::too_many_arguments)]

use crate::extract::{NodeType, TerminalData, TreeTopology};
use crate::gpu::{compute_hand_parallel_shared_mem, GpuHandParallelState, HandParallelKernel};
use crate::solver::{build_mega_terminal_data, build_sorted_topology, upload_or_dummy_f32, upload_or_dummy_i32};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Per-subgame input: initial weights and showdown outcome matrices.
#[derive(Clone)]
pub struct SubgameSpec {
    /// Per-player initial weights, length `num_hands` each.
    pub initial_weights: [Vec<f32>; 2],
    /// Pre-scaled showdown outcomes for player 0 traversal: `[num_showdowns * H * H]`.
    pub showdown_outcomes_p0: Vec<f32>,
    /// Pre-scaled showdown outcomes for player 1 traversal: `[num_showdowns * H * H]`.
    pub showdown_outcomes_p1: Vec<f32>,
}

impl SubgameSpec {
    /// Build a `SubgameSpec` from a `PostFlopGame`, reusing `build_mega_terminal_data`.
    ///
    /// Initial weights are padded to `num_hands` length (zero-padded for shorter player).
    pub fn from_game(
        game: &range_solver::PostFlopGame,
        topo: &TreeTopology,
        term: &TerminalData,
        num_hands: usize,
    ) -> Self {
        use range_solver::interface::Game;
        let raw_weights: [Vec<f32>; 2] = [
            game.initial_weights(0).to_vec(),
            game.initial_weights(1).to_vec(),
        ];
        // Pad to num_hands
        let initial_weights: [Vec<f32>; 2] = [
            {
                let mut w = raw_weights[0].clone();
                w.resize(num_hands, 0.0);
                w
            },
            {
                let mut w = raw_weights[1].clone();
                w.resize(num_hands, 0.0);
                w
            },
        ];
        let mtd = build_mega_terminal_data(topo, term, &raw_weights, num_hands);
        SubgameSpec {
            initial_weights,
            showdown_outcomes_p0: mtd.showdown_outcomes_p0,
            showdown_outcomes_p1: mtd.showdown_outcomes_p1,
        }
    }
}

/// Per-subgame result: strategy sums from the GPU solve.
pub struct SubgameResult {
    /// Strategy sum: `[E * H]` for this subgame, same layout as hand-parallel kernel output.
    pub strategy_sum: Vec<f32>,
}

/// Persistent GPU batch solver that reuses CUDA context, compiled kernel, and topology.
pub struct GpuBatchSolver {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel: HandParallelKernel,
    // Topology on GPU (constant across batches)
    d_edge_parent: CudaSlice<i32>,
    d_edge_child: CudaSlice<i32>,
    d_edge_player: CudaSlice<i32>,
    d_actions_per_node: CudaSlice<f32>,
    d_level_starts: CudaSlice<i32>,
    d_level_counts: CudaSlice<i32>,
    // Fold terminal data on GPU (constant across batches for same topology)
    d_fold_node_ids: CudaSlice<i32>,
    d_fold_payoffs_p0: CudaSlice<f32>,
    d_fold_payoffs_p1: CudaSlice<f32>,
    d_fold_depths: CudaSlice<i32>,
    // Showdown structural data (constant - node IDs, depths, num_player)
    d_showdown_node_ids: CudaSlice<i32>,
    d_showdown_num_player: CudaSlice<i32>,
    d_showdown_depths: CudaSlice<i32>,
    // Card data (constant for same topology)
    d_player_card1: CudaSlice<i32>,
    d_player_card2: CudaSlice<i32>,
    d_opp_card1: CudaSlice<i32>,
    d_opp_card2: CudaSlice<i32>,
    d_same_hand_idx: CudaSlice<i32>,
    // Leaf injection (empty for river)
    d_leaf_cfv_p0: CudaSlice<f32>,
    d_leaf_cfv_p1: CudaSlice<f32>,
    d_leaf_node_ids: CudaSlice<i32>,
    d_leaf_depths: CudaSlice<i32>,
    // Dimensions
    num_nodes: usize,
    num_edges: usize,
    num_hands: usize,
    max_depth: usize,
    max_iterations: u32,
    num_folds: usize,
    num_showdowns: usize,
    num_hands_p0: usize,
    num_hands_p1: usize,
    max_batch: usize,
    shared_mem_bytes: u32,
    // Persistent state for incremental solving
    active_state: Option<GpuHandParallelState>,
    active_d_initial_weights: Option<CudaSlice<f32>>,
    active_d_showdown_p0: Option<CudaSlice<f32>>,
    active_d_showdown_p1: Option<CudaSlice<f32>>,
    active_batch_size: usize,
    /// Number of leaf nodes for leaf injection (0 for river).
    pub num_leaves: usize,
}

impl GpuBatchSolver {
    /// Create a new batch solver with persistent CUDA context and pre-uploaded topology.
    ///
    /// `max_batch` is the maximum number of subgames per batch call.
    pub fn new(
        topo: &TreeTopology,
        term: &TerminalData,
        max_batch: usize,
        num_hands: usize,
        max_iterations: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let kernel = HandParallelKernel::compile(&ctx)?;

        // Build sorted topology
        let (parent_i32, child_i32, player_i32, _ape, level_starts_i32, level_counts_i32) =
            build_sorted_topology(topo);
        let d_edge_parent = stream.clone_htod(&parent_i32)?;
        let d_edge_child = stream.clone_htod(&child_i32)?;
        let d_edge_player = stream.clone_htod(&player_i32)?;
        let d_level_starts = stream.clone_htod(&level_starts_i32)?;
        let d_level_counts = stream.clone_htod(&level_counts_i32)?;

        // Actions per node
        let actions_per_node: Vec<f32> = (0..topo.num_nodes)
            .map(|n| topo.node_num_actions[n] as f32)
            .collect();
        let d_actions_per_node = stream.clone_htod(&actions_per_node)?;

        // Build terminal data using a dummy initial_weights (fold/card data doesn't depend on it)
        let dummy_weights: [Vec<f32>; 2] = [
            vec![1.0f32; num_hands],
            vec![1.0f32; num_hands],
        ];
        let mtd = build_mega_terminal_data(topo, term, &dummy_weights, num_hands);

        // Upload fold data (constant across batches)
        let d_fold_node_ids = upload_or_dummy_i32(&stream, &mtd.fold_node_ids)?;
        let d_fold_payoffs_p0 = upload_or_dummy_f32(&stream, &mtd.fold_payoffs_p0)?;
        let d_fold_payoffs_p1 = upload_or_dummy_f32(&stream, &mtd.fold_payoffs_p1)?;
        let d_fold_depths = upload_or_dummy_i32(&stream, &mtd.fold_depths)?;

        // Upload showdown structural data (constant)
        let d_showdown_node_ids = upload_or_dummy_i32(&stream, &mtd.showdown_node_ids)?;
        let d_showdown_num_player = upload_or_dummy_i32(&stream, &mtd.showdown_num_player)?;
        let d_showdown_depths = upload_or_dummy_i32(&stream, &mtd.showdown_depths)?;

        // Upload card data (constant)
        let d_player_card1 = stream.clone_htod(&mtd.player_card1)?;
        let d_player_card2 = stream.clone_htod(&mtd.player_card2)?;
        let d_opp_card1 = stream.clone_htod(&mtd.opp_card1)?;
        let d_opp_card2 = stream.clone_htod(&mtd.opp_card2)?;
        let d_same_hand_idx = stream.clone_htod(&mtd.same_hand_idx)?;

        // Leaf injection (empty for river)
        let d_leaf_cfv_p0 = upload_or_dummy_f32(&stream, &[])?;
        let d_leaf_cfv_p1 = upload_or_dummy_f32(&stream, &[])?;
        let d_leaf_node_ids = upload_or_dummy_i32(&stream, &[])?;
        let d_leaf_depths = upload_or_dummy_i32(&stream, &[])?;

        let shared_mem_bytes =
            compute_hand_parallel_shared_mem(topo.num_edges, topo.max_depth, topo.num_nodes) as u32;

        Ok(Self {
            ctx,
            stream,
            kernel,
            d_edge_parent,
            d_edge_child,
            d_edge_player,
            d_actions_per_node,
            d_level_starts,
            d_level_counts,
            d_fold_node_ids,
            d_fold_payoffs_p0,
            d_fold_payoffs_p1,
            d_fold_depths,
            d_showdown_node_ids,
            d_showdown_num_player,
            d_showdown_depths,
            d_player_card1,
            d_player_card2,
            d_opp_card1,
            d_opp_card2,
            d_same_hand_idx,
            d_leaf_cfv_p0,
            d_leaf_cfv_p1,
            d_leaf_node_ids,
            d_leaf_depths,
            num_nodes: topo.num_nodes,
            num_edges: topo.num_edges,
            num_hands,
            max_depth: topo.max_depth,
            max_iterations,
            num_folds: mtd.fold_node_ids.len(),
            num_showdowns: mtd.showdown_node_ids.len(),
            num_hands_p0: term.hand_cards[0].len(),
            num_hands_p1: term.hand_cards[1].len(),
            max_batch,
            shared_mem_bytes,
            active_state: None,
            active_d_initial_weights: None,
            active_d_showdown_p0: None,
            active_d_showdown_p1: None,
            active_batch_size: 0,
            num_leaves: 0,
        })
    }

    /// Solve a batch of subgames on GPU. Returns one `SubgameResult` per spec.
    ///
    /// Each spec provides per-subgame initial weights and showdown outcomes.
    /// The topology and fold data are shared across all subgames in the batch.
    pub fn solve_batch(
        &mut self,
        specs: &[SubgameSpec],
    ) -> Result<Vec<SubgameResult>, Box<dyn std::error::Error>> {
        if specs.is_empty() {
            return Ok(Vec::new());
        }
        self.prepare_batch(specs)?;
        self.run_iterations(0, self.max_iterations)?;
        self.extract_results()
    }

    /// Upload leaf node IDs and depths, pre-allocate leaf CFV buffers.
    ///
    /// Call before `prepare_batch` for turn solving with leaf injection.
    /// `node_ids` and `depths` are parallel arrays of leaf nodes in the topology.
    pub fn set_leaf_injection(&mut self, node_ids: &[i32], depths: &[i32]) {
        assert_eq!(
            node_ids.len(),
            depths.len(),
            "node_ids and depths must have same length"
        );
        self.num_leaves = node_ids.len();
        let leaf_cfv_size = (self.num_leaves * self.num_hands).max(1);
        self.d_leaf_node_ids = self.stream.clone_htod(node_ids).expect("upload leaf_node_ids");
        self.d_leaf_depths = self.stream.clone_htod(depths).expect("upload leaf_depths");
        self.d_leaf_cfv_p0 = self
            .stream
            .alloc_zeros::<f32>(leaf_cfv_size)
            .expect("alloc leaf_cfv_p0");
        self.d_leaf_cfv_p1 = self
            .stream
            .alloc_zeros::<f32>(leaf_cfv_size)
            .expect("alloc leaf_cfv_p1");
    }

    /// Upload per-subgame data and initialize solver state.
    ///
    /// Call before `run_iterations()`. Stores state on self for incremental use.
    pub fn prepare_batch(
        &mut self,
        specs: &[SubgameSpec],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let batch_size = specs.len();
        assert!(
            batch_size <= self.max_batch,
            "batch size {} exceeds max_batch {}",
            batch_size,
            self.max_batch
        );
        let h = self.num_hands;
        let e = self.num_edges;
        let n = self.num_nodes;

        // Allocate solver state for this batch
        let state = GpuHandParallelState::new(&self.stream, batch_size, n, e, h)?;

        // Build batched initial_weights: [B * 2 * H]
        let mut weights_flat = vec![0.0f32; batch_size * 2 * h];
        for (b, spec) in specs.iter().enumerate() {
            for p in 0..2 {
                for (hi, &w) in spec.initial_weights[p].iter().enumerate() {
                    if hi < h {
                        weights_flat[b * 2 * h + p * h + hi] = w;
                    }
                }
            }
        }
        let d_initial_weights = self.stream.clone_htod(&weights_flat)?;

        // Build batched showdown outcomes: [B * num_showdowns * H * H] for each player
        let hh = h * h;
        let sd_per_batch = self.num_showdowns * hh;
        let mut batched_sd_p0 = vec![0.0f32; batch_size * sd_per_batch];
        let mut batched_sd_p1 = vec![0.0f32; batch_size * sd_per_batch];
        for (b, spec) in specs.iter().enumerate() {
            let base = b * sd_per_batch;
            let src_len = spec.showdown_outcomes_p0.len().min(sd_per_batch);
            batched_sd_p0[base..base + src_len]
                .copy_from_slice(&spec.showdown_outcomes_p0[..src_len]);
            let src_len = spec.showdown_outcomes_p1.len().min(sd_per_batch);
            batched_sd_p1[base..base + src_len]
                .copy_from_slice(&spec.showdown_outcomes_p1[..src_len]);
        }
        let d_showdown_p0 = upload_or_dummy_f32(&self.stream, &batched_sd_p0)?;
        let d_showdown_p1 = upload_or_dummy_f32(&self.stream, &batched_sd_p1)?;

        self.active_state = Some(state);
        self.active_d_initial_weights = Some(d_initial_weights);
        self.active_d_showdown_p0 = Some(d_showdown_p0);
        self.active_d_showdown_p1 = Some(d_showdown_p1);
        self.active_batch_size = batch_size;
        Ok(())
    }

    /// Launch the DCFR kernel for iteration range `[start, end)`.
    ///
    /// Requires `prepare_batch()` to have been called first. Uses stored GPU state.
    pub fn run_iterations(
        &mut self,
        start: u32,
        end: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let state = self
            .active_state
            .as_mut()
            .expect("run_iterations called before prepare_batch");
        let d_initial_weights = self
            .active_d_initial_weights
            .as_ref()
            .expect("no active initial_weights");
        let d_showdown_p0 = self
            .active_d_showdown_p0
            .as_ref()
            .expect("no active showdown_p0");
        let d_showdown_p1 = self
            .active_d_showdown_p1
            .as_ref()
            .expect("no active showdown_p1");

        let batch_size = self.active_batch_size;
        let h = self.num_hands;
        let n = self.num_nodes;
        let e = self.num_edges;

        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (h as u32, 1, 1),
            shared_mem_bytes: self.shared_mem_bytes,
        };

        let b_i32 = batch_size as i32;
        let n_i32 = n as i32;
        let e_i32 = e as i32;
        let h_i32 = h as i32;
        let max_depth_i32 = self.max_depth as i32;
        let start_iter_i32 = start as i32;
        let end_iter_i32 = end as i32;
        let num_folds_i32 = self.num_folds as i32;
        let num_showdowns_i32 = self.num_showdowns as i32;
        let num_leaves_i32 = self.num_leaves as i32;
        let num_hands_p0_i32 = self.num_hands_p0 as i32;
        let num_hands_p1_i32 = self.num_hands_p1 as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&self.kernel.cfr_solve);
            builder.arg(&mut state.regrets);
            builder.arg(&mut state.strategy_sum);
            builder.arg(&mut state.reach);
            builder.arg(&mut state.cfv);
            builder.arg(&self.d_edge_parent);
            builder.arg(&self.d_edge_child);
            builder.arg(&self.d_edge_player);
            builder.arg(&self.d_actions_per_node);
            builder.arg(&self.d_level_starts);
            builder.arg(&self.d_level_counts);
            builder.arg(&self.d_fold_node_ids);
            builder.arg(&self.d_fold_payoffs_p0);
            builder.arg(&self.d_fold_payoffs_p1);
            builder.arg(&self.d_fold_depths);
            builder.arg(&self.d_showdown_node_ids);
            builder.arg(d_showdown_p0);
            builder.arg(d_showdown_p1);
            builder.arg(&self.d_showdown_num_player);
            builder.arg(&self.d_showdown_depths);
            builder.arg(&self.d_player_card1);
            builder.arg(&self.d_player_card2);
            builder.arg(&self.d_opp_card1);
            builder.arg(&self.d_opp_card2);
            builder.arg(&self.d_same_hand_idx);
            builder.arg(d_initial_weights);
            builder.arg(&self.d_leaf_cfv_p0);
            builder.arg(&self.d_leaf_cfv_p1);
            builder.arg(&self.d_leaf_node_ids);
            builder.arg(&self.d_leaf_depths);
            builder.arg(&b_i32);
            builder.arg(&n_i32);
            builder.arg(&e_i32);
            builder.arg(&h_i32);
            builder.arg(&max_depth_i32);
            builder.arg(&start_iter_i32);
            builder.arg(&end_iter_i32);
            builder.arg(&num_folds_i32);
            builder.arg(&num_showdowns_i32);
            builder.arg(&num_leaves_i32);
            builder.arg(&num_hands_p0_i32);
            builder.arg(&num_hands_p1_i32);
            builder.launch(cfg)?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// Download the reach array `[B * N * H]` from GPU to CPU.
    pub fn download_reach(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let state = self
            .active_state
            .as_ref()
            .expect("download_reach called before prepare_batch");
        let reach: Vec<f32> = self.stream.clone_dtoh(&state.reach)?;
        Ok(reach)
    }

    /// Copy new leaf CFV data into the pre-allocated GPU buffers.
    ///
    /// `p0` and `p1` are `[num_leaves * num_hands]` each.
    pub fn update_leaf_cfvs(
        &mut self,
        p0: &[f32],
        p1: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let expected = self.num_leaves * self.num_hands;
        assert_eq!(
            p0.len(),
            expected,
            "p0 len {} != num_leaves*num_hands {}",
            p0.len(),
            expected
        );
        assert_eq!(
            p1.len(),
            expected,
            "p1 len {} != num_leaves*num_hands {}",
            p1.len(),
            expected
        );
        self.d_leaf_cfv_p0 = self.stream.clone_htod(p0)?;
        self.d_leaf_cfv_p1 = self.stream.clone_htod(p1)?;
        Ok(())
    }

    /// Download strategy_sum and split per subgame.
    pub fn extract_results(&self) -> Result<Vec<SubgameResult>, Box<dyn std::error::Error>> {
        let state = self
            .active_state
            .as_ref()
            .expect("extract_results called before prepare_batch");
        let e = self.num_edges;
        let h = self.num_hands;
        let batch_size = self.active_batch_size;

        let strategy_sum_all: Vec<f32> = self.stream.clone_dtoh(&state.strategy_sum)?;

        let eh = e * h;
        let results = (0..batch_size)
            .map(|b| {
                let start = b * eh;
                SubgameResult {
                    strategy_sum: strategy_sum_all[start..start + eh].to_vec(),
                }
            })
            .collect();

        Ok(results)
    }
}

/// Compute per-player per-hand expected values from GPU strategy_sum.
///
/// Performs a CPU tree walk using the normalized average strategy from strategy_sum.
/// Returns `[oop_evs, ip_evs]` where each is a `Vec<f32>` of length `num_hands`.
///
/// The strategy_sum layout is `[E * H]` where E = num_edges, H = num_hands.
/// Edges are sorted by parent depth (same ordering as `build_sorted_topology`).
pub fn compute_evs_from_strategy_sum(
    topo: &TreeTopology,
    term: &TerminalData,
    strategy_sum: &[f32],
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
) -> [Vec<f32>; 2] {
    let n = topo.num_nodes;
    let e = topo.num_edges;
    let h = num_hands;

    // Build sorted edge mapping (same as kernel uses)
    let mut edges_by_depth: Vec<Vec<usize>> = vec![Vec::new(); topo.max_depth + 1];
    for edge in 0..e {
        let parent_depth = topo.node_depth[topo.edge_parent[edge]];
        edges_by_depth[parent_depth].push(edge);
    }
    let mut sorted_edges: Vec<usize> = Vec::with_capacity(e);
    for depth_edges in &edges_by_depth {
        sorted_edges.extend(depth_edges);
    }
    // sorted_edge_idx -> original edge index
    // strategy_sum[sorted_idx * H + hand] = strategy_sum value

    // Normalize strategy_sum to average strategy per sorted edge
    // For each parent node, normalize across its child edges
    let mut avg_strategy = vec![0.0f32; e * h];
    let mut sorted_idx = 0;
    for depth_edges in &edges_by_depth {
        // Group edges by parent node within this depth
        let mut idx = 0;
        while idx < depth_edges.len() {
            let parent = topo.edge_parent[depth_edges[idx]];
            let n_actions = topo.node_num_actions[parent];
            let group_start = sorted_idx + idx;
            for hand in 0..h {
                let mut total = 0.0f32;
                for a in 0..n_actions {
                    total += strategy_sum[(group_start + a) * h + hand].max(0.0);
                }
                if total > 1e-30 {
                    for a in 0..n_actions {
                        avg_strategy[(group_start + a) * h + hand] =
                            strategy_sum[(group_start + a) * h + hand].max(0.0) / total;
                    }
                } else {
                    let uniform = 1.0 / n_actions as f32;
                    for a in 0..n_actions {
                        avg_strategy[(group_start + a) * h + hand] = uniform;
                    }
                }
            }
            idx += n_actions;
        }
        sorted_idx += depth_edges.len();
    }

    // Build reverse mapping: sorted_idx -> original_edge_idx
    let mut original_edge_for_sorted: Vec<usize> = Vec::with_capacity(e);
    for depth_edges in &edges_by_depth {
        original_edge_for_sorted.extend(depth_edges.iter().copied());
    }

    // Build node -> sorted edge start mapping for its children
    let mut node_first_sorted_edge: Vec<usize> = vec![usize::MAX; n];
    for (sorted_i, &orig_e) in original_edge_for_sorted.iter().enumerate() {
        let parent = topo.edge_parent[orig_e];
        if node_first_sorted_edge[parent] == usize::MAX {
            node_first_sorted_edge[parent] = sorted_i;
        }
    }

    let mut result = [vec![0.0f32; h], vec![0.0f32; h]];

    for player in 0..2 {
        let opp = player ^ 1;

        // Forward pass: compute reach probabilities using avg strategy
        let mut reach = vec![0.0f32; n * h];
        // Root reach = opponent's initial weights
        for hand in 0..initial_weights[opp].len().min(h) {
            reach[hand] = initial_weights[opp][hand];
        }

        // CFV accumulator for each node
        let mut cfv = vec![0.0f32; n * h];

        // Process tree depth by depth (top-down for reach, bottom-up for CFV)
        // First: propagate reach top-down
        for depth in 0..=topo.max_depth {
            for &node_id in &topo.level_nodes[depth] {
                let n_actions = topo.node_num_actions[node_id];
                if n_actions == 0 {
                    continue;
                }
                let sorted_start = node_first_sorted_edge[node_id];
                if sorted_start == usize::MAX {
                    continue;
                }

                match topo.node_type[node_id] {
                    NodeType::Player { player: acting } => {
                        for a in 0..n_actions {
                            let sorted_i = sorted_start + a;
                            let child = topo.edge_child[original_edge_for_sorted[sorted_i]];
                            for hand in 0..h {
                                let parent_reach = reach[node_id * h + hand];
                                if acting == player {
                                    // Player's action: multiply by strategy
                                    reach[child * h + hand] =
                                        parent_reach * avg_strategy[sorted_i * h + hand];
                                } else {
                                    // Opponent's action: pass reach through
                                    reach[child * h + hand] += parent_reach;
                                }
                            }
                        }
                    }
                    NodeType::Chance => {
                        // Chance node: equal weight to each child (or could be card-specific)
                        for a in 0..n_actions {
                            let sorted_i = sorted_start + a;
                            let child = topo.edge_child[original_edge_for_sorted[sorted_i]];
                            for hand in 0..h {
                                reach[child * h + hand] += reach[node_id * h + hand];
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Evaluate terminal nodes
        for (fold_idx, &node_id) in topo.fold_nodes.iter().enumerate() {
            let fd = &term.fold_payoffs[fold_idx];
            let payoff = if fd.folded_player == opp {
                fd.amount_win as f32
            } else {
                fd.amount_lose as f32
            };

            // Fold: CFV = payoff * sum(opp_reach) for each player hand
            // But we need card blocking: exclude opponent hands that share cards
            let player_cards = &term.hand_cards[player];
            let opp_cards = &term.hand_cards[opp];

            for h_p in 0..player_cards.len().min(h) {
                let (pc1, pc2) = player_cards[h_p];
                let pc_mask = (1u64 << pc1) | (1u64 << pc2);

                let mut opp_reach_sum = 0.0f32;
                for h_o in 0..opp_cards.len().min(h) {
                    let (oc1, oc2) = opp_cards[h_o];
                    let oc_mask = (1u64 << oc1) | (1u64 << oc2);
                    if pc_mask & oc_mask != 0 {
                        continue; // card collision
                    }
                    opp_reach_sum += reach[node_id * h + h_o];
                }
                cfv[node_id * h + h_p] = payoff * opp_reach_sum;
            }
        }

        for (sd_idx, &node_id) in topo.showdown_nodes.iter().enumerate() {
            let sd = &term.showdown_outcomes[sd_idx];
            let num_p = sd.num_player_hands[player];
            let num_o = sd.num_player_hands[opp];

            for h_p in 0..num_p.min(h) {
                let mut ev = 0.0f32;
                for h_o in 0..num_o.min(h) {
                    let outcome = if player == 0 {
                        sd.outcome_matrix_p0[h_p * num_o + h_o]
                    } else {
                        // IP perspective: negate the OOP outcome
                        -sd.outcome_matrix_p0[h_o * num_p + h_p]
                    };

                    let payoff = if outcome > 0.0 {
                        sd.amount_win as f32
                    } else if outcome < 0.0 {
                        sd.amount_lose as f32
                    } else {
                        0.0f32
                    };

                    ev += payoff * reach[node_id * h + h_o];
                }
                cfv[node_id * h + h_p] = ev;
            }
        }

        // Backward pass: propagate CFV from children to parents
        for depth in (0..=topo.max_depth).rev() {
            for &node_id in &topo.level_nodes[depth] {
                let n_actions = topo.node_num_actions[node_id];
                if n_actions == 0 {
                    continue;
                }
                let sorted_start = node_first_sorted_edge[node_id];
                if sorted_start == usize::MAX {
                    continue;
                }

                match topo.node_type[node_id] {
                    NodeType::Player { player: acting } => {
                        for a in 0..n_actions {
                            let sorted_i = sorted_start + a;
                            let child = topo.edge_child[original_edge_for_sorted[sorted_i]];
                            for hand in 0..h {
                                if acting == player {
                                    // Player's own action: weight by strategy
                                    cfv[node_id * h + hand] +=
                                        avg_strategy[sorted_i * h + hand]
                                            * cfv[child * h + hand];
                                } else {
                                    // Opponent's action: sum all children
                                    cfv[node_id * h + hand] += cfv[child * h + hand];
                                }
                            }
                        }
                    }
                    NodeType::Chance => {
                        for a in 0..n_actions {
                            let sorted_i = sorted_start + a;
                            let child = topo.edge_child[original_edge_for_sorted[sorted_i]];
                            for hand in 0..h {
                                cfv[node_id * h + hand] += cfv[child * h + hand];
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Root CFVs
        for hand in 0..h {
            result[player][hand] = cfv[hand];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use crate::extract::{extract_terminal_data, extract_topology};
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{card_from_str, flop_from_str, CardConfig};

    fn make_river_game() -> range_solver::PostFlopGame {
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
        let mut game =
            range_solver::PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);
        game
    }

    #[test]
    fn batch_solver_constructs_from_topology() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let solver = GpuBatchSolver::new(&topo, &term, 4, num_hands, 100);
        assert!(solver.is_ok(), "GpuBatchSolver::new failed: {:?}", solver.err());
    }

    #[test]
    fn batch_solver_solve_single_subgame() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let mut solver = GpuBatchSolver::new(&topo, &term, 4, num_hands, 500).unwrap();

        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        let results = solver.solve_batch(&[spec]).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].strategy_sum.len(), topo.num_edges * num_hands);
        // Strategy sum should have non-zero values after 500 iterations
        let total: f32 = results[0].strategy_sum.iter().map(|x| x.abs()).sum();
        assert!(total > 0.0, "strategy_sum should be non-zero after solving");
    }

    #[test]
    fn batch_solver_solve_multiple_subgames() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let mut solver = GpuBatchSolver::new(&topo, &term, 4, num_hands, 500).unwrap();

        // Solve same game 3 times in one batch
        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        let specs = vec![spec.clone(), spec.clone(), spec];
        let results = solver.solve_batch(&specs).unwrap();

        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.strategy_sum.len(), topo.num_edges * num_hands);
            let total: f32 = r.strategy_sum.iter().map(|x| x.abs()).sum();
            assert!(total > 0.0, "each subgame should have non-zero strategy_sum");
        }
    }

    #[test]
    fn batch_solver_results_match_single_solve() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        // Solve with batch solver
        let mut batch = GpuBatchSolver::new(&topo, &term, 4, num_hands, 500).unwrap();
        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        let batch_results = batch.solve_batch(&[spec]).unwrap();

        // Solve with single hand-parallel solver
        let config = crate::GpuSolverConfig {
            max_iterations: 500,
            target_exploitability: 0.0,
            print_progress: false,
        };
        let single_result = crate::gpu_solve_hand_parallel(&game, &config);

        // Both should produce valid, similar root strategies
        assert!(!single_result.root_strategy.is_empty());
        assert!(!batch_results[0].strategy_sum.is_empty());

        // Compare root strategy derived from each
        let n_actions = topo.node_num_actions[0];
        let start = 0usize;
        for h in 0..num_hands {
            let mut batch_total = 0.0f32;
            for a in 0..n_actions {
                batch_total += batch_results[0].strategy_sum[(start + a) * num_hands + h];
            }
            if batch_total > 1e-30 {
                for a in 0..n_actions {
                    let batch_prob =
                        batch_results[0].strategy_sum[(start + a) * num_hands + h] / batch_total;
                    let single_prob = single_result.root_strategy[a * num_hands + h];
                    let diff = (batch_prob - single_prob).abs();
                    assert!(
                        diff < 0.01,
                        "hand {h} action {a}: batch={batch_prob:.4} single={single_prob:.4} diff={diff:.4}"
                    );
                }
            }
        }
    }

    #[test]
    fn batch_solver_reusable_across_batches() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let mut solver = GpuBatchSolver::new(&topo, &term, 4, num_hands, 100).unwrap();
        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);

        // First batch
        let r1 = solver.solve_batch(&[spec.clone()]).unwrap();
        assert_eq!(r1.len(), 1);

        // Second batch - solver should be reusable
        let r2 = solver.solve_batch(&[spec.clone(), spec]).unwrap();
        assert_eq!(r2.len(), 2);
    }

    #[test]
    fn batch_solver_empty_batch_returns_empty() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let mut solver = GpuBatchSolver::new(&topo, &term, 4, num_hands, 100).unwrap();
        let results = solver.solve_batch(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn subgame_spec_from_game_has_correct_dimensions() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);

        assert_eq!(spec.initial_weights[0].len(), num_hands);
        assert_eq!(spec.initial_weights[1].len(), num_hands);
        let num_showdowns = topo.showdown_nodes.len();
        assert_eq!(spec.showdown_outcomes_p0.len(), num_showdowns * num_hands * num_hands);
        assert_eq!(spec.showdown_outcomes_p1.len(), num_showdowns * num_hands * num_hands);
    }

    #[test]
    fn compute_evs_from_strategy_sum_produces_finite_values() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let mut solver = GpuBatchSolver::new(&topo, &term, 4, num_hands, 500).unwrap();
        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        let results = solver.solve_batch(&[spec.clone()]).unwrap();

        let evs = compute_evs_from_strategy_sum(
            &topo,
            &term,
            &results[0].strategy_sum,
            &spec.initial_weights,
            num_hands,
        );

        // Should produce EVs for both players
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].len(), num_hands);
        assert_eq!(evs[1].len(), num_hands);

        // All EVs should be finite
        for p in 0..2 {
            for h in 0..num_hands {
                assert!(evs[p][h].is_finite(), "EV[{p}][{h}] is not finite: {}", evs[p][h]);
            }
        }
    }

    #[test]
    fn incremental_matches_batch() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        // Baseline: single solve_batch call with 150 iterations
        let mut solver1 = GpuBatchSolver::new(&topo, &term, 4, num_hands, 150).unwrap();
        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        let baseline = solver1.solve_batch(&[spec.clone()]).unwrap();

        // Incremental: prepare + 3 chunks of 50 iterations
        let mut solver2 = GpuBatchSolver::new(&topo, &term, 4, num_hands, 150).unwrap();
        solver2.prepare_batch(&[spec]).unwrap();
        solver2.run_iterations(0, 50).unwrap();
        solver2.run_iterations(50, 100).unwrap();
        solver2.run_iterations(100, 150).unwrap();
        let incremental = solver2.extract_results().unwrap();

        assert_eq!(baseline.len(), incremental.len());
        assert_eq!(baseline[0].strategy_sum.len(), incremental[0].strategy_sum.len());

        // strategy_sum must match within 1e-4
        for (i, (b, inc)) in baseline[0]
            .strategy_sum
            .iter()
            .zip(incremental[0].strategy_sum.iter())
            .enumerate()
        {
            let diff = (b - inc).abs();
            assert!(
                diff < 1e-4,
                "strategy_sum[{i}] mismatch: baseline={b} incremental={inc} diff={diff}"
            );
        }
    }

    #[test]
    fn turn_leaf_injection_incremental() {
        use super::*;
        use crate::extract::{extract_terminal_data, extract_topology, NodeType};

        // Use a river game and pick some internal nodes as "leaf injection" targets
        // to verify the machinery works end-to-end.
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        // Find some non-root, non-terminal nodes to use as fake leaf injection points
        let mut leaf_nodes = Vec::new();
        for nid in 1..topo.num_nodes {
            if topo.node_num_actions[nid] > 0 {
                if let NodeType::Player { .. } = topo.node_type[nid] {
                    leaf_nodes.push(nid);
                    if leaf_nodes.len() >= 2 {
                        break;
                    }
                }
            }
        }
        assert!(
            !leaf_nodes.is_empty(),
            "need at least one internal node for leaf injection test"
        );
        let leaf_node_ids: Vec<i32> = leaf_nodes.iter().map(|&n| n as i32).collect();
        let leaf_depths: Vec<i32> = leaf_nodes
            .iter()
            .map(|&n| topo.node_depth[n] as i32)
            .collect();
        let num_leaves = leaf_node_ids.len();

        let mut solver = GpuBatchSolver::new(&topo, &term, 4, num_hands, 100).unwrap();
        solver.set_leaf_injection(&leaf_node_ids, &leaf_depths);

        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        solver.prepare_batch(&[spec]).unwrap();

        // Run 50 iters with zero leaf CFVs
        solver.run_iterations(0, 50).unwrap();

        // Update leaf CFVs with non-zero dummy values
        let leaf_cfv_size = num_leaves * num_hands;
        let p0_cfvs = vec![1.0f32; leaf_cfv_size];
        let p1_cfvs = vec![1.0f32; leaf_cfv_size];
        solver.update_leaf_cfvs(&p0_cfvs, &p1_cfvs).unwrap();

        // Run 50 more
        solver.run_iterations(50, 100).unwrap();

        let results = solver.extract_results().unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].strategy_sum.is_empty());
        // All values should be finite
        for (i, &v) in results[0].strategy_sum.iter().enumerate() {
            assert!(v.is_finite(), "strategy_sum[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn set_leaf_injection_allocates_buffers() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let mut solver = GpuBatchSolver::new(&topo, &term, 4, num_hands, 100).unwrap();

        let leaf_node_ids = vec![1i32, 2];
        let leaf_depths = vec![1i32, 1];
        solver.set_leaf_injection(&leaf_node_ids, &leaf_depths);

        assert_eq!(solver.num_leaves, 2);
    }

    #[test]
    fn download_reach_has_correct_size() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let mut solver = GpuBatchSolver::new(&topo, &term, 4, num_hands, 50).unwrap();
        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);

        solver.prepare_batch(&[spec]).unwrap();
        solver.run_iterations(0, 50).unwrap();

        let reach = solver.download_reach().unwrap();
        // B=1, N=num_nodes, H=num_hands
        assert_eq!(reach.len(), 1 * topo.num_nodes * num_hands);
    }

}
