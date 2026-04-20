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

/// Per-subgame input: initial weights, showdown outcome matrices, and fold payoffs.
#[derive(Clone)]
pub struct SubgameSpec {
    /// Per-player initial weights, length `num_hands` each.
    pub initial_weights: [Vec<f32>; 2],
    /// Pre-scaled showdown outcomes for player 0 traversal: `[num_showdowns * H * H]`.
    /// `None` means this spec has no showdowns — used by turn datagen where leaf
    /// injection from BoundaryNet supplies all CFVs at boundary nodes. When `None`,
    /// `prepare_batch` skips host concat and H2D upload, and the kernel sees
    /// `num_showdowns=0` so its showdown loop is a no-op.
    pub showdown_outcomes_p0: Option<Vec<f32>>,
    /// Pre-scaled showdown outcomes for player 1 traversal: `[num_showdowns * H * H]`.
    /// See `showdown_outcomes_p0` for `None` semantics.
    pub showdown_outcomes_p1: Option<Vec<f32>>,
    /// Per-game fold payoffs for P0 traversal: `[num_folds]`.
    pub fold_payoffs_p0: Vec<f32>,
    /// Per-game fold payoffs for P1 traversal: `[num_folds]`.
    pub fold_payoffs_p1: Vec<f32>,
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
            showdown_outcomes_p0: Some(mtd.showdown_outcomes_p0),
            showdown_outcomes_p1: Some(mtd.showdown_outcomes_p1),
            fold_payoffs_p0: mtd.fold_payoffs_p0,
            fold_payoffs_p1: mtd.fold_payoffs_p1,
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
    // Leaf injection (empty for river). Buffers are per-batch sized:
    // `[max_batch * num_leaves * num_hands]`, indexed by the kernel as
    // `leaf_cfv[bid * num_leaves * H + li * H + h]`.
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
    active_d_fold_payoffs_p0: Option<CudaSlice<f32>>,
    active_d_fold_payoffs_p1: Option<CudaSlice<f32>>,
    active_batch_size: usize,
    /// Number of showdowns active in the current batch. Equal to `self.num_showdowns`
    /// when the batch has real outcomes, or `0` when all specs set `showdown_outcomes_*`
    /// to `None` (turn datagen with leaf injection). Set by `prepare_batch`, consumed
    /// by `run_iterations`.
    active_num_showdowns: usize,
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
            active_d_fold_payoffs_p0: None,
            active_d_fold_payoffs_p1: None,
            active_batch_size: 0,
            active_num_showdowns: 0,
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

    /// Upload leaf node IDs and depths, pre-allocate per-batch leaf CFV buffers.
    ///
    /// Call before `prepare_batch` for turn solving with leaf injection.
    /// `node_ids` and `depths` are parallel arrays of leaf nodes in the topology.
    ///
    /// Allocates `d_leaf_cfv_p0/p1` at `max_batch * num_leaves * num_hands` so
    /// each game in a batch has its own slot. Later `update_leaf_cfvs` calls
    /// upload `[active_batch_size * num_leaves * num_hands]` via `clone_htod`.
    pub fn set_leaf_injection(
        &mut self,
        node_ids: &[i32],
        depths: &[i32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            node_ids.len(),
            depths.len(),
            "node_ids and depths must have same length"
        );
        self.num_leaves = node_ids.len();
        let leaf_cfv_size = (self.max_batch * self.num_leaves * self.num_hands).max(1);
        self.d_leaf_node_ids = self.stream.clone_htod(node_ids)?;
        self.d_leaf_depths = self.stream.clone_htod(depths)?;
        self.d_leaf_cfv_p0 = self.stream.alloc_zeros::<f32>(leaf_cfv_size)?;
        self.d_leaf_cfv_p1 = self.stream.alloc_zeros::<f32>(leaf_cfv_size)?;
        Ok(())
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

        // Build batched showdown outcomes: [B * num_showdowns * H * H] for each player.
        // If all specs have `None` outcomes (turn datagen with leaf injection), skip the
        // allocation + upload entirely and set active_num_showdowns=0 so the kernel's
        // showdown loop is a no-op. Mixed Some/None in one batch is a bug.
        let any_some_p0 = specs.iter().any(|s| s.showdown_outcomes_p0.is_some());
        let any_some_p1 = specs.iter().any(|s| s.showdown_outcomes_p1.is_some());
        let all_some_p0 = specs.iter().all(|s| s.showdown_outcomes_p0.is_some());
        let all_some_p1 = specs.iter().all(|s| s.showdown_outcomes_p1.is_some());
        debug_assert_eq!(
            any_some_p0, all_some_p0,
            "SubgameSpec.showdown_outcomes_p0 must be uniformly Some or None across a batch"
        );
        debug_assert_eq!(
            any_some_p1, all_some_p1,
            "SubgameSpec.showdown_outcomes_p1 must be uniformly Some or None across a batch"
        );
        debug_assert_eq!(
            any_some_p0, any_some_p1,
            "SubgameSpec showdown_outcomes_p0 and _p1 must have the same Some/None state"
        );

        let (d_showdown_p0, d_showdown_p1, active_num_showdowns) = if all_some_p0 {
            let hh = h * h;
            let sd_per_batch = self.num_showdowns * hh;
            let mut batched_sd_p0 = vec![0.0f32; batch_size * sd_per_batch];
            let mut batched_sd_p1 = vec![0.0f32; batch_size * sd_per_batch];
            for (b, spec) in specs.iter().enumerate() {
                let base = b * sd_per_batch;
                if let Some(sd) = spec.showdown_outcomes_p0.as_deref() {
                    let src_len = sd.len().min(sd_per_batch);
                    batched_sd_p0[base..base + src_len].copy_from_slice(&sd[..src_len]);
                }
                if let Some(sd) = spec.showdown_outcomes_p1.as_deref() {
                    let src_len = sd.len().min(sd_per_batch);
                    batched_sd_p1[base..base + src_len].copy_from_slice(&sd[..src_len]);
                }
            }
            (
                upload_or_dummy_f32(&self.stream, &batched_sd_p0)?,
                upload_or_dummy_f32(&self.stream, &batched_sd_p1)?,
                self.num_showdowns,
            )
        } else {
            // No showdowns to compute. Dummy device buffers keep the kernel-arg plumbing
            // happy; active_num_showdowns=0 makes the kernel's showdown loop zero-iteration.
            (
                upload_or_dummy_f32(&self.stream, &[])?,
                upload_or_dummy_f32(&self.stream, &[])?,
                0,
            )
        };

        // Build batched fold payoffs: [B * num_folds] for each player
        let num_folds = self.num_folds;
        let mut batched_fold_p0 = vec![0.0f32; batch_size * num_folds];
        let mut batched_fold_p1 = vec![0.0f32; batch_size * num_folds];
        for (b, spec) in specs.iter().enumerate() {
            let base = b * num_folds;
            let src_len = spec.fold_payoffs_p0.len().min(num_folds);
            batched_fold_p0[base..base + src_len]
                .copy_from_slice(&spec.fold_payoffs_p0[..src_len]);
            let src_len = spec.fold_payoffs_p1.len().min(num_folds);
            batched_fold_p1[base..base + src_len]
                .copy_from_slice(&spec.fold_payoffs_p1[..src_len]);
        }
        let d_fold_p0 = upload_or_dummy_f32(&self.stream, &batched_fold_p0)?;
        let d_fold_p1 = upload_or_dummy_f32(&self.stream, &batched_fold_p1)?;

        self.active_state = Some(state);
        self.active_d_initial_weights = Some(d_initial_weights);
        self.active_d_showdown_p0 = Some(d_showdown_p0);
        self.active_d_showdown_p1 = Some(d_showdown_p1);
        self.active_d_fold_payoffs_p0 = Some(d_fold_p0);
        self.active_d_fold_payoffs_p1 = Some(d_fold_p1);
        self.active_batch_size = batch_size;
        self.active_num_showdowns = active_num_showdowns;
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

        let block_threads = (h as u32).min(1024);
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (block_threads, 1, 1),
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
        let num_showdowns_i32 = self.active_num_showdowns as i32;
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
            let fold_p0 = self
                .active_d_fold_payoffs_p0
                .as_ref()
                .unwrap_or(&self.d_fold_payoffs_p0);
            let fold_p1 = self
                .active_d_fold_payoffs_p1
                .as_ref()
                .unwrap_or(&self.d_fold_payoffs_p1);
            builder.arg(&self.d_fold_node_ids);
            builder.arg(fold_p0);
            builder.arg(fold_p1);
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

    /// Copy new per-batch leaf CFV data to GPU. Uses `clone_htod` (new allocation)
    /// rather than in-place copy because cudarc does not expose a host-to-device
    /// memcpy into an existing slice. The old buffer is freed on drop. This runs
    /// only at boundary re-evaluation intervals, so the alloc overhead is negligible.
    ///
    /// `p0` and `p1` are each `[active_batch_size * num_leaves * num_hands]`,
    /// matching the layout the kernel reads with
    /// `leaf_cfv[bid * num_leaves * H + li * H + h]`.
    ///
    /// Must be called after `prepare_batch` so `active_batch_size` reflects the
    /// current batch dimension.
    pub fn update_leaf_cfvs(
        &mut self,
        p0: &[f32],
        p1: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert!(
            self.active_batch_size > 0,
            "update_leaf_cfvs called before prepare_batch (active_batch_size = 0)"
        );
        let expected = self.active_batch_size * self.num_leaves * self.num_hands;
        assert_eq!(
            p0.len(),
            expected,
            "p0 len {} != active_batch_size*num_leaves*num_hands {}",
            p0.len(),
            expected
        );
        assert_eq!(
            p1.len(),
            expected,
            "p1 len {} != active_batch_size*num_leaves*num_hands {}",
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

/// Shared setup for avg-strategy tree walks: normalizes `strategy_sum` and builds edge mappings.
struct AvgStrategySetup {
    avg_strategy: Vec<f32>,
    original_edge_for_sorted: Vec<usize>,
    node_first_sorted_edge: Vec<usize>,
}

/// Build the normalized average strategy and edge mappings from `strategy_sum`.
///
/// This is shared by `compute_evs_from_strategy_sum` and `compute_reach_at_nodes`.
fn build_avg_strategy_setup(
    topo: &TreeTopology,
    strategy_sum: &[f32],
    num_hands: usize,
) -> AvgStrategySetup {
    let n = topo.num_nodes;
    let e = topo.num_edges;
    let h = num_hands;

    // Build sorted edge mapping (same as kernel uses)
    let mut edges_by_depth: Vec<Vec<usize>> = vec![Vec::new(); topo.max_depth + 1];
    for edge in 0..e {
        let parent_depth = topo.node_depth[topo.edge_parent[edge]];
        edges_by_depth[parent_depth].push(edge);
    }

    // Normalize strategy_sum to average strategy per sorted edge
    let mut avg_strategy = vec![0.0f32; e * h];
    let mut sorted_idx = 0;
    for depth_edges in &edges_by_depth {
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

    AvgStrategySetup {
        avg_strategy,
        original_edge_for_sorted,
        node_first_sorted_edge,
    }
}

/// Forward-walk reach propagation for one player using the average strategy.
///
/// Returns `reach[node * h + hand]` for all nodes. When `player=0`, reach holds
/// P1's (opponent's) reach; when `player=1`, reach holds P0's reach.
fn forward_walk_reach(
    topo: &TreeTopology,
    setup: &AvgStrategySetup,
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
    player: usize,
) -> Vec<f32> {
    let n = topo.num_nodes;
    let h = num_hands;
    let opp = player ^ 1;

    let mut reach = vec![0.0f32; n * h];
    // Root reach = opponent's initial weights
    for hand in 0..initial_weights[opp].len().min(h) {
        reach[hand] = initial_weights[opp][hand];
    }

    for depth in 0..=topo.max_depth {
        for &node_id in &topo.level_nodes[depth] {
            let n_actions = topo.node_num_actions[node_id];
            if n_actions == 0 {
                continue;
            }
            let sorted_start = setup.node_first_sorted_edge[node_id];
            if sorted_start == usize::MAX {
                continue;
            }

            match topo.node_type[node_id] {
                NodeType::Player { player: acting } => {
                    for a in 0..n_actions {
                        let sorted_i = sorted_start + a;
                        let child =
                            topo.edge_child[setup.original_edge_for_sorted[sorted_i]];
                        for hand in 0..h {
                            let parent_reach = reach[node_id * h + hand];
                            if acting == player {
                                reach[child * h + hand] =
                                    parent_reach * setup.avg_strategy[sorted_i * h + hand];
                            } else {
                                reach[child * h + hand] += parent_reach;
                            }
                        }
                    }
                }
                NodeType::Chance => {
                    for a in 0..n_actions {
                        let sorted_i = sorted_start + a;
                        let child =
                            topo.edge_child[setup.original_edge_for_sorted[sorted_i]];
                        for hand in 0..h {
                            reach[child * h + hand] += reach[node_id * h + hand];
                        }
                    }
                }
                _ => {}
            }
        }
    }

    reach
}

/// Compute per-player reach probabilities at specific nodes using the average strategy.
///
/// Normalizes `strategy_sum` into an average strategy, then forward-walks the tree
/// to compute reach at each node. Returns `[oop_reach, ip_reach]` where each is
/// `[target_nodes.len() * num_hands]` -- reach values at the target nodes for the
/// opponent of each player (i.e., `result[0]` is P1's reach from P0's traversal,
/// `result[1]` is P0's reach from P1's traversal).
pub fn compute_reach_at_nodes(
    topo: &TreeTopology,
    strategy_sum: &[f32],
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
    target_nodes: &[usize],
) -> [Vec<f32>; 2] {
    let h = num_hands;
    let setup = build_avg_strategy_setup(topo, strategy_sum, h);

    let mut result = [
        vec![0.0f32; target_nodes.len() * h],
        vec![0.0f32; target_nodes.len() * h],
    ];

    for player in 0..2 {
        let reach = forward_walk_reach(topo, &setup, initial_weights, h, player);
        for (ti, &node_id) in target_nodes.iter().enumerate() {
            for hand in 0..h {
                result[player][ti * h + hand] = reach[node_id * h + hand];
            }
        }
    }

    result
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
    let h = num_hands;

    let setup = build_avg_strategy_setup(topo, strategy_sum, h);

    let mut result = [vec![0.0f32; h], vec![0.0f32; h]];

    for player in 0..2 {
        let opp = player ^ 1;

        // Forward pass: compute reach probabilities using avg strategy
        let reach = forward_walk_reach(topo, &setup, initial_weights, h, player);

        // CFV accumulator for each node
        let mut cfv = vec![0.0f32; n * h];

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
                let sorted_start = setup.node_first_sorted_edge[node_id];
                if sorted_start == usize::MAX {
                    continue;
                }

                match topo.node_type[node_id] {
                    NodeType::Player { player: acting } => {
                        for a in 0..n_actions {
                            let sorted_i = sorted_start + a;
                            let child = topo.edge_child[setup.original_edge_for_sorted[sorted_i]];
                            for hand in 0..h {
                                if acting == player {
                                    // Player's own action: weight by strategy
                                    cfv[node_id * h + hand] +=
                                        setup.avg_strategy[sorted_i * h + hand]
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
                            let child = topo.edge_child[setup.original_edge_for_sorted[sorted_i]];
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
        assert_eq!(
            spec.showdown_outcomes_p0.as_ref().map(Vec::len),
            Some(num_showdowns * num_hands * num_hands)
        );
        assert_eq!(
            spec.showdown_outcomes_p1.as_ref().map(Vec::len),
            Some(num_showdowns * num_hands * num_hands)
        );
        let num_folds = topo.fold_nodes.len();
        assert_eq!(spec.fold_payoffs_p0.len(), num_folds);
        assert_eq!(spec.fold_payoffs_p1.len(), num_folds);
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
        solver.set_leaf_injection(&leaf_node_ids, &leaf_depths).unwrap();

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
        solver.set_leaf_injection(&leaf_node_ids, &leaf_depths).unwrap();

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

    #[test]
    fn compute_reach_at_nodes_matches_evs_function() {
        use super::*;
        use crate::extract::{extract_terminal_data, extract_topology};

        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        // Solve to get strategy_sum.
        let mut solver = GpuBatchSolver::new(&topo, &term, 1, num_hands, 100).unwrap();
        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        let results = solver.solve_batch(&[spec.clone()]).unwrap();

        // Pick some internal nodes (non-root, non-terminal).
        let target_nodes: Vec<usize> = topo
            .level_nodes
            .iter()
            .flat_map(|level| level.iter())
            .filter(|&&n| topo.node_num_actions[n] > 0 && n > 0)
            .take(3)
            .copied()
            .collect();

        if target_nodes.is_empty() {
            return;
        }

        let reach = compute_reach_at_nodes(
            &topo,
            &results[0].strategy_sum,
            &spec.initial_weights,
            num_hands,
            &target_nodes,
        );

        // Verify: reach values are non-negative, and at least some are positive.
        for player in 0..2 {
            assert_eq!(reach[player].len(), target_nodes.len() * num_hands);
            assert!(reach[player].iter().all(|&r| r >= 0.0 && r.is_finite()));
            assert!(
                reach[player].iter().any(|&r| r > 0.0),
                "player {player} should have some positive reach"
            );
        }
    }

    fn make_wide_turn_game() -> range_solver::PostFlopGame {
        // Full ranges on a turn board: C(48,2) = 1128 hands per player (> 1024).
        let oop_range = range_solver::range::Range::ones();
        let ip_range = range_solver::range::Range::ones();
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
        let mut game =
            range_solver::PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);
        game
    }

    #[test]
    fn solve_batch_handles_more_than_1024_hands() {
        use super::*;

        let game = make_wide_turn_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        // This should be > 1024 for a full-range turn game.
        assert!(
            num_hands > 1024,
            "expected >1024 hands for full-range turn game, got {num_hands}"
        );

        let mut solver = GpuBatchSolver::new(&topo, &term, 1, num_hands, 50).unwrap();
        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        let results = solver.solve_batch(&[spec]).unwrap();

        // Verify we get a result with non-zero strategy_sum.
        assert!(!results[0].strategy_sum.is_empty());
        assert!(
            results[0].strategy_sum.iter().any(|&v| v != 0.0),
            "strategy_sum should have non-zero values after solving"
        );
    }

    #[test]
    fn batched_fold_payoffs_matches_single() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);

        // Path 1: solve_batch (calls prepare_batch internally, per-batch fold payoffs)
        let mut solver1 = GpuBatchSolver::new(&topo, &term, 4, num_hands, 100).unwrap();
        let result1 = solver1.solve_batch(&[spec.clone()]).unwrap();

        // Path 2: prepare_batch + run_iterations + extract_results (explicit per-batch)
        let mut solver2 = GpuBatchSolver::new(&topo, &term, 4, num_hands, 100).unwrap();
        solver2.prepare_batch(&[spec]).unwrap();
        solver2.run_iterations(0, 100).unwrap();
        let result2 = solver2.extract_results().unwrap();

        // Strategy sums must match within 1e-4
        assert_eq!(result1[0].strategy_sum.len(), result2[0].strategy_sum.len());
        for (i, (a, b)) in result1[0]
            .strategy_sum
            .iter()
            .zip(&result2[0].strategy_sum)
            .enumerate()
        {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-4,
                "strategy_sum[{i}] mismatch: solve_batch={a} vs prepare+run={b} diff={diff}"
            );
        }
    }

    #[test]
    fn multi_game_batch_different_payoffs() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let scales = [1.0f32, 2.0, 0.5, 3.0];
        let mut specs = Vec::new();
        for &scale in &scales {
            let mut spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
            // Scale fold payoffs to simulate different pot sizes
            for v in &mut spec.fold_payoffs_p0 {
                *v *= scale;
            }
            for v in &mut spec.fold_payoffs_p1 {
                *v *= scale;
            }
            specs.push(spec);
        }

        let mut solver = GpuBatchSolver::new(&topo, &term, 4, num_hands, 100).unwrap();
        solver.prepare_batch(&specs).unwrap();
        solver.run_iterations(0, 100).unwrap();
        let results = solver.extract_results().unwrap();

        assert_eq!(results.len(), 4);
        // Different payoff scales should produce different strategies
        assert!(
            results[0].strategy_sum != results[1].strategy_sum,
            "scale 1.0 and 2.0 should produce different strategies"
        );
    }

    /// Identity check for per-batch leaf CFV indexing at `bid=0`.
    ///
    /// Two independent batch-of-1 solves with the same leaf CFV values must
    /// produce identical strategy_sums. Validates that the kernel reads
    /// `leaf_cfv[0 * num_leaves * H + li * H + h]` rather than something else.
    ///
    /// Uses `showdown_nodes` as leaf injection points — they're genuine
    /// terminal nodes with no outgoing edges, so the injected values survive
    /// the backward pass and propagate up as the true terminal CFV at those
    /// nodes (overriding the showdown eval that would have run at the same
    /// depth immediately before leaf injection).
    #[test]
    fn batched_leaf_cfvs_matches_single() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        assert!(
            !topo.showdown_nodes.is_empty(),
            "river topology must have showdown nodes to use as leaf injection points"
        );
        let leaf_node_ids: Vec<i32> = topo
            .showdown_nodes
            .iter()
            .take(3)
            .map(|&n| n as i32)
            .collect();
        let leaf_depths: Vec<i32> = topo
            .showdown_nodes
            .iter()
            .take(3)
            .map(|&n| topo.node_depth[n] as i32)
            .collect();
        let num_leaves = leaf_node_ids.len();

        // Deterministic non-trivial leaf CFVs (batch_size = 1 slot).
        let leaf_cfv_size = num_leaves * num_hands;
        let p0_cfvs: Vec<f32> = (0..leaf_cfv_size)
            .map(|i| 0.25 + 0.01 * (i as f32))
            .collect();
        let p1_cfvs: Vec<f32> = (0..leaf_cfv_size)
            .map(|i| -0.1 - 0.005 * (i as f32))
            .collect();

        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);

        // Solver A: batch of 1, per-batch leaf CFV upload.
        let mut solver_a = GpuBatchSolver::new(&topo, &term, 4, num_hands, 100).unwrap();
        solver_a
            .set_leaf_injection(&leaf_node_ids, &leaf_depths)
            .unwrap();
        solver_a.prepare_batch(&[spec.clone()]).unwrap();
        solver_a.update_leaf_cfvs(&p0_cfvs, &p1_cfvs).unwrap();
        solver_a.run_iterations(0, 100).unwrap();
        let result_a = solver_a.extract_results().unwrap();

        // Solver B: identical setup.
        let mut solver_b = GpuBatchSolver::new(&topo, &term, 4, num_hands, 100).unwrap();
        solver_b
            .set_leaf_injection(&leaf_node_ids, &leaf_depths)
            .unwrap();
        solver_b.prepare_batch(&[spec]).unwrap();
        solver_b.update_leaf_cfvs(&p0_cfvs, &p1_cfvs).unwrap();
        solver_b.run_iterations(0, 100).unwrap();
        let result_b = solver_b.extract_results().unwrap();

        assert_eq!(result_a.len(), 1);
        assert_eq!(result_b.len(), 1);
        assert_eq!(
            result_a[0].strategy_sum.len(),
            result_b[0].strategy_sum.len()
        );
        for (i, (a, b)) in result_a[0]
            .strategy_sum
            .iter()
            .zip(&result_b[0].strategy_sum)
            .enumerate()
        {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-4,
                "strategy_sum[{i}] mismatch across two identical batch-of-1 solves: {a} vs {b} diff={diff}"
            );
        }
        // And confirm the solve actually produced non-trivial output.
        assert!(
            result_a[0].strategy_sum.iter().any(|&v| v != 0.0),
            "strategy_sum should be non-trivial"
        );
    }

    /// 4 games in a single batch, each with different leaf CFVs, must produce
    /// different strategy_sums across games. Proves the kernel reads per-batch
    /// leaf CFV slots (`bid * num_leaves * H + ...`), not a shared one.
    ///
    /// Uses `showdown_nodes` as leaf injection points so the injected values
    /// actually propagate up the tree and influence regrets / strategies.
    #[test]
    fn multi_game_batch_different_leaf_cfvs() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        // We need at least `batch_size` showdown leaves so we can pair each
        // batch with a distinct "favored" leaf index and force different
        // strategy preferences across batches.
        let batch_size = topo.showdown_nodes.len().min(4);
        assert!(
            batch_size >= 2,
            "river topology must have at least 2 showdown nodes (got {})",
            topo.showdown_nodes.len()
        );
        let leaf_node_ids: Vec<i32> = topo
            .showdown_nodes
            .iter()
            .take(batch_size)
            .map(|&n| n as i32)
            .collect();
        let leaf_depths: Vec<i32> = topo
            .showdown_nodes
            .iter()
            .take(batch_size)
            .map(|&n| topo.node_depth[n] as i32)
            .collect();
        let num_leaves = leaf_node_ids.len();

        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        let specs: Vec<SubgameSpec> = (0..batch_size).map(|_| spec.clone()).collect();

        // Distinct leaf CFV values per game. To force *different* strategy
        // preferences (not just different magnitudes of the same preference),
        // each batch picks a different "favored" leaf index. Because
        // num_leaves >= batch_size, each batch has a unique favored index
        // so no two batches end up with identical CFV layouts.
        let per_game = num_leaves * num_hands;
        let mut p0_batched = vec![0.0f32; batch_size * per_game];
        let mut p1_batched = vec![0.0f32; batch_size * per_game];
        for b in 0..batch_size {
            let favored_leaf = b; // unique per batch since b < batch_size <= num_leaves
            for li in 0..num_leaves {
                let leaf_value_p0 = if li == favored_leaf { 100.0 } else { -100.0 };
                let leaf_value_p1 = if li == favored_leaf { -80.0 } else { 80.0 };
                for h in 0..num_hands {
                    p0_batched[b * per_game + li * num_hands + h] = leaf_value_p0;
                    p1_batched[b * per_game + li * num_hands + h] = leaf_value_p1;
                }
            }
        }

        let mut solver =
            GpuBatchSolver::new(&topo, &term, batch_size, num_hands, 100).unwrap();
        solver
            .set_leaf_injection(&leaf_node_ids, &leaf_depths)
            .unwrap();
        solver.prepare_batch(&specs).unwrap();
        solver.update_leaf_cfvs(&p0_batched, &p1_batched).unwrap();
        solver.run_iterations(0, 100).unwrap();
        let results = solver.extract_results().unwrap();

        assert_eq!(results.len(), batch_size);

        // Every strategy_sum must be finite.
        for (bi, r) in results.iter().enumerate() {
            for (i, &v) in r.strategy_sum.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "strategy_sum[batch={bi}][{i}] not finite: {v}"
                );
            }
        }

        // Pairwise different: each pair must differ somewhere (different leaf
        // CFVs must yield different strategies).
        for i in 0..batch_size {
            for j in (i + 1)..batch_size {
                assert_ne!(
                    results[i].strategy_sum, results[j].strategy_sum,
                    "batch {i} and {j} should have different strategy_sums (different leaf CFVs)"
                );
            }
        }
    }

    /// A batch with `None` showdown outcomes + leaf-injected CFVs must produce
    /// non-trivial, different strategy_sums per batch. This protects the
    /// turn-datagen OOM fix: when `showdown_outcomes_{p0,p1}` are `None`,
    /// `prepare_batch` skips the host alloc / H2D upload, `active_num_showdowns`
    /// is set to 0, and the kernel's showdown loop runs zero iterations — all
    /// terminal CFVs must come from leaf injection.
    ///
    /// Uses the river topology (same as `batched_leaf_cfvs_matches_single`) with
    /// `showdown_nodes` as leaf-injection points. The specs' `showdown_outcomes_*`
    /// are set to `None` instead of `Some(mtd.showdown_outcomes_*)`, so no
    /// showdown-pass contributes to CFVs — only the injected leaf CFVs do.
    #[test]
    fn batch_with_none_showdowns_returns_leaf_injection_values() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        assert!(
            !topo.showdown_nodes.is_empty(),
            "river topology must have showdown nodes to use as leaf injection points"
        );
        // Use every showdown node as a leaf injection point so all terminal
        // values come from injection.
        let leaf_node_ids: Vec<i32> =
            topo.showdown_nodes.iter().map(|&n| n as i32).collect();
        let leaf_depths: Vec<i32> = topo
            .showdown_nodes
            .iter()
            .map(|&n| topo.node_depth[n] as i32)
            .collect();
        let num_leaves = leaf_node_ids.len();

        // Build a spec with None showdown outcomes. Reuse fold payoffs and initial
        // weights from the standard river spec so the non-showdown paths still work.
        let base_spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        let spec_none = SubgameSpec {
            initial_weights: base_spec.initial_weights.clone(),
            showdown_outcomes_p0: None,
            showdown_outcomes_p1: None,
            fold_payoffs_p0: base_spec.fold_payoffs_p0.clone(),
            fold_payoffs_p1: base_spec.fold_payoffs_p1.clone(),
        };

        // Two specs in the batch, each with distinct leaf CFV values so we can
        // prove per-batch slot indexing is correct.
        let batch_size = 2;
        let specs: Vec<SubgameSpec> =
            (0..batch_size).map(|_| spec_none.clone()).collect();

        // Distinct per-batch leaf CFVs to force different strategies.
        let per_game = num_leaves * num_hands;
        let mut p0_batched = vec![0.0f32; batch_size * per_game];
        let mut p1_batched = vec![0.0f32; batch_size * per_game];
        for b in 0..batch_size {
            let favored_leaf = b % num_leaves;
            for li in 0..num_leaves {
                let v_p0 = if li == favored_leaf { 100.0 } else { -100.0 };
                let v_p1 = if li == favored_leaf { -80.0 } else { 80.0 };
                for h in 0..num_hands {
                    p0_batched[b * per_game + li * num_hands + h] = v_p0;
                    p1_batched[b * per_game + li * num_hands + h] = v_p1;
                }
            }
        }

        let mut solver =
            GpuBatchSolver::new(&topo, &term, batch_size, num_hands, 100).unwrap();
        solver
            .set_leaf_injection(&leaf_node_ids, &leaf_depths)
            .unwrap();
        solver.prepare_batch(&specs).unwrap();
        solver.update_leaf_cfvs(&p0_batched, &p1_batched).unwrap();
        solver.run_iterations(0, 100).unwrap();
        let results = solver.extract_results().unwrap();

        assert_eq!(results.len(), batch_size);

        // All strategy_sums must be finite.
        for (bi, r) in results.iter().enumerate() {
            for (i, &v) in r.strategy_sum.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "strategy_sum[batch={bi}][{i}] not finite: {v}"
                );
            }
        }

        // At least one strategy_sum must be non-zero — proves the leaf-injected
        // CFVs propagated through the tree despite num_showdowns=0.
        for (bi, r) in results.iter().enumerate() {
            assert!(
                r.strategy_sum.iter().any(|&v| v != 0.0),
                "batch {bi} strategy_sum should be non-trivial (leaf injection must propagate)"
            );
        }

        // Pairwise different: distinct leaf CFVs must yield distinct strategies.
        for i in 0..batch_size {
            for j in (i + 1)..batch_size {
                assert_ne!(
                    results[i].strategy_sum, results[j].strategy_sum,
                    "batch {i} and {j} should differ (different leaf CFVs with None showdowns)"
                );
            }
        }
    }

    /// Enforces that a batch must be uniformly Some or None across all specs.
    /// In debug mode `prepare_batch` asserts this invariant. In the canonical-
    /// topology batched-datagen model all specs in a batch share the same
    /// terminal structure, so mixed states are a bug.
    #[test]
    #[should_panic(expected = "uniformly Some or None")]
    fn prepare_batch_panics_on_mixed_showdown_presence() {
        use super::*;
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let mut solver =
            GpuBatchSolver::new(&topo, &term, 2, num_hands, 10).unwrap();

        let base_spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        let spec_none = SubgameSpec {
            initial_weights: base_spec.initial_weights.clone(),
            showdown_outcomes_p0: None,
            showdown_outcomes_p1: None,
            fold_payoffs_p0: base_spec.fold_payoffs_p0.clone(),
            fold_payoffs_p1: base_spec.fold_payoffs_p1.clone(),
        };
        let spec_some = base_spec; // already Some(...) from SubgameSpec::from_game

        // Mixed batch must panic in debug.
        solver.prepare_batch(&[spec_none, spec_some]).unwrap();
    }

}
