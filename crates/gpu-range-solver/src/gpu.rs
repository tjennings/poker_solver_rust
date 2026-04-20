//! GPU device management: memory allocation, kernel compilation, solver state.
#![allow(clippy::too_many_arguments)]

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig};
use std::sync::Arc;

pub const BLOCK_SIZE: u32 = 256;

/// Compute grid size for `n` threads with `block_size` threads per block.
pub fn grid_size(n: usize, block_size: u32) -> u32 {
    (n as u32).div_ceil(block_size)
}

/// Compiled CUDA kernels, loaded once per solve.
pub struct CfrKernels {
    pub zero_f32: CudaFunction,
    pub regret_match_accum: CudaFunction,
    pub regret_match_normalize: CudaFunction,
    pub forward_pass_level: CudaFunction,
    pub backward_pass_level: CudaFunction,
    pub regret_update_level: CudaFunction,
    pub dcfr_discount: CudaFunction,
    pub fold_eval: CudaFunction,
    pub showdown_eval: CudaFunction,
    pub best_response_max_level: CudaFunction,
    pub set_reach_root: CudaFunction,
}

impl CfrKernels {
    pub fn compile(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(crate::kernels::CFR_KERNELS_SOURCE)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            zero_f32: module.load_function("zero_f32")?,
            regret_match_accum: module.load_function("regret_match_accum")?,
            regret_match_normalize: module.load_function("regret_match_normalize")?,
            forward_pass_level: module.load_function("forward_pass_level")?,
            backward_pass_level: module.load_function("backward_pass_level")?,
            regret_update_level: module.load_function("regret_update_level")?,
            dcfr_discount: module.load_function("dcfr_discount")?,
            fold_eval: module.load_function("fold_eval")?,
            showdown_eval: module.load_function("showdown_eval")?,
            best_response_max_level: module.load_function("best_response_max_level")?,
            set_reach_root: module.load_function("set_reach_root")?,
        })
    }
}

/// All GPU device memory for the solver.
pub struct GpuSolverState {
    // Topology (constant after upload)
    pub edge_parent: CudaSlice<i32>,
    pub edge_child: CudaSlice<i32>,
    pub edge_player: CudaSlice<i32>,
    pub actions_per_edge: CudaSlice<f32>,

    // Solver state (mutable)
    pub regrets: CudaSlice<f32>,
    pub strategy_sum: CudaSlice<f32>,
    pub strategy: CudaSlice<f32>,
    pub reach: CudaSlice<f32>,
    pub cfv: CudaSlice<f32>,
    pub denom: CudaSlice<f32>,

    // Dimensions
    pub num_nodes: usize,
    pub num_edges: usize,
    pub num_hands: usize,

    // Level dispatch info
    pub level_edge_start: Vec<usize>,
    pub level_edge_count: Vec<usize>,
    pub max_depth: usize,
}

impl GpuSolverState {
    pub fn new(
        _ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        num_nodes: usize,
        num_edges: usize,
        num_hands: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            edge_parent: stream.alloc_zeros::<i32>(num_edges.max(1))?,
            edge_child: stream.alloc_zeros::<i32>(num_edges.max(1))?,
            edge_player: stream.alloc_zeros::<i32>(num_edges.max(1))?,
            actions_per_edge: stream.alloc_zeros::<f32>(num_edges.max(1))?,
            regrets: stream.alloc_zeros::<f32>((num_edges * num_hands).max(1))?,
            strategy_sum: stream.alloc_zeros::<f32>((num_edges * num_hands).max(1))?,
            strategy: stream.alloc_zeros::<f32>((num_edges * num_hands).max(1))?,
            reach: stream.alloc_zeros::<f32>((num_nodes * num_hands).max(1))?,
            cfv: stream.alloc_zeros::<f32>((num_nodes * num_hands).max(1))?,
            denom: stream.alloc_zeros::<f32>((num_nodes * num_hands).max(1))?,
            num_nodes,
            num_edges,
            num_hands,
            level_edge_start: Vec::new(),
            level_edge_count: Vec::new(),
            max_depth: 0,
        })
    }

    /// Upload topology arrays from extracted tree data.
    pub fn upload_topology(
        &mut self,
        stream: &Arc<CudaStream>,
        edge_parent: &[i32],
        edge_child: &[i32],
        edge_player: &[i32],
        actions_per_edge: &[f32],
        level_edge_start: Vec<usize>,
        level_edge_count: Vec<usize>,
        max_depth: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.edge_parent = stream.clone_htod(edge_parent)?;
        self.edge_child = stream.clone_htod(edge_child)?;
        self.edge_player = stream.clone_htod(edge_player)?;
        self.actions_per_edge = stream.clone_htod(actions_per_edge)?;
        self.level_edge_start = level_edge_start;
        self.level_edge_count = level_edge_count;
        self.max_depth = max_depth;
        Ok(())
    }
}

/// Compiled cooperative mega-kernel: a single `cfr_solve` function.
pub struct MegaKernel {
    pub cfr_solve: CudaFunction,
}

impl MegaKernel {
    pub fn compile(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            crate::kernels::CFR_MEGA_KERNEL_SOURCE,
            crate::kernels::mega_kernel_compile_opts(),
        )?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            cfr_solve: module.load_function("cfr_solve")?,
        })
    }

    /// Query the max number of blocks for a cooperative launch of cfr_solve.
    pub fn max_cooperative_blocks(&self, block_size: u32) -> u32 {
        self.cfr_solve
            .occupancy_max_active_blocks_per_multiprocessor(block_size, 0, None)
            .unwrap_or(1)
    }
}

/// All GPU device memory for the cooperative mega-kernel solver.
pub struct GpuMegaState {
    // Topology (constant after upload, shared across batch)
    pub edge_parent: CudaSlice<i32>,
    pub edge_child: CudaSlice<i32>,
    pub edge_player: CudaSlice<i32>,
    pub actions_per_edge: CudaSlice<f32>,
    pub level_starts_gpu: CudaSlice<i32>,
    pub level_counts_gpu: CudaSlice<i32>,

    // Solver state (mutable) — [B * E * H] or [B * N * H]
    pub regrets: CudaSlice<f32>,
    pub strategy_sum: CudaSlice<f32>,
    pub strategy: CudaSlice<f32>,
    pub reach: CudaSlice<f32>,
    pub cfv: CudaSlice<f32>,
    pub denom: CudaSlice<f32>,

    // Terminal data — fold
    pub fold_node_ids: CudaSlice<i32>,
    pub fold_payoffs_p0: CudaSlice<f32>,
    pub fold_payoffs_p1: CudaSlice<f32>,
    pub fold_depths: CudaSlice<i32>,

    // Terminal data — showdown
    pub showdown_node_ids: CudaSlice<i32>,
    pub showdown_outcomes_p0: CudaSlice<f32>,
    pub showdown_outcomes_p1: CudaSlice<f32>,
    pub showdown_num_player: CudaSlice<i32>,
    pub showdown_depths: CudaSlice<i32>,

    // Card data for fold eval — [2 * H]
    pub player_card1: CudaSlice<i32>,
    pub player_card2: CudaSlice<i32>,
    pub opp_card1: CudaSlice<i32>,
    pub opp_card2: CudaSlice<i32>,
    pub same_hand_idx: CudaSlice<i32>,

    // Initial weights — [B * 2 * H]
    pub initial_weights: CudaSlice<f32>,

    // Leaf value injection (for two-pass turn solve)
    pub leaf_cfv_p0: CudaSlice<f32>,
    pub leaf_cfv_p1: CudaSlice<f32>,
    pub leaf_node_ids: CudaSlice<i32>,
    pub leaf_depths: CudaSlice<i32>,

    // Dimensions
    pub batch_size: usize,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub num_hands: usize,
    pub max_depth: usize,
    pub num_folds: usize,
    pub num_showdowns: usize,
    pub num_leaves: usize,

    // CPU-side level info (for exploitability computation)
    pub level_edge_start: Vec<usize>,
    pub level_edge_count: Vec<usize>,
}

impl GpuMegaState {
    pub fn new(
        stream: &Arc<CudaStream>,
        batch_size: usize,
        num_nodes: usize,
        num_edges: usize,
        num_hands: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let beh = (batch_size * num_edges * num_hands).max(1);
        let bnh = (batch_size * num_nodes * num_hands).max(1);
        Ok(Self {
            edge_parent: stream.alloc_zeros::<i32>(num_edges.max(1))?,
            edge_child: stream.alloc_zeros::<i32>(num_edges.max(1))?,
            edge_player: stream.alloc_zeros::<i32>(num_edges.max(1))?,
            actions_per_edge: stream.alloc_zeros::<f32>(num_edges.max(1))?,
            level_starts_gpu: stream.alloc_zeros::<i32>(1)?,
            level_counts_gpu: stream.alloc_zeros::<i32>(1)?,
            regrets: stream.alloc_zeros::<f32>(beh)?,
            strategy_sum: stream.alloc_zeros::<f32>(beh)?,
            strategy: stream.alloc_zeros::<f32>(beh)?,
            reach: stream.alloc_zeros::<f32>(bnh)?,
            cfv: stream.alloc_zeros::<f32>(bnh)?,
            denom: stream.alloc_zeros::<f32>(bnh)?,
            fold_node_ids: stream.alloc_zeros::<i32>(1)?,
            fold_payoffs_p0: stream.alloc_zeros::<f32>(1)?,
            fold_payoffs_p1: stream.alloc_zeros::<f32>(1)?,
            fold_depths: stream.alloc_zeros::<i32>(1)?,
            showdown_node_ids: stream.alloc_zeros::<i32>(1)?,
            showdown_outcomes_p0: stream.alloc_zeros::<f32>(1)?,
            showdown_outcomes_p1: stream.alloc_zeros::<f32>(1)?,
            showdown_num_player: stream.alloc_zeros::<i32>(1)?,
            showdown_depths: stream.alloc_zeros::<i32>(1)?,
            player_card1: stream.alloc_zeros::<i32>(1)?,
            player_card2: stream.alloc_zeros::<i32>(1)?,
            opp_card1: stream.alloc_zeros::<i32>(1)?,
            opp_card2: stream.alloc_zeros::<i32>(1)?,
            same_hand_idx: stream.alloc_zeros::<i32>(1)?,
            initial_weights: stream.alloc_zeros::<f32>(1)?,
            leaf_cfv_p0: stream.alloc_zeros::<f32>(1)?,
            leaf_cfv_p1: stream.alloc_zeros::<f32>(1)?,
            leaf_node_ids: stream.alloc_zeros::<i32>(1)?,
            leaf_depths: stream.alloc_zeros::<i32>(1)?,
            batch_size,
            num_nodes,
            num_edges,
            num_hands,
            max_depth: 0,
            num_folds: 0,
            num_showdowns: 0,
            num_leaves: 0,
            level_edge_start: Vec::new(),
            level_edge_count: Vec::new(),
        })
    }

    /// Upload topology arrays to GPU (level_starts and level_counts as i32 slices).
    pub fn upload_topology(
        &mut self,
        stream: &Arc<CudaStream>,
        edge_parent: &[i32],
        edge_child: &[i32],
        edge_player: &[i32],
        actions_per_edge: &[f32],
        level_starts: &[i32],
        level_counts: &[i32],
        max_depth: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.edge_parent = stream.clone_htod(edge_parent)?;
        self.edge_child = stream.clone_htod(edge_child)?;
        self.edge_player = stream.clone_htod(edge_player)?;
        self.actions_per_edge = stream.clone_htod(actions_per_edge)?;
        self.level_starts_gpu = stream.clone_htod(level_starts)?;
        self.level_counts_gpu = stream.clone_htod(level_counts)?;
        self.level_edge_start = level_starts.iter().map(|&v| v as usize).collect();
        self.level_edge_count = level_counts.iter().map(|&v| v as usize).collect();
        self.max_depth = max_depth;
        Ok(())
    }
}

/// Compiled hand-parallel kernel: a single `cfr_solve` function.
/// No cooperative groups — standard kernel launch.
pub struct HandParallelKernel {
    pub cfr_solve: CudaFunction,
}

impl HandParallelKernel {
    pub fn compile(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            crate::kernels::HAND_PARALLEL_KERNEL_SOURCE,
            crate::kernels::hand_parallel_compile_opts(),
        )?;
        let module = ctx.load_module(ptx)?;
        let cfr_solve = module.load_function("cfr_solve")?;

        // Opt into Ada's extended per-block dynamic smem (99 KB) as a
        // defense-in-depth measure. Current smem budget is ~26 KB — well
        // under the 48 KB default — so this only matters if the topology
        // grows. Failure is not fatal (pre-Ampere GPUs may not support it).
        // See bean poker_solver_rust-oox2.
        let _ = cfr_solve.set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            99 * 1024,
        );

        Ok(Self { cfr_solve })
    }
}

/// GPU state for the hand-parallel solver.
/// Batched layout: `[B * E * H]` for edge-indexed, `[B * N * H]` for node-indexed.
pub struct GpuHandParallelState {
    pub regrets: CudaSlice<f32>,
    pub strategy_sum: CudaSlice<f32>,
    pub reach: CudaSlice<f32>,
    pub cfv: CudaSlice<f32>,
    pub batch_size: usize,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub num_hands: usize,
}

impl GpuHandParallelState {
    pub fn new(
        stream: &Arc<CudaStream>,
        batch: usize,
        nodes: usize,
        edges: usize,
        hands: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let beh = (batch * edges * hands).max(1);
        let bnh = (batch * nodes * hands).max(1);
        Ok(Self {
            regrets: stream.alloc_zeros::<f32>(beh)?,
            strategy_sum: stream.alloc_zeros::<f32>(beh)?,
            reach: stream.alloc_zeros::<f32>(bnh)?,
            cfv: stream.alloc_zeros::<f32>(bnh)?,
            batch_size: batch,
            num_nodes: nodes,
            num_edges: edges,
            num_hands: hands,
        })
    }
}

/// Compute dynamic shared memory size needed for the hand-parallel kernel.
/// Dynamic layout (extern __shared__):
///   level_starts[max_depth+1] + level_counts[max_depth+1] (int each)
///   + actions_per_node[N] (float)
/// Note: edge_parent/edge_child/edge_player are read directly from global
/// memory (uniform-broadcast access pattern, L1-cached). They used to live
/// in smem but that pushed the canonical turn tree over CUDA's per-block
/// limit (~103 KB vs 48 KB default). See bean poker_solver_rust-oox2.
/// Note: card_reach[52] and total_reach are static __shared__ (not dynamic).
pub fn compute_hand_parallel_shared_mem(
    _num_edges: usize,
    max_depth: usize,
    num_nodes: usize,
) -> usize {
    let levels = 2 * (max_depth + 1) * 4; // starts + counts as int
    let actions = num_nodes * 4; // actions_per_node as float
    levels + actions
}

/// Launch helper: standard grid config for `n` elements.
pub fn launch_cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid_size(n, BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_compile_and_load_all_functions() {
        let ctx = CudaContext::new(0).unwrap();
        let kernels = CfrKernels::compile(&ctx);
        assert!(kernels.is_ok(), "kernel compilation failed: {:?}", kernels.err());
    }

    #[test]
    fn mega_kernel_compiles_and_loads() {
        let ctx = CudaContext::new(0).unwrap();
        let kernel = MegaKernel::compile(&ctx);
        assert!(kernel.is_ok(), "mega-kernel compilation failed: {:?}", kernel.err());
    }

    #[test]
    fn mega_kernel_max_cooperative_blocks_positive() {
        let ctx = CudaContext::new(0).unwrap();
        let kernel = MegaKernel::compile(&ctx).unwrap();
        let max_blocks = kernel.max_cooperative_blocks(BLOCK_SIZE);
        assert!(max_blocks > 0, "max_cooperative_blocks must be > 0, got {}", max_blocks);
    }

    #[test]
    fn gpu_state_allocates_correct_sizes() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let state = GpuSolverState::new(&ctx, &stream, 5, 8, 3);
        assert!(state.is_ok(), "state allocation failed: {:?}", state.err());
        let state = state.unwrap();
        assert_eq!(state.num_nodes, 5);
        assert_eq!(state.num_edges, 8);
        assert_eq!(state.num_hands, 3);
    }

    #[test]
    fn gpu_state_batched_allocates_correct_sizes() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let state = GpuMegaState::new(&stream, 2, 5, 8, 3);
        assert!(state.is_ok(), "batched state allocation failed: {:?}", state.err());
        let state = state.unwrap();
        assert_eq!(state.batch_size, 2);
        assert_eq!(state.num_nodes, 5);
        assert_eq!(state.num_edges, 8);
        assert_eq!(state.num_hands, 3);
    }

    #[test]
    fn gpu_state_batched_upload_topology() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut state = GpuMegaState::new(&stream, 1, 3, 2, 4).unwrap();

        let parent = vec![0i32, 0];
        let child = vec![1i32, 2];
        let player = vec![0i32, 0];
        let ape = vec![2.0f32, 2.0];
        let level_starts = vec![0i32];
        let level_counts = vec![2i32];

        let result = state.upload_topology(
            &stream, &parent, &child, &player, &ape,
            &level_starts, &level_counts, 0,
        );
        assert!(result.is_ok());
        assert_eq!(state.max_depth, 0);
    }

    #[test]
    fn gpu_state_upload_topology_roundtrips() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut state = GpuSolverState::new(&ctx, &stream, 3, 2, 4).unwrap();

        let parent = vec![0i32, 0];
        let child = vec![1i32, 2];
        let player = vec![0i32, 0];
        let ape = vec![2.0f32, 2.0];

        let result = state.upload_topology(
            &stream, &parent, &child, &player, &ape,
            vec![0], vec![2], 0,
        );
        assert!(result.is_ok());
        assert_eq!(state.max_depth, 0);
        assert_eq!(state.level_edge_start, vec![0]);
        assert_eq!(state.level_edge_count, vec![2]);
    }

    #[test]
    fn grid_size_rounds_up() {
        assert_eq!(grid_size(1, 256), 1);
        assert_eq!(grid_size(256, 256), 1);
        assert_eq!(grid_size(257, 256), 2);
        assert_eq!(grid_size(512, 256), 2);
        assert_eq!(grid_size(0, 256), 0);
    }

    #[test]
    fn launch_cfg_produces_valid_config() {
        let cfg = launch_cfg(1000);
        assert_eq!(cfg.block_dim, (BLOCK_SIZE, 1, 1));
        assert_eq!(cfg.grid_dim, (4, 1, 1));
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    // === Hand-parallel GPU state tests ===

    #[test]
    fn hand_parallel_kernel_compiles_and_loads() {
        let ctx = CudaContext::new(0).unwrap();
        let kernel = HandParallelKernel::compile(&ctx);
        assert!(kernel.is_ok(), "hand-parallel kernel compilation failed: {:?}", kernel.err());
    }

    #[test]
    fn hand_parallel_state_allocates() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let state = GpuHandParallelState::new(&stream, 4, 10, 15, 8);
        assert!(state.is_ok(), "hand-parallel state allocation failed: {:?}", state.err());
        let state = state.unwrap();
        assert_eq!(state.batch_size, 4);
        assert_eq!(state.num_nodes, 10);
        assert_eq!(state.num_edges, 15);
        assert_eq!(state.num_hands, 8);
    }

    #[test]
    fn hand_parallel_state_single_batch() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let state = GpuHandParallelState::new(&stream, 1, 5, 3, 16);
        assert!(state.is_ok());
        let state = state.unwrap();
        assert_eq!(state.batch_size, 1);
    }

    #[test]
    fn hand_parallel_shared_mem_size_computation() {
        // E=10 edges (unused after oox2 fix), max_depth=3 (4 levels), N=5 nodes
        let size = compute_hand_parallel_shared_mem(10, 3, 5);
        // Dynamic layout: 2*(max_depth+1)*4 + N*4 (edges now in global memory)
        let expected = 2 * 4 * 4 + 5 * 4;
        assert_eq!(size, expected);
    }

    #[test]
    fn canonical_turn_tree_smem_fits_under_cuda_default_limit() {
        // Regression test for bean oox2: the canonical turn tree
        // (SPR=100, bet sizes [25%, 50%, 100%, a] × [25%, 75%, a]) has
        // num_edges=6590, num_nodes=6591, max_depth=16. With edges stored
        // in dynamic shared memory, the required smem was ~103 KB which
        // exceeds CUDA's 48 KB default per-block limit, causing
        // CUDA_ERROR_INVALID_VALUE at kernel launch.
        //
        // After moving edge_parent/child/player out of smem and reading
        // them directly from global memory, the kernel's dynamic smem
        // must fit under the 48 KB default on any CUDA-capable GPU.
        const CUDA_DEFAULT_SMEM_PER_BLOCK: usize = 48 * 1024;
        let size = compute_hand_parallel_shared_mem(6590, 16, 6591);
        assert!(
            size <= CUDA_DEFAULT_SMEM_PER_BLOCK,
            "canonical turn tree smem {} bytes must fit under CUDA default {} bytes",
            size,
            CUDA_DEFAULT_SMEM_PER_BLOCK
        );
    }
}
