//! CPU precomputation for tabular (range-based) GPU CFR.
//!
//! Instead of processing individual deals in batches, the tabular approach
//! maintains factored reach vectors indexed by trajectory IDs. Terminal
//! computations become matrix-vector multiplies over coupling matrices.
//!
//! # Key structures
//!
//! - **Trajectories**: unique per-street hand encodings `[u32; 4]` for each player
//! - **Coupling matrices**: `W[t1][t2]`, `WE[t1][t2]`, `W_neg[t1][t2]`
//! - **Info ID table**: maps `(decision_node, trajectory) → dense_info_set_id`

use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;

use poker_solver_core::cfr::game_tree::{GameTree, NodeType};
use poker_solver_core::cfr::DealInfo;
use poker_solver_core::game::Player;

use crate::{
    align_up, collect_decision_nodes, create_buffer_init, create_buffer_zeroed,
    create_pipeline, encode_tree, entry, DecisionNodeInfo, GpuCfrConfig, GpuError,
    HAND_SHIFT, WORKGROUP_SIZE,
};

/// CPU-side precomputed data for tabular CFR.
pub struct TabularData {
    /// Unique P1 trajectory encodings: `traj_p1[id] = [preflop, flop, turn, river]`.
    pub traj_p1: Vec<[u32; 4]>,
    /// Unique P2 trajectory encodings.
    pub traj_p2: Vec<[u32; 4]>,
    /// Number of P1 trajectories.
    pub n_traj_p1: u32,
    /// Number of P2 trajectories.
    pub n_traj_p2: u32,
    /// `max(n_traj_p1, n_traj_p2)` — stride for info_id_table.
    pub max_traj: u32,

    /// Weight matrix `W[t1 * n_traj_p2 + t2]` — sum of deal weights for pair `(t1, t2)`.
    pub w_mat: Vec<f32>,
    /// Equity-weight matrix `WE[t1 * n_traj_p2 + t2]` = `Σ(w × equity)`.
    pub we_mat: Vec<f32>,
    /// Lose-weight matrix `W_neg[t1 * n_traj_p2 + t2]` = `Σ(w × (1 - equity))`.
    pub w_neg_mat: Vec<f32>,
    /// Transposed weight matrix `W^T[t2 * n_traj_p1 + t1]`.
    pub w_mat_t: Vec<f32>,
    /// Transposed equity-weight matrix.
    pub we_mat_t: Vec<f32>,
    /// Transposed lose-weight matrix.
    pub w_neg_mat_t: Vec<f32>,

    /// Info set ID lookup: `info_id_table[decision_idx * max_traj + traj_id]`.
    /// For P1 nodes, `traj_id` is a P1 trajectory index.
    /// For P2 nodes, `traj_id` is a P2 trajectory index.
    /// Unused slots contain `u32::MAX`.
    pub info_id_table: Vec<u32>,
    /// Number of actions per info set: `info_set_num_actions[dense_id]`.
    pub info_set_num_actions: Vec<u32>,
    /// Reverse mapping: `dense_to_key[dense_id] → u64 info set key`.
    pub dense_to_key: Vec<u64>,
    /// Total number of unique info sets.
    pub num_info_sets: u32,

    /// Weight sums for strategy-sum accumulation.
    /// `weight_sum_p1[t1] = Σ_{t2} W[t1][t2]`.
    pub weight_sum_p1: Vec<f32>,
    /// `weight_sum_p2[t2] = Σ_{t1} W[t1][t2]`.
    pub weight_sum_p2: Vec<f32>,

    /// Indices of decision nodes in the tree, sorted by node_idx.
    pub decision_node_indices: Vec<usize>,
}

/// Precompute all tabular data from deals and game tree.
///
/// Extracts unique trajectories, builds coupling matrices, and computes
/// the info set ID mapping table.
pub fn precompute_tabular(
    tree: &GameTree,
    deals: &[DealInfo],
    decision_nodes: &[DecisionNodeInfo],
) -> TabularData {
    let (traj_p1, traj_p1_map) = extract_unique_trajectories_p1(deals);
    let (traj_p2, traj_p2_map) = extract_unique_trajectories_p2(deals);
    let n1 = traj_p1.len();
    let n2 = traj_p2.len();

    let (w_mat, we_mat, w_neg_mat) =
        build_coupling_matrices(deals, &traj_p1_map, &traj_p2_map, n1, n2);

    let w_mat_t = transpose(&w_mat, n1, n2);
    let we_mat_t = transpose(&we_mat, n1, n2);
    let w_neg_mat_t = transpose(&w_neg_mat, n1, n2);

    let weight_sum_p1 = row_sums(&w_mat, n1, n2);
    let weight_sum_p2 = col_sums(&w_mat, n1, n2);

    let max_traj = n1.max(n2) as u32;

    let decision_node_indices: Vec<usize> = decision_nodes.iter().map(|dn| dn.node_idx).collect();

    let (info_id_table, info_set_num_actions, dense_to_key, num_info_sets) =
        build_info_id_table(
            tree,
            decision_nodes,
            &traj_p1,
            &traj_p2,
            max_traj,
        );

    TabularData {
        traj_p1,
        traj_p2,
        n_traj_p1: n1 as u32,
        n_traj_p2: n2 as u32,
        max_traj,
        w_mat,
        we_mat,
        w_neg_mat,
        w_mat_t,
        we_mat_t,
        w_neg_mat_t,
        info_id_table,
        info_set_num_actions,
        dense_to_key,
        num_info_sets,
        weight_sum_p1,
        weight_sum_p2,
        decision_node_indices,
    }
}

// ---------------------------------------------------------------------------
// Trajectory extraction
// ---------------------------------------------------------------------------

fn extract_unique_trajectories_p1(
    deals: &[DealInfo],
) -> (Vec<[u32; 4]>, FxHashMap<[u32; 4], u32>) {
    let mut map = FxHashMap::default();
    let mut trajs = Vec::new();
    for deal in deals {
        map.entry(deal.hand_bits_p1).or_insert_with(|| {
            let id = trajs.len() as u32;
            trajs.push(deal.hand_bits_p1);
            id
        });
    }
    (trajs, map)
}

fn extract_unique_trajectories_p2(
    deals: &[DealInfo],
) -> (Vec<[u32; 4]>, FxHashMap<[u32; 4], u32>) {
    let mut map = FxHashMap::default();
    let mut trajs = Vec::new();
    for deal in deals {
        map.entry(deal.hand_bits_p2).or_insert_with(|| {
            let id = trajs.len() as u32;
            trajs.push(deal.hand_bits_p2);
            id
        });
    }
    (trajs, map)
}

// ---------------------------------------------------------------------------
// Coupling matrices
// ---------------------------------------------------------------------------

/// Build the three coupling matrices `W`, `WE`, `W_neg` from deals.
///
/// - `W[t1][t2]`    = `Σ deal.weight` for all deals with `(traj_p1=t1, traj_p2=t2)`
/// - `WE[t1][t2]`   = `Σ deal.weight × deal.p1_equity`
/// - `W_neg[t1][t2]` = `Σ deal.weight × (1 - deal.p1_equity)`
fn build_coupling_matrices(
    deals: &[DealInfo],
    traj_p1_map: &FxHashMap<[u32; 4], u32>,
    traj_p2_map: &FxHashMap<[u32; 4], u32>,
    n1: usize,
    n2: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let sz = n1 * n2;
    let mut w = vec![0.0f32; sz];
    let mut we = vec![0.0f32; sz];
    let mut w_neg = vec![0.0f32; sz];

    for deal in deals {
        let t1 = traj_p1_map[&deal.hand_bits_p1] as usize;
        let t2 = traj_p2_map[&deal.hand_bits_p2] as usize;
        let idx = t1 * n2 + t2;
        let dw = deal.weight as f32;
        let eq = deal.p1_equity as f32;
        w[idx] += dw;
        we[idx] += dw * eq;
        w_neg[idx] += dw * (1.0 - eq);
    }

    (w, we, w_neg)
}

/// Transpose a row-major matrix `[rows × cols]` → `[cols × rows]`.
fn transpose(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = mat[r * cols + c];
        }
    }
    out
}

/// Row sums of a row-major `[rows × cols]` matrix.
fn row_sums(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| mat[r * cols..(r + 1) * cols].iter().sum())
        .collect()
}

/// Column sums of a row-major `[rows × cols]` matrix.
fn col_sums(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut sums = vec![0.0f32; cols];
    for r in 0..rows {
        for c in 0..cols {
            sums[c] += mat[r * cols + c];
        }
    }
    sums
}

// ---------------------------------------------------------------------------
// Info set ID table
// ---------------------------------------------------------------------------

/// Build the info ID lookup table and per-info-set action counts.
///
/// For each decision node `d` and trajectory `t`, computes the full info set
/// key and assigns a dense ID. The table maps
/// `(decision_idx, traj_id) → dense_info_id`.
fn build_info_id_table(
    tree: &GameTree,
    decision_nodes: &[DecisionNodeInfo],
    traj_p1: &[[u32; 4]],
    traj_p2: &[[u32; 4]],
    max_traj: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u64>, u32) {
    let n_decision = decision_nodes.len();
    let max_t = max_traj as usize;
    let mut table = vec![u32::MAX; n_decision * max_t];

    let mut key_to_dense: FxHashMap<u64, u32> = FxHashMap::default();
    let mut dense_to_key: Vec<u64> = Vec::new();
    let mut info_set_num_actions: Vec<u32> = Vec::new();

    for (d_idx, dn) in decision_nodes.iter().enumerate() {
        let (trajs, n_traj): (&[[u32; 4]], usize) = match dn.player {
            Player::Player1 => (traj_p1, traj_p1.len()),
            Player::Player2 => (traj_p2, traj_p2.len()),
        };
        let num_children = tree.nodes[dn.node_idx].children.len() as u32;

        for t in 0..n_traj {
            let hand_bits = trajs[t][dn.street as usize];
            let key = dn.position_key | ((hand_bits as u64) << HAND_SHIFT);
            let dense_id = *key_to_dense.entry(key).or_insert_with(|| {
                let id = dense_to_key.len() as u32;
                dense_to_key.push(key);
                info_set_num_actions.push(num_children);
                id
            });
            table[d_idx * max_t + t] = dense_id;
        }
    }

    let num_info_sets = dense_to_key.len() as u32;
    (table, info_set_num_actions, dense_to_key, num_info_sets)
}

// ---------------------------------------------------------------------------
// Level partitioning (decision vs terminal nodes)
// ---------------------------------------------------------------------------

/// Partition tree levels into decision and terminal node lists.
///
/// Returns `(decision_flat, decision_offsets, decision_counts,
///           terminal_flat, terminal_offsets, terminal_counts)`.
pub fn partition_levels(tree: &GameTree) -> LevelPartition {
    let mut dec_flat = Vec::new();
    let mut dec_offsets = Vec::new();
    let mut dec_counts = Vec::new();
    let mut term_flat = Vec::new();
    let mut term_offsets = Vec::new();
    let mut term_counts = Vec::new();

    for level in &tree.levels {
        dec_offsets.push(dec_flat.len() as u32);
        term_offsets.push(term_flat.len() as u32);
        let mut dc = 0u32;
        let mut tc = 0u32;
        for &node_idx in level {
            match &tree.nodes[node_idx as usize].node_type {
                NodeType::Decision { .. } => {
                    dec_flat.push(node_idx);
                    dc += 1;
                }
                NodeType::Terminal { .. } => {
                    term_flat.push(node_idx);
                    tc += 1;
                }
            }
        }
        dec_counts.push(dc);
        term_counts.push(tc);
    }

    // Ensure non-empty buffers (wgpu requires non-zero)
    if dec_flat.is_empty() {
        dec_flat.push(0);
    }
    if term_flat.is_empty() {
        term_flat.push(0);
    }

    LevelPartition {
        decision_flat: dec_flat,
        decision_offsets: dec_offsets,
        decision_counts: dec_counts,
        terminal_flat: term_flat,
        terminal_offsets: term_offsets,
        terminal_counts: term_counts,
    }
}

/// Partitioned level data for separate terminal/decision dispatch.
pub struct LevelPartition {
    pub decision_flat: Vec<u32>,
    pub decision_offsets: Vec<u32>,
    pub decision_counts: Vec<u32>,
    pub terminal_flat: Vec<u32>,
    pub terminal_offsets: Vec<u32>,
    pub terminal_counts: Vec<u32>,
}

// ---------------------------------------------------------------------------
// GPU solver
// ---------------------------------------------------------------------------

/// Uniform parameters for the convergence metric shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ConvergenceUniforms {
    num_info_sets: u32,
    max_actions: u32,
    pass_id: u32,
    num_workgroups: u32,
}

/// Uniform parameters for tabular CFR shaders. 64 bytes, 16-aligned.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TabularUniforms {
    level_start: u32,
    level_count: u32,
    num_nodes: u32,
    n_traj_p1: u32,
    n_traj_p2: u32,
    max_traj: u32,
    num_info_sets: u32,
    max_actions: u32,
    iteration: u32,
    dcfr_alpha: f32,
    dcfr_beta: f32,
    dcfr_gamma: f32,
    strategy_discount: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Tabular (range-based) GPU CFR solver.
///
/// Instead of processing individual deals in batches, maintains factored reach
/// vectors indexed by trajectory IDs. Terminal computations are matrix-vector
/// multiplies over coupling matrices. No batching needed — all state fits in
/// GPU memory at once.
pub struct TabularGpuCfrSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Tree metadata
    num_nodes: u32,
    max_actions: u32,
    num_info_sets: u32,
    n_traj_p1: u32,
    n_traj_p2: u32,
    max_traj: u32,

    // CPU-side level partition
    decision_offsets: Vec<u32>,
    decision_counts: Vec<u32>,
    terminal_offsets: Vec<u32>,
    terminal_counts: Vec<u32>,
    num_levels: usize,

    // Info set key mapping (for extracting strategies)
    dense_to_key: Vec<u64>,

    // GPU buffers — Group 0: tree + uniforms
    node_buffer: wgpu::Buffer,
    children_buffer: wgpu::Buffer,
    decision_level_nodes_buffer: wgpu::Buffer,
    terminal_level_nodes_buffer: wgpu::Buffer,
    info_set_num_actions_buffer: wgpu::Buffer,
    uniform_buffers: [wgpu::Buffer; 2],
    uniform_stride: u32,

    // GPU buffers — Group 1: CFR state
    regret_buffer: wgpu::Buffer,
    strategy_buffer: wgpu::Buffer,
    strategy_sum_buffer: wgpu::Buffer,
    regret_delta_buffer: wgpu::Buffer,
    strat_sum_delta_buffer: wgpu::Buffer,

    // GPU buffers — Group 2: reach/utility/info
    reach_p1_buffer: wgpu::Buffer,
    reach_p2_buffer: wgpu::Buffer,
    util_p1_buffer: wgpu::Buffer,
    util_p2_buffer: wgpu::Buffer,
    info_id_table_buffer: wgpu::Buffer,
    weight_sum_buffer: wgpu::Buffer,

    // GPU buffers — Group 3: coupling matrices
    w_mat_buffer: wgpu::Buffer,
    we_mat_buffer: wgpu::Buffer,
    w_neg_mat_buffer: wgpu::Buffer,
    w_mat_t_buffer: wgpu::Buffer,
    we_mat_t_buffer: wgpu::Buffer,
    w_neg_mat_t_buffer: wgpu::Buffer,

    // Cached bind groups — two sets for double-buffered uniform pipelining
    bg0_dec: [wgpu::BindGroup; 2],
    bg0_term: [wgpu::BindGroup; 2],
    bg1: wgpu::BindGroup,
    bg2: wgpu::BindGroup,
    bg3: wgpu::BindGroup,

    // Compute pipelines
    init_pipeline: wgpu::ComputePipeline,
    merge_regret_match_pipeline: wgpu::ComputePipeline,
    forward_pipeline: wgpu::ComputePipeline,
    backward_terminal_pipeline: wgpu::ComputePipeline,
    backward_decision_pipeline: wgpu::ComputePipeline,

    // Convergence metric
    convergence_pipeline: wgpu::ComputePipeline,
    convergence_buffer: wgpu::Buffer,
    convergence_uniform_buffer: wgpu::Buffer,
    convergence_bg: wgpu::BindGroup,

    config: GpuCfrConfig,
    iterations: u64,
}

impl TabularGpuCfrSolver {
    /// Create a new tabular GPU CFR solver.
    ///
    /// Precomputes trajectories and coupling matrices from the deals,
    /// then uploads all static data to GPU. No batching needed.
    pub fn new(
        tree: &GameTree,
        deals: Vec<DealInfo>,
        config: GpuCfrConfig,
    ) -> Result<Self, GpuError> {
        if deals.is_empty() {
            return Err(GpuError::NoDeals);
        }

        let step = Instant::now();
        let (device, queue) = crate::init_gpu()?;
        println!("  GPU device acquired in {:?}", step.elapsed());

        // Encode tree
        let (gpu_nodes, children_flat, _level_nodes_flat, _level_offsets, _level_counts) =
            encode_tree(tree);
        let num_nodes = tree.nodes.len() as u32;

        // Partition levels into decision/terminal lists
        let partition = partition_levels(tree);
        let num_levels = partition.decision_counts.len();

        // Collect decision nodes and precompute tabular data
        let decision_nodes = collect_decision_nodes(tree);
        let step = Instant::now();
        let tab = precompute_tabular(tree, &deals, &decision_nodes);
        println!(
            "  Tabular precomputation: {} P1 traj, {} P2 traj, {} info sets in {:?}",
            tab.n_traj_p1, tab.n_traj_p2, tab.num_info_sets, step.elapsed()
        );

        // Find max_actions across all info sets
        let max_actions = tab.info_set_num_actions.iter().copied().max().unwrap_or(1);
        let num_info_sets = tab.num_info_sets;
        let n1 = tab.n_traj_p1;
        let n2 = tab.n_traj_p2;
        let max_traj = tab.max_traj;

        // Compute buffer sizes
        let state_size = u64::from(num_info_sets) * u64::from(max_actions) * 4;
        let reach_p1_size = u64::from(num_nodes) * u64::from(n1) * 4;
        let reach_p2_size = u64::from(num_nodes) * u64::from(n2) * 4;
        let mat_size = u64::from(n1) * u64::from(n2) * 4;
        println!(
            "  Buffers: state={:.1}MB, reach_p1={:.1}MB, reach_p2={:.1}MB, matrices={:.1}MB",
            state_size as f64 / 1_048_576.0,
            reach_p1_size as f64 / 1_048_576.0,
            reach_p2_size as f64 / 1_048_576.0,
            (mat_size * 6) as f64 / 1_048_576.0,
        );

        // Create GPU buffers
        let step = Instant::now();

        // Group 0: tree
        let node_buffer = create_buffer_init(
            &device, "tab_nodes", bytemuck::cast_slice(&gpu_nodes),
            wgpu::BufferUsages::STORAGE,
        );
        let children_buffer = create_buffer_init(
            &device, "tab_children", bytemuck::cast_slice(&children_flat),
            wgpu::BufferUsages::STORAGE,
        );
        let decision_level_nodes_buffer = create_buffer_init(
            &device, "tab_dec_level", bytemuck::cast_slice(&partition.decision_flat),
            wgpu::BufferUsages::STORAGE,
        );
        let terminal_level_nodes_buffer = create_buffer_init(
            &device, "tab_term_level", bytemuck::cast_slice(&partition.terminal_flat),
            wgpu::BufferUsages::STORAGE,
        );
        let info_set_num_actions_buffer = create_buffer_init(
            &device, "tab_is_nactions", bytemuck::cast_slice(&tab.info_set_num_actions),
            wgpu::BufferUsages::STORAGE,
        );

        // Group 1: CFR state
        let regret_buffer = create_buffer_zeroed(
            &device, "tab_regret", state_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let strategy_buffer = create_buffer_zeroed(
            &device, "tab_strategy", state_size,
            wgpu::BufferUsages::STORAGE,
        );
        let strategy_sum_buffer = create_buffer_zeroed(
            &device, "tab_strat_sum", state_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let regret_delta_buffer = create_buffer_zeroed(
            &device, "tab_regret_d", state_size,
            wgpu::BufferUsages::STORAGE,
        );
        let strat_sum_delta_buffer = create_buffer_zeroed(
            &device, "tab_ss_delta", state_size,
            wgpu::BufferUsages::STORAGE,
        );

        // Group 2: reach/utility/info
        let reach_p1_buffer = create_buffer_zeroed(
            &device, "tab_reach_p1", reach_p1_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let reach_p2_buffer = create_buffer_zeroed(
            &device, "tab_reach_p2", reach_p2_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let util_p1_buffer = create_buffer_zeroed(
            &device, "tab_util_p1", reach_p1_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let util_p2_buffer = create_buffer_zeroed(
            &device, "tab_util_p2", reach_p2_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let info_id_table_buffer = create_buffer_init(
            &device, "tab_info_id", bytemuck::cast_slice(&tab.info_id_table),
            wgpu::BufferUsages::STORAGE,
        );
        // Combine weight_sum_p1 and weight_sum_p2 into one buffer
        let mut weight_sum_combined = tab.weight_sum_p1.clone();
        weight_sum_combined.extend_from_slice(&tab.weight_sum_p2);
        let weight_sum_buffer = create_buffer_init(
            &device, "tab_weight_sum", bytemuck::cast_slice(&weight_sum_combined),
            wgpu::BufferUsages::STORAGE,
        );

        // Group 3: coupling matrices
        let w_mat_buffer = create_buffer_init(
            &device, "tab_w", bytemuck::cast_slice(&tab.w_mat),
            wgpu::BufferUsages::STORAGE,
        );
        let we_mat_buffer = create_buffer_init(
            &device, "tab_we", bytemuck::cast_slice(&tab.we_mat),
            wgpu::BufferUsages::STORAGE,
        );
        let w_neg_mat_buffer = create_buffer_init(
            &device, "tab_wn", bytemuck::cast_slice(&tab.w_neg_mat),
            wgpu::BufferUsages::STORAGE,
        );
        let w_mat_t_buffer = create_buffer_init(
            &device, "tab_wt", bytemuck::cast_slice(&tab.w_mat_t),
            wgpu::BufferUsages::STORAGE,
        );
        let we_mat_t_buffer = create_buffer_init(
            &device, "tab_wet", bytemuck::cast_slice(&tab.we_mat_t),
            wgpu::BufferUsages::STORAGE,
        );
        let w_neg_mat_t_buffer = create_buffer_init(
            &device, "tab_wnt", bytemuck::cast_slice(&tab.w_neg_mat_t),
            wgpu::BufferUsages::STORAGE,
        );

        // Double-buffered uniform buffers for pipelined chunk encoding
        // Slots: merge_regret_match(1) + forward(L) + backward_term(L) + backward_dec(L)
        let max_uniform_slots = (1 + 3 * num_levels) as u64;
        let min_align = device.limits().min_uniform_buffer_offset_alignment;
        let uniform_stride = align_up(std::mem::size_of::<TabularUniforms>() as u32, min_align);
        let uniform_buf_size = max_uniform_slots * u64::from(uniform_stride);
        let uniform_buffers = [
            create_buffer_zeroed(
                &device, "tab_uniforms_a", uniform_buf_size,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            create_buffer_zeroed(
                &device, "tab_uniforms_b", uniform_buf_size,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
        ];

        println!("  GPU buffers allocated in {:?}", step.elapsed());

        // Create bind group layouts and pipelines
        let step = Instant::now();
        let group0_layout = create_tab_group0_layout(&device);
        let group1_layout = create_tab_group1_layout(&device);
        let group2_layout = create_tab_group2_layout(&device);
        let group3_layout = create_tab_group3_layout(&device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tabular_pipeline_layout"),
            bind_group_layouts: &[&group0_layout, &group1_layout, &group2_layout, &group3_layout],
            immediate_size: 0,
        });

        let init_pipeline = create_pipeline(
            &device, &pipeline_layout, "tab_init",
            include_str!("shaders/tabular_init.wgsl"),
        );
        let merge_regret_match_pipeline = create_pipeline(
            &device, &pipeline_layout, "tab_merge_rm",
            include_str!("shaders/tabular_merge_regret_match.wgsl"),
        );
        let forward_pipeline = create_pipeline(
            &device, &pipeline_layout, "tab_forward",
            include_str!("shaders/tabular_forward.wgsl"),
        );
        let backward_terminal_pipeline = create_pipeline(
            &device, &pipeline_layout, "tab_bwd_term",
            include_str!("shaders/tabular_backward_terminal.wgsl"),
        );
        let backward_decision_pipeline = create_pipeline(
            &device, &pipeline_layout, "tab_bwd_dec",
            include_str!("shaders/tabular_backward_decision.wgsl"),
        );

        // Convergence pipeline (separate layout: uniform + regret + scratch)
        let conv_wgs = num_info_sets.div_ceil(WORKGROUP_SIZE);
        let conv_scratch_size = u64::from(conv_wgs.max(1)) * 4;
        let convergence_buffer = create_buffer_zeroed(
            &device, "tab_conv_scratch", conv_scratch_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let conv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tab_conv_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::layout_entry(1, crate::storage_rw()), // regret
                crate::layout_entry(2, crate::storage_rw()), // scratch
            ],
        });
        let conv_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tab_conv_pl"),
            bind_group_layouts: &[&conv_layout],
            immediate_size: 0,
        });
        let convergence_pipeline = create_pipeline(
            &device, &conv_pipeline_layout, "tab_conv",
            include_str!("shaders/tabular_convergence.wgsl"),
        );
        // Convergence uniform buffer (16 bytes, rewritten each call)
        let conv_uniform_buffer = create_buffer_zeroed(
            &device, "tab_conv_uni",
            std::mem::size_of::<ConvergenceUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let convergence_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tab_conv_bg"),
            layout: &conv_layout,
            entries: &[
                entry(0, &conv_uniform_buffer),
                entry(1, &regret_buffer),
                entry(2, &convergence_buffer),
            ],
        });
        println!("  Shader pipelines compiled in {:?}", step.elapsed());

        // Create double-buffered bind groups for uniform pipelining
        let uniform_binding_size =
            std::num::NonZero::new(std::mem::size_of::<TabularUniforms>() as u64);
        let make_bg0 = |buf: &wgpu::Buffer, level_buf: &wgpu::Buffer, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &group0_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: buf,
                            offset: 0,
                            size: uniform_binding_size,
                        }),
                    },
                    entry(1, &node_buffer),
                    entry(2, &children_buffer),
                    entry(3, level_buf),
                    entry(4, &info_set_num_actions_buffer),
                ],
            })
        };
        let bg0_dec = [
            make_bg0(&uniform_buffers[0], &decision_level_nodes_buffer, "tab_g0_dec_a"),
            make_bg0(&uniform_buffers[1], &decision_level_nodes_buffer, "tab_g0_dec_b"),
        ];
        let bg0_term = [
            make_bg0(&uniform_buffers[0], &terminal_level_nodes_buffer, "tab_g0_term_a"),
            make_bg0(&uniform_buffers[1], &terminal_level_nodes_buffer, "tab_g0_term_b"),
        ];
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tab_g1"),
            layout: &group1_layout,
            entries: &[
                entry(0, &regret_buffer),
                entry(1, &strategy_buffer),
                entry(2, &strategy_sum_buffer),
                entry(3, &regret_delta_buffer),
                entry(4, &strat_sum_delta_buffer),
            ],
        });
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tab_g2"),
            layout: &group2_layout,
            entries: &[
                entry(0, &reach_p1_buffer),
                entry(1, &reach_p2_buffer),
                entry(2, &util_p1_buffer),
                entry(3, &util_p2_buffer),
                entry(4, &info_id_table_buffer),
                entry(5, &weight_sum_buffer),
            ],
        });
        let bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tab_g3"),
            layout: &group3_layout,
            entries: &[
                entry(0, &w_mat_buffer),
                entry(1, &we_mat_buffer),
                entry(2, &w_neg_mat_buffer),
                entry(3, &w_mat_t_buffer),
                entry(4, &we_mat_t_buffer),
                entry(5, &w_neg_mat_t_buffer),
            ],
        });

        Ok(Self {
            device,
            queue,
            num_nodes,
            max_actions,
            num_info_sets,
            n_traj_p1: n1,
            n_traj_p2: n2,
            max_traj,
            decision_offsets: partition.decision_offsets,
            decision_counts: partition.decision_counts,
            terminal_offsets: partition.terminal_offsets,
            terminal_counts: partition.terminal_counts,
            num_levels,
            dense_to_key: tab.dense_to_key,
            node_buffer,
            children_buffer,
            decision_level_nodes_buffer,
            terminal_level_nodes_buffer,
            info_set_num_actions_buffer,
            uniform_buffers,
            uniform_stride,
            regret_buffer,
            strategy_buffer,
            strategy_sum_buffer,
            regret_delta_buffer,
            strat_sum_delta_buffer,
            reach_p1_buffer,
            reach_p2_buffer,
            util_p1_buffer,
            util_p2_buffer,
            info_id_table_buffer,
            weight_sum_buffer,
            w_mat_buffer,
            we_mat_buffer,
            w_neg_mat_buffer,
            w_mat_t_buffer,
            we_mat_t_buffer,
            w_neg_mat_t_buffer,
            bg0_dec,
            bg0_term,
            bg1,
            bg2,
            bg3,
            init_pipeline,
            merge_regret_match_pipeline,
            forward_pipeline,
            backward_terminal_pipeline,
            backward_decision_pipeline,
            convergence_pipeline,
            convergence_buffer,
            convergence_uniform_buffer: conv_uniform_buffer,
            convergence_bg,
            config,
            iterations: 0,
        })
    }

    /// Run `num_iterations` of tabular CFR training on GPU.
    ///
    /// Uses double-buffered uniform pipelining: while the GPU executes chunk N
    /// (using buffer A), the CPU encodes uniforms for chunk N+1 (into buffer B).
    pub fn train(&mut self, num_iterations: u64) {
        let chunk_size = 10u64;
        let mut remaining = num_iterations;
        let mut buf_idx = 0usize;

        // Encode and submit first chunk
        if remaining > 0 {
            let n = remaining.min(chunk_size);
            let cmd = self.encode_chunk(n, buf_idx);
            self.queue.submit(std::iter::once(cmd));
            remaining -= n;
            self.iterations += n;

            // Pipeline: encode next chunk while GPU runs previous
            while remaining > 0 {
                let n = remaining.min(chunk_size);
                let next_buf = 1 - buf_idx;
                let cmd = self.encode_chunk(n, next_buf);
                // Wait for previous GPU work, then submit next
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
                self.queue.submit(std::iter::once(cmd));
                remaining -= n;
                self.iterations += n;
                buf_idx = next_buf;
            }

            // Wait for final chunk
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }

        // Flush: merge the last iteration's pending deltas
        self.flush_deltas(buf_idx);
    }

    /// Encode `n` iterations into a single GPU command buffer using the given
    /// uniform buffer index. Returns the finished command buffer for submission.
    fn encode_chunk(&mut self, n: u64, buf_idx: usize) -> wgpu::CommandBuffer {
        let slots_per_iter = 1 + 3 * self.num_levels;
        let total_slots = n as usize * slots_per_iter;

        // Resize uniform buffers if needed (both must stay the same size)
        let needed_bytes = total_slots as u64 * u64::from(self.uniform_stride);
        if needed_bytes > self.uniform_buffers[buf_idx].size() {
            self.rebuild_uniform_buffers(needed_bytes);
        }

        // Pre-upload ALL uniforms for n iterations into the selected buffer
        for i in 0..n as u32 {
            let iter_num = self.iterations + u64::from(i);
            let strategy_discount = Self::compute_strategy_discount_for(iter_num, self.config.dcfr_gamma);
            let base_slot = i as usize * slots_per_iter;

            // Slot 0: merge_regret_match
            self.write_uniform_slot(buf_idx, base_slot as u32, 0, 0, strategy_discount, iter_num as u32);

            // Forward slots
            for level in 0..self.num_levels {
                let slot = base_slot + 1 + level;
                self.write_uniform_slot(
                    buf_idx, slot as u32, self.decision_offsets[level], self.decision_counts[level],
                    strategy_discount, iter_num as u32,
                );
            }

            // Backward terminal slots
            for (j, level) in (0..self.num_levels).rev().enumerate() {
                let slot = base_slot + 1 + self.num_levels + j;
                self.write_uniform_slot(
                    buf_idx, slot as u32, self.terminal_offsets[level], self.terminal_counts[level],
                    strategy_discount, iter_num as u32,
                );
            }

            // Backward decision slots
            for (j, level) in (0..self.num_levels).rev().enumerate() {
                let slot = base_slot + 1 + 2 * self.num_levels + j;
                self.write_uniform_slot(
                    buf_idx, slot as u32, self.decision_offsets[level], self.decision_counts[level],
                    strategy_discount, iter_num as u32,
                );
            }
        }

        // Encode all n iterations into a single command buffer
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tabular_chunk") },
        );

        let r1_bytes = u64::from(self.num_nodes) * u64::from(self.n_traj_p1) * 4;
        let r2_bytes = u64::from(self.num_nodes) * u64::from(self.n_traj_p2) * 4;
        let stride = self.uniform_stride;

        for i in 0..n as u32 {
            let base_slot = i as usize * slots_per_iter;
            let mrm_offset = base_slot as u32 * stride;

            // Fused merge + regret match
            {
                let mut pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor { label: Some("tab_mrm"), ..Default::default() },
                );
                pass.set_pipeline(&self.merge_regret_match_pipeline);
                pass.set_bind_group(0, &self.bg0_dec[buf_idx], &[mrm_offset]);
                pass.set_bind_group(1, &self.bg1, &[]);
                pass.set_bind_group(2, &self.bg2, &[]);
                pass.set_bind_group(3, &self.bg3, &[]);
                pass.dispatch_workgroups(self.num_info_sets.div_ceil(WORKGROUP_SIZE), 1, 1);
            }

            // Clear reach and utility
            encoder.clear_buffer(&self.reach_p1_buffer, 0, Some(r1_bytes));
            encoder.clear_buffer(&self.reach_p2_buffer, 0, Some(r2_bytes));
            encoder.clear_buffer(&self.util_p1_buffer, 0, Some(r1_bytes));
            encoder.clear_buffer(&self.util_p2_buffer, 0, Some(r2_bytes));

            // Init root reach
            {
                let mut pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor { label: Some("tab_init"), ..Default::default() },
                );
                pass.set_pipeline(&self.init_pipeline);
                pass.set_bind_group(0, &self.bg0_dec[buf_idx], &[mrm_offset]);
                pass.set_bind_group(1, &self.bg1, &[]);
                pass.set_bind_group(2, &self.bg2, &[]);
                pass.set_bind_group(3, &self.bg3, &[]);
                pass.dispatch_workgroups(self.max_traj.div_ceil(WORKGROUP_SIZE), 1, 1);
            }

            // Forward pass
            for level in 0..self.num_levels {
                let count = self.decision_counts[level];
                if count == 0 { continue; }
                let offset = (base_slot + 1 + level) as u32 * stride;
                let threads = count * self.max_traj;
                let mut pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor { label: Some("tab_fwd"), ..Default::default() },
                );
                pass.set_pipeline(&self.forward_pipeline);
                pass.set_bind_group(0, &self.bg0_dec[buf_idx], &[offset]);
                pass.set_bind_group(1, &self.bg1, &[]);
                pass.set_bind_group(2, &self.bg2, &[]);
                pass.set_bind_group(3, &self.bg3, &[]);
                pass.dispatch_workgroups(threads.div_ceil(WORKGROUP_SIZE), 1, 1);
            }

            // Backward pass (reversed levels)
            for (j, level) in (0..self.num_levels).rev().enumerate() {
                let term_count = self.terminal_counts[level];
                if term_count > 0 {
                    let offset = (base_slot + 1 + self.num_levels + j) as u32 * stride;
                    let mut pass = encoder.begin_compute_pass(
                        &wgpu::ComputePassDescriptor { label: Some("tab_bwd_t"), ..Default::default() },
                    );
                    pass.set_pipeline(&self.backward_terminal_pipeline);
                    pass.set_bind_group(0, &self.bg0_term[buf_idx], &[offset]);
                    pass.set_bind_group(1, &self.bg1, &[]);
                    pass.set_bind_group(2, &self.bg2, &[]);
                    pass.set_bind_group(3, &self.bg3, &[]);
                    pass.dispatch_workgroups(
                        term_count,
                        self.max_traj.div_ceil(WORKGROUP_SIZE),
                        1,
                    );
                }

                let dec_count = self.decision_counts[level];
                if dec_count > 0 {
                    let offset = (base_slot + 1 + 2 * self.num_levels + j) as u32 * stride;
                    let threads = dec_count * self.max_traj;
                    let mut pass = encoder.begin_compute_pass(
                        &wgpu::ComputePassDescriptor { label: Some("tab_bwd_d"), ..Default::default() },
                    );
                    pass.set_pipeline(&self.backward_decision_pipeline);
                    pass.set_bind_group(0, &self.bg0_dec[buf_idx], &[offset]);
                    pass.set_bind_group(1, &self.bg1, &[]);
                    pass.set_bind_group(2, &self.bg2, &[]);
                    pass.set_bind_group(3, &self.bg3, &[]);
                    pass.dispatch_workgroups(threads.div_ceil(WORKGROUP_SIZE), 1, 1);
                }
            }
        }

        encoder.finish()
    }

    /// Rebuild both uniform buffers and their bind groups after a resize.
    fn rebuild_uniform_buffers(&mut self, needed_bytes: u64) {
        let group0_layout = create_tab_group0_layout(&self.device);
        let uniform_binding_size =
            std::num::NonZero::new(std::mem::size_of::<TabularUniforms>() as u64);

        for idx in 0..2 {
            self.uniform_buffers[idx] = create_buffer_zeroed(
                &self.device,
                if idx == 0 { "tab_uniforms_a" } else { "tab_uniforms_b" },
                needed_bytes,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );

            let make_bg = |level_buf: &wgpu::Buffer, label: &str| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &group0_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &self.uniform_buffers[idx],
                                offset: 0,
                                size: uniform_binding_size,
                            }),
                        },
                        entry(1, &self.node_buffer),
                        entry(2, &self.children_buffer),
                        entry(3, level_buf),
                        entry(4, &self.info_set_num_actions_buffer),
                    ],
                })
            };
            self.bg0_dec[idx] = make_bg(
                &self.decision_level_nodes_buffer,
                if idx == 0 { "tab_g0_dec_a" } else { "tab_g0_dec_b" },
            );
            self.bg0_term[idx] = make_bg(
                &self.terminal_level_nodes_buffer,
                if idx == 0 { "tab_g0_term_a" } else { "tab_g0_term_b" },
            );
        }
    }

    /// Merge pending deltas without running a full iteration.
    fn flush_deltas(&self, buf_idx: usize) {
        let strategy_discount = Self::compute_strategy_discount_for(self.iterations, self.config.dcfr_gamma);
        self.write_uniform_slot(buf_idx, 0, 0, 0, strategy_discount, self.iterations as u32);
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("flush_deltas") },
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("tab_flush"), ..Default::default() },
            );
            pass.set_pipeline(&self.merge_regret_match_pipeline);
            pass.set_bind_group(0, &self.bg0_dec[buf_idx], &[0]);
            pass.set_bind_group(1, &self.bg1, &[]);
            pass.set_bind_group(2, &self.bg2, &[]);
            pass.set_bind_group(3, &self.bg3, &[]);
            pass.dispatch_workgroups(self.num_info_sets.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Number of completed iterations.
    #[must_use]
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// Number of unique info sets.
    #[must_use]
    pub fn num_info_sets(&self) -> u32 {
        self.num_info_sets
    }

    /// Return average strategies for all info sets.
    #[must_use]
    pub fn all_strategies(&self) -> FxHashMap<u64, Vec<f64>> {
        let strat_sum = self.download_f32_buffer(
            &self.strategy_sum_buffer,
            self.num_info_sets as usize * self.max_actions as usize,
        );
        let a = self.max_actions as usize;
        let mut result = FxHashMap::default();

        for (dense_id, &key) in self.dense_to_key.iter().enumerate() {
            let base = dense_id * a;
            let sums = &strat_sum[base..base + a];
            let total: f32 = sums.iter().sum();
            if total > 0.0 {
                let probs: Vec<f64> = sums.iter().map(|&s| f64::from(s / total)).collect();
                result.insert(key, probs);
            }
        }
        result
    }

    /// Compute max positive regret on GPU (upper bound on exploitability).
    ///
    /// Dispatches a two-pass reduction shader and downloads a single f32.
    /// Much cheaper than downloading the entire regret buffer.
    #[must_use]
    pub fn max_regret(&self) -> f32 {
        let num_wgs = self.num_info_sets.div_ceil(WORKGROUP_SIZE);

        // Pass 1: per-workgroup max
        let pass1_uniforms = ConvergenceUniforms {
            num_info_sets: self.num_info_sets,
            max_actions: self.max_actions,
            pass_id: 0,
            num_workgroups: num_wgs,
        };
        self.queue.write_buffer(
            &self.convergence_uniform_buffer, 0,
            bytemuck::bytes_of(&pass1_uniforms),
        );

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("conv_pass1") },
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("conv_p1"), ..Default::default() },
            );
            pass.set_pipeline(&self.convergence_pipeline);
            pass.set_bind_group(0, &self.convergence_bg, &[]);
            pass.dispatch_workgroups(num_wgs, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        // Pass 2: reduce partial results
        let pass2_uniforms = ConvergenceUniforms {
            num_info_sets: self.num_info_sets,
            max_actions: self.max_actions,
            pass_id: 1,
            num_workgroups: num_wgs,
        };
        self.queue.write_buffer(
            &self.convergence_uniform_buffer, 0,
            bytemuck::bytes_of(&pass2_uniforms),
        );

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("conv_pass2") },
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("conv_p2"), ..Default::default() },
            );
            pass.set_pipeline(&self.convergence_pipeline);
            pass.set_bind_group(0, &self.convergence_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Copy result to staging and download
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.convergence_buffer, 0, &staging, 0, 4);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range();
        let result: f32 = *bytemuck::from_bytes(&data[..4]);
        drop(data);
        staging.unmap();
        result
    }

    // --- Internal methods ---

    fn write_uniform_slot(
        &self, buf_idx: usize, slot: u32, level_start: u32, level_count: u32,
        strategy_discount: f32, iteration: u32,
    ) {
        let uniforms = TabularUniforms {
            level_start,
            level_count,
            num_nodes: self.num_nodes,
            n_traj_p1: self.n_traj_p1,
            n_traj_p2: self.n_traj_p2,
            max_traj: self.max_traj,
            num_info_sets: self.num_info_sets,
            max_actions: self.max_actions,
            iteration,
            dcfr_alpha: self.config.dcfr_alpha as f32,
            dcfr_beta: self.config.dcfr_beta as f32,
            dcfr_gamma: self.config.dcfr_gamma as f32,
            strategy_discount,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let offset = u64::from(slot * self.uniform_stride);
        self.queue.write_buffer(&self.uniform_buffers[buf_idx], offset, bytemuck::bytes_of(&uniforms));
    }

    fn compute_strategy_discount_for(iteration: u64, dcfr_gamma: f64) -> f32 {
        let t = iteration as f64;
        let ratio = t / (t + 1.0);
        ratio.powf(dcfr_gamma) as f32
    }

    /// Download current regret values from GPU (for diagnostics/testing).
    #[must_use]
    pub fn download_regrets(&self) -> Vec<f32> {
        self.download_f32_buffer(
            &self.regret_buffer,
            self.num_info_sets as usize * self.max_actions as usize,
        )
    }

    /// Download current strategy sum values from GPU (for diagnostics/testing).
    #[must_use]
    pub fn download_strategy_sums(&self) -> Vec<f32> {
        self.download_f32_buffer(
            &self.strategy_sum_buffer,
            self.num_info_sets as usize * self.max_actions as usize,
        )
    }

    /// Max actions per info set.
    #[must_use]
    pub fn max_actions(&self) -> u32 {
        self.max_actions
    }

    fn download_f32_buffer(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let size = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tab_staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tab_download") },
        );
        enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }
}

// ---------------------------------------------------------------------------
// Bind group layouts for tabular solver
// ---------------------------------------------------------------------------

fn create_tab_group0_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    use crate::{layout_entry, storage_rw};
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tab_g0_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            },
            layout_entry(1, storage_rw()), // nodes
            layout_entry(2, storage_rw()), // children
            layout_entry(3, storage_rw()), // level_nodes (decision or terminal)
            layout_entry(4, storage_rw()), // info_set_num_actions
        ],
    })
}

fn create_tab_group1_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    use crate::{layout_entry, storage_rw};
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tab_g1_layout"),
        entries: &[
            layout_entry(0, storage_rw()), // regret
            layout_entry(1, storage_rw()), // strategy
            layout_entry(2, storage_rw()), // strategy_sum
            layout_entry(3, storage_rw()), // regret_delta (atomic)
            layout_entry(4, storage_rw()), // strat_sum_delta (atomic)
        ],
    })
}

fn create_tab_group2_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    use crate::{layout_entry, storage_rw};
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tab_g2_layout"),
        entries: &[
            layout_entry(0, storage_rw()), // reach_p1
            layout_entry(1, storage_rw()), // reach_p2
            layout_entry(2, storage_rw()), // util_p1
            layout_entry(3, storage_rw()), // util_p2
            layout_entry(4, storage_rw()), // info_id_table
            layout_entry(5, storage_rw()), // weight_sum
        ],
    })
}

fn create_tab_group3_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    use crate::{layout_entry, storage_rw};
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tab_g3_layout"),
        entries: &[
            layout_entry(0, storage_rw()), // w_mat
            layout_entry(1, storage_rw()), // we_mat
            layout_entry(2, storage_rw()), // w_neg_mat
            layout_entry(3, storage_rw()), // w_mat_t
            layout_entry(4, storage_rw()), // we_mat_t
            layout_entry(5, storage_rw()), // w_neg_mat_t
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::cfr::game_tree::materialize_postflop;
    use poker_solver_core::game::{AbstractionMode, HunlPostflop, PostflopConfig};
    use poker_solver_core::info_key::InfoKey;
    use poker_solver_core::Game;

    fn build_test_data() -> (GameTree, Vec<DealInfo>, Vec<DecisionNodeInfo>) {
        let config = PostflopConfig {
            stack_depth: 10,
            bet_sizes: vec![1.0],
            max_raises_per_street: 2,
        };
        let game = HunlPostflop::new(
            config,
            Some(AbstractionMode::HandClassV2 {
                strength_bits: 0,
                equity_bits: 0,
            }),
            42,
        );
        let states = game.initial_states();
        let tree = materialize_postflop(&game, &states[0]);

        let deals: Vec<DealInfo> = states
            .iter()
            .take(100)
            .map(|state| {
                let hand_bits_p1 = InfoKey::from_raw(game.info_set_key(state)).hand_bits();
                let first_action = game.actions(state)[0];
                let next_state = game.next_state(state, first_action);
                let hand_bits_p2 =
                    InfoKey::from_raw(game.info_set_key(&next_state)).hand_bits();
                let p1_equity = match (&state.p1_cache.rank, &state.p2_cache.rank) {
                    (Some(r1), Some(r2)) => match r1.cmp(r2) {
                        std::cmp::Ordering::Greater => 1.0,
                        std::cmp::Ordering::Less => 0.0,
                        std::cmp::Ordering::Equal => 0.5,
                    },
                    _ => 0.5,
                };
                DealInfo {
                    hand_bits_p1: [hand_bits_p1; 4],
                    hand_bits_p2: [hand_bits_p2; 4],
                    p1_equity,
                    weight: 1.0,
                }
            })
            .collect();

        let decision_nodes = crate::collect_decision_nodes(&tree);
        (tree, deals, decision_nodes)
    }

    #[test]
    fn trajectory_extraction_deduplicates() {
        let deals = vec![
            DealInfo {
                hand_bits_p1: [1, 2, 3, 4],
                hand_bits_p2: [5, 6, 7, 8],
                p1_equity: 0.6,
                weight: 1.0,
            },
            DealInfo {
                hand_bits_p1: [1, 2, 3, 4], // same as first
                hand_bits_p2: [9, 10, 11, 12],
                p1_equity: 0.4,
                weight: 2.0,
            },
            DealInfo {
                hand_bits_p1: [13, 14, 15, 16],
                hand_bits_p2: [5, 6, 7, 8], // same as first
                p1_equity: 0.5,
                weight: 1.0,
            },
        ];

        let (p1_trajs, p1_map) = extract_unique_trajectories_p1(&deals);
        let (p2_trajs, p2_map) = extract_unique_trajectories_p2(&deals);

        assert_eq!(p1_trajs.len(), 2, "P1 should have 2 unique trajectories");
        assert_eq!(p2_trajs.len(), 2, "P2 should have 2 unique trajectories");
        assert_eq!(p1_map[&[1, 2, 3, 4]], 0);
        assert_eq!(p1_map[&[13, 14, 15, 16]], 1);
        assert_eq!(p2_map[&[5, 6, 7, 8]], 0);
        assert_eq!(p2_map[&[9, 10, 11, 12]], 1);
    }

    #[test]
    fn coupling_matrices_accumulate_correctly() {
        let deals = vec![
            DealInfo {
                hand_bits_p1: [1, 0, 0, 0],
                hand_bits_p2: [2, 0, 0, 0],
                p1_equity: 0.8,
                weight: 3.0,
            },
            DealInfo {
                hand_bits_p1: [1, 0, 0, 0],
                hand_bits_p2: [2, 0, 0, 0],
                p1_equity: 0.6,
                weight: 2.0,
            },
        ];

        let (_, p1_map) = extract_unique_trajectories_p1(&deals);
        let (_, p2_map) = extract_unique_trajectories_p2(&deals);
        let (w, we, w_neg) = build_coupling_matrices(&deals, &p1_map, &p2_map, 1, 1);

        assert!((w[0] - 5.0).abs() < 1e-6, "W = 3.0 + 2.0 = 5.0");
        let expected_we = 3.0 * 0.8 + 2.0 * 0.6;
        assert!(
            (we[0] - expected_we as f32).abs() < 1e-5,
            "WE = 3*0.8 + 2*0.6 = {expected_we}"
        );
        let expected_wn = 3.0 * 0.2 + 2.0 * 0.4;
        assert!(
            (w_neg[0] - expected_wn as f32).abs() < 1e-5,
            "W_neg = 3*0.2 + 2*0.4 = {expected_wn}"
        );
    }

    #[test]
    fn weight_sums_match_matrices() {
        let deals = vec![
            DealInfo {
                hand_bits_p1: [1, 0, 0, 0],
                hand_bits_p2: [2, 0, 0, 0],
                p1_equity: 0.5,
                weight: 3.0,
            },
            DealInfo {
                hand_bits_p1: [1, 0, 0, 0],
                hand_bits_p2: [3, 0, 0, 0],
                p1_equity: 0.5,
                weight: 7.0,
            },
        ];

        let (_, p1_map) = extract_unique_trajectories_p1(&deals);
        let (_, p2_map) = extract_unique_trajectories_p2(&deals);
        let (w, _, _) = build_coupling_matrices(&deals, &p1_map, &p2_map, 1, 2);

        let ws_p1 = row_sums(&w, 1, 2);
        let ws_p2 = col_sums(&w, 1, 2);

        assert!((ws_p1[0] - 10.0).abs() < 1e-6, "row sum = 3+7 = 10");
        assert!((ws_p2[0] - 3.0).abs() < 1e-6, "col sum[0] = 3");
        assert!((ws_p2[1] - 7.0).abs() < 1e-6, "col sum[1] = 7");
    }

    #[test]
    fn transpose_correctness() {
        let mat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let t = transpose(&mat, 2, 3);
        // Expected 3×2: [1,4, 2,5, 3,6]
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn info_id_table_consistent_with_tree() {
        let (tree, deals, decision_nodes) = build_test_data();
        let data = precompute_tabular(&tree, &deals, &decision_nodes);

        assert!(data.num_info_sets > 0, "should have info sets");
        assert_eq!(
            data.info_set_num_actions.len(),
            data.num_info_sets as usize
        );
        assert_eq!(data.dense_to_key.len(), data.num_info_sets as usize);

        // Every decision node should have valid info IDs for its trajectories
        for (d_idx, dn) in decision_nodes.iter().enumerate() {
            let n_traj = match dn.player {
                Player::Player1 => data.n_traj_p1 as usize,
                Player::Player2 => data.n_traj_p2 as usize,
            };
            for t in 0..n_traj {
                let info_id = data.info_id_table[d_idx * data.max_traj as usize + t];
                assert_ne!(
                    info_id,
                    u32::MAX,
                    "decision node {d_idx} traj {t} should have valid info ID"
                );
                assert!(
                    info_id < data.num_info_sets,
                    "info ID {info_id} out of range (max {})",
                    data.num_info_sets
                );
            }
        }
    }

    #[test]
    fn precompute_tabular_produces_valid_data() {
        let (tree, deals, decision_nodes) = build_test_data();
        let data = precompute_tabular(&tree, &deals, &decision_nodes);

        assert!(data.n_traj_p1 > 0);
        assert!(data.n_traj_p2 > 0);
        assert_eq!(data.max_traj, data.n_traj_p1.max(data.n_traj_p2));

        let n1 = data.n_traj_p1 as usize;
        let n2 = data.n_traj_p2 as usize;
        assert_eq!(data.w_mat.len(), n1 * n2);
        assert_eq!(data.we_mat.len(), n1 * n2);
        assert_eq!(data.w_neg_mat.len(), n1 * n2);
        assert_eq!(data.w_mat_t.len(), n2 * n1);
        assert_eq!(data.weight_sum_p1.len(), n1);
        assert_eq!(data.weight_sum_p2.len(), n2);

        // W = WE + W_neg (since eq + (1-eq) = 1)
        for i in 0..n1 * n2 {
            let diff = (data.w_mat[i] - data.we_mat[i] - data.w_neg_mat[i]).abs();
            assert!(diff < 1e-4, "W != WE + W_neg at index {i}: diff={diff}");
        }
    }

    #[test]
    fn level_partition_covers_all_nodes() {
        let (tree, _, _) = build_test_data();
        let partition = partition_levels(&tree);

        let total_decision: u32 = partition.decision_counts.iter().sum();
        let total_terminal: u32 = partition.terminal_counts.iter().sum();

        let expected_decision = tree
            .nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Decision { .. }))
            .count() as u32;
        let expected_terminal = tree
            .nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Terminal { .. }))
            .count() as u32;

        assert_eq!(total_decision, expected_decision);
        assert_eq!(total_terminal, expected_terminal);
        assert_eq!(
            partition.decision_counts.len(),
            partition.terminal_counts.len()
        );
        assert_eq!(partition.decision_counts.len(), tree.levels.len());
    }

    // -----------------------------------------------------------------------
    // GPU integration tests
    // -----------------------------------------------------------------------

    fn build_gpu_test_tree_and_deals() -> (GameTree, Vec<DealInfo>) {
        let config = PostflopConfig {
            stack_depth: 10,
            bet_sizes: vec![1.0],
            max_raises_per_street: 2,
        };
        let game = HunlPostflop::new(
            config,
            Some(AbstractionMode::HandClassV2 {
                strength_bits: 0,
                equity_bits: 0,
            }),
            42,
        );
        let states = game.initial_states();
        let tree = materialize_postflop(&game, &states[0]);

        let deals: Vec<DealInfo> = states
            .iter()
            .map(|state| {
                let hand_bits_p1 = InfoKey::from_raw(game.info_set_key(state)).hand_bits();
                let first_action = game.actions(state)[0];
                let next_state = game.next_state(state, first_action);
                let hand_bits_p2 =
                    InfoKey::from_raw(game.info_set_key(&next_state)).hand_bits();
                let p1_equity = match (&state.p1_cache.rank, &state.p2_cache.rank) {
                    (Some(r1), Some(r2)) => match r1.cmp(r2) {
                        std::cmp::Ordering::Greater => 1.0,
                        std::cmp::Ordering::Less => 0.0,
                        std::cmp::Ordering::Equal => 0.5,
                    },
                    _ => 0.5,
                };
                DealInfo {
                    hand_bits_p1: [hand_bits_p1; 4],
                    hand_bits_p2: [hand_bits_p2; 4],
                    p1_equity,
                    weight: 1.0,
                }
            })
            .collect();

        (tree, deals)
    }

    #[test]
    fn tabular_gpu_constructs() {
        let (tree, deals) = build_gpu_test_tree_and_deals();
        let config = GpuCfrConfig::default();

        let solver = TabularGpuCfrSolver::new(&tree, deals, config);
        assert!(solver.is_ok(), "TabularGpuCfrSolver should construct: {:?}", solver.err());
        let solver = solver.unwrap();

        assert!(solver.num_info_sets() > 0, "should have info sets");
        assert_eq!(solver.iterations(), 0, "no iterations yet");
    }

    #[test]
    fn tabular_gpu_runs_one_iteration() {
        let (tree, deals) = build_gpu_test_tree_and_deals();
        let config = GpuCfrConfig::default();
        let mut solver = TabularGpuCfrSolver::new(&tree, deals, config).unwrap();

        solver.train(1);
        assert_eq!(solver.iterations(), 1);

        // After 1 iteration, regrets should be non-zero
        let regrets = solver.download_regrets();
        let has_nonzero = regrets.iter().any(|&r| r.abs() > 1e-10);
        assert!(has_nonzero, "regrets should be non-zero after 1 iteration");
    }

    #[test]
    fn tabular_gpu_strategies_sum_to_one() {
        let (tree, deals) = build_gpu_test_tree_and_deals();
        let config = GpuCfrConfig::default();
        let mut solver = TabularGpuCfrSolver::new(&tree, deals, config).unwrap();

        solver.train(20);
        let strategies = solver.all_strategies();

        assert!(!strategies.is_empty(), "should have strategies after training");

        for (key, probs) in &strategies {
            let total: f64 = probs.iter().sum();
            assert!(
                (total - 1.0).abs() < 0.01,
                "strategy for info set {key:#x} sums to {total}, expected ~1.0"
            );
            for (i, &p) in probs.iter().enumerate() {
                assert!(
                    p >= -0.001,
                    "negative probability {p} at action {i} for info set {key:#x}"
                );
            }
        }
    }

    #[test]
    fn tabular_gpu_strategy_delta_decreases() {
        let (tree, deals) = build_gpu_test_tree_and_deals();
        let config = GpuCfrConfig::default();
        let mut solver = TabularGpuCfrSolver::new(&tree, deals, config).unwrap();

        solver.train(50);
        let strat1 = solver.all_strategies();

        solver.train(100);
        let strat2 = solver.all_strategies();

        solver.train(200);
        let strat3 = solver.all_strategies();

        fn strategy_delta(
            a: &rustc_hash::FxHashMap<u64, Vec<f64>>,
            b: &rustc_hash::FxHashMap<u64, Vec<f64>>,
        ) -> f64 {
            let mut total = 0.0;
            let mut count = 0;
            for (key, pa) in a {
                if let Some(pb) = b.get(key) {
                    let diff: f64 = pa.iter().zip(pb.iter()).map(|(x, y)| (x - y).abs()).sum();
                    total += diff;
                    count += 1;
                }
            }
            if count > 0 { total / count as f64 } else { 0.0 }
        }

        let delta_1_2 = strategy_delta(&strat1, &strat2);
        let delta_2_3 = strategy_delta(&strat2, &strat3);

        println!("  strategy delta: {delta_1_2:.6} -> {delta_2_3:.6}");
        assert!(
            delta_2_3 < delta_1_2,
            "strategy delta should decrease: {delta_1_2:.6} -> {delta_2_3:.6}"
        );
    }

    #[test]
    fn tabular_gpu_regrets_accumulated_after_training() {
        let (tree, deals) = build_gpu_test_tree_and_deals();
        let config = GpuCfrConfig::default();
        let mut solver = TabularGpuCfrSolver::new(&tree, deals, config).unwrap();

        solver.train(10);

        let regrets = solver.download_regrets();
        let max_actions = solver.max_actions() as usize;
        let num_info = solver.num_info_sets() as usize;

        // Check that regrets have sensible structure
        let mut positive_count = 0;
        let mut negative_count = 0;
        for i in 0..num_info {
            for a in 0..max_actions {
                let r = regrets[i * max_actions + a];
                if r > 1e-10 {
                    positive_count += 1;
                } else if r < -1e-10 {
                    negative_count += 1;
                }
            }
        }

        assert!(
            positive_count > 0,
            "should have some positive regrets after 10 iterations"
        );
        assert!(
            negative_count > 0,
            "should have some negative regrets after 10 iterations"
        );
        // No NaN or Inf
        for &r in &regrets {
            assert!(r.is_finite(), "regret should be finite, got {r}");
        }
    }

    #[test]
    fn tabular_gpu_strategy_sums_increase_over_iterations() {
        let (tree, deals) = build_gpu_test_tree_and_deals();
        let config = GpuCfrConfig::default();
        let mut solver = TabularGpuCfrSolver::new(&tree, deals, config).unwrap();

        solver.train(10);
        let sums_10: f32 = solver.download_strategy_sums().iter().sum();

        solver.train(40);
        let sums_50: f32 = solver.download_strategy_sums().iter().sum();

        assert!(
            sums_50 > sums_10,
            "strategy sums should grow: {sums_10} -> {sums_50}"
        );
    }

    #[test]
    fn tabular_gpu_max_regret_matches_download() {
        let (tree, deals) = build_gpu_test_tree_and_deals();
        let config = GpuCfrConfig::default();
        let mut solver = TabularGpuCfrSolver::new(&tree, deals, config).unwrap();

        solver.train(20);

        let gpu_max = solver.max_regret();
        let regrets = solver.download_regrets();
        let cpu_max = regrets.iter().copied().fold(0.0f32, f32::max);

        let diff = (gpu_max - cpu_max).abs();
        assert!(
            diff < 1e-4,
            "GPU max_regret ({gpu_max}) should match CPU max ({cpu_max}), diff={diff}"
        );
    }

    #[test]
    fn tabular_gpu_max_regret_decreases() {
        let (tree, deals) = build_gpu_test_tree_and_deals();
        let config = GpuCfrConfig::default();
        let mut solver = TabularGpuCfrSolver::new(&tree, deals, config).unwrap();

        solver.train(50);
        let mr1 = solver.max_regret();

        solver.train(200);
        let mr2 = solver.max_regret();

        println!("  max_regret: {mr1:.4} -> {mr2:.4}");
        assert!(
            mr2 < mr1,
            "max_regret should decrease: {mr1:.4} -> {mr2:.4}"
        );
    }
}
