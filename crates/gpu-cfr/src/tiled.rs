//! Tiled tabular GPU CFR solver.
//!
//! Tiles the trajectory dimension so coupling matrices, reach vectors, and
//! utility buffers fit in GPU/CPU memory. Only `backward_terminal` couples
//! P1 and P2 trajectories; forward and backward_decision process tiles
//! independently. Reach is recomputed per-tile (cheap forward pass) to
//! avoid storing full reach arrays on CPU.
//!
//! The `tile_size` parameter controls GPU memory usage and can be tuned:
//! larger tiles = fewer dispatches but more VRAM.

use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;

use poker_solver_core::cfr::game_tree::GameTree;
use poker_solver_core::cfr::DealInfo;

use crate::tabular::{
    build_info_id_table, extract_unique_trajectories_p1, extract_unique_trajectories_p2,
    partition_levels,
};
use crate::{
    align_up, collect_decision_nodes, create_buffer_init, create_buffer_zeroed, create_pipeline,
    encode_tree, entry, layout_entry, storage_rw, DecisionNodeInfo, GpuCfrConfig, GpuError,
    WORKGROUP_SIZE,
};

// ---------------------------------------------------------------------------
// Compact deal entries & coupling tile builder
// ---------------------------------------------------------------------------

/// Compact representation of a deal for tiled coupling tile construction.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CompactDealEntry {
    pub t1: u32,
    pub t2: u32,
    pub weight: f32,
    pub equity: f32,
}

/// Builds coupling matrix tiles on-demand from bucket-sorted deal entries.
///
/// Stores all deals in compact form sorted by `(p1_tile, p2_tile)` bucket,
/// with a prefix-sum index for O(1) bucket lookup. The `build_tile` method
/// materializes a `tile_size × tile_size` coupling tile for a given
/// (own_tile, opp_tile) pair and player direction.
pub struct CouplingTileBuilder {
    entries: Vec<CompactDealEntry>,
    /// Prefix-sum offsets: `bucket_offsets[bucket]..bucket_offsets[bucket+1]`
    /// gives the range of entries in that bucket.
    bucket_offsets: Vec<usize>,
    pub n1: u32,
    pub n2: u32,
    pub tile_size: u32,
    pub num_p1_tiles: u32,
    pub num_p2_tiles: u32,
}

impl CouplingTileBuilder {
    /// Create a new tile builder from deals and trajectory maps.
    ///
    /// Converts deals to compact entries, bucket-sorts by tile pair,
    /// and builds a prefix-sum index.
    pub fn new(
        deals: &[DealInfo],
        traj_p1_map: &FxHashMap<[u32; 4], u32>,
        traj_p2_map: &FxHashMap<[u32; 4], u32>,
        n1: u32,
        n2: u32,
        tile_size: u32,
    ) -> Self {
        let num_p1_tiles = n1.div_ceil(tile_size);
        let num_p2_tiles = n2.div_ceil(tile_size);
        let num_buckets = (num_p1_tiles * num_p2_tiles) as usize;

        // Convert deals to compact entries
        let mut entries: Vec<CompactDealEntry> = deals
            .iter()
            .map(|d| CompactDealEntry {
                t1: traj_p1_map[&d.hand_bits_p1],
                t2: traj_p2_map[&d.hand_bits_p2],
                weight: d.weight as f32,
                equity: d.p1_equity as f32,
            })
            .collect();

        // Counting sort by bucket
        let mut counts = vec![0usize; num_buckets];
        for e in &entries {
            let bucket = (e.t1 / tile_size) * num_p2_tiles + (e.t2 / tile_size);
            counts[bucket as usize] += 1;
        }

        let mut bucket_offsets = vec![0usize; num_buckets + 1];
        for i in 0..num_buckets {
            bucket_offsets[i + 1] = bucket_offsets[i] + counts[i];
        }

        // Place entries in sorted order
        let mut write_pos = bucket_offsets[..num_buckets].to_vec();
        let mut sorted = vec![
            CompactDealEntry {
                t1: 0,
                t2: 0,
                weight: 0.0,
                equity: 0.0,
            };
            entries.len()
        ];
        for e in &entries {
            let bucket = ((e.t1 / tile_size) * num_p2_tiles + (e.t2 / tile_size)) as usize;
            sorted[write_pos[bucket]] = *e;
            write_pos[bucket] += 1;
        }

        entries = sorted;

        Self {
            entries,
            bucket_offsets,
            n1,
            n2,
            tile_size,
            num_p1_tiles,
            num_p2_tiles,
        }
    }

    /// Build a coupling tile for the given (own_tile, opp_tile) pair.
    ///
    /// Returns three flat arrays `(w, we, w_neg)` of size `tile_size × tile_size`.
    /// When `for_p1` is true, own=P1 and opp=P2 (direct slice).
    /// When `for_p1` is false, own=P2 and opp=P1 (transposed slice).
    pub fn build_tile(&self, own_tile: u32, opp_tile: u32, for_p1: bool) -> CouplingTile {
        let (p1_tile, p2_tile) = if for_p1 {
            (own_tile, opp_tile)
        } else {
            (opp_tile, own_tile)
        };

        let bucket = (p1_tile * self.num_p2_tiles + p2_tile) as usize;
        let start = self.bucket_offsets[bucket];
        let end = self.bucket_offsets[bucket + 1];

        let ts = self.tile_size as usize;
        let tile_elems = ts * ts;
        let mut w = vec![0.0f32; tile_elems];
        let mut we = vec![0.0f32; tile_elems];
        let mut w_neg = vec![0.0f32; tile_elems];

        let p1_base = p1_tile * self.tile_size;
        let p2_base = p2_tile * self.tile_size;

        for entry in &self.entries[start..end] {
            let local_p1 = (entry.t1 - p1_base) as usize;
            let local_p2 = (entry.t2 - p2_base) as usize;

            // Index as [local_own][local_opp]
            let idx = if for_p1 {
                local_p1 * ts + local_p2
            } else {
                local_p2 * ts + local_p1
            };

            w[idx] += entry.weight;
            we[idx] += entry.weight * entry.equity;
            w_neg[idx] += entry.weight * (1.0 - entry.equity);
        }

        CouplingTile { w, we, w_neg }
    }

    /// Total number of compact deal entries stored.
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }
}

/// A materialized coupling tile (three flat arrays).
pub struct CouplingTile {
    pub w: Vec<f32>,
    pub we: Vec<f32>,
    pub w_neg: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Tiled precompute
// ---------------------------------------------------------------------------

/// Precomputed data for the tiled tabular solver.
pub struct TiledPrecomputeData {
    pub n1: u32,
    pub n2: u32,
    pub max_traj: u32,
    pub tile_builder: CouplingTileBuilder,
    pub info_id_table: Vec<u32>,
    pub info_set_num_actions: Vec<u32>,
    pub dense_to_key: Vec<u64>,
    pub num_info_sets: u32,
    pub weight_sum_p1: Vec<f32>,
    pub weight_sum_p2: Vec<f32>,
}

/// Precompute tiled tabular data without building full coupling matrices.
///
/// Extracts trajectories, builds the `CouplingTileBuilder` from compact deal
/// entries, computes weight sums from the tile builder, and builds the info
/// ID table.
pub fn precompute_tiled(
    tree: &GameTree,
    deals: &[DealInfo],
    decision_nodes: &[DecisionNodeInfo],
    tile_size: u32,
) -> TiledPrecomputeData {
    let step = Instant::now();
    let (traj_p1, traj_p1_map) = extract_unique_trajectories_p1(deals);
    let (traj_p2, traj_p2_map) = extract_unique_trajectories_p2(deals);
    let n1 = traj_p1.len() as u32;
    let n2 = traj_p2.len() as u32;
    let max_traj = n1.max(n2);
    println!(
        "  Trajectories: n1={n1}, n2={n2}, extracted in {:?}",
        step.elapsed()
    );

    let num_p1_tiles = n1.div_ceil(tile_size);
    let num_p2_tiles = n2.div_ceil(tile_size);
    println!(
        "  Tiling: tile_size={tile_size}, p1_tiles={num_p1_tiles}, p2_tiles={num_p2_tiles}"
    );

    // Build coupling tile builder (bucket-sorted compact entries)
    let step = Instant::now();
    let tile_builder =
        CouplingTileBuilder::new(deals, &traj_p1_map, &traj_p2_map, n1, n2, tile_size);
    println!(
        "  CouplingTileBuilder: {} entries, {:.1} MB, built in {:?}",
        tile_builder.num_entries(),
        tile_builder.num_entries() as f64 * 16.0 / 1_048_576.0,
        step.elapsed()
    );

    // Compute weight sums by iterating over compact entries
    let step = Instant::now();
    let mut ws_p1 = vec![0.0f32; n1 as usize];
    let mut ws_p2 = vec![0.0f32; n2 as usize];
    for e in &tile_builder.entries {
        ws_p1[e.t1 as usize] += e.weight;
        ws_p2[e.t2 as usize] += e.weight;
    }
    println!("  Weight sums computed in {:?}", step.elapsed());

    // Build info ID table
    let step = Instant::now();
    let (info_id_table, info_set_num_actions, dense_to_key, num_info_sets) =
        build_info_id_table(tree, decision_nodes, &traj_p1, &traj_p2, max_traj);
    println!(
        "  Info ID table: {} info sets, {:.1} GB, built in {:?}",
        num_info_sets,
        info_id_table.len() as f64 * 4.0 / 1e9,
        step.elapsed()
    );

    TiledPrecomputeData {
        n1,
        n2,
        max_traj,
        tile_builder,
        info_id_table,
        info_set_num_actions,
        dense_to_key,
        num_info_sets,
        weight_sum_p1: ws_p1,
        weight_sum_p2: ws_p2,
    }
}

// ---------------------------------------------------------------------------
// Tiled uniforms
// ---------------------------------------------------------------------------

/// Uniform parameters for tiled tabular CFR shaders. 80 bytes, 16-aligned.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TiledTabularUniforms {
    level_start: u32,
    level_count: u32,
    num_nodes: u32,
    n_traj_own: u32,
    n_traj_opp: u32,
    max_traj: u32,
    num_info_sets: u32,
    max_actions: u32,
    iteration: u32,
    dcfr_alpha: f32,
    dcfr_beta: f32,
    dcfr_gamma: f32,
    strategy_discount: f32,
    player: u32,
    tile_offset: u32,
    tile_size: u32,
    opp_tile_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ---------------------------------------------------------------------------
// Convergence uniforms (reused from tabular)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ConvergenceUniforms {
    num_info_sets: u32,
    max_actions: u32,
    pass_id: u32,
    num_workgroups: u32,
}

// ---------------------------------------------------------------------------
// Tiled tabular GPU CFR solver
// ---------------------------------------------------------------------------

/// Tiled tabular GPU CFR solver.
///
/// Tiles the trajectory dimension so that coupling matrices, reach vectors,
/// and utility buffers fit within GPU memory. Processes one player direction
/// at a time, with reach recomputation per tile to avoid CPU-side reach storage.
#[allow(dead_code)]
pub struct TiledTabularGpuCfrSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Tree metadata
    num_nodes: u32,
    max_actions: u32,
    num_info_sets: u32,
    n1: u32,
    n2: u32,
    max_traj: u32,
    tile_size: u32,
    num_p1_tiles: u32,
    num_p2_tiles: u32,

    // CPU-side level partition
    decision_offsets: Vec<u32>,
    decision_counts: Vec<u32>,
    terminal_offsets: Vec<u32>,
    terminal_counts: Vec<u32>,
    num_levels: usize,

    // CPU-side coupling tile builder
    tile_builder: CouplingTileBuilder,

    // Info set key mapping
    dense_to_key: Vec<u64>,

    // GPU buffers -- Group 0: tree + uniforms
    node_buffer: wgpu::Buffer,
    children_buffer: wgpu::Buffer,
    decision_level_nodes_buffer: wgpu::Buffer,
    terminal_level_nodes_buffer: wgpu::Buffer,
    info_set_num_actions_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_stride: u32,

    // GPU buffers -- Group 1: CFR state (full-sized)
    regret_buffer: wgpu::Buffer,
    strategy_buffer: wgpu::Buffer,
    strategy_sum_buffer: wgpu::Buffer,
    regret_delta_buffer: wgpu::Buffer,
    strat_sum_delta_buffer: wgpu::Buffer,

    // GPU buffers -- Group 2: tile-sized reach/util + full info
    /// Tile-sized buffer: used for reach_own or reach_opp depending on phase.
    reach_buffer: wgpu::Buffer,
    /// Tile-sized buffer: used for util_own.
    util_buffer: wgpu::Buffer,
    /// Small dummy buffer for unused bind group slots.
    dummy_buffer: wgpu::Buffer,
    /// Full-sized info ID table.
    info_id_table_buffer: wgpu::Buffer,
    /// Combined weight sums [p1..., p2...].
    weight_sum_buffer: wgpu::Buffer,

    // GPU buffers -- Group 3: tile-sized coupling matrices
    w_tile_buffer: wgpu::Buffer,
    we_tile_buffer: wgpu::Buffer,
    wn_tile_buffer: wgpu::Buffer,

    // Bind group layouts
    group0_layout: wgpu::BindGroupLayout,
    group1_layout: wgpu::BindGroupLayout,
    group2_layout: wgpu::BindGroupLayout,
    group3_layout: wgpu::BindGroupLayout,

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

impl TiledTabularGpuCfrSolver {
    /// Create a new tiled tabular GPU CFR solver.
    pub fn new(
        tree: &GameTree,
        deals: Vec<DealInfo>,
        config: GpuCfrConfig,
    ) -> Result<Self, GpuError> {
        if deals.is_empty() {
            return Err(GpuError::NoDeals);
        }

        let tile_size = config.tile_size.unwrap_or(32768);

        let step = Instant::now();
        let (device, queue) = crate::init_gpu_large_buffers()?;
        println!("  GPU device acquired in {:?}", step.elapsed());

        // Encode tree
        let (gpu_nodes, children_flat, _lnf, _lo, _lc) = encode_tree(tree);
        let num_nodes = tree.nodes.len() as u32;

        // Partition levels
        let partition = partition_levels(tree);
        let num_levels = partition.decision_counts.len();

        // Precompute tiled data
        let decision_nodes = collect_decision_nodes(tree);
        let step = Instant::now();
        let tab = precompute_tiled(tree, &deals, &decision_nodes, tile_size);
        println!(
            "  Tiled precomputation complete in {:?}",
            step.elapsed()
        );

        let max_actions = tab.info_set_num_actions.iter().copied().max().unwrap_or(1);
        let num_info_sets = tab.num_info_sets;
        let n1 = tab.n1;
        let n2 = tab.n2;
        let max_traj = tab.max_traj;
        let num_p1_tiles = n1.div_ceil(tile_size);
        let num_p2_tiles = n2.div_ceil(tile_size);

        // Buffer sizes
        let state_size = u64::from(num_info_sets) * u64::from(max_actions) * 4;
        let tile_reach_size = u64::from(num_nodes) * u64::from(tile_size) * 4;
        let coupling_tile_size = u64::from(tile_size) * u64::from(tile_size) * 4;
        let info_id_size = tab.info_id_table.len() as u64 * 4;

        println!(
            "  Buffers: state={:.1}MB, tile_reach={:.1}GB, coupling_tile={:.1}GB (x3), info_id={:.1}GB",
            state_size as f64 / 1_048_576.0,
            tile_reach_size as f64 / 1e9,
            coupling_tile_size as f64 / 1e9,
            info_id_size as f64 / 1e9,
        );

        // Create GPU buffers
        let step = Instant::now();

        // Group 0: tree
        let node_buffer = create_buffer_init(
            &device, "tiled_nodes", bytemuck::cast_slice(&gpu_nodes),
            wgpu::BufferUsages::STORAGE,
        );
        let children_buffer = create_buffer_init(
            &device, "tiled_children", bytemuck::cast_slice(&children_flat),
            wgpu::BufferUsages::STORAGE,
        );
        let decision_level_nodes_buffer = create_buffer_init(
            &device, "tiled_dec_level", bytemuck::cast_slice(&partition.decision_flat),
            wgpu::BufferUsages::STORAGE,
        );
        let terminal_level_nodes_buffer = create_buffer_init(
            &device, "tiled_term_level", bytemuck::cast_slice(&partition.terminal_flat),
            wgpu::BufferUsages::STORAGE,
        );
        let info_set_num_actions_buffer = create_buffer_init(
            &device, "tiled_is_nactions", bytemuck::cast_slice(&tab.info_set_num_actions),
            wgpu::BufferUsages::STORAGE,
        );

        // Group 1: CFR state
        let regret_buffer = create_buffer_zeroed(
            &device, "tiled_regret", state_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let strategy_buffer = create_buffer_zeroed(
            &device, "tiled_strategy", state_size,
            wgpu::BufferUsages::STORAGE,
        );
        let strategy_sum_buffer = create_buffer_zeroed(
            &device, "tiled_strat_sum", state_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let regret_delta_buffer = create_buffer_zeroed(
            &device, "tiled_regret_d", state_size,
            wgpu::BufferUsages::STORAGE,
        );
        let strat_sum_delta_buffer = create_buffer_zeroed(
            &device, "tiled_ss_delta", state_size,
            wgpu::BufferUsages::STORAGE,
        );

        // Group 2: tile-sized reach/util + full info
        let reach_buffer = create_buffer_zeroed(
            &device, "tiled_reach", tile_reach_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let util_buffer = create_buffer_zeroed(
            &device, "tiled_util", tile_reach_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let dummy_buffer = create_buffer_zeroed(
            &device, "tiled_dummy", 4,
            wgpu::BufferUsages::STORAGE,
        );
        let info_id_table_buffer = create_buffer_init(
            &device, "tiled_info_id", bytemuck::cast_slice(&tab.info_id_table),
            wgpu::BufferUsages::STORAGE,
        );
        let mut weight_sum_combined = tab.weight_sum_p1;
        weight_sum_combined.extend_from_slice(&tab.weight_sum_p2);
        let weight_sum_buffer = create_buffer_init(
            &device, "tiled_weight_sum", bytemuck::cast_slice(&weight_sum_combined),
            wgpu::BufferUsages::STORAGE,
        );

        // Group 3: tile-sized coupling matrices
        let w_tile_buffer = create_buffer_zeroed(
            &device, "tiled_w", coupling_tile_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let we_tile_buffer = create_buffer_zeroed(
            &device, "tiled_we", coupling_tile_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let wn_tile_buffer = create_buffer_zeroed(
            &device, "tiled_wn", coupling_tile_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        // Uniform buffer
        let max_uniform_slots = (1 + 3 * num_levels) as u64;
        let min_align = device.limits().min_uniform_buffer_offset_alignment;
        let uniform_stride = align_up(
            std::mem::size_of::<TiledTabularUniforms>() as u32,
            min_align,
        );
        let uniform_buf_size = max_uniform_slots * u64::from(uniform_stride);
        let uniform_buffer = create_buffer_zeroed(
            &device, "tiled_uniforms", uniform_buf_size,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        println!("  GPU buffers allocated in {:?}", step.elapsed());

        // Create bind group layouts and pipelines
        let step = Instant::now();
        let group0_layout = create_tiled_group0_layout(&device);
        let group1_layout = create_tiled_group1_layout(&device);
        let group2_layout = create_tiled_group2_layout(&device);
        let group3_layout = create_tiled_group3_layout(&device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tiled_pipeline_layout"),
            bind_group_layouts: &[&group0_layout, &group1_layout, &group2_layout, &group3_layout],
            immediate_size: 0,
        });

        let init_pipeline = create_pipeline(
            &device, &pipeline_layout, "tiled_init",
            include_str!("shaders/tiled_init.wgsl"),
        );
        let merge_regret_match_pipeline = create_pipeline(
            &device, &pipeline_layout, "tiled_mrm",
            include_str!("shaders/tabular_merge_regret_match.wgsl"),
        );
        let forward_pipeline = create_pipeline(
            &device, &pipeline_layout, "tiled_fwd",
            include_str!("shaders/tiled_forward.wgsl"),
        );
        let backward_terminal_pipeline = create_pipeline(
            &device, &pipeline_layout, "tiled_bwd_t",
            include_str!("shaders/tiled_backward_terminal.wgsl"),
        );
        let backward_decision_pipeline = create_pipeline(
            &device, &pipeline_layout, "tiled_bwd_d",
            include_str!("shaders/tiled_backward_decision.wgsl"),
        );

        // Convergence pipeline
        let conv_wgs = num_info_sets.div_ceil(WORKGROUP_SIZE);
        let conv_scratch_size = u64::from(conv_wgs.max(1)) * 4;
        let convergence_buffer = create_buffer_zeroed(
            &device, "tiled_conv_scratch", conv_scratch_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let conv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tiled_conv_layout"),
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
                layout_entry(1, storage_rw()),
                layout_entry(2, storage_rw()),
            ],
        });
        let conv_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tiled_conv_pl"),
            bind_group_layouts: &[&conv_layout],
            immediate_size: 0,
        });
        let convergence_pipeline = create_pipeline(
            &device, &conv_pipeline_layout, "tiled_conv",
            include_str!("shaders/tabular_convergence.wgsl"),
        );
        let convergence_uniform_buffer = create_buffer_zeroed(
            &device, "tiled_conv_uni",
            std::mem::size_of::<ConvergenceUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let convergence_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tiled_conv_bg"),
            layout: &conv_layout,
            entries: &[
                entry(0, &convergence_uniform_buffer),
                entry(1, &regret_buffer),
                entry(2, &convergence_buffer),
            ],
        });

        println!("  Shader pipelines compiled in {:?}", step.elapsed());

        Ok(Self {
            device,
            queue,
            num_nodes,
            max_actions,
            num_info_sets,
            n1,
            n2,
            max_traj,
            tile_size,
            num_p1_tiles,
            num_p2_tiles,
            decision_offsets: partition.decision_offsets,
            decision_counts: partition.decision_counts,
            terminal_offsets: partition.terminal_offsets,
            terminal_counts: partition.terminal_counts,
            num_levels,
            tile_builder: tab.tile_builder,
            dense_to_key: tab.dense_to_key,
            node_buffer,
            children_buffer,
            decision_level_nodes_buffer,
            terminal_level_nodes_buffer,
            info_set_num_actions_buffer,
            uniform_buffer,
            uniform_stride,
            regret_buffer,
            strategy_buffer,
            strategy_sum_buffer,
            regret_delta_buffer,
            strat_sum_delta_buffer,
            reach_buffer,
            util_buffer,
            dummy_buffer,
            info_id_table_buffer,
            weight_sum_buffer,
            w_tile_buffer,
            we_tile_buffer,
            wn_tile_buffer,
            group0_layout,
            group1_layout,
            group2_layout,
            group3_layout,
            init_pipeline,
            merge_regret_match_pipeline,
            forward_pipeline,
            backward_terminal_pipeline,
            backward_decision_pipeline,
            convergence_pipeline,
            convergence_buffer,
            convergence_uniform_buffer,
            convergence_bg,
            config,
            iterations: 0,
        })
    }

    /// Run `num_iterations` of training with a per-iteration callback.
    pub fn train_with_callback<F: FnMut(u64)>(&mut self, num_iterations: u64, mut cb: F) {
        for _ in 0..num_iterations {
            self.run_iteration();
            cb(self.iterations);
        }
    }

    /// Run `num_iterations` of training.
    pub fn train(&mut self, num_iterations: u64) {
        for _ in 0..num_iterations {
            self.run_iteration();
        }
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

    /// Max actions per info set.
    #[must_use]
    pub fn max_actions(&self) -> u32 {
        self.max_actions
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

    /// GPU-computed max positive regret (upper bound on exploitability).
    #[must_use]
    pub fn max_regret(&self) -> f32 {
        let num_wgs = self.num_info_sets.div_ceil(WORKGROUP_SIZE);

        // Pass 1
        let pass1 = ConvergenceUniforms {
            num_info_sets: self.num_info_sets,
            max_actions: self.max_actions,
            pass_id: 0,
            num_workgroups: num_wgs,
        };
        self.queue.write_buffer(
            &self.convergence_uniform_buffer, 0,
            bytemuck::bytes_of(&pass1),
        );
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("conv_p1") },
        );
        {
            let mut pass = enc.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("conv_p1"), ..Default::default() },
            );
            pass.set_pipeline(&self.convergence_pipeline);
            pass.set_bind_group(0, &self.convergence_bg, &[]);
            pass.dispatch_workgroups(num_wgs, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        // Pass 2
        let pass2 = ConvergenceUniforms {
            num_info_sets: self.num_info_sets,
            max_actions: self.max_actions,
            pass_id: 1,
            num_workgroups: num_wgs,
        };
        self.queue.write_buffer(
            &self.convergence_uniform_buffer, 0,
            bytemuck::bytes_of(&pass2),
        );
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("conv_p2") },
        );
        {
            let mut pass = enc.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("conv_p2"), ..Default::default() },
            );
            pass.set_pipeline(&self.convergence_pipeline);
            pass.set_bind_group(0, &self.convergence_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&self.convergence_buffer, 0, &staging, 0, 4);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range();
        let result: f32 = *bytemuck::from_bytes(&data[..4]);
        drop(data);
        staging.unmap();
        result
    }

    // -----------------------------------------------------------------------
    // Iteration logic
    // -----------------------------------------------------------------------

    fn run_iteration(&mut self) {
        let iter_start = Instant::now();
        let strategy_discount = self.compute_strategy_discount();

        // Step 1: Merge deltas + regret match
        self.dispatch_merge_regret_match(strategy_discount);

        // Step 2-3: Process P1 direction then P2 direction
        self.run_player_direction(0, strategy_discount); // P1
        self.run_player_direction(1, strategy_discount); // P2

        self.iterations += 1;
        if self.iterations <= 3 || self.iterations % 10 == 0 {
            println!(
                "  tiled iter {} complete in {:.1}s",
                self.iterations,
                iter_start.elapsed().as_secs_f64(),
            );
            flush_stdout();
        }
    }

    /// Process one player direction: backward_terminal (doubly tiled) + backward_decision.
    fn run_player_direction(&mut self, player: u32, strategy_discount: f32) {
        let (n_own, n_opp) = if player == 0 {
            (self.n1, self.n2)
        } else {
            (self.n2, self.n1)
        };
        let num_own_tiles = n_own.div_ceil(self.tile_size);
        let num_opp_tiles = n_opp.div_ceil(self.tile_size);
        let tile_reach_elems = self.num_nodes as usize * self.tile_size as usize;

        // Allocate CPU-side utility accumulator for this player direction
        let cpu_util_size = self.num_nodes as usize * n_own as usize;
        let mut cpu_util = vec![0.0f32; cpu_util_size];

        // Phase 2: Backward terminal (doubly tiled)
        for own_tile in 0..num_own_tiles {
            let own_offset = own_tile * self.tile_size;
            let own_actual = (n_own - own_offset).min(self.tile_size);

            // Clear util buffer on GPU
            let util_bytes = tile_reach_elems as u64 * 4;
            let mut enc = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("clear_util") },
            );
            enc.clear_buffer(&self.util_buffer, 0, Some(util_bytes));
            self.queue.submit(std::iter::once(enc.finish()));

            for opp_tile in 0..num_opp_tiles {
                let opp_offset = opp_tile * self.tile_size;
                let opp_actual = (n_opp - opp_offset).min(self.tile_size);

                // Recompute opponent reach via forward pass
                self.run_forward_pass(1 - player, opp_offset, opp_actual, n_opp, n_own);

                // Build coupling tile on CPU
                let for_p1 = player == 0;
                let coupling = self.tile_builder.build_tile(own_tile, opp_tile, for_p1);

                // Upload coupling tile to GPU
                self.queue.write_buffer(
                    &self.w_tile_buffer, 0,
                    bytemuck::cast_slice(&coupling.w),
                );
                self.queue.write_buffer(
                    &self.we_tile_buffer, 0,
                    bytemuck::cast_slice(&coupling.we),
                );
                self.queue.write_buffer(
                    &self.wn_tile_buffer, 0,
                    bytemuck::cast_slice(&coupling.w_neg),
                );

                // Dispatch backward_terminal for all terminal levels
                self.dispatch_backward_terminal(
                    player, own_actual, opp_actual, n_own, n_opp, strategy_discount,
                );
            }

            // Download util_own from GPU to cpu_util
            self.download_util_to_cpu(
                &mut cpu_util,
                own_offset as usize,
                own_actual as usize,
                n_own as usize,
            );
        }

        // Phase 3: Backward decision (per tile)
        for own_tile in 0..num_own_tiles {
            let own_offset = own_tile * self.tile_size;
            let own_actual = (n_own - own_offset).min(self.tile_size);

            // Recompute own reach via forward pass
            self.run_forward_pass(player, own_offset, own_actual, n_own, n_opp);

            // Upload util_own from cpu_util to GPU
            self.upload_util_from_cpu(
                &cpu_util,
                own_offset as usize,
                own_actual as usize,
                n_own as usize,
            );

            // Dispatch backward_decision for all levels (reversed)
            self.dispatch_backward_decision(
                player, own_offset, own_actual, n_own, n_opp, strategy_discount,
            );

            // Download updated util_own (parents) back to CPU
            self.download_util_to_cpu(
                &mut cpu_util,
                own_offset as usize,
                own_actual as usize,
                n_own as usize,
            );
        }

        // cpu_util is freed when it goes out of scope
    }

    /// Run forward pass for one player's tile: clear reach buffer, init root, propagate.
    fn run_forward_pass(
        &self,
        player: u32,
        tile_offset: u32,
        tile_actual: u32,
        n_own: u32,
        n_opp: u32,
    ) {
        let reach_bytes = self.num_nodes as u64 * u64::from(self.tile_size) * 4;

        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tiled_fwd") },
        );

        // Clear reach buffer
        enc.clear_buffer(&self.reach_buffer, 0, Some(reach_bytes));

        // Create bind groups
        let bg0 = self.create_bg0_decision(0);
        let bg1 = self.create_bg1();
        let bg2 = self.create_bg2();
        let bg3 = self.create_bg3();

        // Init root reach
        let init_uniforms = self.make_uniforms(
            0, 0, player, tile_offset, tile_actual, 0, n_own, n_opp, 0.0,
        );
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&init_uniforms));

        {
            let mut pass = enc.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("tiled_init"), ..Default::default() },
            );
            pass.set_pipeline(&self.init_pipeline);
            pass.set_bind_group(0, &bg0, &[0]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.set_bind_group(3, &bg3, &[]);
            pass.dispatch_workgroups(tile_actual.div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        // Forward levels
        let stride = self.uniform_stride;
        for level in 0..self.num_levels {
            let count = self.decision_counts[level];
            if count == 0 {
                continue;
            }
            let slot = (1 + level) as u32;
            let fwd_uniforms = self.make_uniforms(
                self.decision_offsets[level], count,
                player, tile_offset, tile_actual, 0,
                n_own, n_opp, 0.0,
            );
            self.queue.write_buffer(
                &self.uniform_buffer,
                u64::from(slot * stride),
                bytemuck::bytes_of(&fwd_uniforms),
            );
            let threads = count * tile_actual;
            let offset = slot * stride;
            let mut pass = enc.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("tiled_fwd_l"), ..Default::default() },
            );
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &bg0, &[offset]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.set_bind_group(3, &bg3, &[]);
            pass.dispatch_workgroups(threads.div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        self.queue.submit(std::iter::once(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Dispatch backward_terminal for all terminal levels.
    fn dispatch_backward_terminal(
        &self,
        player: u32,
        own_tile_size: u32,
        opp_tile_size: u32,
        n_own: u32,
        n_opp: u32,
        strategy_discount: f32,
    ) {
        let bg0 = self.create_bg0_terminal(0);
        let bg1 = self.create_bg1();
        let bg2 = self.create_bg2();
        let bg3 = self.create_bg3();

        let stride = self.uniform_stride;
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tiled_bwd_t") },
        );

        for (j, level) in (0..self.num_levels).rev().enumerate() {
            let term_count = self.terminal_counts[level];
            if term_count == 0 {
                continue;
            }

            let slot = j as u32;
            let uniforms = self.make_uniforms(
                self.terminal_offsets[level], term_count,
                player, 0, own_tile_size, opp_tile_size,
                n_own, n_opp, strategy_discount,
            );
            self.queue.write_buffer(
                &self.uniform_buffer,
                u64::from(slot * stride),
                bytemuck::bytes_of(&uniforms),
            );

            let mut pass = enc.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("tiled_bt"), ..Default::default() },
            );
            pass.set_pipeline(&self.backward_terminal_pipeline);
            pass.set_bind_group(0, &bg0, &[slot * stride]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.set_bind_group(3, &bg3, &[]);
            pass.dispatch_workgroups(
                term_count,
                own_tile_size.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        self.queue.submit(std::iter::once(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Dispatch backward_decision for all levels (reversed).
    fn dispatch_backward_decision(
        &self,
        player: u32,
        tile_offset: u32,
        tile_actual: u32,
        n_own: u32,
        n_opp: u32,
        strategy_discount: f32,
    ) {
        let bg0 = self.create_bg0_decision(0);
        let bg1 = self.create_bg1();
        let bg2 = self.create_bg2();
        let bg3 = self.create_bg3();

        let stride = self.uniform_stride;
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tiled_bwd_d") },
        );

        for (j, level) in (0..self.num_levels).rev().enumerate() {
            let dec_count = self.decision_counts[level];
            if dec_count == 0 {
                continue;
            }

            let slot = j as u32;
            let uniforms = self.make_uniforms(
                self.decision_offsets[level], dec_count,
                player, tile_offset, tile_actual, 0,
                n_own, n_opp, strategy_discount,
            );
            self.queue.write_buffer(
                &self.uniform_buffer,
                u64::from(slot * stride),
                bytemuck::bytes_of(&uniforms),
            );

            let threads = dec_count * tile_actual;
            let mut pass = enc.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("tiled_bd"), ..Default::default() },
            );
            pass.set_pipeline(&self.backward_decision_pipeline);
            pass.set_bind_group(0, &bg0, &[slot * stride]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.set_bind_group(3, &bg3, &[]);
            pass.dispatch_workgroups(threads.div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        self.queue.submit(std::iter::once(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Dispatch fused merge + regret match (unchanged from untiled solver).
    fn dispatch_merge_regret_match(&self, strategy_discount: f32) {
        let uniforms = self.make_uniforms(
            0, 0, 0, 0, 0, 0, self.n1, self.n2, strategy_discount,
        );
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let bg0 = self.create_bg0_decision(0);
        let bg1 = self.create_bg1();
        let bg2 = self.create_bg2();
        let bg3 = self.create_bg3();

        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tiled_mrm") },
        );
        {
            let mut pass = enc.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("tiled_mrm"), ..Default::default() },
            );
            pass.set_pipeline(&self.merge_regret_match_pipeline);
            pass.set_bind_group(0, &bg0, &[0]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.set_bind_group(3, &bg3, &[]);
            pass.dispatch_workgroups(self.num_info_sets.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    // -----------------------------------------------------------------------
    // CPU-GPU data transfer
    // -----------------------------------------------------------------------

    /// Download tile-sized util buffer from GPU into the CPU utility array.
    fn download_util_to_cpu(
        &self,
        cpu_util: &mut [f32],
        own_offset: usize,
        own_actual: usize,
        n_own: usize,
    ) {
        let tile_sz = self.tile_size as usize;
        let nn = self.num_nodes as usize;
        let gpu_elems = nn * tile_sz;
        let gpu_data = self.download_f32_buffer(&self.util_buffer, gpu_elems);

        // Copy from [node * tile_size + lt] to cpu_util[node * n_own + (own_offset + lt)]
        for node in 0..nn {
            let src_base = node * tile_sz;
            let dst_base = node * n_own + own_offset;
            cpu_util[dst_base..dst_base + own_actual]
                .copy_from_slice(&gpu_data[src_base..src_base + own_actual]);
        }
    }

    /// Upload CPU utility data into the tile-sized GPU util buffer.
    fn upload_util_from_cpu(
        &self,
        cpu_util: &[f32],
        own_offset: usize,
        own_actual: usize,
        n_own: usize,
    ) {
        let tile_sz = self.tile_size as usize;
        let nn = self.num_nodes as usize;
        let mut gpu_data = vec![0.0f32; nn * tile_sz];

        // Copy from cpu_util[node * n_own + (own_offset + lt)] to [node * tile_size + lt]
        for node in 0..nn {
            let src_base = node * n_own + own_offset;
            let dst_base = node * tile_sz;
            gpu_data[dst_base..dst_base + own_actual]
                .copy_from_slice(&cpu_util[src_base..src_base + own_actual]);
        }

        self.queue.write_buffer(
            &self.util_buffer, 0,
            bytemuck::cast_slice(&gpu_data),
        );
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_uniforms(
        &self,
        level_start: u32,
        level_count: u32,
        player: u32,
        tile_offset: u32,
        tile_size: u32,
        opp_tile_size: u32,
        n_own: u32,
        n_opp: u32,
        strategy_discount: f32,
    ) -> TiledTabularUniforms {
        TiledTabularUniforms {
            level_start,
            level_count,
            num_nodes: self.num_nodes,
            n_traj_own: n_own,
            n_traj_opp: n_opp,
            max_traj: self.max_traj,
            num_info_sets: self.num_info_sets,
            max_actions: self.max_actions,
            iteration: self.iterations as u32,
            dcfr_alpha: self.config.dcfr_alpha as f32,
            dcfr_beta: self.config.dcfr_beta as f32,
            dcfr_gamma: self.config.dcfr_gamma as f32,
            strategy_discount,
            player,
            tile_offset,
            tile_size,
            opp_tile_size,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    fn compute_strategy_discount(&self) -> f32 {
        let t = self.iterations as f64;
        let ratio = t / (t + 1.0);
        ratio.powf(self.config.dcfr_gamma) as f32
    }

    fn create_bg0_decision(&self, _buf_idx: usize) -> wgpu::BindGroup {
        let uniform_binding_size =
            std::num::NonZero::new(std::mem::size_of::<TiledTabularUniforms>() as u64);
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tiled_g0_dec"),
            layout: &self.group0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: uniform_binding_size,
                    }),
                },
                entry(1, &self.node_buffer),
                entry(2, &self.children_buffer),
                entry(3, &self.decision_level_nodes_buffer),
                entry(4, &self.info_set_num_actions_buffer),
            ],
        })
    }

    fn create_bg0_terminal(&self, _buf_idx: usize) -> wgpu::BindGroup {
        let uniform_binding_size =
            std::num::NonZero::new(std::mem::size_of::<TiledTabularUniforms>() as u64);
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tiled_g0_term"),
            layout: &self.group0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: uniform_binding_size,
                    }),
                },
                entry(1, &self.node_buffer),
                entry(2, &self.children_buffer),
                entry(3, &self.terminal_level_nodes_buffer),
                entry(4, &self.info_set_num_actions_buffer),
            ],
        })
    }

    fn create_bg1(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tiled_g1"),
            layout: &self.group1_layout,
            entries: &[
                entry(0, &self.regret_buffer),
                entry(1, &self.strategy_buffer),
                entry(2, &self.strategy_sum_buffer),
                entry(3, &self.regret_delta_buffer),
                entry(4, &self.strat_sum_delta_buffer),
            ],
        })
    }

    fn create_bg2(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tiled_g2"),
            layout: &self.group2_layout,
            entries: &[
                entry(0, &self.reach_buffer),
                entry(1, &self.dummy_buffer),
                entry(2, &self.util_buffer),
                entry(3, &self.dummy_buffer),
                entry(4, &self.info_id_table_buffer),
                entry(5, &self.weight_sum_buffer),
            ],
        })
    }

    fn create_bg3(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tiled_g3"),
            layout: &self.group3_layout,
            entries: &[
                entry(0, &self.w_tile_buffer),
                entry(1, &self.we_tile_buffer),
                entry(2, &self.wn_tile_buffer),
            ],
        })
    }

    fn download_f32_buffer(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let size = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tiled_download") },
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

fn flush_stdout() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

// ---------------------------------------------------------------------------
// Bind group layouts for tiled solver
// ---------------------------------------------------------------------------

fn create_tiled_group0_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tiled_g0_layout"),
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
            layout_entry(3, storage_rw()), // level_nodes
            layout_entry(4, storage_rw()), // info_set_num_actions
        ],
    })
}

fn create_tiled_group1_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tiled_g1_layout"),
        entries: &[
            layout_entry(0, storage_rw()), // regret
            layout_entry(1, storage_rw()), // strategy
            layout_entry(2, storage_rw()), // strategy_sum
            layout_entry(3, storage_rw()), // regret_delta
            layout_entry(4, storage_rw()), // strat_sum_delta
        ],
    })
}

fn create_tiled_group2_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tiled_g2_layout"),
        entries: &[
            layout_entry(0, storage_rw()), // reach_own (or reach_opp)
            layout_entry(1, storage_rw()), // dummy
            layout_entry(2, storage_rw()), // util_own
            layout_entry(3, storage_rw()), // dummy
            layout_entry(4, storage_rw()), // info_id_table
            layout_entry(5, storage_rw()), // weight_sum
        ],
    })
}

fn create_tiled_group3_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tiled_g3_layout"),
        entries: &[
            layout_entry(0, storage_rw()), // w_tile
            layout_entry(1, storage_rw()), // we_tile
            layout_entry(2, storage_rw()), // w_neg_tile
        ],
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tabular::{
        build_coupling_matrices, extract_unique_trajectories_p1, extract_unique_trajectories_p2,
    };
    use poker_solver_core::cfr::DealInfo;

    #[test]
    fn coupling_tile_builder_sums_match_full_matrix() {
        let deals = vec![
            DealInfo {
                hand_bits_p1: [1, 0, 0, 0],
                hand_bits_p2: [10, 0, 0, 0],
                p1_equity: 0.8,
                weight: 3.0,
            },
            DealInfo {
                hand_bits_p1: [1, 0, 0, 0],
                hand_bits_p2: [20, 0, 0, 0],
                p1_equity: 0.6,
                weight: 2.0,
            },
            DealInfo {
                hand_bits_p1: [2, 0, 0, 0],
                hand_bits_p2: [10, 0, 0, 0],
                p1_equity: 0.5,
                weight: 1.0,
            },
            DealInfo {
                hand_bits_p1: [2, 0, 0, 0],
                hand_bits_p2: [20, 0, 0, 0],
                p1_equity: 0.4,
                weight: 4.0,
            },
        ];

        let (_, p1_map) = extract_unique_trajectories_p1(&deals);
        let (_, p2_map) = extract_unique_trajectories_p2(&deals);
        let n1 = p1_map.len();
        let n2 = p2_map.len();

        // Build full matrices
        let (w_full, we_full, wn_full) =
            build_coupling_matrices(&deals, &p1_map, &p2_map, n1, n2);

        // Build tiled (tile_size=2 covers everything in one tile)
        let builder =
            CouplingTileBuilder::new(&deals, &p1_map, &p2_map, n1 as u32, n2 as u32, 2);
        let tile = builder.build_tile(0, 0, true);

        // Compare
        for i in 0..n1 * n2 {
            assert!(
                (tile.w[i] - w_full[i]).abs() < 1e-6,
                "w mismatch at {i}: {} vs {}",
                tile.w[i],
                w_full[i]
            );
            assert!(
                (tile.we[i] - we_full[i]).abs() < 1e-6,
                "we mismatch at {i}"
            );
            assert!(
                (tile.w_neg[i] - wn_full[i]).abs() < 1e-6,
                "w_neg mismatch at {i}"
            );
        }
    }

    #[test]
    fn coupling_tile_builder_multi_tile() {
        let deals = vec![
            DealInfo {
                hand_bits_p1: [1, 0, 0, 0],
                hand_bits_p2: [10, 0, 0, 0],
                p1_equity: 0.7,
                weight: 5.0,
            },
            DealInfo {
                hand_bits_p1: [2, 0, 0, 0],
                hand_bits_p2: [20, 0, 0, 0],
                p1_equity: 0.3,
                weight: 3.0,
            },
        ];

        let (_, p1_map) = extract_unique_trajectories_p1(&deals);
        let (_, p2_map) = extract_unique_trajectories_p2(&deals);

        // tile_size=1 -> 2 tiles per player, 4 tile pairs
        let builder =
            CouplingTileBuilder::new(&deals, &p1_map, &p2_map, 2, 2, 1);
        assert_eq!(builder.num_p1_tiles, 2);
        assert_eq!(builder.num_p2_tiles, 2);

        // Tile (0,0): t1=0, t2=0 -> first deal
        let t00 = builder.build_tile(0, 0, true);
        assert!((t00.w[0] - 5.0).abs() < 1e-6);

        // Tile (1,1): t1=1, t2=1 -> second deal
        let t11 = builder.build_tile(1, 1, true);
        assert!((t11.w[0] - 3.0).abs() < 1e-6);

        // Tile (0,1): no deals
        let t01 = builder.build_tile(0, 1, true);
        assert!((t01.w[0]).abs() < 1e-6);
    }

    #[test]
    fn coupling_tile_transpose_for_p2() {
        let deals = vec![
            DealInfo {
                hand_bits_p1: [1, 0, 0, 0],
                hand_bits_p2: [10, 0, 0, 0],
                p1_equity: 0.8,
                weight: 3.0,
            },
            DealInfo {
                hand_bits_p1: [2, 0, 0, 0],
                hand_bits_p2: [10, 0, 0, 0],
                p1_equity: 0.5,
                weight: 1.0,
            },
        ];

        let (_, p1_map) = extract_unique_trajectories_p1(&deals);
        let (_, p2_map) = extract_unique_trajectories_p2(&deals);

        let builder =
            CouplingTileBuilder::new(&deals, &p1_map, &p2_map, 2, 1, 2);

        // P1 direction: tile is 2x1 [own=p1, opp=p2]
        let p1_tile = builder.build_tile(0, 0, true);
        // t1=0,t2=0 -> w=3.0; t1=1,t2=0 -> w=1.0
        assert!((p1_tile.w[0] - 3.0).abs() < 1e-6); // [0][0]
        assert!((p1_tile.w[2] - 1.0).abs() < 1e-6); // [1][0] at index 1*2+0=2

        // P2 direction: tile is 1x2 [own=p2, opp=p1]
        let p2_tile = builder.build_tile(0, 0, false);
        // own=t2=0, opp=t1=0 -> w=3.0; own=t2=0, opp=t1=1 -> w=1.0
        assert!((p2_tile.w[0] - 3.0).abs() < 1e-6); // [0][0]
        assert!((p2_tile.w[1] - 1.0).abs() < 1e-6); // [0][1]
    }
}
