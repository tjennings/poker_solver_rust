//! GPU-accelerated CFR solver using wgpu compute shaders.
//!
//! Implements sequence-form CFR with level-by-level GPU dispatch.
//! Each iteration processes batches of deals in parallel on the GPU,
//! with atomic accumulation of regrets across deals.
//!
//! # Architecture
//!
//! ```text
//! GameTree (CPU)  --->  GpuNode[] + children[] + levels[] (GPU static)
//! DealInfo[] (CPU) -->  hand_bits + p1_wins + info_id_lookup[] (GPU per-batch)
//!
//! Per iteration:
//!   regret_match  -->  forward_pass (×L levels)  -->  backward_pass (×L levels)
//!       -->  merge_deltas  -->  dcfr_discount
//! ```

use bytemuck::{Pod, Zeroable};
use pollster::FutureExt as _;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use poker_solver_core::cfr::game_tree::{GameTree, NodeType};
use poker_solver_core::cfr::DealInfo;
use poker_solver_core::game::Player;

const WORKGROUP_SIZE: u32 = 256;

// --- GPU data structures ---

/// GPU-friendly node representation. 32 bytes, aligned to 4.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuNode {
    /// 0 = P1 decision, 1 = P2 decision,
    /// 2 = fold (P1 folded), 3 = fold (P2 folded), 4 = showdown.
    node_type: u32,
    /// Dense position index for info set id. `u32::MAX` for terminals.
    position_index: u32,
    /// Index of first child in `children_flat`.
    first_child: u32,
    /// Number of children (actions).
    num_children: u32,
    /// P1 investment in BB at this terminal. 0 for decisions.
    p1_invested_bb: f32,
    /// P2 investment in BB at this terminal. 0 for decisions.
    p2_invested_bb: f32,
    _pad0: u32,
    _pad1: u32,
}

/// Uniform parameters passed to every shader dispatch.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    level_start: u32,
    level_count: u32,
    num_deals: u32,
    num_nodes: u32,
    num_info_sets: u32,
    max_actions: u32,
    num_hand_classes: u32,
    iteration: u32,
    dcfr_alpha: f32,
    dcfr_beta: f32,
    dcfr_gamma: f32,
    strategy_discount: f32,
}

// --- Public types ---

/// Configuration for GPU CFR training.
#[derive(Debug, Clone)]
pub struct GpuCfrConfig {
    /// DCFR positive regret exponent (default 1.5).
    pub dcfr_alpha: f64,
    /// DCFR negative regret exponent (default 0.5).
    pub dcfr_beta: f64,
    /// DCFR strategy sum exponent (default 2.0).
    pub dcfr_gamma: f64,
    /// Maximum deals per GPU batch. `None` (default) auto-sizes to the
    /// largest batch that fits within GPU buffer binding limits.
    pub max_batch_size: Option<usize>,
}

impl Default for GpuCfrConfig {
    fn default() -> Self {
        Self {
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
            max_batch_size: None,
        }
    }
}

/// Errors from GPU solver initialization or execution.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter found")]
    NoAdapter,
    #[error("failed to request GPU device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
    #[error("no deals provided")]
    NoDeals,
    #[error("GPU buffer map failed")]
    BufferMap,
}

/// GPU-accelerated sequence-form CFR solver.
pub struct GpuCfrSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Tree metadata
    num_nodes: u32,
    max_actions: u32,
    num_info_sets: u32,
    #[allow(dead_code)]
    num_positions: u32,
    num_hand_classes: u32,

    // CPU-side level structure (for dispatch control)
    level_offsets: Vec<u32>,
    level_counts: Vec<u32>,

    // Info set mapping
    #[allow(dead_code)]
    key_to_dense: FxHashMap<u64, u32>,
    dense_to_key: Vec<u64>,

    // Pre-computed per-(deal, node) info set ids — flat layout, stride = num_nodes
    deal_info_ids: Vec<u32>,

    // GPU buffers — static tree
    node_buffer: wgpu::Buffer,
    children_buffer: wgpu::Buffer,
    level_nodes_buffer: wgpu::Buffer,
    position_actions_buffer: wgpu::Buffer,

    // GPU buffers — CFR state (persistent)
    regret_buffer: wgpu::Buffer,
    strategy_buffer: wgpu::Buffer,
    strategy_sum_buffer: wgpu::Buffer,
    regret_delta_buffer: wgpu::Buffer,
    strat_sum_delta_buffer: wgpu::Buffer,

    // GPU buffers — per-batch (resized per batch)
    reach_p1_buffer: wgpu::Buffer,
    reach_p2_buffer: wgpu::Buffer,
    utility_buffer: wgpu::Buffer,
    deal_hand_p1_buffer: wgpu::Buffer,
    deal_hand_p2_buffer: wgpu::Buffer,
    deal_p1_wins_buffer: wgpu::Buffer,
    deal_weight_buffer: wgpu::Buffer,
    info_id_lookup_buffer: wgpu::Buffer,

    // Uniform buffer (array of Uniforms, indexed by dynamic offset)
    uniform_buffer: wgpu::Buffer,
    /// Aligned stride for each Uniforms entry in the dynamic uniform buffer.
    uniform_stride: u32,

    // Bind group layouts
    group0_layout: wgpu::BindGroupLayout,
    group1_layout: wgpu::BindGroupLayout,
    group2_layout: wgpu::BindGroupLayout,

    // Compute pipelines
    regret_match_pipeline: wgpu::ComputePipeline,
    forward_pass_pipeline: wgpu::ComputePipeline,
    backward_pass_pipeline: wgpu::ComputePipeline,
    merge_deltas_pipeline: wgpu::ComputePipeline,
    dcfr_discount_pipeline: wgpu::ComputePipeline,

    // Training state
    config: GpuCfrConfig,
    deals: Vec<DealInfo>,
    batch_size: usize,
    iterations: u64,
}

impl GpuCfrSolver {
    /// Create a new GPU CFR solver.
    ///
    /// Materializes the tree on GPU, pre-computes info set mappings,
    /// and initializes all buffers and compute pipelines.
    pub fn new(
        tree: &GameTree,
        deals: Vec<DealInfo>,
        config: GpuCfrConfig,
    ) -> Result<Self, GpuError> {
        if deals.is_empty() {
            return Err(GpuError::NoDeals);
        }

        let (device, queue) = init_gpu()?;

        // Encode tree for GPU
        let (gpu_nodes, children_flat, level_nodes_flat, level_offsets, level_counts) =
            encode_tree(tree);
        let num_nodes = tree.nodes.len() as u32;

        // Build position index mapping and find max_actions
        let (_position_indices, num_positions, position_num_actions) =
            build_position_indices(tree);

        // Pre-collect decision node indices (avoids repeated NodeType matching)
        let decision_nodes = collect_decision_nodes(tree);

        // Determine hand classes and build info set mapping
        let (key_to_dense, dense_to_key, num_hand_classes) =
            build_info_set_mapping(tree, &deals, &decision_nodes);
        let num_info_sets = dense_to_key.len() as u32;

        let max_actions = position_num_actions.iter().copied().max().unwrap_or(1);

        // Pre-compute per-deal info set id lookups (parallelized, flat layout)
        let deal_info_ids = precompute_deal_info_ids(
            tree, &deals, &decision_nodes, &key_to_dense,
        );

        // Auto-size batch to fit within GPU buffer binding limits.
        // The largest per-batch buffer is info_id_lookup: batch_size × num_nodes × 4 bytes.
        let max_binding = device.limits().max_storage_buffer_binding_size as u64;
        let max_batch_for_gpu = (max_binding / (num_nodes as u64 * 4)) as usize;
        let batch_size = config.max_batch_size
            .unwrap_or(max_batch_for_gpu)
            .min(deals.len())
            .min(max_batch_for_gpu);

        let state_size = (num_info_sets as u64) * (max_actions as u64) * 4;
        let batch_node_size = (batch_size as u64) * (num_nodes as u64) * 4;

        // Create GPU buffers
        let node_buffer = create_buffer_init(&device, "nodes", bytemuck::cast_slice(&gpu_nodes),
            wgpu::BufferUsages::STORAGE);
        let children_buffer = create_buffer_init(&device, "children",
            bytemuck::cast_slice(&children_flat), wgpu::BufferUsages::STORAGE);
        let level_nodes_buffer = create_buffer_init(&device, "level_nodes",
            bytemuck::cast_slice(&level_nodes_flat), wgpu::BufferUsages::STORAGE);
        let position_actions_buffer = create_buffer_init(&device, "position_actions",
            bytemuck::cast_slice(&position_num_actions), wgpu::BufferUsages::STORAGE);

        let regret_buffer = create_buffer_zeroed(&device, "regret",
            state_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let strategy_buffer = create_buffer_zeroed(&device, "strategy",
            state_size, wgpu::BufferUsages::STORAGE);
        let strategy_sum_buffer = create_buffer_zeroed(&device, "strategy_sum",
            state_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let regret_delta_buffer = create_buffer_zeroed(&device, "regret_delta",
            state_size, wgpu::BufferUsages::STORAGE);
        let strat_sum_delta_buffer = create_buffer_zeroed(&device, "strat_sum_delta",
            state_size, wgpu::BufferUsages::STORAGE);

        let reach_p1_buffer = create_buffer_zeroed(&device, "reach_p1",
            batch_node_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let reach_p2_buffer = create_buffer_zeroed(&device, "reach_p2",
            batch_node_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let utility_buffer = create_buffer_zeroed(&device, "utility",
            batch_node_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);

        let batch_deal_size = (batch_size as u64) * 4;
        let batch_lookup_size = (batch_size as u64) * (num_nodes as u64) * 4;
        let deal_hand_p1_buffer = create_buffer_zeroed(&device, "deal_hand_p1",
            batch_deal_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let deal_hand_p2_buffer = create_buffer_zeroed(&device, "deal_hand_p2",
            batch_deal_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let deal_p1_wins_buffer = create_buffer_zeroed(&device, "deal_p1_equity",
            batch_deal_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let deal_weight_buffer = create_buffer_zeroed(&device, "deal_weight",
            batch_deal_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let info_id_lookup_buffer = create_buffer_zeroed(&device, "info_id_lookup",
            batch_lookup_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);

        // Uniform buffer: array of Uniforms with dynamic offsets.
        // Each dispatch indexes a different slot. We need enough slots for:
        //   1 (regret_match) + L (forward) + L (backward) + 1 (merge/discount)
        // where L = number of tree levels.
        let num_levels = level_counts.len();
        let max_uniform_slots = (1 + num_levels + num_levels + 1) as u64;
        let min_align = device.limits().min_uniform_buffer_offset_alignment;
        let uniform_stride = align_up(std::mem::size_of::<Uniforms>() as u32, min_align);
        let uniform_buffer = create_buffer_zeroed(&device, "uniforms",
            max_uniform_slots * u64::from(uniform_stride),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);

        // Create bind group layouts
        let group0_layout = create_group0_layout(&device);
        let group1_layout = create_group1_layout(&device);
        let group2_layout = create_group2_layout(&device);

        // Create compute pipelines
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cfr_pipeline_layout"),
            bind_group_layouts: &[&group0_layout, &group1_layout, &group2_layout],
            immediate_size: 0,
        });

        let regret_match_pipeline = create_pipeline(&device, &pipeline_layout,
            "regret_match", include_str!("shaders/regret_match.wgsl"));
        let forward_pass_pipeline = create_pipeline(&device, &pipeline_layout,
            "forward_pass", include_str!("shaders/forward_pass.wgsl"));
        let backward_pass_pipeline = create_pipeline(&device, &pipeline_layout,
            "backward_pass", include_str!("shaders/backward_pass.wgsl"));
        let merge_deltas_pipeline = create_pipeline(&device, &pipeline_layout,
            "merge_deltas", include_str!("shaders/merge_deltas.wgsl"));
        let dcfr_discount_pipeline = create_pipeline(&device, &pipeline_layout,
            "dcfr_discount", include_str!("shaders/dcfr_discount.wgsl"));

        Ok(Self {
            device,
            queue,
            num_nodes,
            max_actions,
            num_info_sets,
            num_positions,
            num_hand_classes,
            level_offsets,
            level_counts,
            key_to_dense,
            dense_to_key,
            deal_info_ids,
            node_buffer,
            children_buffer,
            level_nodes_buffer,
            position_actions_buffer,
            regret_buffer,
            strategy_buffer,
            strategy_sum_buffer,
            regret_delta_buffer,
            strat_sum_delta_buffer,
            reach_p1_buffer,
            reach_p2_buffer,
            utility_buffer,
            deal_hand_p1_buffer,
            deal_hand_p2_buffer,
            deal_p1_wins_buffer,
            deal_weight_buffer,
            info_id_lookup_buffer,
            uniform_buffer,
            uniform_stride,
            group0_layout,
            group1_layout,
            group2_layout,
            regret_match_pipeline,
            forward_pass_pipeline,
            backward_pass_pipeline,
            merge_deltas_pipeline,
            dcfr_discount_pipeline,
            config,
            deals,
            batch_size,
            iterations: 0,
        })
    }

    /// Run `num_iterations` of CFR training on GPU.
    pub fn train(&mut self, num_iterations: u64) {
        for _ in 0..num_iterations {
            self.run_iteration();
        }
    }

    /// Run `num_iterations` with a per-iteration callback.
    pub fn train_with_callback<F: FnMut(u64)>(&mut self, num_iterations: u64, mut cb: F) {
        for _ in 0..num_iterations {
            self.run_iteration();
            cb(self.iterations);
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

    /// Deals per GPU batch (may be smaller than requested if GPU limits require it).
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Return average strategies for all info sets.
    ///
    /// Downloads `strategy_sum` from GPU, normalizes, and maps back to
    /// `u64` info set keys.
    #[must_use]
    pub fn all_strategies(&self) -> FxHashMap<u64, Vec<f64>> {
        let strategy_sum = self.download_f32_buffer(&self.strategy_sum_buffer,
            self.num_info_sets as usize * self.max_actions as usize);

        let mut result = FxHashMap::default();
        let a = self.max_actions as usize;

        for (dense_id, &key) in self.dense_to_key.iter().enumerate() {
            let base = dense_id * a;
            let sums = &strategy_sum[base..base + a];
            let total: f32 = sums.iter().sum();

            if total > 0.0 {
                let probs: Vec<f64> = sums.iter().map(|&s| f64::from(s / total)).collect();
                result.insert(key, probs);
            }
        }

        result
    }

    // --- Internal methods ---

    fn run_iteration(&mut self) {
        let num_deals = self.deals.len();
        let strategy_discount = self.compute_strategy_discount();

        // Process deals in batches
        let mut deal_offset = 0;
        while deal_offset < num_deals {
            let batch_end = (deal_offset + self.batch_size).min(num_deals);
            let batch_count = batch_end - deal_offset;

            self.upload_batch(deal_offset, batch_count);
            self.dispatch_batch(batch_count as u32, strategy_discount);

            deal_offset = batch_end;
        }

        // After all batches: merge deltas and apply DCFR discount
        self.dispatch_merge_and_discount();

        self.iterations += 1;
    }

    fn upload_batch(&self, deal_offset: usize, batch_count: usize) {
        let n = self.num_nodes as usize;

        // Upload deal data (hand_bits not used on GPU — info IDs are pre-computed)
        let p1_equity: Vec<f32> = (deal_offset..deal_offset + batch_count)
            .map(|i| self.deals[i].p1_equity as f32)
            .collect();
        let deal_weight: Vec<f32> = (deal_offset..deal_offset + batch_count)
            .map(|i| self.deals[i].weight as f32)
            .collect();

        self.queue.write_buffer(&self.deal_p1_wins_buffer, 0, bytemuck::cast_slice(&p1_equity));
        self.queue.write_buffer(&self.deal_weight_buffer, 0, bytemuck::cast_slice(&deal_weight));

        // Upload info_id_lookup for this batch (flat copy)
        let mut lookup = vec![u32::MAX; batch_count * n];
        for (batch_idx, deal_idx) in (deal_offset..deal_offset + batch_count).enumerate() {
            let src = deal_idx * n;
            let dst = batch_idx * n;
            lookup[dst..dst + n].copy_from_slice(&self.deal_info_ids[src..src + n]);
        }
        self.queue.write_buffer(&self.info_id_lookup_buffer, 0, bytemuck::cast_slice(&lookup));

        // Zero reach and utility arrays
        let batch_node_bytes = (batch_count * n * 4) as u64;
        let zeros = vec![0u8; batch_node_bytes as usize];
        self.queue.write_buffer(&self.reach_p1_buffer, 0, &zeros);
        self.queue.write_buffer(&self.reach_p2_buffer, 0, &zeros);
        self.queue.write_buffer(&self.utility_buffer, 0, &zeros);

        // Set root reach = 1.0 for each deal
        let one_bytes = 1.0_f32.to_ne_bytes();
        for d in 0..batch_count {
            let offset = (d * n * 4) as u64;
            self.queue.write_buffer(&self.reach_p1_buffer, offset, &one_bytes);
            self.queue.write_buffer(&self.reach_p2_buffer, offset, &one_bytes);
        }
    }

    fn dispatch_batch(&self, num_deals: u32, strategy_discount: f32) {
        let bg0 = self.create_bind_group0();
        let bg1 = self.create_bind_group1();
        let bg2 = self.create_bind_group2();
        let num_levels = self.level_counts.len();
        let stride = self.uniform_stride;

        // Pre-upload ALL uniforms for this batch into the uniform buffer array.
        // Slot layout: [regret_match, fwd_0..fwd_L-1, bwd_L-1..bwd_0]
        let mut slot = 0u32;

        // Slot 0: regret_match
        self.write_uniform_slot(slot, 0, 0, num_deals, strategy_discount);
        let regret_match_offset = slot * stride;
        slot += 1;

        // Slots 1..L: forward pass levels
        let mut fwd_offsets = Vec::with_capacity(num_levels);
        let mut fwd_threads = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            let level_start = self.level_offsets[level];
            let level_count = self.level_counts[level];
            self.write_uniform_slot(slot, level_start, level_count, num_deals, strategy_discount);
            fwd_offsets.push(slot * stride);
            fwd_threads.push(num_deals * level_count);
            slot += 1;
        }

        // Slots L+1..2L: backward pass levels (reversed)
        let mut bwd_offsets = Vec::with_capacity(num_levels);
        let mut bwd_threads = Vec::with_capacity(num_levels);
        for level in (0..num_levels).rev() {
            let level_start = self.level_offsets[level];
            let level_count = self.level_counts[level];
            self.write_uniform_slot(slot, level_start, level_count, num_deals, strategy_discount);
            bwd_offsets.push(slot * stride);
            bwd_threads.push(num_deals * level_count);
            slot += 1;
        }

        // Single encoder for the entire batch
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("cfr_batch") });

        // Step 1: Regret matching
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.regret_match_pipeline);
            pass.set_bind_group(0, &bg0, &[regret_match_offset]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.dispatch_workgroups(div_ceil(self.num_info_sets, WORKGROUP_SIZE), 1, 1);
        }

        // Step 2: Forward pass (root to leaves)
        for (i, (&offset, &threads)) in fwd_offsets.iter().zip(fwd_threads.iter()).enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(if i == 0 { "fwd_0" } else { "fwd" }),
                ..Default::default()
            });
            pass.set_pipeline(&self.forward_pass_pipeline);
            pass.set_bind_group(0, &bg0, &[offset]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.dispatch_workgroups(div_ceil(threads, WORKGROUP_SIZE), 1, 1);
        }

        // Step 3: Backward pass (leaves to root)
        for (i, (&offset, &threads)) in bwd_offsets.iter().zip(bwd_threads.iter()).enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(if i == 0 { "bwd_0" } else { "bwd" }),
                ..Default::default()
            });
            pass.set_pipeline(&self.backward_pass_pipeline);
            pass.set_bind_group(0, &bg0, &[offset]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.dispatch_workgroups(div_ceil(threads, WORKGROUP_SIZE), 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn dispatch_merge_and_discount(&self) {
        let bg0 = self.create_bind_group0();
        let bg1 = self.create_bind_group1();
        let bg2 = self.create_bind_group2();

        // Write uniforms at slot 0
        self.write_uniform_slot(0, 0, 0, 0, 0.0);

        let total_entries = self.num_info_sets * self.max_actions;
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("merge_discount") });

        // Merge deltas
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("merge_deltas"), ..Default::default() });
            pass.set_pipeline(&self.merge_deltas_pipeline);
            pass.set_bind_group(0, &bg0, &[0]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.dispatch_workgroups(div_ceil(total_entries, WORKGROUP_SIZE), 1, 1);
        }

        // DCFR discount
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("dcfr_discount"), ..Default::default() });
            pass.set_pipeline(&self.dcfr_discount_pipeline);
            pass.set_bind_group(0, &bg0, &[0]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.dispatch_workgroups(div_ceil(total_entries, WORKGROUP_SIZE), 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Write a Uniforms struct at the given slot index in the dynamic uniform buffer.
    fn write_uniform_slot(&self, slot: u32, level_start: u32, level_count: u32, num_deals: u32, strategy_discount: f32) {
        let uniforms = Uniforms {
            level_start,
            level_count,
            num_deals,
            num_nodes: self.num_nodes,
            num_info_sets: self.num_info_sets,
            max_actions: self.max_actions,
            num_hand_classes: self.num_hand_classes,
            iteration: self.iterations as u32,
            dcfr_alpha: self.config.dcfr_alpha as f32,
            dcfr_beta: self.config.dcfr_beta as f32,
            dcfr_gamma: self.config.dcfr_gamma as f32,
            strategy_discount,
        };
        let offset = u64::from(slot * self.uniform_stride);
        self.queue.write_buffer(&self.uniform_buffer, offset, bytemuck::bytes_of(&uniforms));
    }

    fn compute_strategy_discount(&self) -> f32 {
        let t = self.iterations as f64;
        let ratio = t / (t + 1.0);
        ratio.powf(self.config.dcfr_gamma) as f32
    }

    fn create_bind_group0(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("group0"),
            layout: &self.group0_layout,
            entries: &[
                // Dynamic uniform: bind the whole buffer but specify the
                // size of one Uniforms struct. The dynamic offset selects
                // which slot to read.
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: std::num::NonZero::new(std::mem::size_of::<Uniforms>() as u64),
                    }),
                },
                entry(1, &self.node_buffer),
                entry(2, &self.children_buffer),
                entry(3, &self.level_nodes_buffer),
                entry(4, &self.position_actions_buffer),
            ],
        })
    }

    fn create_bind_group1(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("group1"),
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

    fn create_bind_group2(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("group2"),
            layout: &self.group2_layout,
            entries: &[
                entry(0, &self.deal_hand_p1_buffer),
                entry(1, &self.deal_hand_p2_buffer),
                entry(2, &self.deal_p1_wins_buffer),
                entry(3, &self.info_id_lookup_buffer),
                entry(4, &self.reach_p1_buffer),
                entry(5, &self.reach_p2_buffer),
                entry(6, &self.utility_buffer),
                entry(7, &self.deal_weight_buffer),
            ],
        })
    }

    fn download_f32_buffer(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let size = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_read"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("download") });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

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

// --- Free functions ---

fn init_gpu() -> Result<(wgpu::Device, wgpu::Queue), GpuError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })
    .block_on()
    .map_err(|_| GpuError::NoAdapter)?;

    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("cfr_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        },
    )
    .block_on()?;

    Ok((device, queue))
}

/// Encoded tree: (gpu_nodes, children_flat, level_nodes, level_offsets, level_counts).
type EncodedTree = (Vec<GpuNode>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);

fn encode_tree(tree: &GameTree) -> EncodedTree {
    let num_nodes = tree.nodes.len();
    let mut gpu_nodes = Vec::with_capacity(num_nodes);
    let mut children_flat: Vec<u32> = Vec::new();

    for node in &tree.nodes {
        let first_child = children_flat.len() as u32;
        for &child in &node.children {
            children_flat.push(child);
        }

        let (node_type, p1_invested_bb, p2_invested_bb) = match &node.node_type {
            NodeType::Decision { player, .. } => {
                let nt = match player {
                    Player::Player1 => 0u32,
                    Player::Player2 => 1u32,
                };
                (nt, 0.0, 0.0)
            }
            NodeType::Terminal {
                is_fold, fold_player, stacks, starting_stack, ..
            } => {
                let nt = if *is_fold {
                    match fold_player {
                        Player::Player1 => 2u32,
                        Player::Player2 => 3u32,
                    }
                } else {
                    4u32
                };
                let p1_inv = f64::from(*starting_stack - stacks[0]) / 2.0;
                let p2_inv = f64::from(*starting_stack - stacks[1]) / 2.0;
                (nt, p1_inv as f32, p2_inv as f32)
            }
        };

        gpu_nodes.push(GpuNode {
            node_type,
            position_index: u32::MAX,
            first_child,
            num_children: node.children.len() as u32,
            p1_invested_bb,
            p2_invested_bb,
            _pad0: 0,
            _pad1: 0,
        });
    }

    // Ensure children_flat is non-empty (wgpu requires non-zero buffer sizes)
    if children_flat.is_empty() {
        children_flat.push(0);
    }

    // Build level arrays
    let mut level_nodes_flat: Vec<u32> = Vec::new();
    let mut level_offsets: Vec<u32> = Vec::new();
    let mut level_counts: Vec<u32> = Vec::new();

    for level in &tree.levels {
        level_offsets.push(level_nodes_flat.len() as u32);
        level_counts.push(level.len() as u32);
        level_nodes_flat.extend_from_slice(level);
    }

    // Ensure non-empty
    if level_nodes_flat.is_empty() {
        level_nodes_flat.push(0);
    }

    (gpu_nodes, children_flat, level_nodes_flat, level_offsets, level_counts)
}

fn build_position_indices(tree: &GameTree) -> (Vec<u32>, u32, Vec<u32>) {
    let mut key_to_index: FxHashMap<u64, u32> = FxHashMap::default();
    let mut position_num_actions: Vec<u32> = Vec::new();
    let mut indices = vec![u32::MAX; tree.nodes.len()];
    let mut next_index = 0u32;

    for (i, node) in tree.nodes.iter().enumerate() {
        if let NodeType::Decision { .. } = &node.node_type {
            // Position key = info_set_key with hand_bits=0
            let position_key = tree.info_set_key(i as u32, 0);
            let index = *key_to_index.entry(position_key).or_insert_with(|| {
                let idx = next_index;
                next_index += 1;
                position_num_actions.push(node.children.len() as u32);
                idx
            });
            indices[i] = index;
        }
    }

    (indices, next_index, position_num_actions)
}

/// Pre-collected decision node metadata to avoid repeated `NodeType` matching.
#[derive(Clone, Copy)]
struct DecisionNodeInfo {
    node_idx: usize,
    player: Player,
    street: u8,
}

/// Scan the tree once and extract decision node indices with their metadata.
fn collect_decision_nodes(tree: &GameTree) -> Vec<DecisionNodeInfo> {
    tree.nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| match &node.node_type {
            NodeType::Decision { player, street, .. } => Some(DecisionNodeInfo {
                node_idx: i,
                player: *player,
                street: *street,
            }),
            _ => None,
        })
        .collect()
}

/// Build the info set mapping from unique (position, hand_bits) combos.
///
/// Instead of iterating `deals × nodes` (O(D×N)), this:
/// 1. Collects unique `hand_bits` per `(player, street)` in O(D) time.
/// 2. Iterates `decision_nodes × unique_hands` which is much smaller.
fn build_info_set_mapping(
    tree: &GameTree,
    deals: &[DealInfo],
    decision_nodes: &[DecisionNodeInfo],
) -> (FxHashMap<u64, u32>, Vec<u64>, u32) {
    // 8 buckets: player(0..2) × street(0..4)
    let mut unique_per_bucket: [FxHashSet<u32>; 8] = [
        FxHashSet::default(), FxHashSet::default(),
        FxHashSet::default(), FxHashSet::default(),
        FxHashSet::default(), FxHashSet::default(),
        FxHashSet::default(), FxHashSet::default(),
    ];

    // Pass 1: collect unique hand_bits per (player, street) — O(deals)
    for deal in deals {
        for street in 0..4usize {
            unique_per_bucket[street].insert(deal.hand_bits_p1[street]);
            unique_per_bucket[4 + street].insert(deal.hand_bits_p2[street]);
        }
    }

    // Pass 2: iterate decision_nodes × unique_hands — O(D_nodes × U_hands)
    let mut key_to_dense: FxHashMap<u64, u32> = FxHashMap::default();
    let mut dense_to_key: Vec<u64> = Vec::new();
    let mut hand_classes_seen = FxHashSet::default();

    for dn in decision_nodes {
        let bucket = match dn.player {
            Player::Player1 => dn.street as usize,
            Player::Player2 => 4 + dn.street as usize,
        };
        for &hand_bits in &unique_per_bucket[bucket] {
            hand_classes_seen.insert(hand_bits);
            let key = tree.info_set_key(dn.node_idx as u32, hand_bits);
            key_to_dense.entry(key).or_insert_with(|| {
                let id = dense_to_key.len() as u32;
                dense_to_key.push(key);
                id
            });
        }
    }

    let num_hand_classes = hand_classes_seen.len().max(1) as u32;
    (key_to_dense, dense_to_key, num_hand_classes)
}

/// Pre-compute per-(deal, node) info set dense IDs.
///
/// Returns a flat `Vec<u32>` with stride `num_nodes`. Each deal's slice is
/// independent, so computation is parallelized with rayon.
fn precompute_deal_info_ids(
    tree: &GameTree,
    deals: &[DealInfo],
    decision_nodes: &[DecisionNodeInfo],
    key_to_dense: &FxHashMap<u64, u32>,
) -> Vec<u32> {
    let num_nodes = tree.nodes.len();
    let mut flat = vec![u32::MAX; deals.len() * num_nodes];

    flat.par_chunks_mut(num_nodes)
        .zip(deals.par_iter())
        .for_each(|(ids, deal)| {
            for dn in decision_nodes {
                let hand_bits = match dn.player {
                    Player::Player1 => deal.hand_bits_p1[dn.street as usize],
                    Player::Player2 => deal.hand_bits_p2[dn.street as usize],
                };
                let key = tree.info_set_key(dn.node_idx as u32, hand_bits);
                if let Some(&dense_id) = key_to_dense.get(&key) {
                    ids[dn.node_idx] = dense_id;
                }
            }
        });

    flat
}

fn create_buffer_init(device: &wgpu::Device, label: &str, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage,
    })
}

fn create_buffer_zeroed(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    let size = size.max(4); // wgpu requires non-zero buffer size
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_group0_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("group0_layout"),
        entries: &[
            // Uniform binding with dynamic offset
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
            layout_entry(1, storage_rw()),
            layout_entry(2, storage_rw()),
            layout_entry(3, storage_rw()),
            layout_entry(4, storage_rw()),
        ],
    })
}

fn create_group1_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("group1_layout"),
        entries: &[
            layout_entry(0, storage_rw()), // regret
            layout_entry(1, storage_rw()), // strategy
            layout_entry(2, storage_rw()), // strategy_sum
            layout_entry(3, storage_rw()), // regret_delta (atomic)
            layout_entry(4, storage_rw()), // strat_sum_delta (atomic)
        ],
    })
}

fn create_group2_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("group2_layout"),
        entries: &[
            layout_entry(0, storage_rw()), // deal_hand_p1
            layout_entry(1, storage_rw()), // deal_hand_p2
            layout_entry(2, storage_rw()), // deal_p1_equity
            layout_entry(3, storage_rw()), // info_id_lookup
            layout_entry(4, storage_rw()), // reach_p1
            layout_entry(5, storage_rw()), // reach_p2
            layout_entry(6, storage_rw()), // utility
            layout_entry(7, storage_rw()), // deal_weight
        ],
    })
}

fn layout_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw() -> wgpu::BufferBindingType {
    wgpu::BufferBindingType::Storage { read_only: false }
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    label: &str,
    wgsl_source: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_source)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Round `value` up to the next multiple of `alignment`.
fn align_up(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::cfr::game_tree::materialize_postflop;
    use poker_solver_core::game::{AbstractionMode, HunlPostflop, PostflopConfig};
    use poker_solver_core::info_key::InfoKey;
    use poker_solver_core::Game;

    #[test]
    fn gpu_init_succeeds() {
        match init_gpu() {
            Ok((device, _queue)) => {
                let info = device.features();
                println!("GPU initialized, features: {info:?}");
            }
            Err(GpuError::NoAdapter) => {
                println!("No GPU adapter available, skipping test");
            }
            Err(e) => panic!("Unexpected GPU error: {e}"),
        }
    }

    #[test]
    fn gpu_node_layout_is_32_bytes() {
        assert_eq!(std::mem::size_of::<GpuNode>(), 32);
    }

    #[test]
    fn uniforms_layout_is_48_bytes() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 48);
    }

    fn build_test_game() -> (HunlPostflop, Vec<DealInfo>) {
        let config = PostflopConfig {
            stack_depth: 10,
            bet_sizes: vec![1.0],
            max_raises_per_street: 2,
        };
        let game = HunlPostflop::new(config, Some(AbstractionMode::HandClassV2 {
            strength_bits: 0,
            equity_bits: 0,
        }), 42);
        let states = game.initial_states();

        let deals: Vec<DealInfo> = states.iter().take(100).map(|state| {
            let hand_bits_p1 = InfoKey::from_raw(game.info_set_key(state)).hand_bits();

            let first_action = game.actions(state)[0];
            let next_state = game.next_state(state, first_action);
            let hand_bits_p2 = InfoKey::from_raw(game.info_set_key(&next_state)).hand_bits();

            let p1_equity = match (&state.p1_cache.rank, &state.p2_cache.rank) {
                (Some(r1), Some(r2)) => {
                    use std::cmp::Ordering;
                    match r1.cmp(r2) {
                        Ordering::Greater => 1.0,
                        Ordering::Less => 0.0,
                        Ordering::Equal => 0.5,
                    }
                }
                _ => 0.5,
            };

            DealInfo {
                hand_bits_p1: [hand_bits_p1; 4],
                hand_bits_p2: [hand_bits_p2; 4],
                p1_equity,
                weight: 1.0,
            }
        }).collect();

        (game, deals)
    }

    #[test]
    fn gpu_solver_constructs_successfully() {
        let (game, deals) = build_test_game();
        let tree = materialize_postflop(&game, &game.initial_states()[0]);

        match GpuCfrSolver::new(&tree, deals, GpuCfrConfig::default()) {
            Ok(solver) => {
                println!("GPU solver created: {} info sets, {} nodes",
                    solver.num_info_sets(), solver.num_nodes);
            }
            Err(GpuError::NoAdapter) => {
                println!("No GPU, skipping");
            }
            Err(e) => panic!("Failed to create GPU solver: {e}"),
        }
    }

    #[test]
    fn gpu_solver_runs_iterations() {
        let (game, deals) = build_test_game();
        let tree = materialize_postflop(&game, &game.initial_states()[0]);

        match GpuCfrSolver::new(&tree, deals, GpuCfrConfig { max_batch_size: Some(50), ..Default::default() }) {
            Ok(mut solver) => {
                // First iteration has strategy_discount=0, so strategy_sum stays empty.
                // Train 5 iterations so strategies accumulate.
                solver.train(5);
                assert_eq!(solver.iterations(), 5);

                let strategies = solver.all_strategies();
                println!("After 5 iterations: {} info sets with strategies", strategies.len());
                assert!(!strategies.is_empty(), "Should have produced strategies");

                // Verify strategies sum to ~1.0
                for (&key, probs) in &strategies {
                    let sum: f64 = probs.iter().sum();
                    assert!(
                        (sum - 1.0).abs() < 0.01,
                        "Strategy for key {key:#x} doesn't sum to 1: {sum}"
                    );
                }
            }
            Err(GpuError::NoAdapter) => {
                println!("No GPU, skipping");
            }
            Err(e) => panic!("Failed: {e}"),
        }
    }

    #[test]
    fn gpu_solver_trains_multiple_iterations() {
        let (game, deals) = build_test_game();
        let tree = materialize_postflop(&game, &game.initial_states()[0]);

        match GpuCfrSolver::new(&tree, deals, GpuCfrConfig { max_batch_size: Some(50), ..Default::default() }) {
            Ok(mut solver) => {
                solver.train(10);
                assert_eq!(solver.iterations(), 10);

                let strategies = solver.all_strategies();
                assert!(!strategies.is_empty());
                println!("After 10 iterations: {} info sets", strategies.len());
            }
            Err(GpuError::NoAdapter) => {
                println!("No GPU, skipping");
            }
            Err(e) => panic!("Failed: {e}"),
        }
    }
}
