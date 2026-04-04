# cudarc GPU Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace burn with cudarc + custom CUDA kernels in the gpu-range-solver crate, reducing per-operation overhead from ~500μs (burn tensor) to ~5μs (raw CUDA launch) to match CPU solver speed.

**Architecture:** All solver state in `CudaSlice<f32>` on GPU. Custom CUDA kernels compiled at runtime via nvrtc. CPU drives the iteration loop, launching ~20 kernels per iteration (vs ~600 burn tensor ops). Data stays on GPU for the entire solve.

**Tech Stack:** Rust, cudarc 0.19 (driver + nvrtc features), CUDA 12.1, custom .cu kernels

**Design doc:** `docs/plans/2026-04-03-cudarc-gpu-solver-design.md`

---

## Task 1: Update Dependencies — Drop burn, Add cudarc

**Files:**
- Modify: `crates/gpu-range-solver/Cargo.toml`

**Step 1: Update Cargo.toml**

Replace the burn dependency with cudarc:

```toml
[package]
name = "gpu-range-solver"
version.workspace = true
edition.workspace = true

[dependencies]
range-solver = { path = "../range-solver" }
cudarc = { version = "0.19", features = ["driver", "nvrtc"] }

[dev-dependencies]
rand = "0.8"
```

**Step 2: Verify it compiles (extract.rs only)**

Temporarily comment out the `pub mod tensors; pub mod solver; pub mod terminal;` lines in `lib.rs` and the `gpu_solve_game` function body. Keep `pub mod extract;` since it has no burn dependency.

Run: `cargo check -p gpu-range-solver`
Expected: Compiles (extract.rs has no burn imports)

**Step 3: Commit**

```bash
git commit -m "chore(gpu-range-solver): replace burn with cudarc dependency"
```

---

## Task 2: CUDA Kernel Source

Write all CUDA kernels in a single source string. This is pure CUDA C — no Rust, no framework. The kernels implement the core CFR operations.

**Files:**
- Create: `crates/gpu-range-solver/src/kernels.rs`

**Step 1: Write the kernel source**

Create `crates/gpu-range-solver/src/kernels.rs` with the CUDA source as a constant string:

```rust
//! CUDA kernel source and compilation.

/// All CUDA kernels for the CFR solver, compiled at runtime via nvrtc.
pub const CFR_KERNELS_SOURCE: &str = r#"
extern "C" {

// ============================================================
// Utility: zero a float array
// ============================================================
__global__ void zero_f32(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 0.0f;
}

// ============================================================
// Regret matching pass 1: accumulate clipped regrets to denom
// ============================================================
// Thread: one per (edge * H + hand) in the FULL edge array
// Writes: denom[parent[e]*H + h] += max(regrets[e*H+h], 0)
__global__ void regret_match_accum(
    const float* regrets,       // [E * H]
    float* denom,               // [N * H] — must be zeroed before launch
    const int* edge_parent,     // [E]
    int E, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E * H) return;
    int e = idx / H;
    int h = idx % H;
    float clipped = fmaxf(regrets[e * H + h], 0.0f);
    if (clipped > 0.0f) {
        atomicAdd(&denom[edge_parent[e] * H + h], clipped);
    }
}

// ============================================================
// Regret matching pass 2: normalize to get strategy
// ============================================================
__global__ void regret_match_normalize(
    const float* regrets,          // [E * H]
    const float* denom,            // [N * H]
    float* strategy,               // [E * H] output
    const int* edge_parent,        // [E]
    const float* actions_per_edge, // [E]
    int E, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E * H) return;
    int e = idx / H;
    int h = idx % H;
    float clipped = fmaxf(regrets[e * H + h], 0.0f);
    float d = denom[edge_parent[e] * H + h];
    strategy[e * H + h] = (d > 1e-30f) ? (clipped / d) : (1.0f / actions_per_edge[e]);
}

// ============================================================
// Forward pass: propagate reach for one level's edges
// ============================================================
// traverser_player: 0 or 1 (the player doing the current traversal)
// CHANCE_PLAYER = 2
__global__ void forward_pass_level(
    float* reach,              // [N * H]
    const float* strategy,     // [E * H]
    const int* edge_parent,    // [E]
    const int* edge_child,     // [E]
    const int* edge_player,    // [E]
    int level_start,           // first edge index for this level
    int level_count,           // number of edges at this level
    int traverser_player,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= level_count * H) return;
    int local_e = idx / H;
    int h = idx % H;
    int e = level_start + local_e;

    int p = edge_parent[e];
    int c = edge_child[e];
    int player = edge_player[e];

    float parent_reach = reach[p * H + h];

    if (player == traverser_player) {
        // Traverser: counterfactual reach (don't multiply by own strategy)
        reach[c * H + h] = parent_reach;
    } else if (player == 2) {
        // Chance node: divide by number of children (actions_per_edge encodes this)
        // Card blocking handled separately
        reach[c * H + h] = parent_reach;  // chance factor applied by caller or separate kernel
    } else {
        // Opponent: multiply by strategy
        reach[c * H + h] = parent_reach * strategy[e * H + h];
    }
}

// ============================================================
// Backward pass: scatter-add weighted child CFVs to parents
// ============================================================
__global__ void backward_pass_level(
    float* cfv,                // [N * H]
    const float* strategy,     // [E * H]
    const int* edge_parent,    // [E]
    const int* edge_child,     // [E]
    const int* edge_player,    // [E]
    int level_start,
    int level_count,
    int traverser_player,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= level_count * H) return;
    int local_e = idx / H;
    int h = idx % H;
    int e = level_start + local_e;

    int p = edge_parent[e];
    int c = edge_child[e];
    int player = edge_player[e];

    float child_cfv = cfv[c * H + h];

    if (player == traverser_player) {
        // Traverser node: weight by strategy
        atomicAdd(&cfv[p * H + h], strategy[e * H + h] * child_cfv);
    } else {
        // Opponent or chance: sum unweighted
        atomicAdd(&cfv[p * H + h], child_cfv);
    }
}

// ============================================================
// Regret update: after backward_pass_level, parent CFVs are final
// ============================================================
__global__ void regret_update_level(
    float* regrets,            // [E * H]
    float* strategy_sum,       // [E * H]
    const float* cfv,          // [N * H]
    const float* strategy,     // [E * H]
    const int* edge_parent,    // [E]
    const int* edge_child,     // [E]
    const int* edge_player,    // [E]
    int level_start,
    int level_count,
    int traverser_player,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= level_count * H) return;
    int local_e = idx / H;
    int h = idx % H;
    int e = level_start + local_e;

    if (edge_player[e] != traverser_player) return;

    int p = edge_parent[e];
    int c = edge_child[e];

    float instant_regret = cfv[c * H + h] - cfv[p * H + h];
    regrets[e * H + h] += instant_regret;
    strategy_sum[e * H + h] += strategy[e * H + h];
}

// ============================================================
// DCFR discount: scale regrets and strategy_sum
// ============================================================
__global__ void dcfr_discount(
    float* regrets,        // [E * H]
    float* strategy_sum,   // [E * H]
    float alpha,
    float beta,
    float gamma,
    int total              // E * H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float r = regrets[idx];
    regrets[idx] = (r >= 0.0f) ? r * alpha : r * beta;
    strategy_sum[idx] *= gamma;
}

// ============================================================
// Fold evaluation: compute CFV for fold terminal nodes
// ============================================================
// Called per fold node. opp_reach is at node_id in the reach array.
// Writes result to cfv at node_id.
// Uses card blocking: cfv[h] = payoff * (total_opp_reach - blocking[h])
//
// card_reach_scratch: [52] scratch per node (caller provides)
// hand_card1, hand_card2: per-hand card indices for the PLAYER
// opp_hand_card1, opp_hand_card2: per-hand card indices for OPPONENT
// same_hand_index: per player hand, opponent hand index or -1
__global__ void fold_eval(
    float* cfv,                   // [N * H] — write to cfv[node_id * H + h]
    const float* reach,           // [N * H] — read opp reach from reach[node_id * H + h]
    int node_id,                  // which node to evaluate
    float payoff,                 // amount_win or amount_lose depending on who folded
    const int* player_card1,      // [num_player_hands] — card 1 per player hand
    const int* player_card2,      // [num_player_hands]
    const int* opp_card1,         // [num_opp_hands] — card 1 per opp hand
    const int* opp_card2,         // [num_opp_hands]
    const int* same_hand_idx,     // [num_player_hands] — opp index or -1
    int num_player_hands,
    int num_opp_hands,
    int H                         // padded hand dimension
) {
    // Phase 1: compute card_reach[52] and total_reach (use shared memory)
    __shared__ float card_reach[52];
    __shared__ float total_reach;

    int tid = threadIdx.x;
    if (tid < 52) card_reach[tid] = 0.0f;
    if (tid == 0) total_reach = 0.0f;
    __syncthreads();

    // Each thread handles one opponent hand
    if (tid < num_opp_hands) {
        float r = reach[node_id * H + tid];
        atomicAdd(&total_reach, r);
        atomicAdd(&card_reach[opp_card1[tid]], r);
        atomicAdd(&card_reach[opp_card2[tid]], r);
    }
    __syncthreads();

    // Phase 2: compute CFV for each player hand
    if (tid < num_player_hands) {
        int c1 = player_card1[tid];
        int c2 = player_card2[tid];
        float blocking = card_reach[c1] + card_reach[c2];
        int same = same_hand_idx[tid];
        if (same >= 0) {
            blocking -= reach[node_id * H + same];
        }
        cfv[node_id * H + tid] = payoff * (total_reach - blocking);
    }
}

// ============================================================
// Showdown evaluation: matmul outcome × opp_reach
// ============================================================
// cfv[h] = sum_opp(outcome[h * num_opp + opp] * opp_reach[opp])
// where outcome is from player's perspective (+1 win, -1 loss, 0 tie/blocked)
// Then scale by amount_win/amount_lose
__global__ void showdown_eval(
    float* cfv,                   // [N * H]
    const float* reach,           // [N * H]
    int node_id,
    const float* outcome,         // [num_player * num_opp] — pre-computed outcome matrix
    float amount_win,
    float amount_lose,
    int num_player_hands,
    int num_opp_hands,
    int H
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_player_hands) return;

    float win_sum = 0.0f;
    float lose_sum = 0.0f;
    for (int opp = 0; opp < num_opp_hands; opp++) {
        float opp_r = reach[node_id * H + opp];
        float o = outcome[h * num_opp_hands + opp];
        if (o > 0.0f) win_sum += opp_r;
        else if (o < 0.0f) lose_sum += opp_r;
    }
    cfv[node_id * H + h] = win_sum * amount_win + lose_sum * amount_lose;
}

// ============================================================
// Best-response backward: max over actions instead of weighted sum
// ============================================================
// For traverser nodes at one level: cfv[parent] = max over children cfv
// For opponent/chance: same as backward_pass_level (sum)
// This kernel handles the traverser MAX case
__global__ void best_response_max_level(
    float* cfv,                // [N * H]
    const int* edge_parent,    // [E]
    const int* edge_child,     // [E]
    const int* edge_player,    // [E]
    int level_start,
    int level_count,
    int traverser_player,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= level_count * H) return;
    int local_e = idx / H;
    int h = idx % H;
    int e = level_start + local_e;

    if (edge_player[e] != traverser_player) {
        // Opponent/chance: sum (same as backward_pass_level)
        atomicAdd(&cfv[edge_parent[e] * H + h], cfv[edge_child[e] * H + h]);
        return;
    }

    // Traverser: atomicMax (but float atomicMax doesn't exist in CUDA < sm_90)
    // Use atomicCAS-based float max instead
    int p = edge_parent[e];
    float val = cfv[edge_child[e] * H + h];
    int* addr = (int*)&cfv[p * H + h];
    int old = *addr, assumed;
    do {
        assumed = old;
        float old_f = __int_as_float(assumed);
        float new_f = fmaxf(old_f, val);
        old = atomicCAS(addr, assumed, __float_as_int(new_f));
    } while (assumed != old);
}

// ============================================================
// Copy a slice of host data into reach at root node
// ============================================================
__global__ void set_reach_root(
    float* reach,                // [N * H]
    const float* initial_weights, // [H]
    int H
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < H) reach[h] = initial_weights[h];  // node 0 * H + h = h
}

} // extern "C"
"#;
```

**Step 2: Verify it compiles as a Rust module**

Add `pub mod kernels;` to lib.rs. Run: `cargo check -p gpu-range-solver`
Expected: Compiles (it's just a string constant)

**Step 3: Commit**

```bash
git commit -m "feat(gpu-range-solver): CUDA kernel source for CFR solver"
```

---

## Task 3: GPU Device Management

Create `gpu.rs` — handles CudaDevice setup, kernel compilation, and a `GpuSolverState` struct that owns all device memory.

**Files:**
- Create: `crates/gpu-range-solver/src/gpu.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_state_initializes() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        // Tiny tree: 3 nodes, 2 edges, 4 hands
        let state = GpuSolverState::new(&ctx, &stream, 3, 2, 4);
        assert!(state.is_ok());
    }
}
```

**Step 2: Implement GpuSolverState**

```rust
//! GPU device management: memory allocation, kernel compilation, solver state.

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

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
        let ptx = compile_ptx(crate::kernels::CFR_KERNELS_SOURCE)?;
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
    pub regrets: CudaSlice<f32>,       // [E * H]
    pub strategy_sum: CudaSlice<f32>,  // [E * H]
    pub strategy: CudaSlice<f32>,      // [E * H]
    pub reach: CudaSlice<f32>,         // [N * H]
    pub cfv: CudaSlice<f32>,           // [N * H]
    pub denom: CudaSlice<f32>,         // [N * H] scratch

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
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        num_nodes: usize,
        num_edges: usize,
        num_hands: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            edge_parent: stream.alloc_zeros::<i32>(num_edges)?,
            edge_child: stream.alloc_zeros::<i32>(num_edges)?,
            edge_player: stream.alloc_zeros::<i32>(num_edges)?,
            actions_per_edge: stream.alloc_zeros::<f32>(num_edges)?,
            regrets: stream.alloc_zeros::<f32>(num_edges * num_hands)?,
            strategy_sum: stream.alloc_zeros::<f32>(num_edges * num_hands)?,
            strategy: stream.alloc_zeros::<f32>(num_edges * num_hands)?,
            reach: stream.alloc_zeros::<f32>(num_nodes * num_hands)?,
            cfv: stream.alloc_zeros::<f32>(num_nodes * num_hands)?,
            denom: stream.alloc_zeros::<f32>(num_nodes * num_hands)?,
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
        stream: &CudaStream,
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

/// Helper: compute grid size for N threads with block_size threads per block.
pub fn grid_size(n: usize, block_size: u32) -> u32 {
    ((n as u32) + block_size - 1) / block_size
}

pub const BLOCK_SIZE: u32 = 256;
```

**Step 3: Run test**

Run: `cargo test -p gpu-range-solver -- gpu_state_initializes`
Expected: PASS (requires CUDA device)

**Step 4: Commit**

```bash
git commit -m "feat(gpu-range-solver): GPU device management and solver state"
```

---

## Task 4: Solver Core — DCFR Iteration Loop

Rewrite `solver.rs` to use cudarc kernel launches. This is the heart of the rewrite.

**Files:**
- Rewrite: `crates/gpu-range-solver/src/solver.rs`

**Step 1: Write failing integration test**

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn cudarc_solve_matches_cpu() {
        // Build same river game as lib.rs tests
        let game = make_river_game();
        let mut cpu_game = make_river_game();
        let cpu_expl = range_solver::solve(&mut cpu_game, 200, 0.0, false);

        let config = crate::GpuSolverConfig {
            max_iterations: 200,
            target_exploitability: 0.0,
            print_progress: false,
        };
        let result = crate::gpu_solve_game(&game, &config);

        assert!((result.exploitability - cpu_expl).abs() < 0.5,
            "GPU {:.4} vs CPU {:.4}", result.exploitability, cpu_expl);
    }
}
```

**Step 2: Implement the solver**

The solver function orchestrates the iteration loop:

```rust
//! DCFR iteration loop using cudarc CUDA kernel launches.

use crate::extract::{NodeType, TerminalData, TreeTopology};
use crate::gpu::{CfrKernels, GpuSolverState, grid_size, BLOCK_SIZE};
use cudarc::driver::{CudaContext, CudaStream, LaunchConfig};
use std::sync::Arc;

/// DCFR discount parameters (same formula as CPU solver).
struct DiscountParams { alpha_t: f32, beta_t: f32, gamma_t: f32 }

impl DiscountParams {
    fn new(iteration: u32) -> Self {
        let nearest = match iteration { 0 => 0, x => 1u32 << ((x.leading_zeros() ^ 31) & !1) };
        let ta = (iteration as i32 - 1).max(0) as f64;
        let tg = (iteration - nearest) as f64;
        let pa = ta * ta.sqrt();
        let pg = (tg / (tg + 1.0)).powi(3);
        Self {
            alpha_t: (pa / (pa + 1.0)) as f32,
            beta_t: 0.5,
            gamma_t: pg as f32,
        }
    }
}

/// Build sorted edge arrays from topology and upload to GPU.
fn prepare_topology(
    topo: &TreeTopology,
    stream: &CudaStream,
    state: &mut GpuSolverState,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    // Sort edges by parent depth
    let mut node_depth = vec![0usize; topo.num_nodes];
    for (d, nodes) in topo.level_nodes.iter().enumerate() {
        for &n in nodes { node_depth[n] = d; }
    }

    let mut edges_by_depth: Vec<Vec<usize>> = vec![Vec::new(); topo.max_depth + 1];
    for e in 0..topo.num_edges {
        edges_by_depth[node_depth[topo.edge_parent[e]]].push(e);
    }

    let mut sorted_edges = Vec::with_capacity(topo.num_edges);
    let mut level_starts = Vec::new();
    let mut level_counts = Vec::new();
    for depth_edges in &edges_by_depth {
        level_starts.push(sorted_edges.len());
        level_counts.push(depth_edges.len());
        sorted_edges.extend(depth_edges);
    }

    // Build sorted arrays
    let parent_i32: Vec<i32> = sorted_edges.iter().map(|&e| topo.edge_parent[e] as i32).collect();
    let child_i32: Vec<i32> = sorted_edges.iter().map(|&e| topo.edge_child[e] as i32).collect();
    let player_i32: Vec<i32> = sorted_edges.iter().map(|&e| {
        match topo.node_type[topo.edge_parent[e]] {
            NodeType::Player { player } => player as i32,
            NodeType::Chance => 2i32,
            _ => -1i32,
        }
    }).collect();
    let ape: Vec<f32> = sorted_edges.iter().map(|&e| {
        topo.node_num_actions[topo.edge_parent[e]] as f32
    }).collect();

    state.upload_topology(
        stream, &parent_i32, &child_i32, &player_i32, &ape,
        level_starts, level_counts, topo.max_depth,
    )?;

    Ok(sorted_edges)
}

/// Run the full DCFR solve on GPU.
pub fn gpu_solve_cudarc(
    topo: &TreeTopology,
    term: &TerminalData,
    config: &crate::GpuSolverConfig,
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
) -> Result<crate::GpuSolveResult, Box<dyn std::error::Error>> {
    // 1. Setup CUDA
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let kernels = CfrKernels::compile(&ctx)?;

    // 2. Allocate GPU state
    let mut state = GpuSolverState::new(&ctx, &stream, topo.num_nodes, topo.num_edges, num_hands)?;
    let sorted_edges = prepare_topology(topo, &stream, &mut state)?;

    // 3. Upload terminal eval data (fold/showdown node info, card indices, outcome matrices)
    // ... (upload hand_cards, same_hand_index, outcome matrices for each terminal node)

    let e_times_h = state.num_edges * num_hands;
    let n_times_h = state.num_nodes * num_hands;

    // 4. Iteration loop
    let mut exploitability = f32::MAX;
    let mut iterations_run = 0u32;

    for t in 0..config.max_iterations {
        let params = DiscountParams::new(t);

        // DCFR discount
        unsafe {
            let cfg = LaunchConfig {
                grid_dim: (grid_size(e_times_h, BLOCK_SIZE), 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&kernels.dcfr_discount);
            builder.arg(&mut state.regrets);
            builder.arg(&mut state.strategy_sum);
            builder.arg(&params.alpha_t);
            builder.arg(&params.beta_t);
            builder.arg(&params.gamma_t);
            builder.arg(&(e_times_h as i32));
            builder.launch(cfg)?;
        }

        // Alternating player updates
        for player in 0..2 {
            // Zero reach, cfv, denom
            let zero_cfg = LaunchConfig {
                grid_dim: (grid_size(n_times_h, BLOCK_SIZE), 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                let mut b = stream.launch_builder(&kernels.zero_f32);
                b.arg(&mut state.reach);
                b.arg(&(n_times_h as i32));
                b.launch(zero_cfg)?;

                let mut b = stream.launch_builder(&kernels.zero_f32);
                b.arg(&mut state.cfv);
                b.arg(&(n_times_h as i32));
                b.launch(zero_cfg)?;

                let mut b = stream.launch_builder(&kernels.zero_f32);
                b.arg(&mut state.denom);
                b.arg(&(n_times_h as i32));
                b.launch(zero_cfg)?;
            }

            // Set root reach = opponent initial weights
            let opp = player ^ 1;
            let opp_weights_gpu = stream.clone_htod(&initial_weights[opp])?;
            unsafe {
                let cfg = LaunchConfig {
                    grid_dim: (grid_size(num_hands, BLOCK_SIZE), 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = stream.launch_builder(&kernels.set_reach_root);
                b.arg(&mut state.reach);
                b.arg(&opp_weights_gpu);
                b.arg(&(num_hands as i32));
                b.launch(cfg)?;
            }

            // Regret match (all edges)
            unsafe {
                let cfg = LaunchConfig {
                    grid_dim: (grid_size(e_times_h, BLOCK_SIZE), 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = stream.launch_builder(&kernels.regret_match_accum);
                b.arg(&state.regrets);
                b.arg(&mut state.denom);
                b.arg(&state.edge_parent);
                b.arg(&(state.num_edges as i32));
                b.arg(&(num_hands as i32));
                b.launch(cfg)?;

                let mut b = stream.launch_builder(&kernels.regret_match_normalize);
                b.arg(&state.regrets);
                b.arg(&state.denom);
                b.arg(&mut state.strategy);
                b.arg(&state.edge_parent);
                b.arg(&state.actions_per_edge);
                b.arg(&(state.num_edges as i32));
                b.arg(&(num_hands as i32));
                b.launch(cfg)?;
            }

            // Forward pass: level by level
            for depth in 0..=state.max_depth {
                let count = state.level_edge_count[depth];
                if count == 0 { continue; }
                let start = state.level_edge_start[depth];
                unsafe {
                    let cfg = LaunchConfig {
                        grid_dim: (grid_size(count * num_hands, BLOCK_SIZE), 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = stream.launch_builder(&kernels.forward_pass_level);
                    b.arg(&mut state.reach);
                    b.arg(&state.strategy);
                    b.arg(&state.edge_parent);
                    b.arg(&state.edge_child);
                    b.arg(&state.edge_player);
                    b.arg(&(start as i32));
                    b.arg(&(count as i32));
                    b.arg(&(player as i32));
                    b.arg(&(num_hands as i32));
                    b.launch(cfg)?;
                }
            }

            // Backward pass: level by level (reverse)
            for depth in (0..=state.max_depth).rev() {
                // Terminal evaluation at this depth
                evaluate_terminals_at_depth(
                    &stream, &kernels, &mut state, topo, term,
                    depth, player, num_hands,
                    &sorted_edges,
                )?;

                let count = state.level_edge_count[depth];
                if count == 0 { continue; }
                let start = state.level_edge_start[depth];

                unsafe {
                    let cfg = LaunchConfig {
                        grid_dim: (grid_size(count * num_hands, BLOCK_SIZE), 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = stream.launch_builder(&kernels.backward_pass_level);
                    b.arg(&mut state.cfv);
                    b.arg(&state.strategy);
                    b.arg(&state.edge_parent);
                    b.arg(&state.edge_child);
                    b.arg(&state.edge_player);
                    b.arg(&(start as i32));
                    b.arg(&(count as i32));
                    b.arg(&(player as i32));
                    b.arg(&(num_hands as i32));
                    b.launch(cfg)?;

                    // Regret update (separate launch — parent CFVs are now final)
                    let mut b = stream.launch_builder(&kernels.regret_update_level);
                    b.arg(&mut state.regrets);
                    b.arg(&mut state.strategy_sum);
                    b.arg(&state.cfv);
                    b.arg(&state.strategy);
                    b.arg(&state.edge_parent);
                    b.arg(&state.edge_child);
                    b.arg(&state.edge_player);
                    b.arg(&(start as i32));
                    b.arg(&(count as i32));
                    b.arg(&(player as i32));
                    b.arg(&(num_hands as i32));
                    b.launch(cfg)?;
                }
            }
        }

        iterations_run = t + 1;

        // Exploitability check every 5 iterations
        if iterations_run % 5 == 0 || iterations_run == config.max_iterations {
            exploitability = compute_exploitability_gpu(
                &stream, &kernels, &state, topo, term,
                initial_weights, num_hands, &sorted_edges,
            )?;

            if config.print_progress {
                eprintln!("iteration: {} / {} (exploitability = {:.4e})",
                    iterations_run, config.max_iterations, exploitability);
            }

            if exploitability <= config.target_exploitability { break; }
        }
    }

    // Download and normalize root strategy
    let root_strategy = extract_root_strategy(&stream, &state, topo, num_hands)?;

    Ok(crate::GpuSolveResult {
        exploitability,
        iterations_run,
        root_strategy,
    })
}
```

The `evaluate_terminals_at_depth`, `compute_exploitability_gpu`, and `extract_root_strategy` helper functions follow the same pattern — launching the appropriate kernels with the right arguments.

**Step 3: Run tests**

Run: `cargo test -p gpu-range-solver -- cudarc_solve_matches_cpu`
Expected: PASS

**Step 4: Commit**

```bash
git commit -m "feat(gpu-range-solver): cudarc DCFR solver with custom CUDA kernels"
```

---

## Task 5: Terminal Evaluation Kernel Wrappers

Implement the Rust-side wrappers that upload terminal node data and launch fold/showdown kernels.

**Files:**
- Rewrite: `crates/gpu-range-solver/src/terminal.rs`

**Step 1: Implement fold/showdown launch wrappers**

```rust
//! Terminal evaluation: launch fold_eval and showdown_eval CUDA kernels.

use crate::gpu::{CfrKernels, GpuSolverState, grid_size, BLOCK_SIZE};
use crate::extract::{FoldData, ShowdownData, TerminalData};
use cudarc::driver::{CudaStream, LaunchConfig};

/// Launch fold evaluation kernel for one terminal node.
pub fn launch_fold_eval(
    stream: &CudaStream,
    kernels: &CfrKernels,
    state: &mut GpuSolverState,
    node_id: usize,
    payoff: f32,
    player_card1: &cudarc::driver::CudaSlice<i32>,
    player_card2: &cudarc::driver::CudaSlice<i32>,
    opp_card1: &cudarc::driver::CudaSlice<i32>,
    opp_card2: &cudarc::driver::CudaSlice<i32>,
    same_hand_idx: &cudarc::driver::CudaSlice<i32>,
    num_player_hands: usize,
    num_opp_hands: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // fold_eval uses shared memory (52 floats + 1 float = 212 bytes)
    // and needs at least max(num_opp_hands, num_player_hands) threads
    let block = num_opp_hands.max(num_player_hands).max(64).min(1024) as u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0, // __shared__ is statically allocated in the kernel
    };

    unsafe {
        let mut b = stream.launch_builder(&kernels.fold_eval);
        b.arg(&mut state.cfv);
        b.arg(&state.reach);
        b.arg(&(node_id as i32));
        b.arg(&payoff);
        b.arg(player_card1);
        b.arg(player_card2);
        b.arg(opp_card1);
        b.arg(opp_card2);
        b.arg(same_hand_idx);
        b.arg(&(num_player_hands as i32));
        b.arg(&(num_opp_hands as i32));
        b.arg(&(state.num_hands as i32));
        b.launch(cfg)?;
    }

    Ok(())
}

/// Launch showdown evaluation kernel for one terminal node.
pub fn launch_showdown_eval(
    stream: &CudaStream,
    kernels: &CfrKernels,
    state: &mut GpuSolverState,
    node_id: usize,
    outcome_gpu: &cudarc::driver::CudaSlice<f32>,
    amount_win: f32,
    amount_lose: f32,
    num_player_hands: usize,
    num_opp_hands: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = LaunchConfig {
        grid_dim: (grid_size(num_player_hands, BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut b = stream.launch_builder(&kernels.showdown_eval);
        b.arg(&mut state.cfv);
        b.arg(&state.reach);
        b.arg(&(node_id as i32));
        b.arg(outcome_gpu);
        b.arg(&amount_win);
        b.arg(&amount_lose);
        b.arg(&(num_player_hands as i32));
        b.arg(&(num_opp_hands as i32));
        b.arg(&(state.num_hands as i32));
        b.launch(cfg)?;
    }

    Ok(())
}
```

**Step 2: Commit**

```bash
git commit -m "feat(gpu-range-solver): terminal evaluation CUDA kernel wrappers"
```

---

## Task 6: Wire Up lib.rs and Update Module Structure

Replace the burn-based `gpu_solve_game` with the cudarc implementation. Delete `tensors.rs`.

**Files:**
- Modify: `crates/gpu-range-solver/src/lib.rs`
- Delete: `crates/gpu-range-solver/src/tensors.rs`

**Step 1: Update lib.rs**

```rust
pub mod extract;
pub mod gpu;
pub mod kernels;
pub mod solver;
pub mod terminal;

pub struct GpuSolverConfig {
    pub max_iterations: u32,
    pub target_exploitability: f32,
    pub print_progress: bool,
}

pub struct GpuSolveResult {
    pub exploitability: f32,
    pub iterations_run: u32,
    pub root_strategy: Vec<f32>,
}

/// Solve a postflop game on GPU and return the result.
pub fn gpu_solve_game(
    game: &range_solver::PostFlopGame,
    config: &GpuSolverConfig,
) -> GpuSolveResult {
    use range_solver::interface::Game;

    let topo = extract::extract_topology(game);
    let term = extract::extract_terminal_data(game, &topo);
    let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
    let initial_weights: [Vec<f32>; 2] = [
        game.initial_weights(0).to_vec(),
        game.initial_weights(1).to_vec(),
    ];

    solver::gpu_solve_cudarc(&topo, &term, config, &initial_weights, num_hands)
        .expect("GPU solve failed")
}
```

**Step 2: Delete tensors.rs**

```bash
rm crates/gpu-range-solver/src/tensors.rs
```

**Step 3: Run all tests**

Run: `cargo test -p gpu-range-solver`
Expected: All integration tests pass (gpu_solve_river_game_reduces_exploitability, gpu_solve_matches_cpu_convergence, etc.)

**Step 4: Commit**

```bash
git commit -m "feat(gpu-range-solver): wire cudarc solver into public API, delete burn tensors"
```

---

## Task 7: Benchmark and Verify

**Step 1: Benchmark**

```bash
time cargo run -p poker-solver-trainer --release -- gpu-range-solve \
  --oop-range "QQ+,AKs" --ip-range "JJ-99,AQs" \
  --flop "Qs Jh 2c" --turn "8d" --river "3s" \
  --pot 100 --effective-stack 100 --iterations 500

time cargo run -p poker-solver-trainer --release -- range-solve \
  --oop-range "QQ+,AKs" --ip-range "JJ-99,AQs" \
  --flop "Qs Jh 2c" --turn "8d" --river "3s" \
  --pot 100 --effective-stack 100 --iterations 500
```

Target: GPU ≤ 0.5s (vs CPU 0.3s). For wider ranges GPU should be faster.

**Step 2: Test with wider ranges**

```bash
time cargo run -p poker-solver-trainer --release -- gpu-range-solve \
  --oop-range "22+,A2s+,K9s+,Q9s+,ATo+,KTo+" \
  --ip-range "22+,A2s+,K2s+,Q5s+,A2o+,K5o+" \
  --flop "Qs Jh 2c" --turn "8d" --river "3s" \
  --pot 100 --effective-stack 100 --iterations 500
```

GPU should win on this wider-range benchmark.

**Step 3: Run full test suite**

Run: `cargo test`
Expected: All tests pass

**Step 4: Clippy**

Run: `cargo clippy -p gpu-range-solver`
Expected: Clean (allow unsafe blocks)

**Step 5: Commit**

```bash
git commit -m "perf(gpu-range-solver): cudarc solver verified — matches CPU convergence"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Drop burn, add cudarc | Cargo.toml |
| 2 | CUDA kernel source | kernels.rs (new) |
| 3 | GPU device management | gpu.rs (new) |
| 4 | Solver iteration loop | solver.rs (rewrite) |
| 5 | Terminal eval wrappers | terminal.rs (rewrite) |
| 6 | Wire up lib.rs | lib.rs (update), tensors.rs (delete) |
| 7 | Benchmark + verify | CLI tests |

**Critical path:** 1 → 2 → 3 → 4+5 → 6 → 7. Tasks 4 and 5 can be done in parallel once 3 is done.

**Total CUDA kernel launches per iteration:** ~25 (vs ~600 burn ops). Expected speedup: **10-50x** over burn, competitive with CPU on small problems, faster on large.
