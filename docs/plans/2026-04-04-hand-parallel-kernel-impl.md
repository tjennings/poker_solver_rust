# Hand-Parallel GPU Kernel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the GPU solver kernel from node-parallel (grid.sync, 278x slower than CPU) to hand-parallel (one thread block = one subgame, __syncthreads only, targeting 10-30x faster than CPU on warm solves).

**Architecture:** One CUDA thread block processes an entire game tree sequentially, with 1024 threads covering 1024 hands in parallel. Tree topology in shared memory (~7KB). Solver state in global memory with coalesced access. Multiple blocks = batched board solves. No cooperative groups, no grid.sync.

**Tech Stack:** Rust, cudarc 0.19, custom CUDA kernel (nvrtc), RTX 6000 Ada (CC 8.9, 142 SMs)

**Design doc:** `docs/plans/2026-04-04-hand-parallel-kernel-design.md`

---

## Task 1: Hand-Parallel CUDA Kernel

Write the complete hand-parallel `cfr_solve` kernel. This replaces ALL existing kernels (zero, regret_match, forward, backward, fold_eval, showdown_eval, etc.) with a single kernel that runs the full DCFR iteration loop.

**Files:**
- Rewrite: `crates/gpu-range-solver/src/kernels.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn hand_parallel_kernel_compiles() {
    let ptx = cudarc::nvrtc::compile_ptx(HAND_PARALLEL_KERNEL_SOURCE);
    assert!(ptx.is_ok(), "Hand-parallel kernel must compile: {:?}", ptx.err());
}

#[test]
fn hand_parallel_kernel_has_cfr_solve() {
    assert!(HAND_PARALLEL_KERNEL_SOURCE.contains("cfr_solve"),
        "must contain cfr_solve entry point");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p gpu-range-solver -- hand_parallel_kernel`
Expected: FAIL — `HAND_PARALLEL_KERNEL_SOURCE` doesn't exist

**Step 3: Write the kernel**

Replace the entire `CFR_KERNELS_SOURCE` with `HAND_PARALLEL_KERNEL_SOURCE`. The kernel structure:

```cuda
extern "C" __global__ void cfr_solve(
    // Per-block solver state in global memory
    float* regrets,           // [B * E * H]
    float* strategy_sum,      // [B * E * H]
    float* reach,             // [B * N * H]
    float* cfv,               // [B * N * H]

    // Topology (shared across blocks, loaded into shared mem)
    const short* edge_parent,   // [E] — i16
    const short* edge_child,    // [E] — i16
    const signed char* edge_player, // [E] — i8 (0=p0, 1=p1, 2=chance)
    const float* actions_per_node,  // [N] — f32 (num actions per node, 0 for terminals)
    const int* level_starts,    // [max_depth+1]
    const int* level_counts,    // [max_depth+1]

    // Node classification (CPU-built, for sequential traversal)
    const signed char* node_is_fold,    // [N] — 1 if fold terminal
    const signed char* node_is_showdown, // [N] — 1 if showdown terminal
    const float* fold_payoffs_p0,   // [N] — payoff when p0 traverses (0 for non-fold)
    const float* fold_payoffs_p1,   // [N] — payoff when p1 traverses

    // Showdown outcome matrices [B * num_showdowns * H_opp_max]
    // (packed per-block, per-showdown, per-player-hand: win_reach + lose_reach approach)
    // Actually simpler: per-block, per-showdown-node, full [H_player * H_opp] outcome matrix
    const float* showdown_outcomes, // [B * total_showdown_elements]
    const int* showdown_node_ids,   // [num_showdowns]
    const int* showdown_offsets,    // [B * num_showdowns] — offset into showdown_outcomes per block
    const int* showdown_num_p,      // [num_showdowns * 2] — (num_player_h, num_opp_h) per showdown
    int num_showdowns,

    // Card data for fold eval
    const int* player_card1,    // [2 * H] — card1 per hand, player 0 then player 1
    const int* player_card2,    // [2 * H]
    const int* same_hand_idx,   // [2 * H] — opp index or -1

    // Initial weights per block
    const float* initial_weights, // [B * 2 * H]

    // Dimensions
    int B, int N, int E, int H,
    int max_depth,
    int max_iterations,

    // Per-hand number of hands per player (for fold eval bounds)
    int num_hands_p0, int num_hands_p1
) {
    int bid = blockIdx.x;    // which board
    int tid = threadIdx.x;   // which hand
    // H = blockDim.x

    int EH = E * H;
    int NH = N * H;

    // ============================================
    // Load topology into shared memory
    // ============================================
    extern __shared__ char shared_mem[];
    short* s_parent = (short*)shared_mem;
    short* s_child = s_parent + E;
    signed char* s_player = (signed char*)(s_child + E);
    // ... (level_starts, level_counts, fold/showdown info)

    // Cooperative load: each thread loads a few elements
    for (int i = tid; i < E; i += blockDim.x) {
        s_parent[i] = edge_parent[i];
        s_child[i] = edge_child[i];
        s_player[i] = edge_player[i];
    }
    // ... load level_starts, level_counts, node info ...
    __syncthreads();

    // Scratch for fold eval card blocking
    __shared__ float s_card_reach[52];
    __shared__ float s_total_reach;

    // ============================================
    // DCFR iteration loop
    // ============================================
    for (int iter = 0; iter < max_iterations; iter++) {

        // --- Compute DCFR discount params ---
        float alpha, beta, gamma;
        {
            float ta = (float)max(iter - 1, 0);
            float pa = ta * sqrtf(ta);
            alpha = pa / (pa + 1.0f);
            beta = 0.5f;
            int nearest_p4 = (iter == 0) ? 0 : (1 << ((31 - __clz(iter)) & ~1));
            float tg = (float)(iter - nearest_p4);
            float gb = tg / (tg + 1.0f);
            gamma = gb * gb * gb;
        }

        // --- DCFR discount: per-thread, no sync ---
        for (int e = 0; e < E; e++) {
            int idx = bid * EH + e * H + tid;
            float r = regrets[idx];
            regrets[idx] = (r >= 0.0f) ? r * alpha : r * beta;
            strategy_sum[idx] *= gamma;
        }

        // --- Alternating player updates ---
        for (int player = 0; player < 2; player++) {
            int opp = 1 - player;
            int num_ph = (player == 0) ? num_hands_p0 : num_hands_p1;
            int num_oh = (player == 0) ? num_hands_p1 : num_hands_p0;

            // --- Zero reach and cfv: per-thread ---
            for (int n = 0; n < N; n++) {
                reach[bid * NH + n * H + tid] = 0.0f;
                cfv[bid * NH + n * H + tid] = 0.0f;
            }

            // --- Set root reach = opponent initial weights ---
            reach[bid * NH + tid] = initial_weights[bid * 2 * H + opp * H + tid];

            // --- Regret match: per-thread, loop over nodes ---
            // Strategy stored in-register or in a scratch area in global mem
            // Actually, we compute strategy on-the-fly during forward/backward
            // to avoid needing a separate strategy array.
            // OR: compute strategy into a scratch region of global memory.
            // For simplicity, use a global strategy array:
            // (regrets and strategy share the E*H layout)

            // We'll compute strategy just-in-time at each node during traversal.

            // --- Forward pass: sequential over levels, parallel over hands ---
            for (int depth = 0; depth <= max_depth; depth++) {
                int start = s_level_starts[depth];
                int count = s_level_counts[depth];

                // For each node at this depth, compute strategy then propagate
                // Group edges by parent node
                int e = start;
                while (e < start + count) {
                    int parent = s_parent[e];
                    int n_actions = (int)s_actions_per_node[parent];
                    if (n_actions == 0) { e++; continue; }

                    // Regret match for this node (per-thread, no sync)
                    float denom = 0.0f;
                    for (int a = 0; a < n_actions; a++) {
                        denom += fmaxf(regrets[bid * EH + (e + a) * H + tid], 0.0f);
                    }
                    float uniform = 1.0f / (float)n_actions;

                    for (int a = 0; a < n_actions; a++) {
                        int edge = e + a;
                        int child = s_child[edge];
                        float clipped = fmaxf(regrets[bid * EH + edge * H + tid], 0.0f);
                        float strat = (denom > 1e-30f) ? clipped / denom : uniform;

                        float pr = reach[bid * NH + parent * H + tid];

                        if (s_player[edge] == player) {
                            // Traverser: counterfactual (no strategy multiply)
                            reach[bid * NH + child * H + tid] = pr;
                        } else {
                            // Opponent: multiply by strategy
                            reach[bid * NH + child * H + tid] = pr * strat;
                        }
                    }
                    e += n_actions;
                }
                // No __syncthreads needed — each thread writes its own hand
            }

            // --- Backward pass: sequential over levels (reverse) ---
            for (int depth = max_depth; depth >= 0; depth--) {

                // --- Terminal eval at this depth ---
                // Scan nodes at this depth for fold/showdown
                // (Use level_nodes or iterate all nodes — simpler to check node_depth)
                for (int n_idx = 0; n_idx < N; n_idx++) {
                    // Check if node is at this depth and is terminal
                    // (node_depth info needed in shared mem or computed from topology)

                    if (node_is_fold[n_idx] && node_depth_matches(n_idx, depth)) {
                        // --- FOLD EVAL (3 syncs) ---
                        float payoff = (player == 0) ? fold_payoffs_p0[n_idx] : fold_payoffs_p1[n_idx];
                        if (payoff == 0.0f) continue; // not a fold node

                        if (tid < 52) s_card_reach[tid] = 0.0f;
                        if (tid == 0) s_total_reach = 0.0f;
                        __syncthreads();

                        if (tid < num_oh) {
                            float r = reach[bid * NH + n_idx * H + tid];
                            atomicAdd(&s_total_reach, r);
                            int ob = opp * H;
                            atomicAdd(&s_card_reach[player_card1[ob + tid]], r);
                            atomicAdd(&s_card_reach[player_card2[ob + tid]], r);
                        }
                        __syncthreads();

                        if (tid < num_ph) {
                            int pb = player * H;
                            int c1 = player_card1[pb + tid];
                            int c2 = player_card2[pb + tid];
                            float blocking = s_card_reach[c1] + s_card_reach[c2];
                            int same = same_hand_idx[pb + tid];
                            if (same >= 0)
                                blocking -= reach[bid * NH + n_idx * H + same];
                            cfv[bid * NH + n_idx * H + tid] = payoff * (s_total_reach - blocking);
                        }
                        __syncthreads();
                    }

                    if (node_is_showdown[n_idx] && node_depth_matches(n_idx, depth)) {
                        // --- SHOWDOWN EVAL (no sync) ---
                        // Find this showdown's index and outcome data
                        // Each thread (player hand) loops over opponent hands
                        if (tid < num_ph) {
                            // ... lookup outcome matrix for this block + showdown node
                            float val = 0.0f;
                            for (int oh = 0; oh < num_oh; oh++) {
                                float opp_r = reach[bid * NH + n_idx * H + oh];
                                float outcome_val = /* outcome[tid * num_oh + oh] */;
                                val += outcome_val * opp_r;
                            }
                            cfv[bid * NH + n_idx * H + tid] = val;
                        }
                    }
                }

                // --- CFV accumulation + regret update ---
                int start = s_level_starts[depth];
                int count = s_level_counts[depth];

                int e = start;
                while (e < start + count) {
                    int parent = s_parent[e];
                    int n_actions = (int)s_actions_per_node[parent];
                    if (n_actions == 0) { e++; continue; }

                    // Recompute strategy (same as forward pass)
                    float denom = 0.0f;
                    for (int a = 0; a < n_actions; a++)
                        denom += fmaxf(regrets[bid * EH + (e + a) * H + tid], 0.0f);
                    float uniform = 1.0f / (float)n_actions;

                    if (s_player[e] == player) {
                        // Traverser: cfv[parent] = sum(strat * child_cfv)
                        float node_cfv = 0.0f;
                        for (int a = 0; a < n_actions; a++) {
                            float clipped = fmaxf(regrets[bid * EH + (e + a) * H + tid], 0.0f);
                            float strat = (denom > 1e-30f) ? clipped / denom : uniform;
                            float child_cfv = cfv[bid * NH + s_child[e + a] * H + tid];
                            node_cfv += strat * child_cfv;
                        }
                        cfv[bid * NH + parent * H + tid] = node_cfv;

                        // Regret + strategy_sum update
                        for (int a = 0; a < n_actions; a++) {
                            float clipped = fmaxf(regrets[bid * EH + (e + a) * H + tid], 0.0f);
                            float strat = (denom > 1e-30f) ? clipped / denom : uniform;
                            float child_cfv = cfv[bid * NH + s_child[e + a] * H + tid];
                            int idx = bid * EH + (e + a) * H + tid;
                            regrets[idx] += child_cfv - node_cfv;
                            strategy_sum[idx] += strat;
                        }
                    } else {
                        // Opponent/chance: cfv[parent] = sum(child_cfv)
                        float node_cfv = 0.0f;
                        for (int a = 0; a < n_actions; a++)
                            node_cfv += cfv[bid * NH + s_child[e + a] * H + tid];
                        cfv[bid * NH + parent * H + tid] = node_cfv;
                    }
                    e += n_actions;
                }
                // No sync needed
            }
        } // end player loop
    } // end iteration loop
}
```

**Important design choices in the kernel:**
- Strategy is computed on-the-fly (not stored in a separate array). This saves global memory and an extra pass. Strategy is needed in both forward and backward — recompute it from regrets each time (cheap: just clamp+divide).
- Node depth matching for terminal eval: either store node_depth in shared memory, or iterate nodes at each level. Simplest: store a level_nodes array in shared memory listing which nodes are at each depth.
- The edge ordering must group edges by parent node (all actions of a node are contiguous). This is already the case in `extract.rs`.

**Step 4: Run test**

Run: `cargo test -p gpu-range-solver -- hand_parallel_kernel`
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat(gpu-range-solver): hand-parallel CUDA kernel (Supremus architecture)"
```

---

## Task 2: Simplified GPU State

Update `gpu.rs`: remove `CfrKernels` (11 functions), `MegaKernel`, cooperative launch infrastructure. Replace with a single `HandParallelKernel` holding one `CudaFunction`, and a simplified `GpuState` with batched memory layout.

**Files:**
- Rewrite: `crates/gpu-range-solver/src/gpu.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn hand_parallel_kernel_loads() {
    let ctx = CudaContext::new(0).unwrap();
    let kernel = HandParallelKernel::compile(&ctx);
    assert!(kernel.is_ok(), "kernel compile failed: {:?}", kernel.err());
}

#[test]
fn gpu_state_batched_allocation() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    // B=4 blocks, N=10 nodes, E=15 edges, H=8 hands
    let state = GpuState::new(&stream, 4, 10, 15, 8);
    assert!(state.is_ok());
}
```

**Step 2: Implement**

```rust
pub struct HandParallelKernel {
    pub cfr_solve: CudaFunction,
}

impl HandParallelKernel {
    pub fn compile(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(crate::kernels::HAND_PARALLEL_KERNEL_SOURCE)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            cfr_solve: module.load_function("cfr_solve")?,
        })
    }
}

pub struct GpuState {
    // Solver state [B * E * H] and [B * N * H]
    pub regrets: CudaSlice<f32>,
    pub strategy_sum: CudaSlice<f32>,
    pub reach: CudaSlice<f32>,
    pub cfv: CudaSlice<f32>,
    // Dimensions
    pub batch_size: usize,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub num_hands: usize,
}

impl GpuState {
    pub fn new(stream: &Arc<CudaStream>, batch: usize, nodes: usize, edges: usize, hands: usize)
        -> Result<Self, Box<dyn std::error::Error>>
    {
        Ok(Self {
            regrets: stream.alloc_zeros::<f32>((batch * edges * hands).max(1))?,
            strategy_sum: stream.alloc_zeros::<f32>((batch * edges * hands).max(1))?,
            reach: stream.alloc_zeros::<f32>((batch * nodes * hands).max(1))?,
            cfv: stream.alloc_zeros::<f32>((batch * nodes * hands).max(1))?,
            batch_size: batch,
            num_nodes: nodes,
            num_edges: edges,
            num_hands: hands,
        })
    }
}
```

**Step 3: Commit**

```bash
git commit -m "feat(gpu-range-solver): simplified GPU state for hand-parallel kernel"
```

---

## Task 3: Solver — Single Kernel Launch

Rewrite `solver.rs` to launch the hand-parallel kernel once per street. Delete `terminal.rs` (all terminal eval is inline in the kernel).

**Files:**
- Rewrite: `crates/gpu-range-solver/src/solver.rs`
- Delete: `crates/gpu-range-solver/src/terminal.rs`
- Modify: `crates/gpu-range-solver/src/lib.rs` (remove `pub mod terminal`)

**Step 1: Write the failing integration test**

```rust
#[test]
fn hand_parallel_solve_river() {
    let game = make_river_game();
    let config = GpuSolverConfig {
        max_iterations: 500,
        target_exploitability: 0.0,
        print_progress: false,
    };
    let result = gpu_solve_hand_parallel(&game, &config);
    assert!(result.exploitability.is_finite());
    assert!(result.exploitability < 0.01,
        "should converge near 0, got {}", result.exploitability);
}
```

**Step 2: Implement the solver**

The solver flow:
1. Extract topology (reuse `extract.rs`)
2. Build sorted edge arrays (group by parent depth, edges of same node contiguous)
3. Build terminal data (fold payoffs per node, showdown outcomes per block)
4. Build card data (player_card1/card2, same_hand_idx)
5. Upload everything to GPU
6. Launch `<<<num_blocks, num_hands>>> cfr_solve(...)` — standard launch, NOT cooperative
7. `stream.synchronize()`
8. Download strategy_sum, normalize on CPU → root strategy
9. Compute exploitability on CPU (best-response with average strategy)

```rust
pub fn gpu_solve_hand_parallel(
    game: &range_solver::PostFlopGame,
    config: &crate::GpuSolverConfig,
) -> crate::GpuSolveResult {
    use range_solver::interface::Game;

    let topo = crate::extract::extract_topology(game);
    let term = crate::extract::extract_terminal_data(game, &topo);
    let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let kernel = HandParallelKernel::compile(&ctx).unwrap();

    let batch_size = 1; // River: single board
    let mut state = GpuState::new(&stream, batch_size, topo.num_nodes, topo.num_edges, num_hands).unwrap();

    // Sort edges by parent depth, group by parent node
    let sorted = build_sorted_topology(&topo);

    // Upload topology as i16/i8 arrays
    let d_parent = stream.clone_htod(&sorted.edge_parent_i16).unwrap();
    let d_child = stream.clone_htod(&sorted.edge_child_i16).unwrap();
    let d_player = stream.clone_htod(&sorted.edge_player_i8).unwrap();
    // ... upload level_starts, level_counts, terminal data, card data, initial_weights ...

    // Calculate shared memory size
    let shared_mem_bytes = compute_shared_mem_size(topo.num_edges, topo.max_depth, topo.num_nodes);

    // Launch: <<<batch_size, num_hands, shared_mem_bytes>>>
    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (num_hands as u32, 1, 1),
        shared_mem_bytes,
    };

    unsafe {
        let mut b = stream.launch_builder(&kernel.cfr_solve);
        b.arg(&mut state.regrets);
        b.arg(&mut state.strategy_sum);
        b.arg(&mut state.reach);
        b.arg(&mut state.cfv);
        b.arg(&d_parent);
        b.arg(&d_child);
        b.arg(&d_player);
        // ... all other args ...
        b.arg(&(batch_size as i32));
        b.arg(&(topo.num_nodes as i32));
        b.arg(&(topo.num_edges as i32));
        b.arg(&(num_hands as i32));
        b.arg(&(topo.max_depth as i32));
        b.arg(&(config.max_iterations as i32));
        b.launch(cfg).unwrap();
    }

    stream.synchronize().unwrap();

    // Download and normalize
    let strategy_sum_host: Vec<f32> = stream.clone_dtoh(&state.strategy_sum).unwrap();
    let root_strategy = normalize_root_strategy(&strategy_sum_host, &topo, num_hands);

    // Exploitability: run CPU best-response
    let exploitability = compute_exploitability_cpu(game, &strategy_sum_host, &topo, num_hands);

    crate::GpuSolveResult {
        exploitability,
        iterations_run: config.max_iterations,
        root_strategy,
    }
}
```

**Step 3: Update lib.rs**

```rust
pub mod extract;
pub mod gpu;
pub mod kernels;
pub mod solver;
// pub mod terminal; — DELETED

pub fn gpu_solve_game(...) -> GpuSolveResult {
    solver::gpu_solve_hand_parallel(game, config)
}
```

**Step 4: Run tests**

Run: `cargo test -p gpu-range-solver`
Expected: All integration tests pass

**Step 5: Commit**

```bash
git commit -m "feat(gpu-range-solver): hand-parallel solver with single kernel launch"
```

---

## Task 4: Benchmark and Verify

**Step 1: River benchmark**

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

Target: GPU solve time < 0.1s (excluding CUDA init).

**Step 2: Wider ranges benchmark**

```bash
# Same with wider ranges (600 hands)
time cargo run -p poker-solver-trainer --release -- gpu-range-solve \
  --oop-range "22+,A2s+,K5s+,Q7s+,J8s+,T8s+,98s,A5o+,K9o+,Q9o+,JTo" \
  --ip-range "22+,A2s+,K2s+,Q5s+,J7s+,T7s+,97s+,87s,A2o+,K5o+,Q8o+,J8o+,T8o+,98o" \
  --flop "Qs Jh 2c" --turn "8d" --river "3s" \
  --pot 100 --effective-stack 100 --iterations 500
```

**Step 3: Verify output matches CPU**

Run both with `--target-exploitability 0` and compare strategy tables.

**Step 4: Clippy + full test suite**

```bash
cargo clippy -p gpu-range-solver
cargo test -p gpu-range-solver
```

**Step 5: Commit**

```bash
git commit -m "perf(gpu-range-solver): hand-parallel kernel verified — benchmark results"
```

---

## Summary

| Task | What | Key Change |
|------|------|-----------|
| 1 | Hand-parallel CUDA kernel | Replace 11 kernels with 1 hand-parallel cfr_solve |
| 2 | Simplified GPU state | Remove cooperative launch, batched [B×E×H] allocation |
| 3 | Single-launch solver | One kernel launch per street, delete terminal.rs |
| 4 | Benchmark | Verify GPU matches CPU output, measure speedup |

**Critical path:** 1 → 2 → 3 → 4 (all sequential).

**Expected outcome:** 1000x less sync overhead (300ns/iter vs 365μs/iter). GPU solve time dominated by compute, not sync — competitive with CPU on warm solves, faster on large problems.
