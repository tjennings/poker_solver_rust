# Persistent Batched CUDA Kernel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace 11 separate CUDA kernels with a single persistent cooperative mega-kernel that runs the entire DCFR solve loop, with a batch dimension for parallel runout processing. One kernel launch per street.

**Architecture:** A single `cfr_solve` kernel contains the iteration loop, forward/backward passes, regret matching, terminal evaluation, and DCFR discounting. Threads synchronize between phases via `cooperative_groups::grid_group::sync()`. Batch dimension B processes all board runouts simultaneously. Memory layout `[B × E × H]` and `[B × N × H]`.

**Tech Stack:** Rust, cudarc 0.19 (driver + nvrtc), CUDA cooperative groups, RTX 6000 Ada (CC 8.9, 142 SMs)

**Design doc:** `docs/plans/2026-04-03-persistent-batched-kernel-design.md`

---

## Task 1: Persistent Mega-Kernel with Batch Dimension

Replace the 11 separate kernels in `kernels.rs` with a single `cfr_solve` cooperative kernel. This is the core change — everything else follows from it.

**Files:**
- Rewrite: `crates/gpu-range-solver/src/kernels.rs`

**Step 1: Write the failing test**

Update the existing test in kernels.rs:

```rust
#[test]
fn kernel_source_compiles_with_nvrtc() {
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        CFR_KERNELS_SOURCE,
        cudarc::nvrtc::CompileOptions {
            arch: Some("sm_89"),
            include_paths: vec![],
            ..Default::default()
        },
    );
    assert!(ptx.is_ok(), "CUDA source must compile: {:?}", ptx.err());
}

#[test]
fn kernel_source_contains_cfr_solve() {
    assert!(CFR_KERNELS_SOURCE.contains("cfr_solve"),
        "kernel source must contain cfr_solve mega-kernel");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p gpu-range-solver -- kernel_source_contains_cfr_solve`
Expected: FAIL — old source doesn't contain `cfr_solve`

**Step 3: Write the persistent kernel**

Replace the entire `CFR_KERNELS_SOURCE` with the mega-kernel. The key structure:

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void cfr_solve(
    // Mutable state [B * E * H] and [B * N * H]
    float* regrets,
    float* strategy_sum,
    float* strategy,
    float* reach,
    float* cfv,
    float* denom,
    // Topology (shared across batch, constant)
    const int* edge_parent,    // [E]
    const int* edge_child,     // [E]
    const int* edge_player,    // [E]
    const float* actions_per_edge, // [E]
    const int* level_starts,   // [max_depth+1]
    const int* level_counts,   // [max_depth+1]
    // Terminal data
    const int* fold_node_ids,  // [num_folds]
    const float* fold_payoffs_p0, // [num_folds] — payoff when player 0 is traverser
    const float* fold_payoffs_p1, // [num_folds]
    const int* showdown_node_ids, // [num_showdowns]
    const float* showdown_outcomes_p0, // [num_showdowns * H * H] — packed outcome matrices
    const float* showdown_outcomes_p1,
    const int* showdown_num_player, // [num_showdowns * 2] — (num_p0, num_p1) per showdown
    // Card data for fold eval
    const int* player_card1,   // [2 * H] — [p0_cards..., p1_cards...]
    const int* player_card2,   // [2 * H]
    const int* opp_card1,      // [2 * H]
    const int* opp_card2,      // [2 * H]
    const int* same_hand_idx,  // [2 * H]
    // Initial weights
    const float* initial_weights, // [2 * H] — [p0_weights..., p1_weights...]
    // Dimensions
    int B, int N, int E, int H,
    int max_depth,
    int max_iterations,
    int num_folds,
    int num_showdowns,
    // Terminal node depths (for depth-matching during backward pass)
    const int* fold_depths,    // [num_folds]
    const int* showdown_depths // [num_showdowns]
) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int EH = E * H;
    int NH = N * H;
    int BEH = B * EH;
    int BNH = B * NH;

    for (int iter = 0; iter < max_iterations; iter++) {

        // === DCFR Discount ===
        // Compute alpha, beta, gamma from iteration number
        float t_alpha_f = (float)max(iter - 1, 0);
        float pow_alpha = t_alpha_f * sqrtf(t_alpha_f);
        float alpha = pow_alpha / (pow_alpha + 1.0f);
        float beta = 0.5f;
        // gamma uses nearest lower power of 4
        int nearest_p4 = (iter == 0) ? 0 : (1 << ((31 - __clz(iter)) & ~1));
        float t_gamma = (float)(iter - nearest_p4);
        float gamma_base = t_gamma / (t_gamma + 1.0f);
        float gamma = gamma_base * gamma_base * gamma_base;

        for (int i = tid; i < BEH; i += stride) {
            float r = regrets[i];
            regrets[i] = (r >= 0.0f) ? r * alpha : r * beta;
            strategy_sum[i] *= gamma;
        }
        grid.sync();

        // === Alternating player updates ===
        for (int player = 0; player < 2; player++) {
            int opp = 1 - player;

            // Zero scratch arrays
            for (int i = tid; i < BNH; i += stride) {
                reach[i] = 0.0f;
                cfv[i] = 0.0f;
                denom[i] = 0.0f;
            }
            for (int i = tid; i < BEH; i += stride) {
                strategy[i] = 0.0f;
            }
            grid.sync();

            // Set root reach = opponent initial weights
            for (int i = tid; i < B * H; i += stride) {
                int b = i / H;
                int h = i % H;
                reach[b * NH + h] = initial_weights[opp * H + h];
            }
            grid.sync();

            // === Regret match phase 1: accumulate ===
            for (int i = tid; i < BEH; i += stride) {
                int b = i / EH;
                int local = i % EH;
                int e = local / H;
                int h = local % H;
                float clipped = fmaxf(regrets[i], 0.0f);
                if (clipped > 0.0f) {
                    atomicAdd(&denom[b * NH + edge_parent[e] * H + h], clipped);
                }
            }
            grid.sync();

            // === Regret match phase 2: normalize ===
            for (int i = tid; i < BEH; i += stride) {
                int b = i / EH;
                int local = i % EH;
                int e = local / H;
                int h = local % H;
                float clipped = fmaxf(regrets[i], 0.0f);
                float d = denom[b * NH + edge_parent[e] * H + h];
                strategy[i] = (d > 1e-30f) ? (clipped / d) : (1.0f / actions_per_edge[e]);
            }
            grid.sync();

            // === Forward pass: level by level ===
            for (int depth = 0; depth <= max_depth; depth++) {
                int start = level_starts[depth];
                int count = level_counts[depth];
                for (int i = tid; i < B * count * H; i += stride) {
                    int b = i / (count * H);
                    int local = i % (count * H);
                    int e = start + local / H;
                    int h = local % H;
                    int p = edge_parent[e];
                    int c = edge_child[e];
                    float pr = reach[b * NH + p * H + h];

                    if (edge_player[e] == player) {
                        reach[b * NH + c * H + h] = pr;
                    } else if (edge_player[e] == 2) {
                        reach[b * NH + c * H + h] = pr; // chance: caller handles factor
                    } else {
                        reach[b * NH + c * H + h] = pr * strategy[b * EH + e * H + h];
                    }
                }
                grid.sync();
            }

            // === Backward pass: level by level (reverse) ===
            for (int depth = max_depth; depth >= 0; depth--) {

                // --- Fold terminal evaluation ---
                for (int fi = 0; fi < num_folds; fi++) {
                    if (fold_depths[fi] != depth) continue;
                    int node_id = fold_node_ids[fi];
                    float payoff = (player == 0) ? fold_payoffs_p0[fi] : fold_payoffs_p1[fi];

                    // Card blocking: shared memory approach
                    // Each batch element processes independently
                    // (simplified: use global memory atomics for batch support)
                    int num_opp_h = H; // use H as upper bound
                    int num_player_h = H;

                    // Per-batch fold eval
                    for (int bi = tid; bi < B; bi += stride) {
                        // Compute total opp reach and card_reach[52] for this batch
                        float total_r = 0.0f;
                        float card_r[52];
                        for (int c = 0; c < 52; c++) card_r[c] = 0.0f;

                        int opp_base = opp * H;
                        for (int oh = 0; oh < num_opp_h; oh++) {
                            float r = reach[bi * NH + node_id * H + oh];
                            if (r != 0.0f) {
                                total_r += r;
                                card_r[opp_card1[opp_base + oh]] += r;
                                card_r[opp_card2[opp_base + oh]] += r;
                            }
                        }

                        int player_base = player * H;
                        for (int ph = 0; ph < num_player_h; ph++) {
                            int c1 = player_card1[player_base + ph];
                            int c2 = player_card2[player_base + ph];
                            float blocking = card_r[c1] + card_r[c2];
                            int same = same_hand_idx[player_base + ph];
                            if (same >= 0) {
                                blocking -= reach[bi * NH + node_id * H + same];
                            }
                            cfv[bi * NH + node_id * H + ph] = payoff * (total_r - blocking);
                        }
                    }
                }
                grid.sync();

                // --- Showdown terminal evaluation ---
                for (int si = 0; si < num_showdowns; si++) {
                    if (showdown_depths[si] != depth) continue;
                    int node_id = showdown_node_ids[si];
                    int num_ph = showdown_num_player[si * 2 + player];
                    int num_oh = showdown_num_player[si * 2 + opp];
                    const float* outcome = (player == 0)
                        ? &showdown_outcomes_p0[si * H * H]
                        : &showdown_outcomes_p1[si * H * H];

                    for (int i = tid; i < B * num_ph; i += stride) {
                        int b = i / num_ph;
                        int h = i % num_ph;
                        float win_sum = 0.0f, lose_sum = 0.0f;
                        for (int oh = 0; oh < num_oh; oh++) {
                            float opp_r = reach[b * NH + node_id * H + oh];
                            float o = outcome[h * num_oh + oh];
                            if (o > 0.0f) win_sum += opp_r;
                            else if (o < 0.0f) lose_sum += opp_r;
                        }
                        // amount_win/lose encoded in outcome or passed separately
                        // For now: outcomes are pre-scaled
                        cfv[b * NH + node_id * H + h] = win_sum + lose_sum;
                    }
                }
                grid.sync();

                // --- Scatter-add child CFVs to parents ---
                int start = level_starts[depth];
                int count = level_counts[depth];
                for (int i = tid; i < B * count * H; i += stride) {
                    int b = i / (count * H);
                    int local = i % (count * H);
                    int e = start + local / H;
                    int h = local % H;
                    int p = edge_parent[e];
                    int c = edge_child[e];
                    float cv = cfv[b * NH + c * H + h];

                    if (edge_player[e] == player) {
                        atomicAdd(&cfv[b * NH + p * H + h], strategy[b * EH + e * H + h] * cv);
                    } else {
                        atomicAdd(&cfv[b * NH + p * H + h], cv);
                    }
                }
                grid.sync();

                // --- Regret update (traverser edges only) ---
                for (int i = tid; i < B * count * H; i += stride) {
                    int b = i / (count * H);
                    int local = i % (count * H);
                    int e = start + local / H;
                    int h = local % H;
                    if (edge_player[e] != player) continue;
                    int p = edge_parent[e];
                    int c = edge_child[e];
                    int idx = b * EH + e * H + h;
                    regrets[idx] += cfv[b * NH + c * H + h] - cfv[b * NH + p * H + h];
                    strategy_sum[idx] += strategy[idx];
                }
                grid.sync();
            } // end backward depth loop
        } // end player loop
    } // end iteration loop
}
```

The kernel also needs a small helper for exploitability (not in the main loop — run as a separate cooperative launch after the solve, or compute on CPU).

**Important nvrtc compile flags:** Must pass `--gpu-architecture=sm_89` for cooperative groups support. Use `cudarc::nvrtc::compile_ptx_with_opts` if available, or pass raw nvrtc options.

**Step 4: Run test**

Run: `cargo test -p gpu-range-solver -- kernel_source_compiles`
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat(gpu-range-solver): persistent cooperative mega-kernel with batch dimension"
```

---

## Task 2: Batch-Dimensioned GPU State

Update `gpu.rs` to allocate `[B × E × H]` and `[B × N × H]` arrays. Add cooperative launch infrastructure.

**Files:**
- Modify: `crates/gpu-range-solver/src/gpu.rs`

**Step 1: Update GpuSolverState**

Add `batch_size: usize` field. All mutable arrays become `batch_size * dim`:

```rust
impl GpuSolverState {
    pub fn new(
        _ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        batch_size: usize,
        num_nodes: usize,
        num_edges: usize,
        num_hands: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let beh = (batch_size * num_edges * num_hands).max(1);
        let bnh = (batch_size * num_nodes * num_hands).max(1);
        Ok(Self {
            // ... topology arrays stay [E], [N] (shared across batch)
            regrets: stream.alloc_zeros::<f32>(beh)?,
            strategy_sum: stream.alloc_zeros::<f32>(beh)?,
            strategy: stream.alloc_zeros::<f32>(beh)?,
            reach: stream.alloc_zeros::<f32>(bnh)?,
            cfv: stream.alloc_zeros::<f32>(bnh)?,
            denom: stream.alloc_zeros::<f32>(bnh)?,
            batch_size,
            // ...
        })
    }
}
```

Add a `CfrKernels` method for cooperative launch:

```rust
impl CfrKernels {
    pub fn compile(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            crate::kernels::CFR_KERNELS_SOURCE,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            cfr_solve: module.load_function("cfr_solve")?,
        })
    }
}
```

Add helper for querying max cooperative blocks:

```rust
pub fn max_cooperative_blocks(func: &CudaFunction, block_size: u32) -> u32 {
    // Use CUDA occupancy API to determine max blocks for cooperative launch
    // Fallback: 142 (RTX 6000 Ada SM count)
    142
}
```

**Step 2: Commit**

```bash
git commit -m "feat(gpu-range-solver): batch-dimensioned GPU state + cooperative launch"
```

---

## Task 3: Rewrite Solver with Single Cooperative Launch

Replace the per-iteration multi-launch solver loop with a single cooperative kernel launch per street.

**Files:**
- Rewrite: `crates/gpu-range-solver/src/solver.rs`

**Step 1: Implement the new solver**

The solver becomes dramatically simpler:

```rust
pub fn gpu_solve_cudarc(
    topo: &TreeTopology,
    term: &TerminalData,
    config: &crate::GpuSolverConfig,
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
) -> Result<crate::GpuSolveResult, Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let kernels = CfrKernels::compile(&ctx)?;

    let batch_size = 1; // river: B=1, turn: B=44 (computed from topology)
    let mut state = GpuSolverState::new(&ctx, &stream, batch_size, topo.num_nodes, topo.num_edges, num_hands)?;

    // Upload topology
    prepare_topology(topo, &stream, &mut state)?;

    // Upload terminal data (fold payoffs, showdown outcomes, card indices)
    // as flat arrays matching the kernel's parameter expectations
    upload_all_terminal_data(&stream, topo, term, &mut state, num_hands)?;

    // Upload initial weights [2 * H]
    let mut weights_flat = vec![0.0f32; 2 * num_hands];
    for p in 0..2 {
        for (h, &w) in initial_weights[p].iter().enumerate() {
            if h < num_hands { weights_flat[p * num_hands + h] = w; }
        }
    }
    let initial_weights_gpu = stream.clone_htod(&weights_flat)?;

    // Launch ONE cooperative kernel for the entire solve
    let max_blocks = max_cooperative_blocks(&kernels.cfr_solve, BLOCK_SIZE);
    unsafe {
        cudarc::driver::sys::cuLaunchCooperativeKernel(
            kernels.cfr_solve.cu_function,
            max_blocks, 1, 1,
            BLOCK_SIZE, 1, 1,
            0, // shared mem
            stream.cu_stream,
            kernel_params![
                &mut state.regrets, &mut state.strategy_sum, &mut state.strategy,
                &mut state.reach, &mut state.cfv, &mut state.denom,
                &state.edge_parent, &state.edge_child, &state.edge_player,
                &state.actions_per_edge, &state.level_starts_gpu, &state.level_counts_gpu,
                // ... terminal data pointers ...
                // ... card data pointers ...
                &initial_weights_gpu,
                &(batch_size as i32), &(topo.num_nodes as i32),
                &(topo.num_edges as i32), &(num_hands as i32),
                &(topo.max_depth as i32), &(config.max_iterations as i32),
                // ... num_folds, num_showdowns ...
            ].as_mut_ptr(),
        );
    }
    stream.synchronize()?;

    // Download strategy_sum, normalize on CPU to get root strategy
    let root_strategy = extract_root_strategy(&stream, &state, topo, num_hands)?;

    // Compute exploitability on CPU (download regrets + strategy_sum, run best-response)
    let exploitability = compute_exploitability_cpu(/* ... */)?;

    Ok(crate::GpuSolveResult {
        exploitability,
        iterations_run: config.max_iterations,
        root_strategy,
    })
}
```

**Note on `cuLaunchCooperativeKernel` parameters:** cudarc's sys binding expects raw `*mut c_void` for kernel parameters. The exact parameter marshaling requires building an array of pointers to each argument. This is the trickiest part — each argument must be a pointer to the device pointer (for arrays) or pointer to the scalar (for ints/floats).

**Building kernel params for cudarc sys:**

```rust
// For each CudaSlice<T> argument, pass a pointer to its device pointer.
// For scalar arguments, pass a pointer to the scalar value.
let mut params: Vec<*mut std::ffi::c_void> = Vec::new();

// Device array: pass pointer to the CUdeviceptr
let regrets_ptr = state.regrets.device_ptr();
params.push(&regrets_ptr as *const _ as *mut _);

// Scalar: pass pointer to the value
let batch_i32 = batch_size as i32;
params.push(&batch_i32 as *const _ as *mut _);
```

**Step 2: Delete terminal.rs**

Terminal evaluation is now inline in the mega-kernel. Remove the file and its `pub mod terminal;` from lib.rs.

**Step 3: Run integration tests**

Run: `cargo test -p gpu-range-solver`
Expected: All integration tests pass (gpu_solve_river_game_reduces_exploitability, gpu_solve_matches_cpu_convergence)

**Step 4: Commit**

```bash
git commit -m "feat(gpu-range-solver): single cooperative launch solver, delete terminal.rs"
```

---

## Task 4: Multi-Street Support (Turn)

Add street decomposition for turn solves: launch river mega-kernel with B=44, aggregate results, launch turn mega-kernel with B=1.

**Files:**
- Modify: `crates/gpu-range-solver/src/solver.rs`

**Step 1: Implement street decomposition**

For a turn solve:
1. Identify chance nodes in topology (where turn→river transition happens)
2. Extract the river subtree topology (same for all 44 runouts)
3. Build per-runout terminal data (different boards = different showdown outcomes)
4. Launch river solve with B=44
5. Download river root CFVs: `[44 × H]`
6. Average across runouts: `turn_leaf_cfv[h] = mean over 44 runouts`
7. Launch turn solve with B=1, using averaged CFVs as terminal values at chance children

**The key challenge:** The persistent kernel currently hardcodes terminal evaluation. For the turn street pass, "terminals" are actually the aggregated river CFVs (not fold/showdown). Add a `leaf_cfv` parameter to the kernel — a `[B × num_leaf_nodes × H]` array that provides pre-computed terminal values at leaf nodes (which are the chance children in the turn tree).

**Step 2: Run turn integration test**

Run: `cargo test -p gpu-range-solver -- gpu_solve_turn`
Expected: Turn game tests pass

**Step 3: Commit**

```bash
git commit -m "feat(gpu-range-solver): multi-street support for turn solves (B=44)"
```

---

## Task 5: Benchmark and Verify

**Step 1: Run river benchmark**

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

Target: GPU ≤ 0.3s (matching CPU).

**Step 2: Run turn benchmark at various hand counts**

```bash
# Same benchmark script as before but with turn spots
# Target: GPU within 2x of CPU for turn solves
```

**Step 3: Run full test suite**

Run: `cargo test -p gpu-range-solver`
Expected: All tests pass

**Step 4: Commit**

```bash
git commit -m "perf(gpu-range-solver): persistent kernel verified — benchmarks match CPU"
```

---

## Summary

| Task | What | Key Change |
|------|------|-----------|
| 1 | Persistent mega-kernel | Replace 11 kernels with 1 cooperative `cfr_solve` |
| 2 | Batch GPU state | `[B×E×H]` allocations, cooperative launch infra |
| 3 | Single-launch solver | Entire solve = one `cuLaunchCooperativeKernel` call |
| 4 | Multi-street | Turn: B=44 river launch → aggregate → B=1 turn launch |
| 5 | Benchmark | Verify GPU matches CPU speed |

**Critical path:** 1 → 2 → 3 → 5 (river working). Task 4 (turn support) after river is verified.

**The big win:** 12,500 kernel launches → 1. All data stays in L2 cache. Grid-stride loops handle any batch × edge × hand combination. One cooperative launch per street.
