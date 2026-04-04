# cudarc GPU Solver Design — Custom CUDA Kernels for CFR

*2026-04-03 — Replace burn with cudarc + custom CUDA kernels*

## Problem

The burn-based GPU solver is ~10x slower than CPU due to burn's per-operation tensor abstraction overhead (~1000x overhead per op vs raw arrays, confirmed with NdArray CPU backend at 150x slower than native CPU). The per-op overhead dominates regardless of algorithmic batching.

## Solution

Drop burn entirely. Use cudarc for direct CUDA device memory management and kernel launches. Write ~5 small custom CUDA kernels (20-40 lines each) that implement the CFR operations. Each kernel processes all edges at a level × all hands in parallel. Data stays on GPU for the entire solve.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | cudarc (drop burn) | Zero per-op framework overhead |
| Kernel approach | Custom CUDA (not cuSPARSE) | Per-hand strategy can't be encoded in single SpMM; custom kernels fuse gather+weight+scatter |
| Kernel compilation | nvrtc (runtime compile) | No build-time nvcc dependency; CUDA source embedded as include_str! |
| Memory layout | Flat [E × H] and [N × H] | Row-major CudaSlice<f32>, no tensor wrapper |
| Batch dimension | None (sequential runouts) | Get single-solve perf right first; batching layered later |

## Architecture

```
CPU (Rust)                              GPU (CUDA kernels)
┌──────────────────────┐        ┌──────────────────────────────┐
│ extract.rs           │        │ Constant (uploaded once):     │
│  → TreeTopology      │─upload─│  edge_parent [E]             │
│  → TerminalData      │        │  edge_child  [E]             │
│                      │        │  edge_player [E]             │
│ gpu.rs               │        │  actions_per_edge [E]        │
│  → CudaDevice setup  │        │  fold/showdown payoff data   │
│  → kernel loading    │        │                              │
│                      │        │ Mutable state:               │
│ solver.rs            │        │  regrets      [E × H]        │
│  → iteration loop    │        │  strategy_sum [E × H]        │
│  → kernel launches   │        │  strategy     [E × H]        │
│                      │        │  reach        [N × H]        │
│ kernels.rs           │        │  cfv          [N × H]        │
│  → launch wrappers   │        │  denom        [N × H] scratch│
└──────────────────────┘        └──────────────────────────────┘
```

## GPU Memory Layout

All flat `CudaSlice<f32>` / `CudaSlice<i32>`, row-major `[rows × H]`.

| Array | Shape | Description |
|-------|-------|-------------|
| reach | [N × H] | Counterfactual reach per node per hand |
| cfv | [N × H] | Counterfactual values per node per hand |
| regrets | [E × H] | Cumulative regrets per edge per hand |
| strategy_sum | [E × H] | Cumulative strategy per edge per hand |
| strategy | [E × H] | Current iteration strategy (scratch) |
| denom | [N × H] | Regret-match denominator (scratch) |
| edge_parent | [E] | i32, parent node per edge (constant) |
| edge_child | [E] | i32, child node per edge (constant) |
| edge_player | [E] | i32, 0=player0, 1=player1, 2=chance (constant) |
| actions_per_edge | [E] | f32, parent's action count (constant) |

Edges sorted by parent depth → each level is contiguous slice.
Level dispatch info on CPU: `level_edge_start: Vec<usize>`, `level_edge_count: Vec<usize>`.

N = num_nodes, E = num_edges, H = num_hands.

## CUDA Kernels

All in a single `kernels/cfr.cu` file, compiled at runtime via nvrtc.

### `regret_match_accum`
First pass: clip regrets ≥ 0, atomicAdd to per-node denominator.
```
Thread: one per (edge × hand)
clipped = max(regrets[e*H+h], 0.0)
atomicAdd(&denom[parent[e]*H+h], clipped)
```

### `regret_match_normalize`
Second pass: normalize clipped regrets by denominator.
```
Thread: one per (edge × hand)
clipped = max(regrets[e*H+h], 0.0)
d = denom[parent[e]*H+h]
strategy[e*H+h] = d > 0 ? clipped/d : 1.0/actions_per_edge[e]
```

### `forward_pass_level`
Propagate reach from parent to child for one level's edges.
```
Thread: one per (level_edge × hand)
e = level_start + idx / H
h = idx % H
p = edge_parent[e], c = edge_child[e]
player = edge_player[e]
if player == traverser:
    reach[c*H+h] = reach[p*H+h]
elif player == opponent:
    reach[c*H+h] = reach[p*H+h] * strategy[e*H+h]
elif player == CHANCE:
    reach[c*H+h] = reach[p*H+h] / chance_factor  // with card blocking
```

### `backward_pass_level`
CFV propagation + regret update for one level's edges.
```
Thread: one per (level_edge × hand)
e = level_start + idx / H
h = idx % H
p = edge_parent[e], c = edge_child[e]
player = edge_player[e]
child_cfv = cfv[c*H+h]

// Scatter-add weighted CFV to parent
if player == traverser:
    atomicAdd(&cfv[p*H+h], strategy[e*H+h] * child_cfv)
else:
    atomicAdd(&cfv[p*H+h], child_cfv)

// Regret update (traverser only)
if player == traverser:
    // instant_regret computed after all edges at this level are done
    // (requires parent cfv to be complete — second pass or separate kernel)
```

Note: regret update needs parent CFV to be fully accumulated first. Split into two sub-kernels or use cooperative groups sync.

### `regret_update_level`
Separate kernel after backward_pass_level completes (parent CFVs are final).
```
Thread: one per (level_edge × hand)
if edge_player[e] == traverser:
    regrets[e*H+h] += cfv[child*H+h] - cfv[parent*H+h]
    strategy_sum[e*H+h] += strategy[e*H+h]
```

### `dcfr_discount`
Element-wise discount of regrets and strategy_sum.
```
Thread: one per (edge × hand)
r = regrets[e*H+h]
regrets[e*H+h] = r >= 0 ? r * alpha : r * beta
strategy_sum[e*H+h] *= gamma
```

### Terminal Evaluation Kernels

**Fold eval:** Per fold terminal node:
1. Small reduction kernel: sum opponent reach by card (52 partial sums)
2. Per-hand kernel: `cfv[h] = payoff * (total_reach - card_reach[c1[h]] - card_reach[c2[h]] + same_hand_correction)`

**Showdown eval:** Per showdown node, dense matmul: `cfv[h] = sum_opp(outcome[h,opp] * opp_reach[opp])`. Use cublas sgemm for large hand counts, or inline kernel for small.

## Iteration Flow

```
for t in 0..max_iterations:
    launch dcfr_discount(regrets, strategy_sum, alpha, beta, gamma)
    
    for player in [0, 1]:
        launch zero(reach), zero(cfv), zero(denom)
        set reach[root] = opponent_initial_weights
        
        launch regret_match_accum(regrets, denom, edge_parent)
        launch regret_match_normalize(regrets, denom, strategy, edge_parent, actions_per_edge)
        
        for depth in 0..max_depth:
            launch forward_pass_level(depth_start, depth_count, reach, strategy, ...)
        
        for depth in max_depth..0:
            launch terminal_eval_at_depth(...)    // if terminals exist at this depth
            launch backward_pass_level(depth_start, depth_count, cfv, strategy, ...)
            launch regret_update_level(depth_start, depth_count, regrets, strategy_sum, cfv, strategy, ...)
    
    if t % 5 == 4:
        exploitability = compute_exploitability(...)   // best-response pass
        if exploitability <= target: break

download strategy_sum → CPU, normalize → root_strategy
```

## Expected Performance

| Metric | burn (current) | cudarc (target) |
|--------|---------------|-----------------|
| Per-op overhead | ~50-500μs (tensor alloc/clone/shape check) | ~5μs (raw kernel launch) |
| Ops per iteration | ~600 | ~20-30 |
| Time per iteration (river, 19 hands) | ~56ms | ~0.15ms |
| 500 iterations (river, 19 hands) | 2.6s | ~0.1s |
| CPU reference | | 0.3s |

For larger problems (1326 hands), the GPU should be significantly faster than CPU because each kernel processes E×1326 elements in parallel.

## File Changes

| File | Action |
|------|--------|
| `Cargo.toml` | Drop burn, add cudarc with driver+nvrtc features |
| `extract.rs` | Unchanged (topology extraction) |
| `lib.rs` | Same public API, cudarc internals |
| `gpu.rs` | New: CudaDevice setup, memory alloc, kernel loading |
| `kernels.rs` | New: CUDA source string, launch wrappers |
| `kernels/cfr.cu` | New: all CUDA kernel source |
| `solver.rs` | Rewrite: iteration loop with kernel launches |
| `terminal.rs` | Rewrite: fold/showdown as CUDA kernels |
| `tensors.rs` | Delete (replaced by gpu.rs) |

## What Stays the Same

- `extract.rs` — tree topology + terminal data extraction
- `lib.rs` public API — `GpuSolverConfig`, `GpuSolveResult`, `gpu_solve_game()`
- CLI command in trainer (`gpu-range-solve`)
- Integration tests (GPU vs CPU exploitability comparison)
