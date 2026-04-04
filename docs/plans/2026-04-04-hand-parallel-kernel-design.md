# Hand-Parallel GPU Kernel Design (Supremus Architecture)

*2026-04-04 — Rewrite GPU solver from node-parallel to hand-parallel*

## Problem

The current GPU solver parallelizes over tree nodes and uses `grid.sync()` between levels (~5-10μs each, 73 per iteration = 365μs/iter). On a turn solve this produces 278x slower than CPU. The Supremus paper achieves 1000 iterations in 0.8s on the flop by parallelizing over HANDS instead.

## Solution

One CUDA thread block = one independent subgame solve. 1024 threads handle up to 1024 hands in parallel. The block processes the entire tree sequentially (level-by-level). `__syncthreads()` replaces `grid.sync()` — 2000x cheaper (~20ns vs ~5μs). Multiple blocks solve different boards simultaneously (batching).

## Key Insight

Forward pass, backward pass, regret matching, DCFR discount, and showdown evaluation have **zero cross-hand dependencies**. Each thread processes its own hand independently through the entire tree. The ONLY cross-hand operation is **fold terminal evaluation** (card-blocking requires summing opponent reach across hands → shared memory reduction, ~3 syncs per fold node).

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Parallelism axis | Hands (not nodes) | Zero cross-hand deps except fold eval; __syncthreads is 2000x cheaper than grid.sync |
| Block size | 1024 threads | Handles up to 1024 hands. For 1326: 2 hands/thread |
| Tree traversal | Level-by-level within block | Reuses existing level arrays from extract.rs |
| Tree topology | Shared memory (~7KB) | Loaded once, read-only throughout kernel |
| Solver state | Global memory, coalesced | regrets/strategy/reach/cfv too large for shared mem |
| Batching | One block per board | 44 blocks for turn, 1980 for flop, all run in parallel |
| Cooperative groups | NONE | Not needed — only __syncthreads within a block |

## Kernel Structure

```cuda
__global__ void cfr_solve(
    // Global solver state [B × E × H] and [B × N × H]
    float* regrets, float* strategy_sum, float* reach, float* cfv,
    // Topology (shared across all blocks)
    const int* edge_parent, const int* edge_child, const int* edge_player,
    const int* level_starts, const int* level_counts,
    // Terminal data (per block for showdown, shared for fold)
    // Card data, initial weights, dimensions, iteration count
    ...
) {
    int bid = blockIdx.x;    // which board/subgame
    int tid = threadIdx.x;   // which hand
    int H = blockDim.x;

    // Load tree topology into shared memory (~7KB)
    __shared__ int16_t s_parent[MAX_E], s_child[MAX_E];
    __shared__ int8_t s_player[MAX_E];
    __shared__ int s_level_starts[MAX_D], s_level_counts[MAX_D];
    // ... cooperative load, __syncthreads() ...

    for (int iter = 0; iter < max_iterations; iter++) {
        // DCFR discount — per-thread, no sync
        // Alpha/beta/gamma computed from iter inline

        for (int player = 0; player < 2; player++) {
            // Zero reach/cfv — per-thread loop over nodes
            // Set root reach — per-thread
            // Regret match — per-thread (loop over actions per node, no reduction)
            
            // Forward pass — sequential over levels, parallel over hands
            for (depth 0..max_depth) {
                for (each edge at depth) {
                    // reach[child,tid] = f(reach[parent,tid], strategy[edge,tid])
                }
                // NO sync — each thread writes its own hand only
            }

            // Backward pass — sequential over levels (reverse)
            for (depth max_depth..0) {
                // Fold eval: 3 __syncthreads per fold node (shared mem reduction)
                // Showdown eval: per-thread, no sync
                // CFV accumulate: per-thread
                // Regret update: per-thread
            }
        }
    }
}
```

## Sync Budget Per Iteration

| Operation | __syncthreads | Cost |
|-----------|--------------|------|
| DCFR discount | 0 | per-thread |
| Regret matching | 0 | per-thread loop over actions |
| Forward pass (8 levels) | 0 | per-thread |
| Backward pass (8 levels) | 0 | per-thread |
| Regret update | 0 | per-thread |
| Fold eval (~5 nodes) | 15 | 3 syncs × 5 nodes |
| Showdown eval | 0 | per-thread loop |
| **Total** | **~15** | **~300ns** |

Current mega-kernel: 73 grid.sync × ~5μs = 365μs/iter. **1200x reduction in sync overhead.**

## Memory Model

**Shared memory (48KB available, ~7KB used):**

| Data | Size | Type |
|------|------|------|
| edge_parent[E] | 1KB | i16 (E ≤ 500) |
| edge_child[E] | 1KB | i16 |
| edge_player[E] | 500B | i8 |
| level_starts[D] | 64B | i32 (D ≤ 16) |
| level_counts[D] | 64B | i32 |
| terminal node info | ~2KB | fold/showdown IDs, depths, payoffs |
| card_reach[52] + total_reach | 212B | fold eval scratch |
| **Total** | **~5KB** | |

**Global memory per block (coalesced access):**

| Array | Size per block | Access pattern |
|-------|---------------|----------------|
| regrets [E × H] | 500 × 1024 × 4 = 2MB | `[bid*E*H + e*H + tid]` — coalesced |
| strategy_sum [E × H] | 2MB | same |
| reach [N × H] | 250 × 1024 × 4 = 1MB | same |
| cfv [N × H] | 1MB | same |
| **Total per block** | **~6MB** | |

For 44 blocks (turn): 44 × 6MB = 264MB. For 1980 blocks (flop): 1980 × 6MB = 11.6GB. Fits in 48GB.

## Terminal Evaluation

**Fold (requires cross-hand sync):**
```cuda
__shared__ float s_card_reach[52];
__shared__ float s_total_reach;

// Phase 1: zero accumulators
if (tid < 52) s_card_reach[tid] = 0;
if (tid == 0) s_total_reach = 0;
__syncthreads();

// Phase 2: opponent hands contribute reach by card
if (tid < num_opp_hands) {
    float r = reach[bid*N*H + node*H + tid];
    atomicAdd(&s_total_reach, r);
    atomicAdd(&s_card_reach[opp_c1[tid]], r);
    atomicAdd(&s_card_reach[opp_c2[tid]], r);
}
__syncthreads();

// Phase 3: player hands compute cfv
if (tid < num_player_hands) {
    float blocking = s_card_reach[p_c1[tid]] + s_card_reach[p_c2[tid]];
    if (same_idx[tid] >= 0) blocking -= reach[bid*N*H + node*H + same_idx[tid]];
    cfv[bid*N*H + node*H + tid] = payoff * (s_total_reach - blocking);
}
__syncthreads();
```

**Showdown (no sync needed):**
```cuda
if (tid < num_player_hands) {
    float val = 0;
    for (int opp = 0; opp < num_opp_hands; opp++)
        val += outcome[tid * num_opp + opp] * reach[bid*N*H + node*H + opp];
    cfv[bid*N*H + node*H + tid] = val;
}
```

## Batched Multi-Street Solve

**Turn solve (2 phases):**
```
Phase 1: <<<44 blocks, H threads>>> cfr_solve(river subtree topology, per-board terminals, 1000 iter)
         All 44 river boards solved in parallel: ~3ms
         Download 44 root CFVs, average → turn leaf values

Phase 2: <<<1 block, H threads>>> cfr_solve(turn action tree, leaf values as terminals, 1000 iter)
         Single turn tree with pre-computed leaf CFVs: ~3ms
         Download root strategy
```

**Flop solve (3 phases):**
```
Phase 1: <<<1980 blocks, H threads>>> river subtrees → ~3ms
Phase 2: <<<45 blocks, H threads>>> turn subtrees → ~3ms  
Phase 3: <<<1 block, H threads>>> flop action tree → ~3ms
Total: ~9ms compute + 280ms CUDA init = ~290ms
```

## Expected Performance

| Scenario | Current GPU | Hand-parallel GPU | CPU |
|----------|-----------|-------------------|-----|
| River (19h, 500 iter) | 0.68s | **~280ms** (init-dominated) | 0.00s |
| River (600h, 500 iter) | 1.83s | **~283ms** | 0.03s |
| Turn (135h, 200 iter) | 19s | **~286ms** | 0.08s |
| Turn (600h, 200 iter) | 50s | **~286ms** | 0.18s |

The GPU becomes competitive once CUDA init is amortized (repeated solves or batch of spots).

With a warm CUDA context (init already done):
| Scenario | Hand-parallel GPU | CPU |
|----------|-------------------|-----|
| River (600h, 500 iter) | **~3ms** | 30ms |
| Turn (600h, 200 iter) | **~6ms** | 180ms |

**GPU wins 10-30x on warm solves.** This is the regime Supremus operates in — solving thousands of subgames during play, with CUDA context initialized once at startup.

## File Changes

| File | Change |
|------|--------|
| `kernels.rs` | Complete rewrite: hand-parallel kernel, no cooperative groups |
| `gpu.rs` | Simplify: remove cooperative launch, standard <<<blocks, threads>>> |
| `solver.rs` | Simplify: multi-phase launcher for turn/flop |
| `terminal.rs` | Delete (fold/showdown inline in kernel) |
| `extract.rs` | Unchanged |
| `lib.rs` | Unchanged public API |
