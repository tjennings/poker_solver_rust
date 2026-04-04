# Persistent Batched CUDA Kernel Design

*2026-04-03 — Single kernel launch for entire CFR solve with batched runouts*

## Problem

The cudarc GPU solver launches ~25 kernels per iteration × 500 iterations = 12,500 launches per solve. At ~5μs per launch, dispatch overhead alone is 62ms. Combined with L2 cache eviction between launches, the GPU is 10-58x slower than CPU across all problem sizes.

## Solution

Replace all per-iteration kernel launches with a **single persistent kernel** that runs the entire DCFR solve loop. Threads synchronize between phases using CUDA cooperative groups (`grid.sync()`) instead of separate launches. Add a **batch dimension** to process all board runouts simultaneously.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Kernel strategy | Single persistent mega-kernel | Eliminates 12,499 of 12,500 launches |
| Synchronization | Cooperative groups `grid.sync()` | ~1μs vs ~5μs per launch; all modern GPUs (CC ≥ 6.0) |
| Batch dimension | B runouts in one kernel | Turn: B=44, Flop: B=1980 — massive parallelism |
| Launch API | `cuLaunchCooperativeKernel` | Required for grid-wide sync |
| Thread model | Grid-stride loops | Each thread processes multiple elements; handles any problem size |

## Hardware

- NVIDIA RTX 6000 Ada: 142 SMs, compute 8.9, 48GB VRAM
- Max concurrent threads (cooperative): 142 blocks × 1024 threads = 145,408
- Grid-stride loops cover any work size beyond 145K elements

## Memory Layout

All arrays flat, row-major, with batch as leading dimension:

```
regrets:      [B × E × H]    f32    persistent across iterations
strategy_sum: [B × E × H]    f32    persistent across iterations
strategy:     [B × E × H]    f32    scratch per iteration
reach:        [B × N × H]    f32    scratch per player traversal
cfv:          [B × N × H]    f32    scratch per player traversal
denom:        [B × N × H]    f32    scratch for regret matching

Indexing: array[b * stride + e * H + h]

B = batch size (1 for river, 44 for turn, 1980 for flop)
N = nodes in action subtree
E = edges in action subtree
H = max(num_hands_oop, num_hands_ip)
```

Topology arrays (shared across batch, constant):
```
edge_parent:      [E]     i32
edge_child:       [E]     i32
edge_player:      [E]     i32     (0=player0, 1=player1, 2=chance)
actions_per_edge: [E]     f32
level_starts:     [D+1]   i32     D = max_depth
level_counts:     [D+1]   i32
```

## Kernel Structure

One `extern "C" __global__` function containing the full DCFR loop:

```
cfr_solve(all_arrays, B, N, E, H, max_depth, max_iterations, ...) {
    grid_group grid = this_grid();
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    stride = gridDim.x * blockDim.x;

    for iter in 0..max_iterations:
        // DCFR discount — grid-stride over B*E*H
        grid.sync()

        for player in 0..2:
            // Zero reach, cfv, denom — grid-stride over B*N*H
            grid.sync()

            // Set root reach = opponent weights
            grid.sync()

            // Regret match phase 1: accum clipped to denom — grid-stride over B*E*H
            grid.sync()

            // Regret match phase 2: normalize — grid-stride over B*E*H
            grid.sync()

            // Forward pass: for each depth level
            for depth in 0..max_depth:
                // grid-stride over B * level_count * H
                grid.sync()

            // Backward pass: for each depth level (reverse)
            for depth in max_depth..0:
                // Terminal eval at this depth (inline fold/showdown)
                grid.sync()
                // Scatter-add child CFVs to parents
                grid.sync()
                // Regret update for traverser edges
                grid.sync()
}
```

Approximately **~25 `grid.sync()` calls per iteration** at ~1μs each = 25μs sync overhead per iteration.

## Thread Mapping

All work uses grid-stride loops:

```cuda
int total_work = B * level_count * H;
for (int i = tid; i < total_work; i += stride) {
    int b = i / (level_count * H);
    int local = i % (level_count * H);
    int e = level_start + local / H;
    int h = local % H;
    // ... operate on array[b * E * H + e * H + h]
}
```

## Terminal Evaluation (Inline)

Terminal eval moves inside the mega-kernel:

**Fold:** Per fold node at current depth, uses shared memory for card_reach[52]:
- Phase 1: sum opp reach by card (atomicAdd to shared)
- Phase 2: cfv[h] = payoff * (total - card_reach[c1] - card_reach[c2] + correction)

**Showdown:** Per showdown node, inner loop over opponent hands:
- cfv[h] = sum_opp(outcome[h,opp] * opp_reach[opp])
- Outcome matrices uploaded as flat array indexed by batch and showdown node

## Multi-Street Solve (Turn / Flop)

The persistent kernel solves ONE street's action subtree. Multi-street solves chain kernel launches:

| Solve | Pass 1 (river) | Aggregate | Pass 2 (turn) | Aggregate | Pass 3 (flop) |
|-------|----------------|-----------|---------------|-----------|---------------|
| River | B=1, done | — | — | — | — |
| Turn | B=44 | avg river root CFVs → turn leaf values | B=1, done | — | — |
| Flop | B=1980 (or chunked) | avg → turn leaves | B=45 | avg → flop leaves | B=1, done |

Each pass is ONE cooperative kernel launch. Aggregation is a small CPU-side operation between passes (download B root CFVs, average, upload as terminal values for next street).

**Memory for flop (B=1980, N=50, E=80, H=1326):**
- regrets + strategy_sum + strategy: 3 × 1980 × 80 × 1326 × 4B = 2.5 GB
- reach + cfv + denom: 3 × 1980 × 50 × 1326 × 4B = 1.6 GB
- Total: ~4.1 GB — fits in 48GB easily

If B is too large, chunk: run B=500 four times, accumulate results.

## Cooperative Launch via cudarc

```rust
// Query max blocks for cooperative launch
let max_blocks = cuda_occupancy_max_active_blocks(
    &kernel_func, BLOCK_SIZE, shared_mem
);

// Launch cooperative kernel
unsafe {
    cudarc::driver::sys::cuLaunchCooperativeKernel(
        kernel_func.cu_function,
        max_blocks, 1, 1,        // grid dim
        BLOCK_SIZE, 1, 1,        // block dim
        shared_mem_bytes,
        stream.cu_stream,
        params.as_mut_ptr(),
    );
}
```

## Exploitability

Two options:
1. **Skip in-kernel exploitability** — run all max_iterations, compute exploitability on CPU after download. Simplest.
2. **In-kernel reduction** — every N iterations, run best-response pass inside the kernel, reduce to a scalar, compare to target. More complex but enables early stopping.

Recommend option 1 for v1. Early stopping can be added later.

## Expected Performance

| Scenario | Current | Persistent+Batched |
|----------|---------|-------------------|
| River (B=1, 19h, 500 iter) | 0.30s | ~0.05s |
| Turn (B=44, 600h, 200 iter) | 10.6s | ~0.3s |
| Flop (B=1980, 600h, 200 iter) | N/A (too slow) | ~5s |
| CPU turn reference (600h, 200 iter) | | 0.18s |

The turn solve at ~0.3s would be within 2x of CPU (0.18s), with GPU winning at wider ranges where H→1326.

## File Changes

| File | Change |
|------|--------|
| `kernels.rs` | Replace 11 kernels with 1 persistent `cfr_solve` mega-kernel |
| `gpu.rs` | Batch-dimensioned allocations, cooperative launch wrapper, occupancy query |
| `solver.rs` | Simplify: extract → upload → one launch per street → download → aggregate |
| `terminal.rs` | Delete (terminal eval moved inline into mega-kernel) |
| `extract.rs` | Unchanged |
| `lib.rs` | Unchanged public API |
