# GPUGT vs. Current ReBeL-Style Solver Architecture

*2026-04-02 — Research notes on GPU-accelerated CFR approaches*

## Background

- **GPUGT** (University of Toronto, Juho Kim): Reformulates CFR as sparse matrix-vector operations on GPU. [github.com/uoftcprg/gpugt](https://github.com/uoftcprg/gpugt), paper: arXiv:2408.14778
- **Our range-solver**: Recursive DFS-based exact DCFR with 1326-hand vectorization, CPU-only (rayon parallelism at chance nodes)
- **Supremus** (Zarick et al.): End-to-end GPU NLHE solver with custom CUDA kernels — never open-sourced (arXiv:2007.10442)

## Side-by-Side Comparison

| Dimension | Our Range-Solver (ReBeL-style) | GPUGT |
|-----------|--------------------------------|-------|
| **Traversal** | Recursive DFS (`solve_recursive`) | Level-synchronous sparse matmul |
| **Compute unit** | Per-node: `num_actions × 1326` ops | Per-level: one sparse SPMV over all nodes at depth |
| **Hardware** | CPU (rayon parallelism at flop chance nodes) | GPU (cuSPARSE for all iterations) |
| **Memory layout** | Node arena + contiguous storage buffers, pointers into slabs | CSR sparse matrices (graph, graph2, mask, utilities) |
| **Regret matching** | Per-node loop: `max(regret, 0) / sum` over actions | Vectorized: `mask.T @ (mask @ clip(regrets))` across ALL infosets at once |
| **Forward pass** | Recursive: `cfreach *= strategy[action]` at each node | Matrix: `strategy[level] = behavioral * parent_strategy`, level by level |
| **Backward pass** | Recursive: `node_value += strategy * child_cfv` | Matrix: `V[sources] = graph[sources] @ V + graph2[sources] @ utility` |
| **Info set handling** | Implicit in tree structure (each node = one infoset for 1326 hands) | Explicit `mask` matrix (decision_points × actions) |
| **Discounting** | DCFR (alpha/beta/gamma per iteration) | CFR / CFR+ only (no DCFR) |
| **Compression** | i16 quantization with scale factors | None (f64 on GPU) |
| **NN integration** | Leaf evaluator via cfvnet (burn framework) | None — pure tabular CFR |

## Where GPUGT Wins

### 1. Parallelism Granularity
Our solver parallelizes at the **flop chance node** level (rayon over board runouts). Within each subtree, traversal is sequential. GPUGT parallelizes at **every level of the tree** — all nodes at the same depth process simultaneously in one kernel launch.

### 2. No Recursion Overhead
Our hot path is `solve_recursive` with function call overhead, branch prediction misses at the terminal/chance/player dispatch, and stack pressure from 1326-element scratch buffers. GPUGT replaces all of this with a flat loop over tree levels doing sparse matmuls.

### 3. Memory Access Pattern
Sparse SPMV on GPU has predictable, coalesced memory access. Our solver chases `children_offset` pointers and does scattered reads across `storage1`/`storage2` buffers.

## Where Our Solver Wins

### 1. DCFR
GPUGT only implements CFR and CFR+. Our solver uses DCFR with alpha/beta/gamma weighting, which converges faster in practice for poker. Adding DCFR to GPUGT would require per-iteration weight scaling of the regret accumulation — straightforward but not yet done.

### 2. Neural Net Leaf Evaluation
Our architecture supports depth-bounded solving with cfvnet at leaf nodes. GPUGT is purely tabular — no NN integration. For real-time subgame solving, you need NN-backed leaf evaluation.

### 3. Compression
Our i16 quantized storage cuts memory ~2x, critical for large NLHE trees. GPUGT uses f64 throughout.

### 4. Isomorphism
Our solver exploits suit isomorphisms to reduce work by ~4-24x at chance nodes. GPUGT has no isomorphism support.

### 5. Production-Ready for NLHE
Our solver handles real poker: 1326 combos, board runouts, hand evaluation, rake. GPUGT has only been tested on toy games (Kuhn, Leduc, Liar's Dice).

## The Fundamental Architecture Difference

```
OUR SOLVER:                           GPUGT:

  solve_recursive(node)               for level in reversed(levels):
    if terminal → evaluate              V[nodes_at_level] = G @ V + G2 @ U
    if chance → parallel children
    if player → regret match          regrets += cfv - mask.T@(mask@(σ*cfv))
      for action:                     σ = clip(regrets) / mask.T@(mask@clip(r))
        recurse(child)
      update regrets                  ← entire tree processed per line
      update strategy

  ↑ one node at a time                ↑ all nodes at a level at once
```

## GPUGT Matrix Formulation Details

### Five Key Sparse Structures

| Structure | Shape | Purpose |
|-----------|-------|---------|
| `graph` (CSR) | nodes × nodes | Parent→child adjacency, values set to behavioral strategy |
| `graph2` (CSR) | nodes × sequences | Node→action-sequence mapping |
| `mask` (CSR) | decision_points × actions | Which actions available at each info set |
| `counterfactual` (array) | sequences | Maps sequences → destination nodes |
| `level_sources` (list) | per BFS level | Node indices for level-synchronous traversal |

### One CFR Iteration (~6 sparse matmuls, all GPU)

```
1. REGRET MATCHING (next_strategy):
   numerator = clip(regrets, 0)
   denominator = mask.T @ (mask @ numerator)     ← 2 sparse SPMV
   behavioral = where(denom ≈ 0, uniform, num/denom)
   sequence_strategy = forward_multiply_by_level(behavioral)

2. OPPONENT UTILITY:
   utility = utilities_matrix @ sequence_strategy  ← 1 sparse SPMV

3. COUNTERFACTUAL VALUES (backward pass):
   for level in REVERSED(levels):
       V[sources] = graph[sources] @ V + graph2[sources] @ utility  ← 2 SPMV/level

4. REGRET UPDATE:
   immediate = cfv - mask.T @ (mask @ (strategy * cfv))  ← 2 sparse SPMV
   cumulative_regrets += immediate
```

## Open-Source GPU CFR Landscape

| Project | Open Source? | GPU Framework | CFR on GPU? | Notes |
|---------|-------------|---------------|-------------|-------|
| **GPUGT** (Kim, UofT) | Yes (MIT) | CuPy/cuSPARSE | Yes, matrix formulation | Best reference. Toy games only. |
| **Supremus** (Zarick et al.) | No | CUDA C++ | Yes, end-to-end | 1000 DCFR+ iters in 0.8s on flop. Never released. |
| **GPUCFR** (janrvdolf) | Yes | CUDA C/C++ | Yes, vanilla CFR | Academic quality. Goofspiel only. |
| **GPU EGT** (Kroer/Sandholm) | No | CUDA/cuSPARSE | Yes (EGT, not CFR) | First-order method alternative. |
| **TexasSolverGPU** | No (free binary) | CUDA (presumed) | Yes | Closed source. Claims 4x faster than Pio. |
| **b-inary/postflop-solver** | Yes (Rust) | None (CPU SIMD) | No | Fastest open-source CPU solver. Architecture closest to our range-solver. |
| **ReBeL** (FAIR) | Yes (Liar's Dice) | PyTorch (NN only) | No, CPU CFR + GPU NN | HUNL implementation never released. |

## Hybrid Architecture Path Forward

The most practical approach isn't "pick one" — it's combining the strengths:

1. **Keep our tree structure** — NLHE trees with isomorphism, NN leaves, and DCFR need the flexibility of explicit nodes
2. **GPU-accelerate per-hand vector ops** — the `num_actions × 1326` regret matching, reach probability scaling, and CFV accumulation at each node are embarrassingly parallel across the 1326 hands
3. **Batch node processing by level** — instead of recursive DFS, do BFS level-by-level, dispatching GPU kernels for all nodes at each depth simultaneously (the GPUGT insight applied to our node structure)

The 1326-hand dimension maps directly to GPU threads. Each CFR iteration's hot inner loop (`fma_slices_uninit`, `regret_matching_into`) is already structured as `for hand in 0..1326` — that's a GPU kernel waiting to happen.

### Supremus Architecture Lessons

The Supremus paper describes the gold standard for GPU poker solving:
- "Single on-ramp" loads tree into GPU memory; "single off-ramp" reads results — zero CPU-GPU transfers during iteration
- Simultaneous (not alternating) player updates — fewer NN forward passes needed
- 1,000 DCFR+ iterations in 0.8 seconds on a flop subtree, 6x faster than DeepStack

### GPU Backend Options for Rust

- **wgpu** — cross-platform (Metal on Mac, Vulkan/DX12 on cloud). Good for dev + production portability.
- **cudarc** — Rust bindings to CUDA. Maximum NVIDIA performance. Cloud-only.
- **burn** — already used by cfvnet. Has GPU backend support (wgpu, CUDA via LibTorch).
