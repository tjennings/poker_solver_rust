# GPUGT vs. Current ReBeL-Style Solver Architecture

*2026-04-02 — Research notes on GPU-accelerated CFR approaches*

## Side-by-Side Comparison

| Dimension | Your Range-Solver (ReBeL-style) | GPUGT |
|-|-|-|
| Traversal | Recursive DFS (`solve_recursive`) | Level-synchronous sparse matmul |
| Compute unit | Per-node: `num_actions × 1326` ops | Per-level: one sparse SPMV over all nodes at depth |
| Hardware | CPU (rayon parallelism at flop chance nodes) | GPU (cuSPARSE for all iterations) |
| Memory layout | Node arena + contiguous storage buffers, pointers into slabs | CSR sparse matrices (graph, graph2, mask, utilities) |
| Regret matching | Per-node loop: `max(regret, 0) / sum` over actions | Vectorized: `mask.T @ (mask @ clip(regrets))` across ALL infosets at once |
| Forward pass | Recursive: `cfreach *= strategy[action]` at each node | Matrix: `strategy[level] = behavioral * parent_strategy`, level by level |
| Backward pass | Recursive: `node_value += strategy * child_cfv` | Matrix: `V[sources] = graph[sources] @ V + graph2[sources] @ utility` |
| Info set handling | Implicit in tree structure (each node = one infoset for 1326 hands) | Explicit mask matrix (decision_points × actions) |
| Discounting | DCFR (alpha/beta/gamma per iteration) | CFR / CFR+ only (no DCFR) |
| Compression | i16 quantization with scale factors | None (f64 on GPU) |
| NN integration | Leaf evaluator via cfvnet (burn framework) | None — pure tabular CFR |

## Where GPUGT Wins

1. **Parallelism granularity.** Our solver parallelizes at the flop chance node level (rayon over board runouts). Within each subtree, traversal is sequential. GPUGT parallelizes at every level of the tree — all nodes at the same depth process simultaneously in one kernel launch.

2. **No recursion overhead.** Our hot path is `solve_recursive` with function call overhead, branch prediction misses at the terminal/chance/player dispatch, and stack pressure from 1326-element scratch buffers. GPUGT replaces all of this with a flat loop over tree levels doing sparse matmuls.

3. **Memory access pattern.** Sparse SPMV on GPU has predictable, coalesced memory access. Our solver chases `children_offset` pointers and does scattered reads across `storage1`/`storage2` buffers.

## Where Our Solver Wins

1. **DCFR.** GPUGT only implements CFR and CFR+. Our solver uses DCFR with alpha/beta/gamma weighting, which converges faster in practice for poker. Adding DCFR to GPUGT would require per-iteration weight scaling of the regret accumulation — straightforward but not yet done.

2. **Neural net leaf evaluation.** Our architecture supports depth-bounded solving with cfvnet at leaf nodes. GPUGT is purely tabular — no NN integration. For real-time subgame solving, you need NN-backed leaf evaluation.

3. **Compression.** Our i16 quantized storage cuts memory ~2x, critical for large NLHE trees. GPUGT uses f64 throughout.

4. **Isomorphism.** Our solver exploits suit isomorphisms to reduce work by ~4-24x at chance nodes. GPUGT has no isomorphism support.

5. **Production-ready for NLHE.** Our solver handles real poker: 1326 combos, board runouts, hand evaluation, rake. GPUGT has only been tested on toy games (Kuhn, Leduc, Liar's Dice).

## The Fundamental Architecture Difference

```
OUR SOLVER:                             GPUGT:

  solve_recursive(node)                 for level in reversed(levels):
    if terminal → evaluate                V[nodes_at_level] = G @ V + G2 @ U
    if chance → parallel children
    if player → regret match            regrets += cfv - mask.T@(mask@(σ*cfv))
      for action:                       σ = clip(regrets) / mask.T@(mask@clip(r))
        recurse(child)
      update regrets                    ← entire tree processed per line
      update strategy

  ↑ one node at a time                  ↑ all nodes at a level at once
```

## A Hybrid Path Forward

The most practical approach isn't "pick one" — it's combining the strengths:

1. **Keep our tree structure** — NLHE trees with isomorphism, NN leaves, and DCFR need the flexibility of explicit nodes
2. **GPU-accelerate the per-hand vector ops** — the `num_actions × 1326` regret matching, reach probability scaling, and CFV accumulation at each node are embarrassingly parallel across the 1326 hands
3. **Batch node processing by level** — instead of recursive DFS, do BFS level-by-level, dispatching GPU kernels for all nodes at each depth simultaneously (the GPUGT insight applied to our node structure)

The 1326-hand dimension maps directly to GPU threads. Each CFR iteration's hot inner loop (`fma_slices_uninit`, `regret_matching_into`) is already structured as `for hand in 0..1326` — that's a GPU kernel waiting to happen.
