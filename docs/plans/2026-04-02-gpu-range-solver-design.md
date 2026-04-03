# GPU Range Solver Design

*2026-04-02 — GPUGT-style sparse matrix CFR on GPU via burn*

## Goal

A new CLI command (`gpu-range-solve`) that reimplements the existing `range-solve` using level-synchronous matrix CFR on GPU. Same inputs, same output format, same DCFR algorithm — but runs on GPU with batched board runouts for massive parallelism.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| GPU framework | burn (wgpu + optional cuda) | Already in project (cfvnet, rebel); unified stack |
| Street scope | All streets (flop/turn/river) | Flop is slow but supported; user accepts time cost |
| Isomorphisms | Skipped for v1 | Complicates matrix structure; GPU absorbs extra work |
| CFR variant | DCFR (alpha/beta/gamma) | Matches CPU solver convergence; trivial as element-wise ops |
| Crate | New `gpu-range-solver` | Clean separation from CPU solver; own burn dependency |
| Tree building | Reuse `range-solver`'s `PostFlopGame` | Proven tree builder; extract topology into tensors |
| Batching | Per-street, batch dim = board runouts | Same topology shared across deals; GPU parallelism sweet spot |
| Terminal eval — fold | Card-index gather + scatter ops | O(n) per hand, fully vectorized |
| Terminal eval — showdown | Pre-computed outcome matrix × reach matmul | O(n²) memory per deal; chunk for flop (1,980 deals) |
| Correctness | Similar exploitability (not bit-exact) | f32 GPU vs CPU divergence expected |

## Architecture

### Crate Structure

```
crates/gpu-range-solver/
  Cargo.toml          # depends on range-solver, burn
  src/
    lib.rs            # public API: gpu_solve(), GpuSolverConfig
    extract.rs        # PostFlopGame → GPU tensor conversion
    tensors.rs        # StreetSolver tensor layout definitions
    solver.rs         # level-synchronous DCFR iteration loop
    terminal.rs       # fold + showdown evaluation as tensor ops
    chance.rs         # chance node handling + cross-street aggregation
```

**CLI:** New `GpuRangeSolve` subcommand in `poker-solver-trainer` with identical arguments to `RangeSolve`. Internally:
1. Builds `PostFlopGame` using existing tree builder (same code path as CPU)
2. Calls `gpu_range_solver::gpu_solve(game, config)` instead of CPU solver
3. Outputs same format (header, iteration progress, root strategy table)

**burn dependency:** `burn = { version = "0.16", features = ["wgpu"] }` with optional `cuda = ["burn/cuda-jit"]`, matching cfvnet's pattern.

### Per-Street Batched Solve

The solve decomposes **per-street, bottom-up** (river → turn → flop). Each street has its own action subtree topology (shared across all board runouts at that street). The batch dimension covers all runouts.

**Batch sizes by input:**

| Input | River batch | Turn batch | Flop (no batch) |
|-------|------------|------------|-----------------|
| River (5 cards) | 1 | — | — |
| Turn (4 cards) | 44 | 1 | — |
| Flop (3 cards) | 1,980 | 45 | 1 |

**Cross-street bridge:** After solving all river subtrees, the per-runout root CFVs `[river_batch × hands]` are reshaped and averaged to produce turn leaf values `[turn_batch × hands]`. Turn root CFVs similarly feed up to flop.

### Tree Extraction (CPU → GPU Tensors)

After `PostFlopGame` builds the tree, walk the node arena once on CPU to extract topology into flat arrays.

**Edge-based representation.** Each parent→child link is an "edge" with a global index. Strategy, regrets, and strategy_sum are stored per-edge (not per-node), avoiding padding for variable action counts.

**Extraction produces:**

For each node:
- `node_depth[n]` — depth in tree (for level grouping)
- `node_type[n]` — terminal_fold / terminal_showdown / chance / player0 / player1
- `node_player[n]` — which player acts

For each edge:
- `edge_parent[e]` — parent node index
- `edge_child[e]` — child node index
- `edge_action_index[e]` — action index within parent (0..num_actions-1)

For terminals:
- Fold: `fold_player`, `pot_payoff`
- Showdown: hand strength orderings, pot/rake amounts

For chance nodes:
- `dealt_card[n]` — which card this chance child represents
- `card_valid_mask[card, hand]` — binary `[52 × 1326]` (constant, pre-computed)

**Level grouping:** Nodes sorted by depth → `level_nodes[depth]` and `level_edges[depth]` as index tensors.

### GPU Tensor Layout

```rust
struct StreetSolver {
    // Topology (shared across batch, constant after extraction)
    num_nodes: usize,
    num_edges: usize,
    edge_parent: Tensor<Int, 1>,            // [num_edges]
    edge_child: Tensor<Int, 1>,             // [num_edges]
    node_type: Tensor<Int, 1>,              // [num_nodes]
    level_edge_ranges: Vec<(usize, usize)>, // start..end per depth
    level_node_ranges: Vec<(usize, usize)>, // start..end per depth

    // Per-deal data (batch dim = board runouts)
    hand_mask: Tensor<Float, 2>,            // [batch × num_hands]
    terminal_fold_value: Tensor<Float, 3>,  // [batch × num_fold_nodes × num_hands]
    showdown_outcome: ...,                  // chunked [batch_chunk × hands × hands]

    // Mutable solver state
    regrets: Tensor<Float, 3>,              // [batch × num_edges × num_hands]
    strategy_sum: Tensor<Float, 3>,         // [batch × num_edges × num_hands]
    reach: Tensor<Float, 3>,               // [batch × num_nodes × num_hands]
    cfv: Tensor<Float, 3>,                 // [batch × num_nodes × num_hands]
}
```

**Memory example** (flop solve, river subtree ~50 nodes / 80 edges):
- `regrets + strategy_sum`: 1,980 × 80 × 1,326 × 4B × 2 = 1.6 GB
- `reach + cfv`: 1,980 × 50 × 1,326 × 4B × 2 = 1.0 GB
- Showdown outcomes (chunked): ~310 MB per chunk of 44 deals
- **Total: ~3 GB** — fits a 24GB GPU comfortably

### Core CFR Iteration Loop

Each iteration: alternating player updates (player 0, then player 1). Per player traversal, bottom-up by street (river → turn → flop).

**Within each street solver, one pass per player:**

```
// BACKWARD PASS (bottom-up)
for depth in (max_depth..=0).rev():

    // Terminal nodes: compute CFVs from opponent reach
    evaluate_fold_terminals(depth, player, reach, cfv)
    evaluate_showdown_terminals(depth, player, reach, cfv)

    // Opponent decision nodes:
    //   strategy = regret_match(regrets)
    //   cfv[node] = sum_a(child_cfv[a])
    
    // Traverser's decision nodes:
    //   strategy = regret_match(regrets)
    //   cfv[node] = sum_a(strategy[a] * child_cfv[a])
    //   regrets[a] += discount * (child_cfv[a] - cfv[node])
    //   strategy_sum[a] = strategy_sum[a] * gamma + strategy[a]

// FORWARD PASS (top-down, prepares reach for next iteration)
for depth in 0..=max_depth:
    // Opponent nodes: reach[child] = reach[parent] * strategy[action]
    // Traverser nodes: reach[child] = reach[parent]
    // Chance nodes: reach[child] = reach[parent] / num_cards * hand_mask[deal]
```

**Key burn operations (all batched over deals):**

| Operation | burn expression | Shape |
|-----------|----------------|-------|
| Regret match: clip | `regrets.clamp_min(0.0)` | `[batch × actions × hands]` |
| Regret match: normalize | `clipped / clipped.sum_dim(action_dim)` | `[batch × actions × hands]` |
| CFV accumulate | `(strategy * child_cfvs).sum_dim(action_dim)` | `[batch × hands]` |
| Regret update | `child_cfvs - cfv.unsqueeze(action_dim)` | `[batch × actions × hands]` |
| Reach propagate | `reach.unsqueeze(action_dim) * strategy` | `[batch × actions × hands]` |
| Chance aggregate | `child_cfvs.sum_dim(deal_dim) / num_cards` | `[parent_batch × hands]` |

### Terminal Evaluation on GPU

#### Fold Nodes

```
cfv[h] = pot_payoff * (total_opp_reach - blocking_reach[h])
```

Pre-computed constants:
- `hand_to_cards: [1326 × 2]` — maps hand → (card1, card2)

Per iteration on GPU:
1. `total_opp_reach = opp_reach.sum(hand_dim)` → `[batch]`
2. `card_reach = scatter_add(opp_reach, by card)` → `[batch × 52]`
3. `blocking[h] = card_reach[c1[h]] + card_reach[c2[h]] - same_hand_correction` → `[batch × hands]`
4. `cfv[h] = pot_payoff * (total - blocking[h])` → `[batch × hands]`

#### Showdown Nodes

Pre-computed outcome matrix per deal:
- `outcome[h_player, h_opp]` = +1 (win), 0 (tie/blocked), -1 (loss)
- Shape: `[batch × hands_p × hands_opp]`

Showdown CFV as matmul:
```
cfv = pot_payoff * outcome.matmul(opp_reach.unsqueeze(-1)).squeeze(-1)
```

**Flop chunking:** 1,980 deals × 1326² × 4B = 13.9 GB exceeds VRAM. Process showdown evaluations in chunks of ~200 deals, reusing the outcome buffer.

### DCFR Discounting

Element-wise ops per iteration:

```rust
let positive_mask = regrets.clone().clamp_min(0.0).sign();
let negative_mask = 1.0 - positive_mask;
regrets = regrets * (positive_mask * alpha_t + negative_mask * beta_t);
strategy_sum = strategy_sum * gamma_t;
```

### Exploitability

Best-response traversal: same backward pass, replace strategy-weighted sum with max:
```
br_cfv[node] = max_a(child_cfv[a])
```

`exploitability = (br_value_p0 + br_value_p1) / pot`

Checked every 5 iterations, matching CPU solver. Early stop if below target.

### Finalization & Output

Average strategy:
```
avg_strategy = strategy_sum / strategy_sum.sum(action_dim)
```

Transfer root node's average strategy to CPU. Output identical format:
- Header: board, street, pot, stack, hands, memory, GPU device
- Iteration progress with exploitability
- Per-hand strategy table (hand × action percentages)
- Solve time
