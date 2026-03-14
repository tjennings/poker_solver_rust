# GPU Solver Design — Supremus-Style Continual Resolving

## Overview

A new `gpu-solver` crate implementing DCFR+ solving entirely on NVIDIA GPUs via `cudarc` + custom CUDA kernels. The system serves two purposes:

1. **Real-time resolving** — Solve lookahead subtrees in <1 second for live play decisions
2. **CFVNet training data generation** — Batch-solve thousands of subgames on-GPU to train neural network value functions

The solver uses bucketed hand representations (from the existing clustering pipeline) and the existing `burn-cuda` backend for neural network inference/training.

## Architecture

```
crates/gpu-solver/
├── src/
│   ├── lib.rs           # Public API
│   ├── tree.rs           # Flat-array lookahead tree builder (CPU-side)
│   ├── gpu.rs            # cudarc device management, memory allocation, kernel launches
│   ├── solver.rs         # DCFR+ orchestration: upload → iterate → download
│   ├── datagen.rs        # Batch subgame generation for CFVNet training
│   └── resolve.rs        # Single-tree resolve for live play
├── kernels/              # .cu source files → compiled to PTX
│   ├── regret_match.cu
│   ├── forward_reach.cu
│   ├── terminal_eval.cu
│   ├── leaf_eval.cu
│   ├── backward_cfv.cu
│   ├── update_regrets.cu
│   └── extract_strategy.cu
├── tests/
└── Cargo.toml            # depends on core, cudarc, burn-cuda
```

### Dependencies

- `cudarc` — Rust CUDA driver API wrapper for custom kernel launches
- `burn-cuda` — Neural network inference at leaf nodes (CFVNet)
- `poker-solver-core` — Game tree types, hand evaluation, bucket files
- `range-solver` — Reference implementation for correctness testing

### GPU Memory Layout

All data is stored as flat arrays in GPU global memory, uploaded once before the iteration loop:

| Array | Shape | Description |
|-------|-------|-------------|
| `regrets` | `[num_infosets × max_actions]` | Cumulative regrets (f32) |
| `strategy_sum` | `[num_infosets × max_actions]` | Cumulative weighted strategy (f32) |
| `reach_probs` | `[num_nodes × num_buckets]` | Per-node range vectors (f32) |
| `cfvalues` | `[num_nodes × num_buckets]` | Counterfactual values (f32) |
| `current_strategy` | `[num_infosets × max_actions]` | Current iteration strategy (f32) |
| `node_info` | `[num_nodes]` | Node metadata: type, player, parent, action count, level, pot, etc. (packed struct) |
| `level_starts` | `[depth + 1]` | Index boundaries for each tree level |
| `children` | `[num_edges]` | Child node indices (u32) |
| `child_offsets` | `[num_nodes + 1]` | CSR-style offset into children array |

### Tree Builder (CPU-side)

The lookahead tree is built on CPU using the existing action abstraction config, then flattened into level-order arrays:

1. Build game tree from current state using action abstraction (reuse `game_tree.rs` patterns)
2. Assign nodes to levels (BFS order)
3. Map decision nodes to infosets (player × level × action-sequence hash)
4. Pack into flat arrays with CSR-style child indexing
5. Single `cuMemcpy` upload to GPU

### CUDA Kernels

Each kernel processes all nodes/infosets at a given level in parallel:

**`regret_match`** — One thread per infoset. Reads cumulative regrets, outputs current strategy via regret matching (clamp negatives to zero, normalize).

**`forward_reach`** — One thread per (node, bucket) pair at a given level. Multiplies parent's reach probability by the action probability that leads to this node. Called top-down, one level at a time.

**`terminal_eval`** — One thread per (terminal_node, bucket) pair. Computes payoffs: fold nodes use pot size, showdown nodes use equity lookup tables (precomputed per bucket pair).

**`leaf_eval`** — Gathers bucket distributions at depth-boundary nodes, batches them, invokes CFVNet forward pass via burn-cuda. Writes estimated CFVs back to `cfvalues` array.

**`backward_cfv`** — One thread per (node, bucket) pair at a given level. Aggregates children's CFVs weighted by action probabilities. Computes counterfactual values for the acting player. Called bottom-up, one level at a time.

**`update_regrets`** — One thread per (infoset, action) pair. Updates cumulative regrets and strategy sums with DCFR+ weighting: regret weight = `max(0, t - d)` where `d = 100`, strategy weight = `max(0, t - d)`.

**`extract_strategy`** — One thread per infoset. Normalizes strategy sums into final action probabilities.

### DCFR+ Iteration Loop

```
upload(tree_structure, initial_ranges, equity_tables)

for t in 1..=num_iterations:
    launch regret_match(all_infosets)
    for level in 0..depth:
        launch forward_reach(level)
    launch terminal_eval(all_terminals)
    launch leaf_eval(all_depth_boundaries)  // CFVNet, phases 3+
    for level in (0..depth).rev():
        launch backward_cfv(level)
    launch update_regrets(t, all_infosets)

launch extract_strategy(all_infosets)
download(strategy)
```

No CPU-GPU memory transfers occur between iterations. Kernel launches are enqueued on a single CUDA stream.

### Bucket Equity Tables

For terminal showdown evaluation, we need bucket-vs-bucket equity for each board. These are precomputed on CPU (or GPU in later phases) and uploaded alongside the tree:

- `equity_table: [f32; num_buckets × num_buckets]` — expected value of bucket_i vs bucket_j at showdown
- Computed from the clustering pipeline's bucket assignments + hand evaluator

## Phased Implementation

### Phase 1: GPU DCFR+ Solver

**Goal:** A working GPU DCFR+ solver with a CLI identical to the existing `range-solve` command.

**Scope:**
- Flat-array tree builder for postflop subtrees (single street or multi-street)
- All CUDA kernels except `leaf_eval` (terminals only — fold, showdown)
- cudarc device management, memory allocation, kernel compilation
- DCFR+ iteration loop with delayed averaging (d=100)
- CLI: `gpu-solve` subcommand in trainer binary, same args as `range-solve`
- Bucketed mode: uses existing bucket files for hand abstraction

**Deliverable:** Solve any postflop subtree on GPU. Run same position through both `range-solve` and `gpu-solve`, compare strategies. Convergence rate and final exploitability should be comparable.

**Testing:**
- Solve river, turn, and flop subtrees; compare strategy output vs range-solver
- Exploitability convergence curve vs CPU baseline
- Benchmark: iterations/second at various tree sizes
- Correctness: per-kernel unit tests comparing GPU output to CPU reference calculations

### Phase 2: River CFVNet Training on GPU

**Goal:** Generate river training data and train river CFVNet model entirely on GPU.

**Scope:**
- Batch solver mode: solve N independent river subgames in parallel on GPU
- Random situation sampling: board, ranges, pot/stack (CPU-generated, batch-uploaded)
- CFV extraction: after solving, read counterfactual values at root as training targets
- Direct feed into burn-cuda training loop (GPU tensor → training step, no host round-trip for data)
- CLI: `gpu-datagen river` and `gpu-train river` commands

**Deliverable:** Train a river CFVNet model from scratch using only GPU computation. Compare validation loss against existing CPU-trained river model — should be equivalent or better (more data throughput).

**Testing:**
- Compare GPU-generated CFVs against CPU datagen (`sampler.rs`) for same positions
- Train river model, measure validation loss vs CPU baseline
- Benchmark: training examples/second vs current pipeline
- Solve a river subtree using the GPU-trained model's values manually to sanity-check

### Phase 3: Turn CFVNet Training on GPU

**Goal:** GPU datagen for turn subgames using the Phase 2 river model as leaf evaluator, plus turn model training.

**Scope:**
- `leaf_eval` kernel integration: at river-boundary leaf nodes, batch-invoke river CFVNet via burn-cuda
- Shared CUDA context between cudarc solver kernels and burn-cuda inference
- Turn datagen: solve random turn subgames on GPU with river model at leaves
- Turn model training on GPU
- CLI: `gpu-datagen turn --river-model <path>` and `gpu-train turn`

**Deliverable:** Train a turn CFVNet model using GPU datagen with river model leaf evaluation. Compare against existing CPU turn pipeline output.

**Testing:**
- Compare turn CFVs (GPU) vs CPU turn datagen for same positions
- Train turn model, compare validation loss vs CPU baseline
- End-to-end: solve a turn subtree using the new turn model, compare strategy quality
- Benchmark: throughput improvement over CPU pipeline

### Phase 4: Flop + Auxiliary Preflop Model

**Goal:** Complete the neural network stack with flop and auxiliary preflop models.

**Scope:**
- Flop datagen: solve random flop subgames with turn model at leaves
- Flop CFVNet training on GPU
- Auxiliary preflop network: trained on preflop situations with flop model at leaves
- Full model stack: preflop-aux → flop → turn → river
- CLI: `gpu-datagen flop/preflop` and `gpu-train flop/preflop`

**Deliverable:** Complete set of trained value networks covering all streets, all trained via GPU pipeline. A full hand can be resolved from preflop onward using the model stack.

**Testing:**
- Each model layer validates against exact solving on small trees
- Full-stack resolve from preflop: verify strategy is reasonable (no dominated actions)
- Benchmark full training pipeline end-to-end: time to train all four model layers
- Compare against any existing CPU baselines where available

### Phase 5: Explorer Integration for HU Games

**Goal:** Wire the GPU solver into the Tauri frontend for interactive heads-up resolving.

**Scope:**
- Tauri commands: `gpu_resolve` — given a game state and model stack, resolve and return strategy
- Explorer UI integration: display GPU-resolved strategies in the existing strategy matrix view
- Live resolving: user navigates the game tree, GPU re-resolves at each decision point
- Off-tree action handling: safe resolving when opponent plays a bet size not in the abstraction
- Model management: load/unload CFVNet model stack

**Deliverable:** Open the Explorer, load a model stack, and interactively browse GPU-resolved strategies for any HU situation. Response time <1 second per resolve.

**Testing:**
- Resolve from various game states, verify UI displays correct strategies
- Performance: resolve latency <1s in the Explorer
- Off-tree actions: verify safe resolving produces reasonable strategies
- User testing: navigate a full hand from preflop to river in the Explorer

## Key Design Decisions

1. **cudarc over raw CUDA FFI** — Rust-native memory management with full kernel control
2. **burn-cuda for neural nets only** — The right tool for NN inference; DCFR+ kernels are custom
3. **Flat level-order arrays** — Enables per-level parallel kernel launches, no recursion
4. **Bucketed representation** — Fits in GPU memory, matches Supremus approach, leverages existing clustering pipeline
5. **DCFR+ with delayed averaging** — Faster convergence than vanilla CFR+, proven by Supremus
6. **Shared CUDA context** — cudarc and burn-cuda share the same GPU device to avoid redundant memory copies
7. **CSR-style child indexing** — Compact tree representation with O(1) child lookup
