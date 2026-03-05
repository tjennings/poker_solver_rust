# GPU Vectorization for Large-Depth Game Trees

*Research date: 2026-03-03*

## Summary

There is no viable approach for putting full tabular CFR on GPU for poker-scale trees. The two proven paths are: (1) bounded depth-limited solving with neural value functions (Supremus/GTO Wizard approach), or (2) staying on CPU (PioSOLVER approach). Sequence-form first-order methods are a niche third option for fixed-size endgames.

---

## Approaches Evaluated

### 1. Matrix-Form Tabular CFR (Small Games Only)

**Paper**: [GPU-Accelerated CFR (Kim 2024)](https://arxiv.org/html/2408.14778v1)

Reformulates CFR as sparse matrix-vector multiplications:
- Tree stored as CSR sparse matrices (adjacency, level graphs, infoset mappings)
- Level-synchronous processing: all nodes at depth D in parallel, then D-1, etc.
- Forward pass (reach probs) top-down, backward pass (CF values) bottom-up
- Up to 352x faster than OpenSpiel Python, 22x faster than C++
- **Rejected from ICLR 2025** -- does not scale. HUNL has ~10^14 infosets; matrix representation exceeds GPU memory

[GPUCFR](https://github.com/janrvdolf/gpucfr) -- CUDA vanilla CFR, tested only on Goofspiel. Academic exercise.

A [practical attempt](https://lattice.uptownhr.com/cfr-poker-gpu/gpu-acceleration-findings) found MCCFR was **2.6x slower on GPU** than CPU due to hundreds of tiny kernel launches per hand (10-50us overhead each).

### 2. Depth-Limited Solving + Neural Value Functions (Production Viable)

This is the architecture every superhuman poker AI uses. The GPU accelerates neural inference, not tree traversal.

| System | Approach |
|-|-|
| DeepStack | GPU trains CFVnets, CPU does small subgame CFR, GPU evaluates leaves |
| Supremus/GTO Wizard | DCFR+ fully on GPU for small subgames + neural CFVnets at boundaries. 1000 iters in 0.8s |
| ReBeL | GPU trains value/policy networks, 8 CPU threads per GPU for data gen |
| Deep CFR | CPU generates (infoset, regret) pairs via external sampling MCCFR, GPU trains neural net |

### 3. Sequence-Form First-Order Methods

**Paper**: [Kroer, Farina, Sandholm (NeurIPS 2018)](https://www.mit.edu/~gfarina/2018/gpu-egt-nips18/gpu_egt.nips18.cr.pdf)

- Excessive Gap Technique on sequence-form polytope
- Sequence form maps naturally to sparse matrix ops (unlike CFR's recursive structure)
- Tested on Libratus-scale endgames, competitive with CFR+
- Requires reimplementing solver in sequence form

### 4. CPU-Only (PioSOLVER Approach)

[PioSOLVER explicitly states](https://piosolver.com/docs/faq/hardware/) GPUs don't matter. Pure CPU, scales with core count x frequency. Industry benchmark for tabular postflop solving.

---

## Why Deep Trees Are Hard on GPU

1. **Sequential dependency**: CF values propagate bottom-up, chain length = tree depth (20-40+ for HUNL postflop)
2. **Irregular branching**: fold/check/bet nodes have different fan-out -> warp divergence
3. **MCCFR sampling**: traverses one trajectory at a time -> almost no intra-traversal parallelism
4. **Memory**: full HUNL tree far exceeds GPU VRAM

---

## Supremus / GTO Wizard Architecture (Deep Dive)

### Core Idea

Never solve the full game tree. Instead:

1. Build a **lookahead tree** for the current street only
2. Run **DCFR+** on that small tree (1000 iterations, 0.8s on GPU)
3. At **leaf nodes** (end of street), query a neural network (CFVnet) for estimated counterfactual values
4. As play continues, **continual resolving** (CFR-D) builds a new lookahead from current state

This keeps the tree small enough to fit entirely in GPU memory.

### DCFR+ Algorithm

Modified Discounted CFR:

| | DCFR | DCFR+ |
|-|-|-|
| Avg strategy weight | t^2 | max(0, t-d), d=100 |
| Effect | Quadratic from start | First 100 iters get zero weight (delayed averaging) |

Still converges to Nash at O(1/sqrt(T)). Both players update simultaneously (faster with neural value functions).

### CFVnet Neural Networks

**Architecture**: 7 fully-connected hidden layers, 500 nodes each. External constraint ensures player values sum to zero.

**Input** (length 2,001):
- 1,000 buckets x 2 players = 2,000 probability values (beliefs about each player's hand distribution)
- 1 scalar: pot size as fraction of starting chips

**Output**: Expected counterfactual values for each of 1,000 buckets (as fraction of pot).

**Four networks trained sequentially** (each uses the previous as leaf evaluator):

| Network | Training samples | Huber loss (val) |
|-|-|-|
| River | 50M subgames | 0.015 |
| Turn | 20M | 0.010 |
| Flop | 5M | 0.011 |
| Preflop aux | 10M | 0.000070 |

Each training sample = a random subgame solved with 4,000 DCFR+ iterations. Networks trained offline via supervised learning.

### GPU Implementation

Written in CUDA + C++, runs entirely on GPU:
- Single on-ramp: load the lookahead tree into GPU memory
- All DCFR+ iterations run without CPU-GPU memory transfers
- Single off-ramp: read results when done
- Custom CUDA kernels optimized for throughput

**Performance**: 1,000 iterations in 0.8 seconds (6x faster than DeepStack on same abstraction). Achieved 3 mbb/g exploitability 5,000x faster than DeepStack.

### Action Abstraction

| Action position | Options |
|-|-|
| 1st to act | F, C, 0.33x, 0.5x, 0.75x, 1.0x, 1.25x, 2.0x, All-in |
| 2nd action | F, C, 0.25x, 0.5x, 1.0x, All-in |
| 3rd action | F, C, 0.25x, All-in |
| 4th+ | F, C, 1.0x, All-in |

Out-of-abstraction opponent actions handled via continual resolving.

### Results

- Beat Slumbot by +176 +/- 44 mbb/g over 150K hands
- DeepStack reimplementation *lost* 63 mbb/g against same opponent
- 3 mbb/g exploitability

### GTO Wizard Production System

Builds on Supremus foundation:
- Solves one street at a time using neural value functions for future streets
- Self-play trained over hundreds of millions of hands
- Handles up to 200bb with arbitrary bet sizes in ~3 seconds per street
- April 2025: switched from Nash Equilibrium to QRE for better accuracy at low-frequency nodes

---

## Key Architectural Insight

The architecture that makes GPU viable is **not** "put CFR on GPU." It is:

1. **Bound tree size** via depth-limited solving (one street)
2. **Train neural nets** to replace deep subtrees with value estimates
3. **Now the bounded tree fits in VRAM** -> run DCFR+ entirely on GPU with no CPU-GPU transfers
4. **Continual resolving** handles the full game by chaining small solves

The neural nets are the enabler. Without them, the tree is too large for GPU. With them, you get a small tree perfectly suited to GPU parallelism.

---

## Recent Neural CFR Variants (2024-2025)

- [Deep Predictive DCFR](https://arxiv.org/abs/2511.08174) -- neural DCFR with predictive targets
- [Robust Deep MCCFR](https://arxiv.org/pdf/2509.00923) -- variance-reduced sampled advantages
- [D2CFR](https://ieeexplore.ieee.org/document/10273630/) -- dueling network architecture for state value estimation

---

## Minimal Implementation Options

Four paths to a working depth-limited solver, ordered by what they validate first. All share the same end-state (Supremus-like architecture) but front-load different risks.

### Option A: Equity-Leaf Depth-Limited Solver (No ML, CPU-only)

**Idea**: Build the full depth-limited solving + continual resolving architecture, but use raw equity as the leaf value function instead of neural nets. Equity is cheap to compute and requires zero ML infrastructure.

**What you build**:
1. Single-street lookahead tree builder (given a game state, enumerate one street of actions)
2. Tabular DCFR+ solver that operates on the bounded tree
3. Leaf evaluator that computes equity-vs-range for each hand bucket at street boundaries
4. Continual resolving loop: as play proceeds, rebuild and re-solve from current state

**What this proves**: The entire resolving architecture works end-to-end. You can play a hand of poker with it. Strategy quality will be mediocre (equity != counterfactual value under optimal play), but the *plumbing* is correct.

**Upgrade path**: Swap the equity leaf evaluator for a neural CFVnet. Everything else stays the same — the CFVnet is a drop-in replacement with the same interface (range beliefs + pot → per-bucket values).

**Effort**: ~Medium. No ML, no GPU. Main work is the tree builder and resolving loop.

**Risk**: Low. Well-understood components. Early poker AIs (pre-DeepStack) used exactly this approach.

---

### Option B: River-Up CFVnet Pipeline (ML, CPU)

**Idea**: Build the neural value function pipeline bottom-up, starting from the river. The river is the ideal starting point because it's terminal — no leaf value function needed, just showdown equity. Train a river CFVnet, then use it as leaf values to solve turn subgames, train a turn CFVnet, repeat upward.

**What you build**:
1. River subgame solver: tabular DCFR+ on random river deals, solving to convergence
2. Data generator: solve N random river subgames, collect (range_beliefs, pot) → counterfactual_values pairs
3. River CFVnet: train a neural network on this data (7 FC layers × 500 nodes, or start smaller)
4. Turn solver: depth-limited DCFR+ on turn subgames, querying river CFVnet at leaves
5. Repeat: generate turn data, train turn CFVnet, build flop solver...

**What this proves**: That your value networks are accurate enough to support depth-limited solving. Each layer is independently testable — you can measure river CFVnet accuracy before ever building the turn solver.

**Upgrade path**: Once all four networks are trained, you have the full Supremus pipeline on CPU. GPU acceleration is then a pure performance optimization.

**Effort**: ~High. Requires ML training infrastructure (PyTorch/tch-rs or similar), data generation pipeline, network architecture tuning. But each layer is a self-contained milestone.

**Risk**: Medium. The critical question is whether your river CFVnet is accurate enough. Supremus needed 50M river samples — how many do you need for acceptable quality?

---

### Option C: GPU River Endgame Solver (No ML, GPU)

**Idea**: Validate GPU CFR on the simplest possible case. River endgames are terminal (no leaf values needed), have bounded tree size, and fixed structure. Get DCFR+ running in CUDA for river-only, then extend.

**What you build**:
1. River tree flattener: convert a river endgame tree into GPU-friendly flat arrays (node data, parent/child indices, infoset mappings)
2. CUDA DCFR+ kernels: forward pass (reach probabilities), backward pass (CF values), regret/strategy update — all on GPU
3. Single on-ramp/off-ramp: load tree → run N iterations → read strategy
4. Benchmark against CPU solver on same river endgames

**What this proves**: That GPU CFR works and is faster than CPU for bounded subgames. This is the hardest technical piece — if it works, extending to other streets (with CFVnet leaves) is incremental.

**Upgrade path**: Add a CFVnet inference call at leaf nodes (trivial once the GPU pipeline exists). Extend tree builder to turn/flop. The GPU infrastructure is reusable.

**Effort**: ~High. Requires CUDA expertise, careful memory layout design, kernel optimization. But the scope is small (river only, no ML).

**Risk**: High-ish. GPU kernel development is fiddly. The payoff is only clear once you also have CFVnets — river-only GPU solving isn't useful by itself since river subgames are already fast on CPU.

---

### Option D: Playable Agent with Pluggable Components (Hybrid)

**Idea**: Build a complete playable agent as fast as possible, using the cheapest viable approach for each street. Then upgrade components independently.

**What you build**:
1. **Preflop**: Precomputed lookup table (solve once offline, store strategy)
2. **Flop**: Depth-limited CFR with equity leaves (Option A approach). Coarse action abstraction (3-4 bet sizes)
3. **Turn**: Same as flop, equity leaves pointing to river
4. **River**: Full tabular solve (small enough to solve exactly in real-time)
5. **Resolving glue**: Given opponent's action, re-solve current street from updated beliefs

**What this proves**: End-to-end playability. You have something that can sit at a table and play hands. You can measure its exploitability and head-to-head performance against benchmarks. Each component has a clear upgrade path.

**Upgrade path**:
- Replace equity leaves with CFVnets (one street at a time, river-up)
- Replace CPU solver with GPU solver (one street at a time)
- Refine action abstraction as solving gets faster
- Add opponent modeling / adaptation

**Effort**: ~Medium. No single component is hard. The integration work (belief tracking, re-solving, action translation) is the main challenge.

**Risk**: Low. Every component uses known techniques. Quality will be poor initially but measurable and improvable.

---

### Comparison

| | ML required | GPU required | End-to-end playable | Validates architecture | Validates GPU | Validates neural values |
|-|-|-|-|-|-|-|
| A: Equity leaves | No | No | Yes | Yes | No | No |
| B: River-up CFVnet | Yes | No | Eventually | Partially | No | Yes |
| C: GPU river solver | No | Yes | No | No | Yes | No |
| D: Playable hybrid | No | No | Yes | Yes | No | No |

### Recommended Sequencing

These options aren't mutually exclusive. A natural progression:

1. **Start with A or D** — get the depth-limited resolving architecture working with equity leaves
2. **Layer in B** — train river CFVnet, swap it in as leaf evaluator, measure improvement
3. **Optionally add C** — once subgame solving is the bottleneck, move DCFR+ to GPU

This way each step produces a working system that's strictly better than the previous one.

---

## Training Setup: CFVnet Pipeline

### What You're Training

A CFVnet is a supervised regression model. Given the current beliefs about what hands each player holds (range) and the pot size, it predicts the counterfactual value of each hand bucket. One network per street boundary (river, turn, flop, preflop auxiliary). Networks are trained sequentially bottom-up — each depends on the one below.

### Hand Bucketing

Before anything else, you need a fixed mapping from 1,326 hole card combos to N buckets. This mapping is used everywhere — data generation, training, and inference must all agree.

**Options** (in order of complexity):
1. **EHS/EHS2 clustering** — compute expected hand strength (squared) for each combo given the board, k-means into N buckets. Supremus used 1,000. Standard approach.
2. **OCHS (Opponent Cluster Hand Strength)** — cluster based on equity distribution vs opponent range, not just average equity. Better separation of hands with similar equity but different distributions.
3. **Hand class + equity bin** — use made-hand/draw classification × intra-class equity quantile. Coarser but interpretable. Could start here (e.g., 19 classes × 10 equity bins = 190 buckets) and move to EHS2 later.

The bucket count controls network input/output size. More buckets = more expressive but more data needed. 200-1,000 is the practical range.

### Data Generation

Each training sample is one solved subgame. The pipeline:

1. **Sample a random subgame root**: random board cards, random range beliefs for both players (drawn from realistic distributions — e.g., output of solving the previous street), random pot size
2. **Solve it with tabular DCFR+** to convergence (Supremus used 4,000 iterations per sample). This is the expensive step — every training sample requires a full CFR solve
3. **Record the training pair**:
   - Input: P1 range (N floats) + P2 range (N floats) + pot fraction (1 float) = 2N+1 values
   - Output: counterfactual values per bucket for each player (N floats per player)

This is embarrassingly parallel — every subgame is independent. Distribute across all available CPU cores.

**Scale reference** (Supremus):

| Network | Samples | At 4K iters each |
|-|-|-|
| River | 50M | Cheapest per sample (small trees) |
| Turn | 20M | Uses river CFVnet at leaves |
| Flop | 5M | Uses turn CFVnet at leaves |
| Preflop aux | 10M | Uses flop CFVnet at leaves |

You can likely start much smaller — try 1M river samples and measure validation loss. The quality curve will tell you how much data you actually need.

**Random range generation**: The ranges fed to the solver during data generation should be realistic. Options:
- Uniform random (simple but produces unrealistic ranges)
- Sample from a precomputed preflop solution (more realistic)
- Perturb known equilibrium ranges with noise (best coverage of likely play)

### Network Architecture

**Supremus reference**: 7 fully-connected hidden layers × 500 nodes, ReLU activations, Huber loss.

**Zero-sum constraint**: The weighted sum of predicted values across both players' buckets must equal zero (the game is zero-sum). Supremus enforces this with an external constraint network. Simpler alternative: predict P1 values only, derive P2 values as the negative (weighted by range).

**Minimal starting point**: 3-4 hidden layers × 256 nodes. You can always scale up once the pipeline works. The architecture is not the hard part — data generation is.

### Training Infrastructure

**Framework options for Rust projects**:

| Framework | Language | Rust integration |
|-|-|-|
| PyTorch (Python) | Python | Generate data in Rust, export to files, train in Python. Simple but two-language |
| tch-rs | Rust | Rust bindings to libtorch. Full PyTorch API from Rust. Mature but depends on C++ libtorch |
| candle | Rust | Pure Rust ML framework (HuggingFace). No C++ deps. Newer, lighter |
| burn | Rust | Pure Rust, backend-agnostic (CPU/GPU). Most Rusty API. Active development |
| ONNX Runtime | Any | Train in Python, export ONNX, inference in Rust via ort crate. Clean separation |

**Pragmatic recommendation**: Train in Python/PyTorch (ecosystem is unbeatable for experimentation), export to ONNX, run inference in Rust via `ort`. This separates the training experimentation loop (where Python excels) from the production solver (where Rust excels). You don't want to debug training convergence issues through Rust FFI bindings.

### Training Loop

Standard supervised regression:
1. Load training pairs into memory (or stream from disk if too large)
2. Mini-batch SGD/Adam, Huber loss
3. Validate on held-out set every epoch
4. Early stopping when validation loss plateaus

**Key hyperparameters**: learning rate (~1e-3 to start), batch size (1024-4096), Huber delta (1.0).

### Validation

Network accuracy matters, but the real test is downstream solver quality:

1. **Offline**: Huber loss on held-out subgames. Supremus achieved 0.010-0.015 on river
2. **Functional**: Plug the CFVnet into depth-limited solving, solve known subgames, compare strategies to tabular ground truth
3. **End-to-end**: Measure exploitability of the full agent using the CFVnet. This is the number that actually matters

### Sequential Training Order

```
River CFVnet    ← leaf values are showdown (exact, no dependency)
    ↓
Turn CFVnet     ← leaf values from river CFVnet
    ↓
Flop CFVnet     ← leaf values from turn CFVnet
    ↓
Preflop aux     ← leaf values from flop CFVnet
```

Each network's quality is bounded by the one below it. If the river CFVnet is bad, the turn CFVnet inherits that error. This is why you validate each layer before building on top of it.

---

## Training Setup: CUDA DCFR+ Solver

### What You're Building

A DCFR+ solver that runs entirely on GPU. The tree is loaded once into VRAM, all iterations execute without CPU-GPU transfers, and the converged strategy is read back at the end. This replaces the CPU CFR inner loop — the tree builder and resolving logic stay on CPU.

### Tree Representation on GPU

The game tree must be flattened into contiguous arrays suitable for coalesced GPU memory access. No pointers, no recursion, no dynamic allocation.

**Core data structures** (flat arrays, one entry per node):

| Array | Contents | Type |
|-|-|-|
| `node_type` | Terminal / chance / player0 / player1 | u8 |
| `parent` | Index of parent node | u32 |
| `children_offset` | Start index into children array | u32 |
| `children_count` | Number of child actions | u8 |
| `infoset_id` | Which information set this node belongs to | u32 |
| `terminal_value` | Payoff at terminal nodes (0 elsewhere) | f32 |

**Per-infoset arrays** (one entry per infoset × action):

| Array | Contents | Type |
|-|-|-|
| `cumulative_regret` | Running regret sums | f32 |
| `cumulative_strategy` | Running strategy sums (for average policy) | f32 |
| `current_strategy` | Current iteration's strategy (derived from regrets) | f32 |

**Level index**: A separate array mapping depth → (start_node, end_node) for level-synchronous traversal. Pre-computed on CPU during tree construction.

**Memory layout considerations**:
- Group arrays by access pattern, not by node. During the forward pass you read `current_strategy` for all nodes at a depth — these should be contiguous
- Infoset data is accessed by infoset ID, not node order. Nodes in the same infoset may be scattered across the tree. Consider sorting nodes by infoset to improve locality
- Align to 128-byte boundaries for coalesced access

### Kernel Design

Three kernels, executed in sequence per iteration:

**1. Forward pass (top-down, per depth level)**
```
for depth = 0 to max_depth:
    launch kernel: for each node at this depth in parallel
        reach_prob[node] = reach_prob[parent] * current_strategy[parent_action]
```
One kernel launch per depth level. Each thread handles one node. Threads within a warp process adjacent nodes at the same depth (coalesced reads from level index).

**2. Backward pass (bottom-up, per depth level)**
```
for depth = max_depth to 0:
    launch kernel: for each node at this depth in parallel
        if terminal: cf_value[node] = terminal_value * opponent_reach
        else: cf_value[node] = sum(cf_value[child] * strategy[child_action])
```
Same structure, reverse direction. Terminal nodes are base cases.

**3. Regret and strategy update (per infoset)**
```
launch kernel: for each infoset in parallel
    for each action:
        regret = cf_value[action_child] - cf_value[infoset_node]
        cumulative_regret[infoset][action] += discount(t) * regret
    current_strategy = regret_match(cumulative_regret[infoset])
    if t > delay_threshold:
        cumulative_strategy[infoset] += reach_prob * current_strategy
```
This is where DCFR+ discounting and delayed averaging happen. One thread per infoset.

### DCFR+ Specifics on GPU

- **Discounting**: multiply old regrets by `t^α / (t^α + 1)` and old strategy sums by `(t / (t+1))^β` each iteration. This is a scalar multiply on the cumulative arrays — trivial on GPU
- **Delayed averaging**: skip strategy sum accumulation for first d iterations (d=100). Just a conditional in kernel 3
- **Simultaneous updates**: both players update every iteration (not alternating). The forward/backward passes already compute values for both players

### Integration with CFVnet

At leaf nodes that represent street boundaries (not showdown), the backward pass needs CFVnet values instead of terminal payoffs:

1. **Before DCFR+ loop**: identify all street-boundary leaf nodes, batch their (range, pot) inputs, run CFVnet inference on GPU, store results in a `leaf_cfv` array
2. **During backward pass**: terminal nodes use `terminal_value`, street-boundary leaves use `leaf_cfv[node]`

If the CFVnet is also on GPU (ONNX Runtime with CUDA EP, or a custom network), this is a single GPU-to-GPU transfer. No CPU round-trip.

### Memory Budget

Rough sizing for a single-street lookahead:

| Component | Per-node/infoset | Typical count | Memory |
|-|-|-|-|
| Node arrays (type, parent, children, etc.) | ~20 bytes | 50K-500K nodes | 1-10 MB |
| Infoset arrays (regret, strategy × actions) | ~40 bytes × actions | 10K-100K infosets | 1-20 MB |
| Reach probabilities | 4 bytes | 50K-500K nodes | 0.2-2 MB |
| CF values | 4 bytes | 50K-500K nodes | 0.2-2 MB |
| CFVnet model weights | — | — | 5-20 MB |

**Total**: 10-60 MB per subgame. Comfortably fits in any modern GPU (even a 4GB card). This is exactly why depth-limiting works — you're solving a tiny tree, not the full game.

### Build Infrastructure

**CUDA toolchain options for Rust**:

| Approach | Description |
|-|-|
| Raw CUDA + FFI | Write kernels in .cu files, compile with nvcc, call from Rust via `extern "C"`. Maximum control, most work |
| cudarc | Rust crate wrapping the CUDA driver API. Load PTX modules, launch kernels, manage memory from Rust. No C++ needed at build time if you pre-compile PTX |
| wgpu / vulkano | GPU compute via Vulkan/WebGPU. Cross-platform (no NVIDIA lock-in) but lower-level than CUDA, less mature for compute |
| OpenCL (ocl crate) | Cross-vendor GPU compute. Mature but less ecosystem support than CUDA |

**Pragmatic recommendation**: Write kernels in CUDA (.cu), pre-compile to PTX, load via `cudarc` from Rust. This gives you CUDA's full power for the kernels while keeping the host code in Rust. Avoid wgpu/Vulkan unless cross-vendor support is a hard requirement — the CUDA ecosystem (cuSPARSE, cuBLAS, profiling tools) is significantly ahead for this workload.

### Development Sequence

1. **CPU reference first**: Implement DCFR+ on CPU using the same flat-array representation (no tree pointers). This is your ground-truth for correctness testing
2. **Port to GPU**: Translate the three kernels to CUDA. Compare results against CPU reference on small games (Kuhn, Leduc)
3. **Benchmark on river**: Run on HUNL river endgames. Measure iterations/second vs CPU
4. **Add CFVnet leaves**: Integrate ONNX Runtime (or custom net) for leaf evaluation. Extend to turn subgames
5. **Profile and optimize**: Use Nsight Compute to find bottlenecks — likely memory access patterns and warp divergence at irregular branching points

---

## Sources

- [GPU-Accelerated CFR (Kim 2024)](https://arxiv.org/html/2408.14778v1)
- [GPU-Accelerated CFR OpenReview](https://openreview.net/forum?id=dWsBrgaNzU)
- [GPUCFR](https://github.com/janrvdolf/gpucfr)
- [Solving Large Sequential Games with EGT (Kroer, Farina, Sandholm 2018)](https://www.mit.edu/~gfarina/2018/gpu-egt-nips18/gpu_egt.nips18.cr.pdf)
- [Inside Supremus (poker-ai.org)](https://poker-ai.org/inside-supremus-and-the-evolution-of-poker-ai-research/)
- [Unlocking the Potential of Deep CFVnets (arxiv)](https://ar5iv.labs.arxiv.org/html/2007.10442)
- [GPU Acceleration for CFR Poker - Practical Findings](https://lattice.uptownhr.com/cfr-poker-gpu/gpu-acceleration-findings)
- [GTO Wizard AI Explained](https://blog.gtowizard.com/gto-wizard-ai-explained/)
- [GTO Wizard AI Benchmarks](https://blog.gtowizard.com/gto-wizard-ai-benchmarks/)
- [PioSOLVER Hardware FAQ](https://piosolver.com/docs/faq/hardware/)
- [DeepStack Paper](https://arxiv.org/pdf/1701.01724)
- [ReBeL (Facebook Research)](https://github.com/facebookresearch/rebel)
- [Deep CFR (Steinberger)](https://github.com/EricSteinberger/Deep-CFR)
- [NVIDIA: Thinking Parallel - Tree Traversal on GPU](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/)
- [Depth-Limited Solving (Brown & Sandholm 2018)](https://arxiv.org/pdf/1805.08195)
