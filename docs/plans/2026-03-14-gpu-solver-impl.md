# GPU Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Build a Supremus-style GPU DCFR+ solver for real-time continual resolving and CFVNet training, using `cudarc` for custom CUDA kernels and `burn-cuda` for neural network inference/training.

**Architecture:** New `crates/gpu-solver/` crate. Flat level-order arrays uploaded once to GPU, DCFR+ iteration loop runs entirely on-GPU via custom CUDA kernels launched through `cudarc`. Neural network leaf evaluation via `burn-cuda`. CLI mirrors existing `range-solve` command.

**Tech Stack:** Rust nightly, cudarc (CUDA driver API), burn-cuda (neural net inference), rs_poker (hand evaluation), clap (CLI)

---

## Phase 1: GPU DCFR+ Solver

**Goal:** Solve postflop subtrees on GPU with a CLI identical to `range-solve`. Validate correctness by comparing output against the existing range-solver.

---

### Task 1: Crate Scaffolding

**Files:**
- Create: `crates/gpu-solver/Cargo.toml`
- Create: `crates/gpu-solver/src/lib.rs`
- Modify: `Cargo.toml` (workspace members)

**Step 1: Add crate to workspace**

Add `"crates/gpu-solver"` to workspace members in `/Users/coreco/code/poker_solver_rust/Cargo.toml`.

**Step 2: Create Cargo.toml**

```toml
[package]
name = "poker-solver-gpu"
version = "0.1.0"
edition = "2021"

[dependencies]
cudarc = { version = "0.16", features = ["std", "driver", "nvrtc"] }
poker-solver-core = { path = "../core" }
rs_poker = { workspace = true }
thiserror = { workspace = true }
rayon = { workspace = true }

[dev-dependencies]
approx = "0.5"
tempfile = { workspace = true }
```

**Step 3: Create lib.rs**

```rust
pub mod tree;
pub mod gpu;
pub mod solver;

#[cfg(test)]
mod tests;
```

**Step 4: Create empty modules**

Create empty files:
- `crates/gpu-solver/src/tree.rs`
- `crates/gpu-solver/src/gpu.rs`
- `crates/gpu-solver/src/solver.rs`
- `crates/gpu-solver/src/tests.rs`

**Step 5: Verify it compiles**

Run: `cargo check -p poker-solver-gpu`
Expected: Compiles with no errors.

**Step 6: Commit**

```bash
git add crates/gpu-solver/ Cargo.toml
git commit -m "feat(gpu-solver): scaffold crate with cudarc dependency"
```

---

### Task 2: Flat Tree Data Structures

**Files:**
- Create: `crates/gpu-solver/src/tree.rs`
- Test: `crates/gpu-solver/src/tree.rs` (inline tests)

**Step 1: Write the failing test**

In `crates/gpu-solver/src/tree.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_tree_structure() {
        // Root (OOP decision) -> [Fold, Check]
        //   Fold -> Terminal(fold, IP wins pot=100)
        //   Check -> (IP decision) -> [Check, Bet50]
        //     Check -> Terminal(showdown, pot=100)
        //     Bet50 -> (OOP decision) -> [Fold, Call]
        //       Fold -> Terminal(fold, IP wins pot=150)
        //       Call -> Terminal(showdown, pot=200)

        let tree = FlatTree::new_test_tree();

        // 7 nodes total in BFS order
        assert_eq!(tree.num_nodes(), 7);
        // 3 levels: root, IP decision + 2 terminals, OOP decision + terminal, 2 terminals
        assert_eq!(tree.num_levels(), 4);
        // 3 decision nodes -> 3 infosets (each unique since single board)
        assert_eq!(tree.num_infosets(), 3);
        // Root has 2 children
        assert_eq!(tree.num_children(0), 2);
        // Level 0 has 1 node (root)
        assert_eq!(tree.level_node_count(0), 1);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-gpu test_tiny_tree_structure`
Expected: FAIL — `FlatTree` not defined.

**Step 3: Implement the flat tree types**

```rust
/// Node type discriminant for GPU
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeType {
    DecisionOop = 0,
    DecisionIp = 1,
    TerminalFold = 2,
    TerminalShowdown = 3,
}

/// Flat level-order game tree for GPU upload.
/// All arrays indexed by node_id (BFS order).
#[derive(Debug)]
pub struct FlatTree {
    // Per-node metadata
    pub node_types: Vec<NodeType>,
    pub pots: Vec<f32>,

    // CSR child indexing: children of node i are
    // children[child_offsets[i]..child_offsets[i+1]]
    pub child_offsets: Vec<u32>,
    pub children: Vec<u32>,

    // Parent tracking (for reach propagation)
    // parent_nodes[i] = parent of node i (root has u32::MAX)
    // parent_actions[i] = which action index of parent leads to node i
    pub parent_nodes: Vec<u32>,
    pub parent_actions: Vec<u32>,

    // Level boundaries: nodes at level l are
    // node_ids[level_starts[l]..level_starts[l+1]]
    pub level_starts: Vec<u32>,

    // Infoset mapping: decision node i -> infoset_ids[i]
    // Non-decision nodes have u32::MAX
    pub infoset_ids: Vec<u32>,
    pub infoset_num_actions: Vec<u32>,
    pub num_infosets: usize,

    // Terminal info
    pub terminal_indices: Vec<u32>,

    // Showdown equity: precomputed hand-vs-hand payoff matrix
    // For each showdown terminal, maps to an equity_table_id
    // equity_tables[id] is a num_hands x num_hands matrix
    pub showdown_equity_ids: Vec<u32>, // indexed by terminal ordinal
    pub equity_tables: Vec<Vec<f32>>,  // [table_id][h1 * num_hands + h2] -> payoff

    // Fold payoffs: fold_payoffs[terminal_ordinal][hand] -> payoff
    pub fold_payoffs: Vec<Vec<f32>>,

    pub num_hands: usize,
}

impl FlatTree {
    pub fn num_nodes(&self) -> usize {
        self.node_types.len()
    }

    pub fn num_levels(&self) -> usize {
        self.level_starts.len() - 1
    }

    pub fn num_children(&self, node: usize) -> usize {
        (self.child_offsets[node + 1] - self.child_offsets[node]) as usize
    }

    pub fn level_node_count(&self, level: usize) -> usize {
        (self.level_starts[level + 1] - self.level_starts[level]) as usize
    }

    pub fn player(&self, node: usize) -> u8 {
        match self.node_types[node] {
            NodeType::DecisionOop => 0,
            NodeType::DecisionIp => 1,
            _ => u8::MAX,
        }
    }

    pub fn is_terminal(&self, node: usize) -> bool {
        matches!(
            self.node_types[node],
            NodeType::TerminalFold | NodeType::TerminalShowdown
        )
    }

    /// Build a tiny test tree for unit tests.
    /// Root (OOP) -> [Fold, Check]
    ///   Fold -> Terminal(fold)
    ///   Check -> IP -> [Check, Bet50]
    ///     Check -> Terminal(showdown, pot=100)
    ///     Bet50 -> OOP -> [Fold, Call]
    ///       Fold -> Terminal(fold)
    ///       Call -> Terminal(showdown, pot=200)
    pub fn new_test_tree() -> Self {
        // BFS order: [Root(0), Fold(1), IPDec(2), SDCheck(3), OOPDec(4), FoldBet(5), SDCall(6)]
        let node_types = vec![
            NodeType::DecisionOop,      // 0: root
            NodeType::TerminalFold,     // 1: fold after root
            NodeType::DecisionIp,       // 2: IP decision
            NodeType::TerminalShowdown, // 3: check-check showdown
            NodeType::DecisionOop,      // 4: OOP facing bet
            NodeType::TerminalFold,     // 5: fold to bet
            NodeType::TerminalShowdown, // 6: call showdown
        ];
        let pots = vec![100.0, 100.0, 100.0, 100.0, 150.0, 150.0, 200.0];
        let child_offsets = vec![0, 2, 2, 4, 4, 6, 6, 6];
        let children = vec![1, 2, 3, 4, 5, 6];
        let parent_nodes = vec![u32::MAX, 0, 0, 2, 2, 4, 4];
        let parent_actions = vec![u32::MAX, 0, 1, 0, 1, 0, 1];
        let level_starts = vec![0, 1, 3, 5, 7];
        let infoset_ids = vec![0, u32::MAX, 1, u32::MAX, 2, u32::MAX, u32::MAX];
        let infoset_num_actions = vec![2, 2, 2];
        let terminal_indices = vec![1, 3, 5, 6];

        FlatTree {
            node_types,
            pots,
            child_offsets,
            children,
            parent_nodes,
            parent_actions,
            level_starts,
            infoset_ids,
            infoset_num_actions,
            num_infosets: 3,
            terminal_indices,
            showdown_equity_ids: vec![],
            equity_tables: vec![],
            fold_payoffs: vec![],
            num_hands: 2, // placeholder for test
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-gpu test_tiny_tree_structure`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/gpu-solver/src/tree.rs
git commit -m "feat(gpu-solver): add flat tree data structures"
```

---

### Task 3: Tree Builder from Range-Solver Game Tree

**Files:**
- Modify: `crates/gpu-solver/src/tree.rs`
- Modify: `crates/gpu-solver/Cargo.toml` (add range-solver dependency)
- Test: inline tests

This task builds a `FlatTree` from a `range_solver::PostFlopGame` by walking its recursive tree structure and flattening into BFS-ordered arrays. This is the bridge between the existing solver infrastructure and the GPU layout.

**Step 1: Add range-solver dependency**

In `crates/gpu-solver/Cargo.toml`, add:
```toml
range-solver = { path = "../range-solver" }
```

**Step 2: Write the failing test**

```rust
#[test]
fn test_build_from_postflop_game() {
    use range_solver::{PostFlopGame, CardConfig, ActionTree, TreeConfig, BoardState, Range};

    let oop_range = "AA".parse::<Range>().unwrap();
    let ip_range = "KK".parse::<Range>().unwrap();

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [
            Card::new(12, 3), // As
            Card::new(11, 2), // Kh
            Card::new(0, 0),  // 2c
        ],
        turn: Card::NOT_DEALT,
        river: Card::NOT_DEALT,
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 100,
        effective_stack: 100,
        river_bet_sizes: [
            BetSizeOptions::try_from(("50%,100%", "60%,100%")).unwrap(),
            BetSizeOptions::try_from(("50%,100%", "60%,100%")).unwrap(),
        ],
        // ... other sizes
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.allocate_memory(false);

    let flat = FlatTree::from_postflop_game(&game);

    // Should have at least root + some children
    assert!(flat.num_nodes() > 1);
    // Root should be OOP decision
    assert_eq!(flat.node_types[0], NodeType::DecisionOop);
    // num_hands should match game
    assert_eq!(flat.num_hands, game.num_private_hands(0));
    // Level 0 should have exactly 1 node
    assert_eq!(flat.level_node_count(0), 1);
    // All nodes should have valid parents except root
    for i in 1..flat.num_nodes() {
        assert_ne!(flat.parent_nodes[i], u32::MAX);
    }
}
```

**Step 3: Implement `FlatTree::from_postflop_game`**

Walk the PostFlopGame tree via BFS. For each node:
- Determine type (decision OOP/IP, terminal fold/showdown, chance)
- Record pot, parent, action index
- For decision nodes, assign infoset IDs
- For showdown terminals, precompute the hand-vs-hand equity matrix using `game.evaluate()` or hand strength tables
- For fold terminals, precompute per-hand fold payoffs

Skip chance nodes: since range-solver handles chance by iterating over turn/river cards internally, the GPU solver for Phase 1 operates at a single board state (no chance nodes). Multi-street with chance nodes comes in Phase 3+.

The BFS walk uses the `GameNode::play(action)` method from range-solver to traverse children.

For equity tables at showdown:
- Use the range-solver's hand strength tables (`hand_strength[board_state][player]`)
- For each pair of hands, determine winner
- Build the payoff matrix: `payoff[h1][h2] = pot/2` if h1 wins, `-pot/2` if h1 loses, `0` if tie
- Account for card blocking (hands sharing cards can't co-exist)

For fold payoffs:
- Folding player pays `pot/2` to the other player
- Account for card blocking in the sum

**Step 4: Run test, verify pass**

Run: `cargo test -p poker-solver-gpu test_build_from_postflop_game`
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat(gpu-solver): build FlatTree from PostFlopGame"
```

---

### Task 4: CUDA Device Wrapper

**Files:**
- Create: `crates/gpu-solver/src/gpu.rs`
- Test: inline tests (gated behind `#[cfg(feature = "cuda")]`)

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_round_trip() {
        let gpu = GpuContext::new(0).expect("CUDA device required");
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let gpu_buf = gpu.upload(&data).unwrap();
        let result = gpu.download(&gpu_buf).unwrap();
        assert_eq!(data, result);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-gpu test_gpu_round_trip`
Expected: FAIL — `GpuContext` not defined.

**Step 3: Implement GpuContext**

```rust
use cudarc::driver::{CudaContext, CudaStream, CudaSlice, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

pub struct GpuContext {
    pub ctx: Arc<CudaContext>,
    pub stream: CudaStream,
}

impl GpuContext {
    pub fn new(device_ordinal: usize) -> Result<Self, GpuError> {
        let ctx = CudaContext::new(device_ordinal)?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream })
    }

    pub fn upload<T: cudarc::driver::DeviceRepr + Copy>(
        &self,
        data: &[T],
    ) -> Result<CudaSlice<T>, GpuError> {
        Ok(self.stream.clone_htod(data)?)
    }

    pub fn download<T: cudarc::driver::DeviceRepr + Copy>(
        &self,
        buf: &CudaSlice<T>,
    ) -> Result<Vec<T>, GpuError> {
        Ok(self.stream.clone_dtoh(buf)?)
    }

    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + Copy>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, GpuError> {
        Ok(self.stream.alloc_zeros(len)?)
    }

    pub fn compile_and_load(
        &self,
        source: &str,
        function_name: &str,
    ) -> Result<cudarc::driver::CudaFunction, GpuError> {
        let ptx = compile_ptx(source)?;
        let module = self.ctx.load_module(ptx)?;
        Ok(module.load_function(function_name)?)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("NVRTC compilation error: {0}")]
    Compile(#[from] cudarc::nvrtc::CompileError),
}
```

**Step 4: Run test, verify pass**

Run: `cargo test -p poker-solver-gpu test_gpu_round_trip`
Expected: PASS (on a machine with CUDA)

**Step 5: Commit**

```bash
git commit -m "feat(gpu-solver): add CUDA device wrapper with upload/download"
```

---

### Task 5: Regret Match Kernel

**Files:**
- Create: `crates/gpu-solver/kernels/regret_match.cu`
- Modify: `crates/gpu-solver/src/gpu.rs`
- Test: inline tests

The simplest kernel — one thread per infoset. Reads cumulative regrets, outputs current strategy via regret matching.

**Step 1: Write the failing test**

```rust
#[test]
fn test_regret_match_kernel() {
    let gpu = GpuContext::new(0).unwrap();

    // 2 infosets, each with 3 actions
    let num_infosets: u32 = 2;
    let max_actions: u32 = 3;

    // Infoset 0: regrets [10, -5, 20] -> strategy [10/30, 0, 20/30] = [0.333, 0, 0.667]
    // Infoset 1: regrets [-1, -2, -3] -> all negative -> uniform [0.333, 0.333, 0.333]
    let regrets: Vec<f32> = vec![10.0, -5.0, 20.0, -1.0, -2.0, -3.0];
    let num_actions: Vec<u32> = vec![3, 3];

    let gpu_regrets = gpu.upload(&regrets).unwrap();
    let gpu_num_actions = gpu.upload(&num_actions).unwrap();
    let gpu_strategy = gpu.alloc_zeros::<f32>(6).unwrap();

    gpu.launch_regret_match(
        &gpu_regrets,
        &gpu_num_actions,
        &gpu_strategy,
        num_infosets,
        max_actions,
    ).unwrap();

    let strategy = gpu.download(&gpu_strategy).unwrap();

    let eps = 1e-5;
    // Infoset 0
    assert!((strategy[0] - 1.0 / 3.0).abs() < eps);
    assert!((strategy[1] - 0.0).abs() < eps);
    assert!((strategy[2] - 2.0 / 3.0).abs() < eps);
    // Infoset 1: uniform
    assert!((strategy[3] - 1.0 / 3.0).abs() < eps);
    assert!((strategy[4] - 1.0 / 3.0).abs() < eps);
    assert!((strategy[5] - 1.0 / 3.0).abs() < eps);
}
```

**Step 2: Write the CUDA kernel**

Create `crates/gpu-solver/kernels/regret_match.cu`:

```cuda
extern "C" __global__ void regret_match(
    const float* regrets,       // [num_infosets * max_actions]
    const unsigned int* num_actions, // [num_infosets]
    float* strategy,            // [num_infosets * max_actions]
    unsigned int num_infosets,
    unsigned int max_actions
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_infosets) return;

    unsigned int n = num_actions[i];
    unsigned int base = i * max_actions;

    // Sum positive regrets
    float pos_sum = 0.0f;
    for (unsigned int a = 0; a < n; a++) {
        float r = regrets[base + a];
        if (r > 0.0f) pos_sum += r;
    }

    // Compute strategy
    if (pos_sum > 0.0f) {
        for (unsigned int a = 0; a < n; a++) {
            float r = regrets[base + a];
            strategy[base + a] = (r > 0.0f) ? (r / pos_sum) : 0.0f;
        }
    } else {
        // Uniform
        float uniform = 1.0f / (float)n;
        for (unsigned int a = 0; a < n; a++) {
            strategy[base + a] = uniform;
        }
    }

    // Zero out unused action slots
    for (unsigned int a = n; a < max_actions; a++) {
        strategy[base + a] = 0.0f;
    }
}
```

**Step 3: Add launch method to GpuContext**

```rust
impl GpuContext {
    pub fn launch_regret_match(
        &self,
        regrets: &CudaSlice<f32>,
        num_actions: &CudaSlice<u32>,
        strategy: &mut CudaSlice<f32>,
        num_infosets: u32,
        max_actions: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/regret_match.cu"),
            "regret_match",
        )?;
        let cfg = LaunchConfig::for_num_elems(num_infosets as u32);
        unsafe {
            kernel.launch_on_stream(
                &self.stream,
                cfg,
                (regrets, num_actions, strategy, num_infosets, max_actions),
            )?;
        }
        Ok(())
    }
}
```

**Step 4: Run test, verify pass**

Run: `cargo test -p poker-solver-gpu test_regret_match_kernel`
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat(gpu-solver): add regret_match CUDA kernel"
```

---

### Task 6: Forward Reach Kernel

**Files:**
- Create: `crates/gpu-solver/kernels/forward_reach.cu`
- Modify: `crates/gpu-solver/src/gpu.rs`
- Test: inline tests

Propagates reach probabilities top-down. One thread per (node, hand) pair at a given level. Each non-root node's reach = parent's reach × strategy probability of the action leading to this node.

**Step 1: Write the failing test**

```rust
#[test]
fn test_forward_reach_kernel() {
    let gpu = GpuContext::new(0).unwrap();

    // Tree: Root(0) -> [Child(1), Child(2)]
    // Root is infoset 0 with strategy [0.6, 0.4]
    // 2 hands
    // Initial reach for hand 0: 1.0, hand 1: 0.5

    let num_hands: u32 = 2;
    let num_nodes_this_level: u32 = 2; // children at level 1

    // reach_probs[node * num_hands + hand]
    // Root (node 0) reach: [1.0, 0.5]
    let mut reach = vec![1.0f32, 0.5, 0.0, 0.0]; // nodes 0,1 — child reach starts at 0
    let strategy = vec![0.6f32, 0.4]; // infoset 0: [action0=0.6, action1=0.4]

    let parent_nodes = vec![0u32, 0];    // both children have parent 0
    let parent_actions = vec![0u32, 1];  // child 0 via action 0, child 1 via action 1
    let parent_infosets = vec![0u32, 0]; // parent's infoset
    let max_actions: u32 = 2;

    // Node indices at this level
    let level_nodes = vec![1u32, 2]; // but we're only testing 2 children (nodes 1,2)

    let gpu_reach = gpu.upload(&reach).unwrap();
    let gpu_strategy = gpu.upload(&strategy).unwrap();
    // ... upload other arrays, call kernel

    // Expected:
    // Child 1 (action 0): reach = parent_reach * strategy[0] = [1.0*0.6, 0.5*0.6] = [0.6, 0.3]
    // Child 2 (action 1): reach = parent_reach * strategy[1] = [1.0*0.4, 0.5*0.4] = [0.4, 0.2]
}
```

**Step 2: Write the CUDA kernel**

Create `crates/gpu-solver/kernels/forward_reach.cu`:

```cuda
extern "C" __global__ void forward_reach(
    float* reach_probs,              // [num_nodes * num_hands], read parent, write child
    const float* strategy,           // [num_infosets * max_actions]
    const unsigned int* level_nodes, // node indices at this level
    const unsigned int* parent_nodes,// [num_nodes_this_level] -> parent node index
    const unsigned int* parent_actions,// [num_nodes_this_level] -> action index
    const unsigned int* parent_infosets,// [num_nodes_this_level] -> parent's infoset
    unsigned int num_nodes_this_level,
    unsigned int num_hands,
    unsigned int max_actions
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int node_local = tid / num_hands;
    unsigned int hand = tid % num_hands;

    if (node_local >= num_nodes_this_level) return;

    unsigned int node = level_nodes[node_local];
    unsigned int parent = parent_nodes[node_local];
    unsigned int action = parent_actions[node_local];
    unsigned int infoset = parent_infosets[node_local];

    float parent_reach = reach_probs[parent * num_hands + hand];
    float action_prob = strategy[infoset * max_actions + action];

    reach_probs[node * num_hands + hand] = parent_reach * action_prob;
}
```

**Step 3: Implement launch method and test**

Follow the same pattern as Task 5. Verify GPU output matches hand-computed expected values.

**Step 4: Commit**

```bash
git commit -m "feat(gpu-solver): add forward_reach CUDA kernel"
```

---

### Task 7: Terminal Evaluation Kernel

**Files:**
- Create: `crates/gpu-solver/kernels/terminal_eval.cu`
- Modify: `crates/gpu-solver/src/gpu.rs`
- Test: inline tests

Two sub-kernels: fold evaluation and showdown evaluation.

**Fold evaluation:** For each fold terminal node and each hand h of the winning player:
```
cfv[node][h] = fold_payoff[h] * (total_opp_reach - blocked_opp_reach[h])
```
where `fold_payoff[h]` accounts for pot size and blocking.

**Showdown evaluation:** For each showdown terminal and each hand h:
```
cfv[node][h] = sum_over_opp_hands(equity[h][h'] * reach[opp][h'])
```
This is a matrix-vector multiply: `cfv = equity_matrix @ opp_reach`.

**Step 1: Write the failing test**

Test with 3 hands, known equity relationships. Hand 0 beats Hand 1 beats Hand 2. Verify fold payoffs and showdown CFVs match hand computation.

**Step 2: Write the CUDA kernels**

Fold kernel: one thread per (terminal, hand). Accumulates opponent reach excluding blocked cards.

Showdown kernel: one thread per (terminal, hand). Performs dot product of equity row with opponent reach vector. Use shared memory for the opponent reach vector if `num_hands` is small enough.

```cuda
extern "C" __global__ void terminal_fold_eval(
    float* cfvalues,                    // [num_nodes * num_hands]
    const float* reach_probs,           // [num_nodes * num_hands]
    const unsigned int* terminal_nodes, // which nodes are fold terminals
    const float* fold_payoffs,          // [num_terminals * num_hands]
    const unsigned int* fold_players,   // who folded (0=OOP, 1=IP)
    unsigned int num_terminals,
    unsigned int num_hands
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int term_idx = tid / num_hands;
    unsigned int hand = tid % num_hands;
    if (term_idx >= num_terminals) return;

    unsigned int node = terminal_nodes[term_idx];
    float payoff = fold_payoffs[term_idx * num_hands + hand];

    // Opponent reach sum (excluding blocked hands — handled in payoff precomputation)
    cfvalues[node * num_hands + hand] = payoff;
}

extern "C" __global__ void terminal_showdown_eval(
    float* cfvalues,                     // [num_nodes * num_hands]
    const float* reach_probs,            // [num_nodes * num_hands]
    const unsigned int* terminal_nodes,
    const float* equity_tables,          // [num_tables * num_hands * num_hands]
    const unsigned int* equity_ids,      // terminal -> table
    unsigned int num_terminals,
    unsigned int num_hands
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int term_idx = tid / num_hands;
    unsigned int hand = tid % num_hands;
    if (term_idx >= num_terminals) return;

    unsigned int node = terminal_nodes[term_idx];
    unsigned int table_id = equity_ids[term_idx];
    const float* eq_row = &equity_tables[table_id * num_hands * num_hands + hand * num_hands];

    // Dot product: cfv = sum(equity[hand][opp] * opp_reach[opp])
    float cfv = 0.0f;
    for (unsigned int opp = 0; opp < num_hands; opp++) {
        cfv += eq_row[opp] * reach_probs[node * num_hands + opp];
    }
    cfvalues[node * num_hands + hand] = cfv;
}
```

Note: The actual payoff computation must handle which player is the traverser. The equity table should encode signed payoffs (positive = traverser wins, negative = traverser loses) with card blocking already zeroed out. This is precomputed on CPU during `FlatTree::from_postflop_game`.

**Step 3: Implement, test, commit**

```bash
git commit -m "feat(gpu-solver): add terminal fold and showdown CUDA kernels"
```

---

### Task 8: Backward CFV Kernel

**Files:**
- Create: `crates/gpu-solver/kernels/backward_cfv.cu`
- Test: inline tests

Propagates counterfactual values bottom-up. For each decision node at a given level, computes CFV as weighted sum of children's CFVs. For the acting player's perspective, also computes per-action counterfactual values (needed for regret update).

**Step 1: Write the failing test**

Small tree with known child CFVs. Verify parent CFV = sum(strategy[a] * child_cfv[a]).

**Step 2: Write the CUDA kernel**

```cuda
extern "C" __global__ void backward_cfv(
    float* cfvalues,                    // [num_nodes * num_hands] — children written, parents to write
    const float* strategy,              // [num_infosets * max_actions]
    const unsigned int* level_nodes,    // decision nodes at this level
    const unsigned int* child_offsets,  // CSR child indexing
    const unsigned int* children_arr,
    const unsigned int* infoset_ids,
    unsigned int num_nodes_this_level,
    unsigned int num_hands,
    unsigned int max_actions
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int node_local = tid / num_hands;
    unsigned int hand = tid % num_hands;
    if (node_local >= num_nodes_this_level) return;

    unsigned int node = level_nodes[node_local];
    unsigned int infoset = infoset_ids[node];

    // Skip terminals (their cfvalues are already set)
    if (infoset == 0xFFFFFFFF) return;

    unsigned int first_child = child_offsets[node];
    unsigned int last_child = child_offsets[node + 1];
    unsigned int n_actions = last_child - first_child;

    float node_cfv = 0.0f;
    for (unsigned int a = 0; a < n_actions; a++) {
        unsigned int child = children_arr[first_child + a];
        float child_cfv = cfvalues[child * num_hands + hand];
        float action_prob = strategy[infoset * max_actions + a];
        node_cfv += action_prob * child_cfv;
    }

    cfvalues[node * num_hands + hand] = node_cfv;
}
```

**Step 3: Implement, test, commit**

```bash
git commit -m "feat(gpu-solver): add backward_cfv CUDA kernel"
```

---

### Task 9: Update Regrets Kernel (DCFR+)

**Files:**
- Create: `crates/gpu-solver/kernels/update_regrets.cu`
- Test: inline tests

Updates cumulative regrets and strategy sums with DCFR+ weighting. For each infoset and action:
```
instantaneous_regret[a] = cfv_child[a] - cfv_node
regret[a] = regret[a] * discount + instantaneous_regret[a]
strategy_sum[a] += weight * strategy[a]
```

DCFR+ weighting: `weight = max(0, t - d)` where `d = 100`. Regret discount: positive regrets scaled by `t^alpha / (t^alpha + 1)`, negative by `t^beta / (t^beta + 1)`. For Supremus-style DCFR+: `alpha = 1.5, beta = 0.5`.

**Step 1: Write the failing test**

Initialize known regrets, run one update step with known CFVs, verify regret values match hand computation.

**Step 2: Write the CUDA kernel**

```cuda
extern "C" __global__ void update_regrets(
    float* regrets,                  // [num_infosets * max_actions]
    float* strategy_sum,             // [num_infosets * max_actions]
    const float* strategy,           // [num_infosets * max_actions]
    const float* cfvalues,           // [num_nodes * num_hands]
    const unsigned int* decision_nodes,  // nodes that are infoset owners
    const unsigned int* child_offsets,
    const unsigned int* children_arr,
    const unsigned int* infoset_ids,
    const unsigned int* num_actions_arr,
    unsigned int num_decision_nodes,
    unsigned int num_hands,
    unsigned int max_actions,
    unsigned int iteration,          // current iteration t
    float pos_discount,              // t^alpha / (t^alpha + 1)
    float neg_discount,              // t^beta / (t^beta + 1)
    float strat_weight               // max(0, t - delay)
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dec_local = tid / (max_actions * num_hands);
    unsigned int remainder = tid % (max_actions * num_hands);
    unsigned int action = remainder / num_hands;
    unsigned int hand = remainder % num_hands;

    if (dec_local >= num_decision_nodes) return;

    unsigned int node = decision_nodes[dec_local];
    unsigned int infoset = infoset_ids[node];
    unsigned int n_actions = num_actions_arr[infoset];

    if (action >= n_actions) return;

    unsigned int first_child = child_offsets[node];
    unsigned int child = children_arr[first_child + action];

    float child_cfv = cfvalues[child * num_hands + hand];
    float node_cfv = cfvalues[node * num_hands + hand];
    float inst_regret = child_cfv - node_cfv;

    unsigned int reg_idx = infoset * max_actions + action;

    // DCFR+ regret update
    float old_regret = regrets[reg_idx * num_hands + hand];
    float discount = (old_regret >= 0.0f) ? pos_discount : neg_discount;
    float new_regret = old_regret * discount + inst_regret;
    regrets[reg_idx * num_hands + hand] = new_regret;

    // Strategy sum update
    if (strat_weight > 0.0f) {
        strategy_sum[reg_idx * num_hands + hand] +=
            strat_weight * strategy[infoset * max_actions + action];
    }
}
```

Note: The exact indexing here needs careful attention. Regrets may be indexed per-infoset (shared across hands at the same infoset) or per-infoset-per-hand. In the range-solver, regrets are per-hand: `regrets[action * num_hands + hand]`. For the GPU bucketed solver, regrets would be per-bucket. For Phase 1 (concrete combos), regrets are per-hand.

The key design decision: are regrets accumulated across all hands in an infoset, or per-hand? In standard CFR with concrete hands, regrets are per-hand (each hand has its own regret vector at each infoset). In bucketed CFR, regrets are per-bucket. Since we're using concrete hands in Phase 1, regrets are `[num_infosets × num_actions × num_hands]`.

Revise the kernel accordingly — the regret and strategy arrays should be `[num_infosets * max_actions * num_hands]`.

**Step 3: Implement, test, commit**

```bash
git commit -m "feat(gpu-solver): add DCFR+ update_regrets CUDA kernel"
```

---

### Task 10: Extract Strategy Kernel

**Files:**
- Create: `crates/gpu-solver/kernels/extract_strategy.cu`
- Test: inline tests

Normalizes cumulative strategy sums into final action probabilities.

**Step 1: Write the failing test**

```rust
// strategy_sum = [10, 20, 30] for infoset 0 -> [1/6, 2/6, 3/6]
// strategy_sum = [0, 0, 0] for infoset 1 -> uniform [1/3, 1/3, 1/3]
```

**Step 2: Write the CUDA kernel**

```cuda
extern "C" __global__ void extract_strategy(
    const float* strategy_sum,       // [num_infosets * max_actions]
    const unsigned int* num_actions,
    float* output_strategy,          // [num_infosets * max_actions]
    unsigned int num_infosets,
    unsigned int max_actions
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_infosets) return;

    unsigned int n = num_actions[i];
    unsigned int base = i * max_actions;

    float total = 0.0f;
    for (unsigned int a = 0; a < n; a++) {
        total += strategy_sum[base + a];
    }

    if (total > 0.0f) {
        for (unsigned int a = 0; a < n; a++) {
            output_strategy[base + a] = strategy_sum[base + a] / total;
        }
    } else {
        float uniform = 1.0f / (float)n;
        for (unsigned int a = 0; a < n; a++) {
            output_strategy[base + a] = uniform;
        }
    }

    for (unsigned int a = n; a < max_actions; a++) {
        output_strategy[base + a] = 0.0f;
    }
}
```

**Step 3: Implement, test, commit**

```bash
git commit -m "feat(gpu-solver): add extract_strategy CUDA kernel"
```

---

### Task 11: GPU Solver Orchestration

**Files:**
- Modify: `crates/gpu-solver/src/solver.rs`
- Test: integration test

This ties everything together: upload tree to GPU, run the DCFR+ iteration loop calling kernels in sequence, download final strategy.

**Step 1: Write the failing test**

```rust
#[test]
fn test_gpu_solver_tiny_tree() {
    // Build a small river subtree using range-solver
    // Solve with GPU solver
    // Solve same tree with range-solver
    // Compare strategies — should converge to similar values

    let (game, flat_tree) = build_test_game(); // helper
    let gpu = GpuContext::new(0).unwrap();

    let result = GpuSolver::new(&gpu, &flat_tree)
        .solve(1000, None) // 1000 iterations
        .unwrap();

    // Compare against range-solver
    let mut game_ref = game.clone();
    let exploitability = range_solver::solve(&mut game_ref, 1000, 0.01, false);

    // Strategies should be close (not exact due to different DCFR params,
    // but should agree on which actions are preferred)
    // Check exploitability is reasonable
    assert!(result.exploitability < 5.0, "GPU exploitability too high: {}", result.exploitability);
}
```

**Step 2: Implement GpuSolver**

```rust
pub struct GpuSolver<'a> {
    gpu: &'a GpuContext,
    tree: &'a FlatTree,

    // GPU buffers
    regrets: CudaSlice<f32>,
    strategy_sum: CudaSlice<f32>,
    current_strategy: CudaSlice<f32>,
    reach_probs: CudaSlice<f32>,
    cfvalues: CudaSlice<f32>,

    // Tree structure on GPU
    gpu_tree: GpuTreeBuffers, // all the tree arrays uploaded

    // Compiled kernels
    kernels: GpuKernels,
}

pub struct SolveResult {
    /// Per-infoset action probabilities
    pub strategy: Vec<f32>,
    /// Final exploitability estimate
    pub exploitability: f32,
    /// Number of iterations run
    pub iterations: u32,
}

impl<'a> GpuSolver<'a> {
    pub fn new(gpu: &'a GpuContext, tree: &'a FlatTree) -> Result<Self, GpuError> {
        // 1. Allocate GPU buffers for regrets, strategy, reach, cfvalues
        // 2. Upload tree structure arrays
        // 3. Compile all PTX kernels
        todo!()
    }

    pub fn solve(
        &mut self,
        max_iterations: u32,
        target_exploitability: Option<f32>,
    ) -> Result<SolveResult, GpuError> {
        // Initialize reach_probs with initial ranges at root
        // For each iteration t:
        //   1. regret_match kernel
        //   2. for each level top-down: forward_reach kernel
        //   3. terminal_eval kernels (fold + showdown)
        //   4. for each level bottom-up: backward_cfv kernel
        //   5. update_regrets kernel with DCFR+ params
        // extract_strategy kernel
        // Download strategy
        todo!()
    }
}
```

**Step 3: Implement solve loop**

The iteration loop runs entirely on-GPU. Compute DCFR+ discount factors on CPU each iteration and pass as kernel arguments.

Key detail: the solver must handle **two traversals per iteration** (one for each player as traverser), matching the range-solver's approach. Each traversal computes CFVs from that player's perspective.

DCFR+ parameters per iteration `t`:
```rust
let pos_discount = t_alpha.powi(2) / (t_alpha.powi(2) + 1.0);
let neg_discount = 0.5;
let strat_weight = (t as f32 - 100.0).max(0.0);
```

**Step 4: Run test, iterate until pass**

Run: `cargo test -p poker-solver-gpu test_gpu_solver_tiny_tree`

Debug by comparing intermediate values (reach probs, cfvalues) between GPU and a manual CPU trace of the same tree.

**Step 5: Commit**

```bash
git commit -m "feat(gpu-solver): implement GPU DCFR+ solve loop"
```

---

### Task 12: CLI Matching Range-Solve

**Files:**
- Modify: `crates/trainer/src/main.rs` (add `gpu-solve` subcommand)
- Modify: `crates/trainer/Cargo.toml` (add gpu-solver dependency)

**Step 1: Add the subcommand**

Add a `GpuSolve` variant to the CLI enum with identical arguments to `RangeSolve`:

```rust
/// Solve a postflop subtree on GPU using DCFR+
GpuSolve {
    #[arg(long)]
    oop_range: String,
    #[arg(long)]
    ip_range: String,
    #[arg(long)]
    flop: String,
    #[arg(long)]
    turn: Option<String>,
    #[arg(long)]
    river: Option<String>,
    #[arg(long, default_value_t = 100)]
    pot: i32,
    #[arg(long, default_value_t = 100)]
    effective_stack: i32,
    #[arg(long, default_value_t = 1000)]
    iterations: u32,
    #[arg(long, default_value_t = 0.5)]
    target_exploitability: f32,
    #[arg(long, default_value = "50%,100%")]
    oop_bet_sizes: String,
    #[arg(long, default_value = "60%,100%")]
    oop_raise_sizes: String,
    #[arg(long, default_value = "50%,100%")]
    ip_bet_sizes: String,
    #[arg(long, default_value = "60%,100%")]
    ip_raise_sizes: String,
},
```

**Step 2: Implement the handler**

The handler should:
1. Parse arguments identically to `range-solve`
2. Build `PostFlopGame` the same way
3. Convert to `FlatTree` via `FlatTree::from_postflop_game`
4. Create `GpuContext` and `GpuSolver`
5. Solve and print results in the same format as range-solve

**Step 3: Test manually**

```bash
# CPU reference
cargo run -p poker-solver-trainer --release -- range-solve \
  --oop-range "QQ+,AKs" --ip-range "22+,AK,AQ" \
  --flop "Qs Jh 2c" --river "8d" --pot 100 --effective-stack 100

# GPU solver
cargo run -p poker-solver-trainer --release -- gpu-solve \
  --oop-range "QQ+,AKs" --ip-range "22+,AK,AQ" \
  --flop "Qs Jh 2c" --river "8d" --pot 100 --effective-stack 100
```

Compare output side by side. Strategies should be similar (not identical due to different DCFR parameterization).

**Step 4: Commit**

```bash
git commit -m "feat(gpu-solver): add gpu-solve CLI command mirroring range-solve"
```

---

### Task 13: Integration Tests

**Files:**
- Create: `crates/gpu-solver/tests/compare_range_solver.rs`

**Step 1: Write comparison tests**

```rust
/// Solve the same position with both solvers and compare.
/// Strategies won't be identical (different DCFR params) but
/// exploitability should converge and dominant actions should agree.
#[test]
fn test_river_comparison() {
    // ... build game, solve both ways, compare
}

#[test]
fn test_turn_comparison() {
    // ... same but starting from turn
}

#[test]
fn test_flop_comparison() {
    // ... same but starting from flop
}

#[test]
fn test_convergence_rate() {
    // Run GPU solver at 100, 500, 1000, 5000 iterations
    // Verify exploitability decreases monotonically
}

#[test]
fn test_benchmark_iterations_per_second() {
    // Time 1000 iterations on a river subtree
    // Print iterations/second for baseline measurement
}
```

**Step 2: Run all tests**

Run: `cargo test -p poker-solver-gpu --release`
Expected: All pass.

**Step 3: Commit**

```bash
git commit -m "test(gpu-solver): add integration tests comparing GPU vs range-solver"
```

---

## Phase 2: River CFVNet Training on GPU

**Goal:** Generate river training data and train river CFVNet model entirely on GPU.

---

### Task 14: GPU Batch Solver

**Files:**
- Create: `crates/gpu-solver/src/datagen.rs`
- Test: integration test

Add a batch solving mode: solve N independent river subgames in parallel on a single GPU. Each subgame has its own tree, ranges, and pot/stack, but they share the same CUDA kernels.

**Approach:** Since river subtrees are small and uniform in structure (same action abstraction), we can:
1. Build N flat trees on CPU (one per random situation)
2. Batch them into a single "super-tree" with N independent roots
3. Upload once, run DCFR+ iterations on the batched tree
4. Extract per-situation CFVs

Alternative: If trees vary in structure, solve them sequentially on GPU (still faster than CPU due to hand-level parallelism).

**Step 1: Write failing test**

```rust
#[test]
fn test_batch_solve_river() {
    let gpu = GpuContext::new(0).unwrap();
    let situations = generate_random_river_situations(10);
    let results = batch_solve_river(&gpu, &situations, 1000).unwrap();

    assert_eq!(results.len(), 10);
    for result in &results {
        assert_eq!(result.cfvs_oop.len(), 1326);
        assert_eq!(result.cfvs_ip.len(), 1326);
    }
}
```

**Step 2: Implement batch solver**

Reuse the situation sampling from `crates/cfvnet/src/datagen/sampler.rs`. For each situation, build a FlatTree, then solve on GPU.

**Step 3: Compare against CPU datagen**

Solve the same 100 situations with both GPU batch solver and existing CPU `solve_situation()`. Compare CFVs — should be close (within solver convergence tolerance).

**Step 4: Commit**

```bash
git commit -m "feat(gpu-solver): add batch river solving for datagen"
```

---

### Task 15: GPU-Resident Training Data Pipeline

**Files:**
- Modify: `crates/gpu-solver/src/datagen.rs`
- Create: `crates/gpu-solver/src/training.rs`

Connect batch solver output directly to burn-cuda training without moving data through CPU.

**Step 1: Write failing test**

```rust
#[test]
fn test_gpu_datagen_to_training_tensor() {
    // Generate 100 situations on GPU
    // Convert to burn tensors (already on GPU)
    // Verify tensor shapes match CFVNet input/output expectations
}
```

**Step 2: Implement data pipeline**

After batch solving, extract CFVs and construct training records:
- Input: `[1326 + 1326 + 52 + 13 + 3] = 2720` (OOP range + IP range + board + pot + stack + player)
- Target: `[1326]` CFVs
- Valid mask: `[1326]` (which combos are valid for this board)

These can be assembled as GPU tensors and fed directly to burn-cuda's training loop.

**Step 3: Commit**

```bash
git commit -m "feat(gpu-solver): GPU-resident datagen to training tensor pipeline"
```

---

### Task 16: River CFVNet Training Loop

**Files:**
- Modify: `crates/gpu-solver/src/training.rs`

**Step 1: Write failing test**

```rust
#[test]
fn test_train_river_model_gpu() {
    // Generate 10,000 training examples via GPU datagen
    // Train a small network (3 layers, 100 hidden) for 10 epochs
    // Verify loss decreases
}
```

**Step 2: Implement**

Reuse the existing CFVNet model architecture from `crates/cfvnet/src/model/network.rs` but train entirely on GPU:
1. GPU batch solve → training tensors (on GPU)
2. Shuffle buffer on GPU
3. Forward pass, loss computation, backward pass — all burn-cuda
4. Checkpoint saving

**Step 3: CLI command**

```bash
cargo run -p poker-solver-trainer --release -- gpu-train river \
  --num-samples 1000000 \
  --hidden-layers 7 --hidden-size 500 \
  --epochs 50 --batch-size 1024 \
  --output models/river_gpu.bin
```

**Step 4: Validate against CPU-trained model**

Train same architecture on same data. Compare validation loss — should be equivalent.

**Step 5: Commit**

```bash
git commit -m "feat(gpu-solver): full GPU river CFVNet training pipeline"
```

---

## Phase 3: Turn CFVNet Training on GPU

**Goal:** GPU datagen for turn subgames using Phase 2's river model as leaf evaluator, plus turn model training.

---

### Task 17: Leaf Evaluation Kernel (CFVNet Integration)

**Files:**
- Create: `crates/gpu-solver/kernels/leaf_eval.cu`
- Modify: `crates/gpu-solver/src/solver.rs`

Integrate burn-cuda CFVNet inference at depth-boundary leaf nodes. When the solver encounters a `DepthBoundary` terminal:

1. Gather bucket distributions and game state at that node
2. Build CFVNet input tensor
3. Run burn-cuda forward pass (batched across all boundary nodes)
4. Write predicted CFVs back to the cfvalues array

**Key challenge:** Sharing CUDA context between cudarc (solver kernels) and burn-cuda (neural net). Both must operate on the same GPU without redundant memory copies.

**Step 1: Write failing test**

```rust
#[test]
fn test_solver_with_leaf_eval() {
    // Build a turn subtree with DepthBoundary at river
    // Load a trained river model
    // Solve with GPU solver using model at leaves
    // Verify strategy is reasonable
}
```

**Step 2: Implement**

Modify `GpuSolver::solve()` to detect DepthBoundary nodes and invoke the neural net. The forward pass can be batched: collect all boundary node inputs into a single `[N, 2720]` tensor, run one forward pass, scatter outputs back to cfvalues.

**Step 3: Commit**

```bash
git commit -m "feat(gpu-solver): integrate CFVNet leaf evaluation in GPU solve loop"
```

---

### Task 18: Turn Datagen + Training

**Files:**
- Modify: `crates/gpu-solver/src/datagen.rs`
- Modify: `crates/gpu-solver/src/training.rs`

**Step 1: Implement turn datagen**

Sample random turn situations (4-card boards). Build turn subtrees with `depth_limit=1` (river as DepthBoundary). Solve on GPU using river model at leaves. Extract turn-level CFVs as training targets.

**Step 2: Train turn model**

Same architecture as river model but trained on turn data. Inputs include 4-card boards.

**Step 3: CLI command**

```bash
cargo run -p poker-solver-trainer --release -- gpu-train turn \
  --river-model models/river_gpu.bin \
  --num-samples 500000 \
  --output models/turn_gpu.bin
```

**Step 4: Validate**

Compare GPU-trained turn model's validation loss against existing CPU pipeline output from `turn_generate.rs`.

**Step 5: Commit**

```bash
git commit -m "feat(gpu-solver): GPU turn datagen and training pipeline"
```

---

## Phase 4: Flop + Auxiliary Preflop Model

**Goal:** Complete the neural network stack covering all streets.

---

### Task 19: Flop CFVNet Training

**Files:**
- Modify: `crates/gpu-solver/src/datagen.rs`
- Modify: `crates/gpu-solver/src/training.rs`

Same pattern as turn: sample flop situations, solve with turn model at leaves, train flop model.

**Step 1: Implement flop datagen**

Sample 3-card flop boards. Build flop subtrees with depth boundary at turn. Solve using turn model at leaves.

**Step 2: Train flop model**

```bash
cargo run -p poker-solver-trainer --release -- gpu-train flop \
  --turn-model models/turn_gpu.bin \
  --num-samples 200000 \
  --output models/flop_gpu.bin
```

**Step 3: Commit**

```bash
git commit -m "feat(gpu-solver): GPU flop datagen and training pipeline"
```

---

### Task 20: Auxiliary Preflop Model

**Files:**
- Modify: `crates/gpu-solver/src/datagen.rs`
- Modify: `crates/gpu-solver/src/training.rs`

The auxiliary preflop network handles preflop decision making. Sample preflop situations, resolve using flop model at leaves.

**Step 1: Implement preflop datagen**

Generate preflop starting situations with varying open sizes, 3-bet trees, etc. Solve with flop model at leaves.

**Step 2: Train preflop model**

```bash
cargo run -p poker-solver-trainer --release -- gpu-train preflop \
  --flop-model models/flop_gpu.bin \
  --num-samples 100000 \
  --output models/preflop_gpu.bin
```

**Step 3: Full model stack validation**

Resolve a hand from preflop through river using the complete model stack. Verify strategies are reasonable at each street.

**Step 4: Commit**

```bash
git commit -m "feat(gpu-solver): complete model stack with preflop auxiliary network"
```

---

## Phase 5: Explorer Integration for HU Games

**Goal:** Wire GPU solver into the Tauri frontend for interactive heads-up resolving.

---

### Task 21: GPU Resolve Tauri Command

**Files:**
- Modify: `crates/tauri-app/src/lib.rs`
- Modify: `crates/tauri-app/Cargo.toml`

**Step 1: Add Tauri command**

```rust
#[tauri::command]
async fn gpu_resolve(
    board: Vec<String>,
    pot: f64,
    stacks: f64,
    oop_range: Vec<f32>,
    ip_range: Vec<f32>,
    model_stack: String, // path to model directory
    state: State<'_, GpuState>,
) -> Result<ResolveResult, String> {
    // Build lookahead tree
    // Load appropriate model for leaf evaluation
    // Solve on GPU
    // Return strategy matrix
}
```

**Step 2: Model stack management**

```rust
struct GpuState {
    context: Option<GpuContext>,
    models: ModelStack, // preflop, flop, turn, river models
}
```

Add commands: `gpu_load_models`, `gpu_unload_models`, `gpu_status`.

**Step 3: Commit**

```bash
git commit -m "feat(tauri): add gpu_resolve command for live resolving"
```

---

### Task 22: Frontend Integration

**Files:**
- Modify: `frontend/src/Explorer.tsx`
- Modify: `frontend/src/invoke.ts`

**Step 1: Add GPU resolve toggle**

Add a UI toggle in the Explorer to switch between CPU range-solver and GPU resolver. When GPU is selected, resolves happen via `gpu_resolve` instead of the postflop solver commands.

**Step 2: Off-tree action handling**

When the opponent plays an action not in the abstraction (e.g. a bet size between two abstracted sizes), implement safe resolving:
- Re-solve with the off-tree action added to the abstraction
- Ensure opponent can't exploit the gap

**Step 3: Performance verification**

Target: <1 second per resolve in the Explorer. Add timing display to the UI.

**Step 4: Commit**

```bash
git commit -m "feat(frontend): integrate GPU resolver in Explorer"
```

---

## Notes for the Implementer

### DCFR+ vs Range-Solver's DCFR

The range-solver uses a specific DCFR variant with:
- `alpha_t = sqrt(t-1)`, quadratic positive regret discount
- `beta_t = 0.5` constant
- `gamma_t` based on power-of-4 schedule

Supremus DCFR+ uses:
- Delayed averaging: `weight = max(0, t - 100)`
- Linear regret weighting (not quadratic)
- Simultaneous player updates (not alternating)

The GPU solver should implement Supremus-style DCFR+. This means strategies won't match the range-solver exactly, but exploitability should converge to similar values.

### Testing on macOS

You cannot run CUDA tests on macOS. Options:
- Develop on a Linux machine with NVIDIA GPU
- Use a cloud GPU instance (see `docs/cloud.md`)
- Gate GPU tests behind `#[cfg(feature = "cuda")]` so `cargo test` passes everywhere

### Key Files to Reference

| What | Where |
|------|-------|
| Range-solver DCFR | `crates/range-solver/src/solver.rs` |
| Range-solver game tree | `crates/range-solver/src/game/mod.rs` |
| Range-solver terminal eval | `crates/range-solver/src/game/evaluation.rs` |
| Range-solver CLI | `crates/trainer/src/main.rs:565-782` |
| CFVNet model | `crates/cfvnet/src/model/network.rs` |
| CFVNet datagen sampler | `crates/cfvnet/src/datagen/sampler.rs` |
| Turn datagen (GPU) | `crates/cfvnet/src/datagen/turn_generate.rs` |
| River net evaluator | `crates/cfvnet/src/eval/river_net_evaluator.rs` |
| LeafEvaluator trait | `crates/core/src/blueprint_v2/cfv_subgame_solver.rs` |
| Blueprint game tree | `crates/core/src/blueprint_v2/game_tree.rs` |
| Existing GPU datagen design | `docs/plans/2026-03-14-gpu-turn-datagen-design.md` |
