# Turn CFVNet Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Train a turn CFVNet on GPU using Supremus-style pipeline: solve turn subgames with the trained river model as leaf evaluator at depth boundaries, all GPU-resident.

**Architecture:** Extends the Phase 2 river training pipeline. The turn tree has fold terminals + depth-boundary leaves (no chance nodes, no showdown). At each DCFR+ iteration, reach probabilities at depth boundaries are gathered, encoded into 2720-dim inputs for all (boundary × 48 river cards × spots), batched into one burn-cuda forward pass through the river model, averaged per combo, and scattered back to cfvalues. The rest of the solve loop (regret match, forward pass, backward CFV, regret update) is unchanged.

**Tech Stack:** Rust, cudarc (CUDA kernels), burn 0.16 with cuda-jit (river model inference + turn model training), range-solver (tree building with depth_limit)

---

## Task 1: Turn Tree Builder

**Files:**
- Modify: `crates/gpu-solver/src/tree.rs`
- Modify: `crates/gpu-solver/src/training/sampler.rs`
- Test: inline tests

Build a FlatTree from a turn PostFlopGame with `depth_limit: Some(1)`. This produces depth-boundary leaves instead of expanding through the river.

**Step 1: Add depth_limit support to FlatTree::from_postflop_game**

The current builder panics on chance nodes. With `depth_limit: Some(1)`, the range-solver's ActionTree marks river transitions as `PLAYER_DEPTH_BOUNDARY_FLAG` instead of creating chance nodes. The FlatTree builder needs to recognize these as a new terminal type.

Add `NodeType::DepthBoundary = 4` to the enum. In the BFS walk, check `node.is_depth_boundary()` via the player flags. Treat depth boundaries like terminals (no children) but track them separately from fold/showdown.

Add fields to FlatTree:
```rust
pub boundary_indices: Vec<u32>,  // node IDs of depth-boundary nodes
pub boundary_pots: Vec<f32>,     // pot at each boundary
pub boundary_stacks: Vec<f32>,   // effective stack at each boundary
```

**Step 2: Add turn situation sampler**

In the GPU sampler, add a mode that samples 4-card boards (flop + turn) instead of 5-card (full river). The existing `sample_boards` kernel samples 5 cards. Add a `board_size` parameter.

**Step 3: Test**

Build a turn game with `depth_limit: Some(1)`, verify:
- No chance nodes or showdown terminals
- Fold terminals exist
- Depth-boundary nodes exist
- `boundary_indices` is populated

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): turn tree builder with depth-boundary support"
```

---

## Task 2: Leaf Evaluation Input Encoding Kernel

**Files:**
- Create: `crates/gpu-solver/kernels/encode_leaf_inputs.cu`
- Modify: `crates/gpu-solver/src/gpu.rs`
- Test: inline tests

A CUDA kernel that encodes reach probabilities at depth boundaries into 2720-dim network inputs for all (boundary × river_card × spot) combinations.

**Step 1: The kernel**

One thread per (boundary × river_card × spot) triple. Each thread:
1. Reads OOP and IP reach at the boundary node for all 1326 hands
2. Constructs a 5-card board (4 turn cards + 1 river card)
3. Zeros reach for combos conflicting with the river card
4. Writes the 2720-dim feature vector to the output buffer

```cuda
extern "C" __global__ void encode_leaf_inputs(
    float* output,                    // [num_inputs * 2720]
    const float* reach_oop,           // [num_nodes * num_hands]
    const float* reach_ip,
    const unsigned int* boundary_nodes, // [num_boundaries]
    const unsigned int* turn_board,   // [4] — the 4 turn board cards
    const unsigned int* river_cards,  // [num_rivers] — possible river cards
    const float* boundary_pots,       // [num_boundaries]
    const float* boundary_stacks,     // [num_boundaries]
    const unsigned int* combo_cards,  // [1326 * 2]
    unsigned int traverser,
    unsigned int num_boundaries,
    unsigned int num_rivers,          // typically 48
    unsigned int num_hands,
    unsigned int hands_per_spot
) {
    // tid maps to (boundary_idx, river_idx, spot_idx)
    // Compute the 2720-dim input and write to output[tid * 2720..]
}
```

**Step 2: Launch method in gpu.rs**

**Step 3: Test** — encode a known position, verify the output matches CPU `build_input()` from river_net_evaluator.rs.

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): leaf evaluation input encoding CUDA kernel"
```

---

## Task 3: River Model Inference Integration

**Files:**
- Create: `crates/gpu-solver/src/training/leaf_eval.rs`
- Modify: `crates/gpu-solver/Cargo.toml`
- Test: inline tests

Load the trained river CFVNet and run batched inference on GPU.

**Step 1: GpuLeafEvaluator struct**

```rust
pub struct GpuLeafEvaluator<B: Backend> {
    model: CfvNet<B>,
    device: B::Device,
}

impl<B: Backend> GpuLeafEvaluator<B> {
    pub fn load(model_path: &Path, device: &B::Device) -> Result<Self, String>;

    /// Run batched inference on encoded inputs.
    /// inputs: [batch_size, 2720] — pre-encoded on GPU
    /// Returns: [batch_size, 1326] — raw CFV predictions
    pub fn infer(&self, inputs: &Tensor<B, 2>) -> Tensor<B, 2>;
}
```

**Step 2: Bridge cudarc → burn tensors**

The encoded inputs are in a cudarc `CudaSlice<f32>`. Need to create a burn `Tensor` from this. Use the GPU→CPU→GPU bounce (download cudarc, create burn TensorData, upload to burn device). Same pattern as the Phase 2 trainer.

**Step 3: Test** — load a river model (or create a random one), run inference on synthetic 2720-dim inputs, verify output shape is [batch, 1326].

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): river model inference wrapper for leaf evaluation"
```

---

## Task 4: CFV Averaging and Scatter Kernel

**Files:**
- Create: `crates/gpu-solver/kernels/average_leaf_cfvs.cu`
- Modify: `crates/gpu-solver/src/gpu.rs`
- Test: inline tests

After the river model returns [N, 1326] CFVs (one per boundary × river × spot), average across river cards per combo and write to the cfvalues buffer.

**Step 1: The kernel**

One thread per (boundary, spot, hand). Each thread:
1. Loops over 48 river cards
2. For each river card, checks if this hand's cards conflict with the river card
3. If not conflicting, accumulates the CFV from the inference output
4. Divides by the number of non-conflicting rivers
5. Writes to `cfvalues[boundary_node * num_hands + hand]`

```cuda
extern "C" __global__ void average_leaf_cfvs(
    float* cfvalues,                  // [num_nodes * num_hands] output
    const float* raw_cfvs,            // [num_boundaries * num_rivers * num_spots * 1326]
    const unsigned int* boundary_nodes,
    const unsigned int* river_cards,  // [num_rivers]
    const unsigned int* combo_cards,  // [1326 * 2]
    unsigned int num_boundaries,
    unsigned int num_rivers,
    unsigned int num_hands,
    unsigned int hands_per_spot
) {
    // Average across river cards, handling card conflicts
}
```

**Step 2: Launch method**

**Step 3: Test** — synthetic CFVs, verify averaging logic and card conflict handling.

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): leaf CFV averaging and scatter kernel"
```

---

## Task 5: Turn Batch Solver with Leaf Evaluation

**Files:**
- Create: `crates/gpu-solver/src/training/turn_solver.rs`
- Modify: `crates/gpu-solver/src/batch.rs`
- Test: inline tests

Wire the leaf evaluation into the DCFR+ solve loop for turn games.

**Step 1: TurnBatchSolver struct**

Wraps a `BatchGpuSolver` and adds leaf evaluation:

```rust
pub struct TurnBatchSolver<'a, B: Backend> {
    batch_solver: BatchGpuSolver<'a>,
    leaf_evaluator: GpuLeafEvaluator<B>,
    // Depth boundary metadata
    boundary_nodes: Vec<u32>,
    num_boundaries: u32,
    river_cards: CudaSlice<u32>,     // [48] possible rivers (varies by board)
    num_rivers: u32,
    // Working buffers
    encoded_inputs: CudaSlice<f32>,  // [num_boundaries * 48 * num_spots * 2720]
    raw_cfvs: CudaSlice<f32>,        // [num_boundaries * 48 * num_spots * 1326]
}
```

**Step 2: Modified solve loop**

The solve loop matches Phase 2 but adds leaf evaluation between terminal eval and backward CFV:

```rust
for traverser in 0..2 {
    // Zero cfvalues
    // Fold terminal eval (same as Phase 2)

    // NEW: Leaf evaluation at depth boundaries
    // 1. Encode inputs from reach at boundaries
    gpu.launch_encode_leaf_inputs(...)?;
    // 2. Batched burn-cuda forward pass
    let cfv_predictions = leaf_evaluator.infer(&encoded_tensor)?;
    // 3. Average and scatter to cfvalues
    gpu.launch_average_leaf_cfvs(...)?;

    // Backward CFV (same as Phase 2)
    // Update regrets (same as Phase 2)
}
```

**Step 3: Handle the river card enumeration**

For a 4-card turn board, enumerate the 48 possible river cards (52 - 4 board cards). Upload `river_cards: [48]` once per batch.

**Step 4: Test** — solve a small turn game (narrow ranges, 100 iterations) with a random river model. Verify strategies are valid (sum to 1, no NaN).

**Step 5: Commit**

```bash
git commit -am "feat(gpu-solver): turn batch solver with leaf evaluation"
```

---

## Task 6: Turn Training Pipeline

**Files:**
- Create: `crates/gpu-solver/src/training/turn_pipeline.rs`
- Modify: `crates/gpu-solver/src/training/mod.rs`
- Test: inline tests

The orchestrator for turn model training. Same structure as Phase 2's `train_river_cfvnet` but uses `TurnBatchSolver` and trains a turn model.

**Step 1: Configuration**

```rust
pub struct TurnTrainingConfig {
    // Same fields as RiverTrainingConfig plus:
    pub river_model_path: PathBuf,  // trained river model to use as leaf evaluator
    // All other fields inherited
}
```

**Step 2: Pipeline**

```rust
pub fn train_turn_cfvnet<B: AutodiffBackend>(config, device) {
    // Load river model
    let leaf_eval = GpuLeafEvaluator::load(&config.river_model_path, device)?;

    // Same loop as river pipeline:
    // sample turn situations (4-card boards)
    // build turn solver (with depth boundaries + leaf eval)
    // solve → extract root CFVs → reservoir insert → train step
    // validate periodically
}
```

**Step 3: Validation**

Ground-truth validator pre-solves turn positions using the same river model. Same dual validation (hold-out + ground-truth RMSE).

**Step 4: Test** — small end-to-end: 100 samples, 100 iterations, tiny model. Verify loss is finite.

**Step 5: Commit**

```bash
git commit -am "feat(gpu-solver): turn training pipeline with river leaf evaluator"
```

---

## Task 7: CLI Command

**Files:**
- Modify: `crates/trainer/src/main.rs`
- Modify: `crates/trainer/Cargo.toml`

**Step 1: Add gpu-train-turn subcommand**

```rust
GpuTrainTurn {
    #[arg(long)]
    river_model: PathBuf,           // trained river model path
    #[arg(long, default_value_t = 20_000_000)]
    num_samples: u64,
    #[arg(long, default_value_t = 4000)]
    solve_iterations: u32,
    #[arg(long, default_value_t = 1000)]
    batch_size: usize,
    // ... same args as gpu-train-river
    #[arg(long)]
    output: PathBuf,
}
```

**Step 2: Handler**

```rust
Commands::GpuTrainTurn { river_model, .. } => {
    let config = TurnTrainingConfig { river_model_path: river_model, .. };
    train_turn_cfvnet::<B>(&config, &device)?;
}
```

**Step 3: Test** — verify CLI parses, build succeeds.

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): add gpu-train-turn CLI command"
```

---

## Task 8: Update Beans

```bash
beans update poker_solver_rust-ofou -s in-progress
git add .beans/ && git commit -m "beans: Phase 3 in progress"
```
