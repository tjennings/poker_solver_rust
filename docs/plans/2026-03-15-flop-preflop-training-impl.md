# Flop + Preflop Model Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Complete the Supremus neural network stack by training flop and preflop auxiliary CFVNets on GPU, enabling full-game continual resolving from preflop to river.

**Architecture:** Two sub-pipelines: (A) Flop training reuses the Phase 3 turn pipeline infrastructure with different parameters (3-card boards, ~45 turn cards, turn model at leaves). (B) Preflop auxiliary training is a new inference-only pipeline — no CFR solving, just averaging flop model predictions over all 22,100 possible flops.

**Tech Stack:** Rust, cudarc, CudaNetInference (cuBLAS + custom kernels), burn-cuda (training only)

---

## Task 1: Generalize Turn Pipeline to Flop

**Files:**
- Modify: `crates/gpu-solver/src/training/turn_pipeline.rs`
- Modify: `crates/gpu-solver/src/training/turn_solver.rs`
- Modify: `crates/gpu-solver/src/training/builder.rs`
- Modify: `crates/gpu-solver/src/tree.rs`

The turn pipeline already handles depth-boundary trees with neural leaf evaluation. Generalize it to accept any street.

**Step 1: Parameterize by street**

The `TurnTrainingConfig` currently assumes 4-card boards. Rename/generalize:
- Add `board_size: usize` field (3 for flop, 4 for turn)
- Add `street_name: String` for display
- The `build_turn_game()` helper needs to accept `BoardState::Flop` or `BoardState::Turn`

In `tree.rs`, update `build_turn_game()` to accept a `BoardState` parameter:
```rust
pub fn build_depth_limited_game(
    board_state: BoardState,
    board: &[u8],  // 3 cards for flop, 4 for turn
    pot: i32,
    stack: i32,
    bet_sizes: &BetSizeOptions,
) -> Result<PostFlopGame, String>
```

**Step 2: Parameterize the sampler**

The `GpuTurnSampler` samples 4-card boards. Add a `board_size` parameter (or create `GpuFlopSampler` that samples 3 cards). Simplest: add `board_size` to the existing sampler's kernel.

Actually, the existing `GpuSampler` (Phase 2) already has a `sample_boards` kernel that takes a board size implicitly (samples 5 cards). The `GpuTurnSampler` samples 4. Create a generic sampler or just use the same kernel with `board_size` as a parameter.

**Step 3: Parameterize river card enumeration**

For turn training: 48 possible river cards (52 - 4 board).
For flop training: ~45 possible turn cards (52 - 3 board - ~4 blocked ≈ varies).
Actually: exactly 52 - 3 = 49 possible turn cards per flop.

The `encode_leaf_inputs` and `average_leaf_cfvs` kernels already parameterize `num_rivers`. Just pass 49 instead of 48.

**Step 4: Test**

Build a flop game with `depth_limit: Some(0)`, `BoardState::Flop`. Verify depth boundaries exist, no chance nodes, tree is valid.

**Step 5: Commit**

```bash
git commit -am "feat(gpu-solver): generalize turn pipeline to support flop training"
```

---

## Task 2: Flop Training Pipeline + CLI

**Files:**
- Create: `crates/gpu-solver/src/training/flop_pipeline.rs`
- Modify: `crates/gpu-solver/src/training/mod.rs`
- Modify: `crates/trainer/src/main.rs`

**Step 1: Flop pipeline**

Copy `train_turn_cfvnet_cuda` and adapt:
- Sample 3-card boards
- Load turn model (not river) as leaf evaluator
- 49 possible turn cards per boundary
- `BoardState::Flop` for tree construction

```rust
pub struct FlopTrainingConfig {
    pub turn_model_path: PathBuf,
    pub turn_hidden_layers: usize,
    pub turn_hidden_size: usize,
    // ... same training params as TurnTrainingConfig
}

pub fn train_flop_cfvnet_cuda<B: AutodiffBackend>(
    config: &FlopTrainingConfig,
    device: &B::Device,
) -> Result<(), String>
```

**Step 2: CLI command**

```rust
GpuTrainFlop {
    #[arg(long)]
    turn_model: PathBuf,
    #[arg(long, default_value_t = 7)]
    turn_hidden_layers: usize,
    #[arg(long, default_value_t = 500)]
    turn_hidden_size: usize,
    // ... same args
    #[arg(long)]
    output: PathBuf,
}
```

**Step 3: Test**

Small end-to-end: train a tiny flop model using a randomly-initialized turn model. Verify loss is finite, pipeline completes.

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): flop training pipeline and CLI"
```

---

## Task 3: Preflop Flop Enumeration Kernel

**Files:**
- Create: `crates/gpu-solver/kernels/enumerate_flops.cu`
- Create: `crates/gpu-solver/kernels/average_flop_cfvs.cu`
- Modify: `crates/gpu-solver/src/gpu.rs`

The preflop pipeline needs to enumerate all 22,100 possible flops and batch-forward-pass the flop model.

**Step 1: Precompute flop lookup table**

On CPU, enumerate all C(52,3) = 22,100 possible 3-card flops. Store as `[22100 * 3]` u32 array. Upload once to GPU.

```rust
fn enumerate_all_flops() -> Vec<u32> {
    let mut flops = Vec::with_capacity(22100 * 3);
    for c1 in 0u32..52 {
        for c2 in (c1+1)..52 {
            for c3 in (c2+1)..52 {
                flops.push(c1);
                flops.push(c2);
                flops.push(c3);
            }
        }
    }
    assert_eq!(flops.len(), 22100 * 3);
    flops
}
```

**Step 2: Encode preflop inputs kernel**

For each (flop_idx, spot), encode a 2720-dim input using the spot's ranges + the flop board:

```cuda
extern "C" __global__ void encode_preflop_inputs(
    float* output,                    // [num_flops * num_spots * 2720]
    const float* ranges_oop,          // [num_spots * 1326]
    const float* ranges_ip,
    const unsigned int* all_flops,    // [22100 * 3]
    const unsigned int* combo_cards,  // [1326 * 2]
    const float* pots,                // [num_spots]
    const float* stacks,              // [num_spots]
    unsigned int traverser,
    unsigned int num_flops,           // 22100
    unsigned int num_spots
)
```

One thread per (flop_idx, spot). Each writes 2720 floats: OOP range (zeroed for flop-conflicting combos), IP range (same), board one-hot, rank presence, pot/400, stack/400, traverser.

**Step 3: Average flop CFVs kernel**

After the flop model returns [22100 * num_spots, 1326] CFVs, average across flops per combo:

```cuda
extern "C" __global__ void average_preflop_cfvs(
    float* output,                    // [num_spots * 1326]
    const float* raw_cfvs,            // [num_flops * num_spots * 1326]
    const unsigned int* all_flops,    // [22100 * 3]
    const unsigned int* combo_cards,  // [1326 * 2]
    unsigned int num_flops,
    unsigned int num_spots
)
```

One thread per (spot, combo). Loops over 22,100 flops, skips flops where the combo conflicts with board cards, averages the valid CFVs.

**Step 4: Test**

Verify encoding and averaging on a small subset (e.g., 100 flops, 2 spots).

**Step 5: Commit**

```bash
git commit -am "feat(gpu-solver): preflop flop enumeration and averaging kernels"
```

---

## Task 4: Preflop Training Pipeline

**Files:**
- Create: `crates/gpu-solver/src/training/preflop_pipeline.rs`
- Modify: `crates/gpu-solver/src/training/mod.rs`

The preflop pipeline is fundamentally different — no CFR solver:

```rust
pub struct PreflopTrainingConfig {
    pub flop_model_path: PathBuf,
    pub flop_hidden_layers: usize,
    pub flop_hidden_size: usize,
    pub num_samples: u64,
    pub reservoir_capacity: usize,
    pub hidden_layers: usize,
    pub hidden_size: usize,
    pub train_batch_size: usize,
    pub train_steps_per_batch: usize,
    pub learning_rate: f64,
    pub output_dir: PathBuf,
    pub seed: u64,
}

pub fn train_preflop_cfvnet_cuda<B: AutodiffBackend>(
    config: &PreflopTrainingConfig,
    device: &B::Device,
) -> Result<(), String> {
    // 1. Load flop model as CudaNetInference
    let flop_model = CudaNetInference::load_from_burn(...)?;

    // 2. Pre-enumerate all 22,100 flops, upload to GPU
    let all_flops = enumerate_all_flops();
    let gpu_flops = gpu.upload(&all_flops)?;

    // 3. Main loop (NO CFR solving):
    while total_samples < config.num_samples {
        // Sample random situations (ranges + pot + stack, NO board)
        // For each spot:
        //   Encode 22,100 inputs (same ranges, different boards)
        //   Batch forward pass through flop model
        //   Average CFVs across 22,100 flops per combo
        //   Insert into reservoir
        // Train step from reservoir
    }
}
```

**Batch size consideration**: 22,100 flops × 2720 features × 4 bytes = 240MB per spot. With batch_size=10 spots: 2.4GB for inputs + 1.2GB for outputs = 3.6GB. Fits in 48GB VRAM but limits batch size. Use batch_size=10-50 spots.

Actually, we can process spots sequentially (each spot generates 22,100 inference inputs) but batch the flops within a spot. Each spot is: encode 22,100 inputs → one forward pass → average → insert 2 records (OOP + IP). Very fast per spot.

**Step 1: Implement pipeline**

**Step 2: Test** — small end-to-end with random flop model.

**Step 3: Commit**

```bash
git commit -am "feat(gpu-solver): preflop auxiliary training pipeline (inference-only)"
```

---

## Task 5: Preflop CLI Command

**Files:**
- Modify: `crates/trainer/src/main.rs`

```rust
GpuTrainPreflop {
    #[arg(long)]
    flop_model: PathBuf,
    #[arg(long, default_value_t = 7)]
    flop_hidden_layers: usize,
    #[arg(long, default_value_t = 500)]
    flop_hidden_size: usize,
    #[arg(long, default_value_t = 10_000_000)]
    num_samples: u64,
    // ... training params
    #[arg(long)]
    output: PathBuf,
}
```

**Commit:**

```bash
git commit -am "feat(gpu-solver): add gpu-train-flop and gpu-train-preflop CLI commands"
```

---

## Task 6: Update Beans

```bash
beans update poker_solver_rust-qpms -s in-progress
git add .beans/ && git commit -m "beans: Phase 4 in progress"
```
