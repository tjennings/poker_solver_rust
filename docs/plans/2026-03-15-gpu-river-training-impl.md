# GPU-Resident River CFVNet Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Train a river counterfactual value network entirely on GPU: sample situations, batch-solve with DCFR+, feed CFVs into a GPU reservoir, train with burn-cuda — zero CPU involvement in the hot loop.

**Architecture:** New module `crates/gpu-solver/src/training/` containing the GPU-resident pipeline. Reuses `BatchGpuSolver` for solving, `burn-cuda` for neural net training. A GPU reservoir buffer connects solver output to training input. Situation sampling and hand evaluation run on GPU via custom CUDA kernels.

**Tech Stack:** Rust, cudarc (CUDA kernels), burn 0.16 with `cuda-jit` feature, cuRAND (via cudarc)

---

## Task 1: Simultaneous Traversal

**Files:**
- Modify: `crates/gpu-solver/src/solver.rs`
- Modify: `crates/gpu-solver/src/batch.rs`
- Test: existing unit tests must still pass

Currently the solver does alternating traversal (two passes per iteration — one per player). Switch to simultaneous: one pass per iteration, both players' regrets updated.

**Step 1: Modify the solve loop in solver.rs**

Change the iteration loop from:
```rust
for traverser in 0..2u32 {
    // regret match, init reach, forward pass
    // terminal eval for this traverser
    // backward cfv for this traverser
    // update regrets for this traverser's nodes
}
```
to:
```rust
// regret match (once)
// init reach (once)
// forward pass (once)
// terminal eval for BOTH players (compute two cfvalue arrays)
// backward cfv for BOTH players
// update regrets for ALL nodes (both players simultaneously)
```

This requires two `cfvalues` buffers (one per player perspective) or running terminal+backward twice but updating all regrets in one pass.

**Simplest approach**: Keep the `for traverser in 0..2` loop structure but move regret_match, init_reach, and forward_pass OUTSIDE the traverser loop (computed once). Only terminal eval + backward CFV + regret update run per traverser. This is what the current batch solver already does — regret_match and forward_pass are inside the traverser loop unnecessarily.

Wait — the current solver already moved regret_match + init_reach + forward_pass inside the traverser loop as a bug fix (traverser 1 needs to see traverser 0's regret updates). For simultaneous traversal, we compute ONE strategy from regrets, propagate reach ONCE, then compute CFVs and update regrets for BOTH players using that same strategy.

**Step 2: Validate against CPU range-solver**

Run the 30-position debug_compare test. Strategies may differ slightly from the CPU solver (which uses alternating), but exploitability should be comparable.

**Step 3: Commit**

```bash
git commit -am "feat(gpu-solver): switch to simultaneous traversal"
```

---

## Task 2: CFV Extraction

**Files:**
- Modify: `crates/gpu-solver/src/batch.rs`
- Test: inline tests

After solving, the root node's cfvalues contain the training targets. Add CFV extraction to BatchSolveResult.

**Step 1: Add fields to BatchSolveResult**

```rust
pub struct BatchSolveResult {
    pub strategies: Vec<Vec<f32>>,
    pub iterations: u32,
    pub cfvs_oop: Vec<Vec<f32>>,  // per-spot: [actual_hands] root CFVs for OOP
    pub cfvs_ip: Vec<Vec<f32>>,   // per-spot: [actual_hands] root CFVs for IP
}
```

**Step 2: Download CFVs after final iteration**

On the last iteration, after backward CFV for each traverser, download `cfvalues[0..num_hands]` (root node). Store in per-traverser buffers. After the loop, slice into per-spot arrays (same pattern as strategy extraction).

**Step 3: Test**

```rust
#[test]
fn test_batch_cfv_extraction() {
    // Solve 3 spots, verify cfvs_oop/ip are non-empty and non-zero
}
```

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): extract root CFVs for datagen"
```

---

## Task 3: GPU Situation Sampler

**Files:**
- Create: `crates/gpu-solver/kernels/sample_situations.cu`
- Create: `crates/gpu-solver/src/training/sampler.rs`
- Modify: `crates/gpu-solver/src/gpu.rs`
- Test: inline tests

Generate random river situations entirely on GPU.

**Step 1: CUDA kernel for situation sampling**

```cuda
// Generates N random river situations on GPU.
// Each situation: 5 board cards, 1326 OOP range weights, 1326 IP range weights, pot, stack
extern "C" __global__ void sample_situations(
    float* ranges_oop,        // [N * 1326] output
    float* ranges_ip,         // [N * 1326] output
    unsigned int* boards,     // [N * 5] output (card indices 0..51)
    float* pots,              // [N] output
    float* stacks,            // [N] output
    unsigned int* rng_states, // [N] cuRAND state (or simple LCG state)
    unsigned int num_situations,
    float pot_min, float pot_max,
    float stack_min, float stack_max
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_situations) return;

    // Use LCG or xorshift RNG per thread
    unsigned int state = rng_states[tid];

    // Sample 5 unique board cards
    // Sample range weights (0..1 for non-blocked, 0 for blocked)
    // Sample pot and stack from uniform distribution
    // Write outputs

    rng_states[tid] = state; // save RNG state
}
```

For the RNG: use a simple xorshift32 or LCG — we don't need cryptographic quality. Seed each thread with `base_seed + tid`.

For board card sampling: sample 5 unique cards from 0..51 using rejection sampling (fast since 5/52 is sparse).

For range weights: generate uniform [0, 1] per combo, zero out combos that share a card with the board.

For card blocking: combo index i maps to card pair (c1, c2). Precompute this mapping as a lookup table uploaded to GPU.

**Step 2: Rust wrapper**

```rust
pub struct GpuSampler {
    rng_states: CudaSlice<u32>,
    combo_cards: CudaSlice<u32>,  // [1326 * 2] lookup table
    // output buffers
    ranges_oop: CudaSlice<f32>,
    ranges_ip: CudaSlice<f32>,
    boards: CudaSlice<u32>,
    pots: CudaSlice<f32>,
    stacks: CudaSlice<f32>,
}
```

**Step 3: Test**

```rust
#[test]
fn test_gpu_sampler() {
    let gpu = GpuContext::new(0).unwrap();
    let mut sampler = GpuSampler::new(&gpu, batch_size=100, seed=42);
    let situations = sampler.sample().unwrap();
    assert_eq!(situations.boards.len(), 100 * 5);
    // Boards should have 5 unique cards per situation
    // Ranges should have non-zero weights for non-blocked combos
}
```

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): add GPU situation sampler"
```

---

## Task 4: GPU Hand Strength Evaluator

**Files:**
- Create: `crates/gpu-solver/kernels/evaluate_hands.cu`
- Modify: `crates/gpu-solver/src/training/sampler.rs`
- Test: inline tests

Evaluate hand strengths for all 1326 combos on a given board, entirely on GPU.

**Step 1: Upload hand evaluator lookup table**

The rs_poker hand evaluator uses lookup tables. Upload these to GPU memory once. Alternatively, implement a simple rank-based evaluator on GPU (compare 7-card hands by rank category + kicker).

**Simpler approach**: Use the range-solver's `evaluate_hand_strength()` function pattern but on GPU. For river (7 cards total), evaluate each combo's 7-card hand and assign a u32 strength value where higher = stronger.

The fastest GPU approach: upload a precomputed `[52 choose 2] × [52 choose 5]` → strength lookup, but that's too large (1326 × 2.6M entries). Instead, implement the evaluator logic in CUDA.

**Practical approach**: The `rs_poker` evaluator is complex. For Phase 2, compute hand strengths on CPU for each sampled board and upload. The hand strength computation is O(1326) per board — trivial on CPU (~0.1ms). This keeps GPU for the heavy lifting (solving + training) and avoids porting the evaluator to CUDA.

```rust
// In the per-hand data builder:
// 1. Download boards from GPU (5 * N u32s — tiny)
// 2. Evaluate strengths on CPU for all combos per board
// 3. Upload strengths back to GPU
```

This is a minor CPU→GPU transfer (1326 * N * 4 bytes = ~5MB for 1000 spots) and happens once per batch, not per iteration.

**Step 2: Test**

Compare GPU-computed strengths against CPU reference.

**Step 3: Commit**

```bash
git commit -am "feat(gpu-solver): add hand strength evaluation for sampled boards"
```

---

## Task 5: GPU Per-Hand Data Builder

**Files:**
- Create: `crates/gpu-solver/src/training/builder.rs`
- Create: `crates/gpu-solver/kernels/build_per_hand_data.cu`
- Test: inline tests

Build the per-hand arrays needed by BatchGpuSolver from sampled situations, mostly on GPU.

**Step 1: Compute payoffs on GPU**

Fold and showdown payoffs are simple arithmetic from pot size:
```cuda
// fold_amount_win = 0.5 * pot / num_combinations
// fold_amount_lose = -0.5 * pot / num_combinations
// showdown with rake: amount_win = (0.5 * pot - rake) / num_combinations
```

A kernel computes these for all (terminal, hand) pairs given the per-spot pot sizes.

**Step 2: Compute card blocking on GPU**

The same_hand_index and hand_cards arrays can be computed from the combo_cards lookup table:
```cuda
// For each traverser hand h and opponent hand h':
// blocked = (h.c1 == h'.c1 || h.c1 == h'.c2 || h.c2 == h'.c1 || h.c2 == h'.c2)
```

**Step 3: Build initial reach**

Initial reach = sampled range weights (already on GPU from sampler).

**Step 4: Assemble BatchGpuSolver inputs**

Create a method that takes GPU sampler output + hand strengths and produces a ready-to-solve BatchGpuSolver without going through FlatTree/PostFlopGame on CPU.

This is the key: bypass the CPU entirely for batch construction. The tree topology is shared (same for all spots) and uploaded once. Only per-hand data varies.

**Step 5: Test**

Build per-hand data for 10 sampled spots. Verify payoff signs, reach non-negativity, hand strength ordering.

**Step 6: Commit**

```bash
git commit -am "feat(gpu-solver): GPU per-hand data builder for sampled situations"
```

---

## Task 6: GPU Reservoir Buffer

**Files:**
- Create: `crates/gpu-solver/src/training/reservoir.rs`
- Create: `crates/gpu-solver/kernels/reservoir_insert.cu`
- Test: inline tests

A fixed-capacity circular buffer in GPU memory storing training examples.

**Step 1: Define the reservoir**

```rust
pub struct GpuReservoir {
    inputs: CudaSlice<f32>,    // [capacity * INPUT_SIZE]
    targets: CudaSlice<f32>,   // [capacity * OUTPUT_SIZE]
    masks: CudaSlice<f32>,     // [capacity * OUTPUT_SIZE]
    ranges: CudaSlice<f32>,    // [capacity * OUTPUT_SIZE] (for aux loss)
    game_values: CudaSlice<f32>, // [capacity]
    write_idx: u32,
    size: u32,
    capacity: u32,
}
```

**Step 2: Insert kernel**

After solving a batch, encode the situations and CFVs into training records and insert into the reservoir:

```cuda
extern "C" __global__ void reservoir_insert(
    // destination reservoir buffers
    float* res_inputs, float* res_targets, float* res_masks,
    // source: solved batch data
    const float* ranges_oop, const float* ranges_ip,
    const unsigned int* boards, const float* pots, const float* stacks,
    const float* cfvs,
    // indexing
    unsigned int write_start,
    unsigned int capacity,
    unsigned int num_records,
    unsigned int input_size,
    unsigned int output_size
) {
    // Each thread writes one training record into the reservoir
}
```

**Step 3: Sample mini-batch**

A kernel that randomly selects B indices from [0, size) and gathers them into a contiguous mini-batch:

```cuda
extern "C" __global__ void reservoir_sample_batch(
    // gather from reservoir into contiguous batch tensors
)
```

**Step 4: Test**

Insert 100 records, sample a mini-batch of 32, verify shapes and non-zero values.

**Step 5: Commit**

```bash
git commit -am "feat(gpu-solver): add GPU reservoir buffer for training examples"
```

---

## Task 7: burn-cuda Training Integration

**Files:**
- Create: `crates/gpu-solver/src/training/trainer.rs`
- Modify: `crates/gpu-solver/Cargo.toml` (add burn dependency)
- Test: inline tests

Connect the GPU reservoir to burn-cuda training.

**Step 1: Add burn dependency**

```toml
[dependencies]
burn = { version = "0.16", features = ["train"], optional = true }

[features]
cuda = ["cudarc", "burn/cuda-jit"]
```

**Step 2: Shared CUDA context**

The critical integration point: cudarc (solver kernels) and burn-cuda (training) need to share the same GPU device. Investigate if burn's `CudaDevice` can coexist with cudarc's `CudaContext` on the same GPU.

If they can't share memory directly, the reservoir samples must be copied from cudarc buffers to burn tensors. This is still GPU→GPU (fast) not GPU→CPU→GPU.

**Step 3: Implement GpuTrainer**

```rust
pub struct GpuTrainer<B: AutodiffBackend> {
    model: CfvNet<B>,
    optimizer: Adam<B>,
    device: B::Device,
}

impl<B: AutodiffBackend> GpuTrainer<B> {
    pub fn train_step(&mut self, batch: TrainingBatch) -> f32 {
        // Forward pass
        let output = self.model.forward(batch.inputs);
        // Compute loss
        let loss = cfvnet_loss(output, batch.targets, batch.masks, batch.ranges, batch.game_values);
        // Backward + step
        let grads = loss.backward();
        self.optimizer.step(grads);
        loss.into_scalar()
    }
}
```

**Step 4: Test**

Create a small model, run 10 training steps with synthetic data, verify loss decreases.

**Step 5: Commit**

```bash
git commit -am "feat(gpu-solver): integrate burn-cuda training with GPU reservoir"
```

---

## Task 8: Validation Loop

**Files:**
- Create: `crates/gpu-solver/src/training/validation.rs`
- Test: inline tests

Two-tier validation: continuous hold-out loss + periodic ground-truth comparison.

**Step 1: Hold-out loss**

Reserve 5% of reservoir insertions for validation (mark with a flag). Each training epoch, compute loss on held-out examples without updating weights.

**Step 2: Ground-truth validation set**

Pre-solve 100 fixed river positions on CPU at high iteration count (10,000). Save ground-truth CFVs. Periodically run the model on these inputs and measure prediction error.

```rust
pub struct ValidationSet {
    inputs: Vec<Vec<f32>>,      // [100 * 2720]
    ground_truth: Vec<Vec<f32>>, // [100 * 1326] exact CFVs
    masks: Vec<Vec<f32>>,
}
```

**Step 3: Report metrics**

Print: training loss, validation loss, ground-truth RMSE, learning rate, samples seen, examples/second.

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): add dual validation (hold-out + ground-truth)"
```

---

## Task 9: Orchestrator + CLI

**Files:**
- Create: `crates/gpu-solver/src/training/mod.rs`
- Modify: `crates/trainer/src/main.rs`
- Modify: `crates/trainer/Cargo.toml`

Tie everything together into a single training pipeline.

**Step 1: Training orchestrator**

```rust
pub fn train_river_cfvnet(config: RiverTrainingConfig) -> Result<(), Box<dyn Error>> {
    let gpu = GpuContext::new(0)?;
    let mut sampler = GpuSampler::new(&gpu, config.batch_size, config.seed);
    let mut reservoir = GpuReservoir::new(&gpu, config.reservoir_capacity);
    let mut trainer = GpuTrainer::new(&gpu, config.model_config);
    let validation = ValidationSet::precompute(&gpu, 100);

    let mut total_samples = 0u64;
    while total_samples < config.num_samples {
        // Sample + solve
        let situations = sampler.sample()?;
        let hand_data = build_per_hand_data(&gpu, &situations)?;
        let solve_result = batch_solve(&gpu, &hand_data, config.solve_iterations)?;

        // Insert into reservoir
        reservoir.insert(&situations, &solve_result)?;
        total_samples += config.batch_size as u64;

        // Train
        for _ in 0..config.train_steps_per_batch {
            let batch = reservoir.sample_minibatch(config.train_batch_size)?;
            let loss = trainer.train_step(batch);
        }

        // Validate periodically
        if total_samples % config.validation_interval == 0 {
            let holdout_loss = trainer.validation_loss(&reservoir)?;
            let gt_rmse = validation.evaluate(&trainer)?;
            println!("samples={total_samples} loss={loss:.6} val_loss={holdout_loss:.6} gt_rmse={gt_rmse:.6}");
        }

        // Checkpoint periodically
        if total_samples % config.checkpoint_interval == 0 {
            trainer.save_checkpoint(&config.output_dir, total_samples)?;
        }
    }

    trainer.save_final(&config.output_dir)?;
    Ok(())
}
```

**Step 2: CLI command**

```rust
/// Train a river CFVNet on GPU using Supremus-style pipeline
GpuTrainRiver {
    #[arg(long, default_value_t = 50_000_000)]
    num_samples: u64,
    #[arg(long, default_value_t = 4000)]
    solve_iterations: u32,
    #[arg(long, default_value_t = 1000)]
    batch_size: usize,
    #[arg(long, default_value_t = 100_000)]
    reservoir_capacity: usize,
    #[arg(long, default_value_t = 7)]
    hidden_layers: usize,
    #[arg(long, default_value_t = 500)]
    hidden_size: usize,
    #[arg(long, default_value_t = 1024)]
    train_batch_size: usize,
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f64,
    #[arg(long)]
    output: PathBuf,
}
```

**Step 3: Test end-to-end**

Small-scale integration test: 100 samples, 100 solve iterations, 10 training steps. Verify loss is finite and decreasing.

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): add gpu-train-river CLI with full Supremus pipeline"
```
