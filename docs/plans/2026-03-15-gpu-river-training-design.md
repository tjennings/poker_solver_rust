# GPU-Resident River CFVNet Training — Design

## Overview

Train a river counterfactual value network entirely on GPU using the Supremus approach: sample random subgames, batch-solve with DCFR+, feed root CFVs into a GPU-resident reservoir, and train the network — all without CPU involvement in the hot loop.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Fully GPU-Resident Loop             │
│                                                  │
│  GPU RNG ──► Sample N situations                 │
│       │      (random boards, ranges, pot)        │
│       ▼                                          │
│  Build per-hand data (hand strengths, payoffs,   │
│  card blocking, initial reach) on GPU            │
│       │                                          │
│  BatchGpuSolver.solve(N) ──► Root CFVs           │
│       │                                          │
│       ▼                                          │
│  Reservoir Buffer (GPU memory) ◄── insert        │
│       │                                          │
│       ▼                                          │
│  Random mini-batch ──► burn-cuda train step      │
│       │                                          │
│       ▼                                          │
│  Repeat until convergence                        │
│                                                  │
│  Only CPU involvement: launch control,           │
│  checkpointing, progress reporting               │
└─────────────────────────────────────────────────┘
```

### CPU↔GPU Transfers

- **Start**: upload tree topology template, initial network weights
- **Periodic**: download loss metrics, save model checkpoint
- **End**: download final trained model

Zero transfers in the solve→train hot loop.

## Components

### 1. GPU Situation Sampler

A CUDA kernel that generates random river situations on-device:

- **Board**: 5 unique random cards from 0..51 using GPU cuRAND
- **Ranges**: Random weights in [0, 1] for each of 1326 combos, zeroed for board-blocked combos
- **Pot**: Random integer from configurable interval (e.g., 40-400)
- **Effective stack**: Random integer from configurable interval

Output: GPU buffers containing N situations' data, ready for the batch solver.

Card blocking (which combos conflict with the board) is a simple bitmask operation on GPU — each combo's two cards are checked against the 5 board cards.

### 2. GPU Per-Hand Data Builder

Currently, building per-hand data (hand strengths, payoffs, card blocking matrices, initial reach) happens on CPU via `FlatTree::from_postflop_game()`. For the GPU-resident loop, this must move to GPU:

- **Hand strength evaluation**: For each combo on the given board, evaluate 7-card hand strength. This uses a lookup table or evaluator. The evaluator can be uploaded once and reused.
- **Showdown payoffs**: Derived from pot size (simple arithmetic on GPU)
- **Fold payoffs**: Same — derived from pot
- **Card blocking**: Compare card pairs between traverser and opponent combos
- **Initial reach**: The sampled range weights

This replaces the CPU `FlatTree::from_postflop_game()` for batch datagen.

### 3. Batch GPU Solver (existing, with modifications)

The Phase 1 `BatchGpuSolver` with:

- **Simultaneous traversal** (not alternating) — one traversal per iteration updating both players' regrets
- **CFV extraction** — after solving, download root CFVs for both players as training targets
- **1,326 concrete hands** (no bucketing)
- **4,000 iterations** per batch (Supremus default)

### 4. GPU Reservoir Buffer

A fixed-size circular buffer in GPU global memory holding the most recent M training examples:

```
struct ReservoirBuffer {
    inputs: [M × 2720] f32,   // encoded situation features
    targets: [M × 1326] f32,  // CFVs (training targets)
    masks: [M × 1326] f32,    // valid combo mask
    write_idx: atomic u32,     // circular insertion point
    size: u32,                 // current fill level (up to M)
}
```

Each solved situation produces 2 training examples (OOP and IP perspectives). The reservoir is filled by the solve phase and sampled by the train phase.

Encoding for network input (matching existing cfvnet format):
- 1326 OOP range weights
- 1326 IP range weights
- 52-element board one-hot
- 13-element rank presence
- pot (normalized)
- stack (normalized)
- player indicator (0 or 1)
- Total: 2720 features

### 5. burn-cuda Training

The existing cfvnet network architecture (7×500, Huber loss, Adam optimizer) running on burn-cuda. Instead of loading batches from disk, it reads random mini-batches from the GPU reservoir buffer.

Key integration: cudarc (solver) and burn-cuda (training) must share the same CUDA device context to avoid redundant memory copies. This may require a custom burn backend adapter or manual tensor construction from raw GPU pointers.

### 6. Orchestrator

A Rust main loop that coordinates the pipeline:

```rust
fn train_river_cfvnet(config: TrainingConfig) {
    let gpu = GpuContext::new(0);
    let reservoir = ReservoirBuffer::new(gpu, capacity=100_000);
    let mut model = CfvNet::new(config.model_config);

    for epoch in 0.. {
        // Solve phase: sample + solve a batch
        let situations = gpu_sample_situations(gpu, batch_size=1000);
        let cfvs = batch_solve(gpu, &situations, iterations=4000);
        reservoir.insert(situations, cfvs);

        // Train phase: run training steps from reservoir
        for _ in 0..train_steps_per_batch {
            let batch = reservoir.sample_minibatch(batch_size=1024);
            let loss = model.train_step(batch);
        }

        // Periodic: checkpoint + report
        if epoch % checkpoint_interval == 0 {
            save_checkpoint(&model, epoch);
            report_metrics(loss, reservoir.size());
        }
    }
}
```

### 7. CLI

New subcommand in the trainer:
```
gpu-train river \
  --num-samples 50000000 \
  --solve-iterations 4000 \
  --batch-size 1000 \
  --reservoir-capacity 100000 \
  --hidden-layers 7 \
  --hidden-size 500 \
  --train-batch-size 1024 \
  --learning-rate 0.001 \
  --checkpoint-interval 10000 \
  --output models/river_gpu.bin
```

## Key Design Decisions

1. **Concrete 1,326 hands** — no bucketing. 33% more work than Supremus's 1,000 buckets but eliminates clustering pipeline and abstraction error.

2. **Simultaneous traversal** — one traversal per iteration updating both players. Matches Supremus finding that simultaneous converges faster with neural leaves (and simpler loop).

3. **GPU-resident everything** — sampling, solving, reservoir, training all on GPU. CPU only for launch control and checkpointing.

4. **Reservoir sampling** — overlaps solving and training. Training reads from a buffer that's continuously refreshed by new solves.

5. **Shared CUDA context** — cudarc and burn-cuda share one GPU device to enable zero-copy data flow from solver output to training input.

6. **Same network architecture as existing cfvnet** — 7×500 hidden layers, Huber loss, 2720→1326. Allows direct comparison with CPU-trained models.

## Validation

- Train a river model with GPU pipeline
- Compare validation loss against existing CPU-trained river model
- Solve test positions using both models as leaf evaluators, compare strategy quality
- Benchmark: training examples/second vs current CPU pipeline

## Dependencies

- Phase 1 `BatchGpuSolver` (done)
- `burn-cuda` backend (already in cfvnet dependencies)
- cuRAND or equivalent GPU RNG
- Hand evaluator lookup table for GPU
