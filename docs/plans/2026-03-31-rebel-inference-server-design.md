# ReBeL Inference Server Design

**Date:** 2026-03-31
**Status:** Approved

## Problem

Our current ReBeL implementation evaluates boundary CFVs **once before CFR solving**, then runs pure CPU iterations. The reference implementation queries the value net at **every CFR iteration** because reach probabilities at leaves change as the strategy evolves. Pre-computing freezes leaf values at initial ranges, which is fundamentally wrong for convergence.

Additionally, full-street solving with 200-1024 CFR iterations on large trees (4 bet sizes × raise depths × 1326 combos) is extremely slow — weeks for 500K records.

## Solution

GPU inference server with async request batching, interleaved training on a single GPU.

## Architecture

```
┌──────────────┐     requests      ┌──────────────────┐
│ CPU Worker 1 │────────────────►  │                  │
│ CPU Worker 2 │────────────────►  │  GPU Inference   │
│ CPU Worker 3 │────────────────►  │     Server       │
│     ...      │────────────────►  │  (owns model)    │
│ CPU Worker N │────────────────►  │                  │
└──────────────┘  ◄────────────── └──────────────────┘
                   responses              │
                                          │ every K solves
                                          ▼
                                   ┌──────────────┐
                                   │  Train step  │
                                   │  (same GPU)  │
                                   └──────────────┘
```

### Components

**1. InferenceServer** (`crates/rebel/src/inference_server.rs`)

Dedicated thread owning the CfvNet model. Collects requests from an mpsc channel, batches by size or timeout, runs one GPU forward pass, returns results via per-request oneshot channels. Periodically pauses inference to run a training step from the replay buffer.

```rust
struct InferenceRequest {
    input: Vec<f32>,                          // 2720-element encoded input
    response_tx: oneshot::Sender<Vec<f32>>,   // 1326 CFV outputs
}

struct InferenceServer {
    request_rx: mpsc::Receiver<InferenceRequest>,
    model: CfvNet<Wgpu>,
    device: WgpuDevice,
    batch_size: usize,            // max requests per batch (e.g., 256)
    batch_timeout_us: u64,        // max wait before running undersized batch
    replay_buffer: ReplayBuffer,
    train_every_n_solves: usize,
    solves_since_train: AtomicUsize,
}
```

Server loop:
```
loop {
    collect requests until batch_size or timeout
    run forward pass on batch
    send results back via oneshot channels

    if solves_since_train >= train_every_n_solves:
        sample batch from replay buffer
        run one training step (forward + backward + optimizer)
        reset counter
}
```

**2. ReplayBuffer** (`crates/rebel/src/replay_buffer.rs`)

Thread-safe circular buffer of training examples. Workers append, server samples.

```rust
struct ReplayEntry {
    input: [f32; 2720],   // encoded PBS
    target: [f32; 1326],  // root CFVs from solve
}

struct ReplayBuffer {
    entries: Mutex<VecDeque<ReplayEntry>>,
    capacity: usize,  // ~200K entries (~3GB)
}
```

Operations: `push(entry)` (evicts oldest if full), `sample(n) -> Vec<ReplayEntry>`.

**3. Iterative solver** (modify `crates/rebel/src/subgame_solve.rs`)

New function that calls `solve_step()` in a loop, re-evaluating boundary CFVs each iteration via the inference server.

```rust
fn solve_subgame_iterative(
    pbs: &Pbs,
    config: &SolveConfig,
    inference_tx: &mpsc::Sender<InferenceRequest>,
) -> Result<SubgameSolveResult, String> {
    let mut game = build_game(pbs, config)?;

    for iter in 0..config.solver_iterations {
        // Re-evaluate boundaries with current ranges
        let boundary_cfvs = evaluate_boundaries_via_server(
            &game, pbs, inference_tx,
        );
        set_all_boundary_cfvs(&mut game, &boundary_cfvs);

        solve_step(&game, iter);
    }

    extract_result(&game, pbs)
}
```

Workers call:
```rust
fn evaluate_leaf(tx: &mpsc::Sender<InferenceRequest>, input: Vec<f32>) -> Vec<f32> {
    let (resp_tx, resp_rx) = oneshot::channel();
    tx.send(InferenceRequest { input, response_tx: resp_tx });
    resp_rx.recv()  // blocks until GPU batch completes
}
```

**4. Data generation workers** (modify `crates/rebel/src/self_play.rs`)

Workers sample PBSs, build PostFlopGame, run the iterative solver, push training examples to the replay buffer.

Per subgame solve:
1. Sample PBS (blueprint play for offline, self-play for live)
2. Build PostFlopGame with depth_limit=0 (one street of betting)
3. Run 1024 CFR iterations, each re-evaluating boundaries via inference server
4. Extract root CFVs for both players
5. Push 2 training examples (one per player) to replay buffer

**5. CLI** — `rebel-train` spawns inference server thread, worker threads, runs until budget exhausted.

### Subgame Depth

One full betting round (depth_limit=0), matching the ReBeL paper: "Our agent always solves to the end of the current betting round." Value net evaluates at street transitions (chance nodes) and terminals. This matches our existing tree structure.

### What stays the same

- PostFlopGame, solve_step(), set_boundary_cfvs() — unchanged
- CfvNet model (2720→1326) — unchanged
- PBS struct, belief updates, blueprint sampler — unchanged
- rebel-seed for initial river seeding — unchanged
- Value net input format (2720 features) — unchanged

### What changes

- `subgame_solve.rs` — new iterative solve loop with per-iteration boundary eval
- `self_play.rs` — workers use inference server instead of direct evaluator calls
- New: `inference_server.rs`, `replay_buffer.rs`
- `orchestration.rs` — wire up server + workers + training

### Config additions

```yaml
inference:
  batch_size: 256
  batch_timeout_us: 100
  train_every_n_solves: 50
  replay_capacity: 200000
```

### Performance expectations

Per CFR iteration: ~20-40 boundary queries per subgame (10-20 boundaries × 2 players), batched across 16 worker threads = ~320-640 queries per batch. Single GPU forward pass for a batch of 640 × 2720 inputs is ~1ms. At 1024 iterations per solve, GPU time per solve = ~1 second. With 16 threads solving in parallel, throughput = ~16 solves/second.

Training step every 50 solves = every ~3 seconds. One optimizer step takes ~10ms. Negligible overhead.
