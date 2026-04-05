# GPU-Batched River Datagen Design

*2026-04-04 — Integrate hand-parallel GPU solver into domain datagen pipeline*

## Problem

The domain datagen pipeline solves river subgames one-at-a-time on CPU threads. The GPU can solve 142 subgames simultaneously (one per SM on RTX 6000 Ada). For river datagen (50M+ samples needed), this is the bottleneck.

## Solution

Add a `GpuBatchSolver` to the `gpu-range-solver` crate that holds a persistent CUDA context and solves batches of up to 142 independent river subgames in a single kernel launch. The domain pipeline collects batches of situations, builds per-subgame terminal data, launches the GPU kernel, downloads results, and persists records.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | River only (board_size=5) | Proven GPU kernel, no chance nodes, highest data demand |
| Bet size fuzz | Disabled for GPU batches | All subgames must share tree topology |
| CUDA context | Initialized once, reused | Amortizes 280ms init across all batches |
| Batch size | 142 (configurable) | Matches SM count for full GPU utilization |
| CFV extraction | Download cfv array directly | Root CFVs available after last iteration |
| Config toggle | `backend: "gpu"` in datagen config | Falls back to CPU if not set |

## Architecture

```
Pipeline start:
  CudaContext::new(0)         ← once (280ms)
  compile kernel              ← once
  extract topology            ← once (shared bet sizes)
  allocate GpuBatchSolver     ← once (142 × E × H regrets, etc.)

Per batch (repeat until num_samples exhausted):
  collect 142 situations      ← from SituationGenerator
  build per-subgame data:
    - initial_weights[142 × 2 × H]  (different ranges per board)
    - showdown_outcomes[142 × ...]    (different boards)
    - hand validity masks             (board card blocking)
  upload terminal data        ← to GPU (~1ms)
  zero regrets/strategy_sum   ← on GPU
  <<<142, H>>> cfr_solve()    ← single kernel launch (~3ms for 1000 iter)
  stream.synchronize()
  download cfv[142 × N × H]  ← from GPU (~1ms)
  extract 284 TrainingRecords ← CPU (142 subgames × 2 players)
  write records               ← RecordWriter
```

## GpuBatchSolver API (new in gpu-range-solver crate)

```rust
/// Persistent GPU solver for batched subgame solving.
/// Initialized once, reused across many batches.
pub struct GpuBatchSolver {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel: HandParallelKernel,
    // Pre-uploaded topology (shared across all batches)
    d_edge_parent: CudaSlice<i16>,
    d_edge_child: CudaSlice<i16>,
    d_edge_player: CudaSlice<i8>,
    // ... level arrays, node info ...
    // Pre-allocated solver state (reused per batch)
    d_regrets: CudaSlice<f32>,       // [max_batch × E × H]
    d_strategy_sum: CudaSlice<f32>,  // [max_batch × E × H]
    d_reach: CudaSlice<f32>,         // [max_batch × N × H]
    d_cfv: CudaSlice<f32>,           // [max_batch × N × H]
    // Dimensions
    max_batch: usize,
    num_nodes: usize,
    num_edges: usize,
    num_hands: usize,
    max_depth: usize,
}

impl GpuBatchSolver {
    /// Initialize: create CUDA context, compile kernel, upload topology.
    pub fn new(
        topo: &TreeTopology,
        max_batch: usize,
        num_hands: usize,
        max_iterations: u32,
    ) -> Result<Self, Box<dyn Error>>;

    /// Solve a batch of subgames. Returns per-subgame root CFVs.
    /// Each SubgameSpec provides board-specific terminal data + initial weights.
    pub fn solve_batch(
        &mut self,
        specs: &[SubgameSpec],
    ) -> Result<Vec<SubgameResult>, Box<dyn Error>>;
}

/// Per-subgame specification within a batch.
pub struct SubgameSpec {
    pub initial_weights: [Vec<f32>; 2],     // per-player range weights
    pub showdown_outcomes: Vec<ShowdownData>, // per-showdown-node outcome matrices
    pub fold_payoffs: Vec<FoldData>,         // per-fold-node payoffs
    pub hand_valid_mask: Vec<f32>,           // [H] — 1.0 if hand valid, 0.0 if blocked by board
}

/// Result from one solved subgame.
pub struct SubgameResult {
    pub root_cfvs: [Vec<f32>; 2],    // per-player root CFVs [H]
    pub exploitability: f32,          // computed on CPU after download
}
```

## Pipeline Integration (cfvnet crate)

### Config changes

```yaml
datagen:
  backend: "gpu"          # "cpu" (default) or "gpu"
  gpu_batch_size: 142     # subgames per GPU launch
```

### Pipeline changes (pipeline.rs)

```rust
impl DomainPipeline {
    pub fn run(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
        // ... existing setup ...

        if config.datagen.backend == "gpu" && board_size >= 5 {
            return Self::run_gpu(config, output_path, /* shared objects */);
        }
        
        // ... existing CPU path unchanged ...
    }

    fn run_gpu(config: &CfvnetConfig, output_path: &Path, ...) -> Result<(), String> {
        let batch_size = config.datagen.gpu_batch_size.unwrap_or(142);
        
        // Build topology ONCE from a representative game
        let representative_sit = generator.peek();
        let representative_game = builder.build(&representative_sit);
        let topo = extract_topology(&representative_game);
        let term_template = extract_terminal_data(&representative_game, &topo);
        
        // Initialize GPU solver (CUDA context + kernel compile + topology upload)
        let mut gpu = GpuBatchSolver::new(&topo, batch_size, num_hands, iterations)?;
        
        // Batch loop
        loop {
            // Collect up to batch_size situations
            let batch: Vec<Situation> = (0..batch_size)
                .filter_map(|_| generator.next())
                .collect();
            if batch.is_empty() { break; }
            
            // Build per-subgame specs
            let specs: Vec<SubgameSpec> = batch.iter()
                .map(|sit| build_subgame_spec(sit, &topo, &term_template))
                .collect();
            
            // Solve all on GPU
            let results = gpu.solve_batch(&specs)?;
            
            // Extract and write records
            for (sit, result) in batch.iter().zip(results) {
                let records = build_training_records(sit, &result, num_hands);
                writer.lock().unwrap().write(&records)?;
                pb.inc(1);
            }
        }
        
        Ok(())
    }
}
```

### Building SubgameSpec from Situation

For each situation in the batch:
1. **Initial weights:** Copy from `sit.ranges` — already `[f32; 1326]` per player
2. **Hand validity:** Check each of 1326 hand combos against the 5 board cards. Zero weight if hand shares a card with board.
3. **Showdown outcomes:** Build outcome matrix `[H × H]` per showdown node. This requires hand evaluation against the specific board — reuse `extract_terminal_data` logic but with the batch's board cards.
4. **Fold payoffs:** Computed from pot/stack — same formula as `extract.rs:extract_terminal_data`

### Building TrainingRecords from SubgameResult

For each player p in {0, 1}:
```rust
TrainingRecord {
    board: sit.board.to_vec(),
    pot: sit.pot as f32,
    effective_stack: sit.effective_stack as f32,
    player: p as u8,
    game_value: dot(sit.ranges[p], result.root_cfvs[p]),
    oop_range: sit.ranges[0],
    ip_range: sit.ranges[1],
    cfvs: result.root_cfvs[p],
    valid_mask: compute_valid_mask(&sit.board),
}
```

## Performance Estimate

| Metric | CPU (18 threads) | GPU (142 batch) |
|--------|-----------------|-----------------|
| Subgames/second | ~600 (0.03s each × 18 threads) | ~24,000 (142 per ~6ms launch) |
| Time for 50M samples | ~23 hours | ~35 minutes |
| CUDA init overhead | — | 280ms (once) |

The GPU speedup comes from throughput: 142 simultaneous subgames vs 18 sequential CPU threads. Each GPU batch takes ~6ms (kernel) + ~2ms (upload/download) = ~8ms for 142 subgames = **17,750 subgames/second**.

With CPU overhead for situation generation, record building, and I/O, realistic throughput is probably **10,000-15,000 subgames/second**, or **~1 hour for 50M samples**.

## What Changes

| File | Change |
|------|--------|
| `gpu-range-solver/src/lib.rs` | Export `GpuBatchSolver`, `SubgameSpec`, `SubgameResult` |
| `gpu-range-solver/src/batch.rs` | New: `GpuBatchSolver` implementation |
| `cfvnet/src/datagen/domain/pipeline.rs` | Add `run_gpu()` path |
| `cfvnet/src/config.rs` | Add `backend`, `gpu_batch_size` fields to DatagenConfig |
| `cfvnet/Cargo.toml` | Add `gpu-range-solver` dependency |

## What Stays the Same

| Component | Status |
|-----------|--------|
| `extract.rs` | Unchanged (topology extraction reused) |
| `kernels.rs` | Unchanged (hand-parallel kernel already supports batch via blockIdx.x) |
| CPU datagen path | Unchanged (backend: "cpu" uses existing code) |
| RecordWriter | Unchanged |
| SituationGenerator | Unchanged |
| TrainingRecord format | Unchanged |
| CLI interface | Unchanged (config-driven) |
