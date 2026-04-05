# GPU-Batched River Datagen Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU batch solving to the cfvnet datagen pipeline so 142 river subgames are solved simultaneously on GPU per kernel launch, replacing the per-thread CPU solve.

**Architecture:** New `GpuBatchSolver` in gpu-range-solver crate wraps persistent CUDA context + compiled kernel + pre-uploaded topology + reusable GPU memory. Pipeline collects batches of situations, builds per-subgame terminal data, calls `solve_batch()`, extracts TrainingRecords from downloaded CFVs.

**Tech Stack:** Rust, cudarc 0.19, hand-parallel CUDA kernel (existing), cfvnet datagen domain pipeline

**Design doc:** `docs/plans/2026-04-04-gpu-batched-datagen-design.md`

---

## Task 1: GpuBatchSolver in gpu-range-solver

Add `batch.rs` to the gpu-range-solver crate with a `GpuBatchSolver` that manages persistent CUDA state and solves batches of subgames.

**Files:**
- Create: `crates/gpu-range-solver/src/batch.rs`
- Modify: `crates/gpu-range-solver/src/lib.rs` (add `pub mod batch` and re-exports)

**Step 1: Write the failing test**

In `crates/gpu-range-solver/src/batch.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_solver_creates_and_solves_single() {
        // Build a river game for topology
        let game = make_river_game();
        let topo = crate::extract::extract_topology(&game);
        let term = crate::extract::extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let mut solver = GpuBatchSolver::new(&topo, 1, num_hands, 500).unwrap();

        // Build a single spec from the terminal data
        let spec = SubgameSpec::from_terminal_data(&term, num_hands);
        let results = solver.solve_batch(&[spec]).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].root_cfvs[0].len(), num_hands);
        assert_eq!(results[0].root_cfvs[1].len(), num_hands);
        // CFVs should be finite
        for p in 0..2 {
            for &v in &results[0].root_cfvs[p] {
                assert!(v.is_finite(), "CFV must be finite");
            }
        }
    }

    #[test]
    fn batch_solver_solves_multiple() {
        let game = make_river_game();
        let topo = crate::extract::extract_topology(&game);
        let term = crate::extract::extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let mut solver = GpuBatchSolver::new(&topo, 4, num_hands, 200).unwrap();

        // Same spec replicated 4 times (different boards would have different specs)
        let spec = SubgameSpec::from_terminal_data(&term, num_hands);
        let specs = vec![spec.clone(), spec.clone(), spec.clone(), spec];
        let results = solver.solve_batch(&specs).unwrap();

        assert_eq!(results.len(), 4);
        // All should produce same results (same inputs)
        for i in 1..4 {
            for p in 0..2 {
                for h in 0..num_hands {
                    assert!(
                        (results[0].root_cfvs[p][h] - results[i].root_cfvs[p][h]).abs() < 0.01,
                        "batch {} player {} hand {} differs", i, p, h
                    );
                }
            }
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p gpu-range-solver -- batch_solver_creates`
Expected: Compilation error — `batch` module doesn't exist

**Step 3: Implement GpuBatchSolver**

```rust
//! Persistent GPU batch solver for solving many independent subgames simultaneously.

use crate::extract::{TreeTopology, TerminalData, NodeType};
use crate::gpu::HandParallelKernel;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Specification for one subgame within a batch.
/// All subgames in a batch share the same tree topology.
#[derive(Clone)]
pub struct SubgameSpec {
    /// Per-player initial weights [2][H]. Includes board card blocking (zeroed hands).
    pub initial_weights: [Vec<f32>; 2],
    /// Showdown outcome matrices (pre-scaled). Same layout as MegaTerminalData.
    pub showdown_outcomes_p0: Vec<f32>,
    pub showdown_outcomes_p1: Vec<f32>,
}

/// Result from one solved subgame.
pub struct SubgameResult {
    /// Per-player root CFVs [2][H].
    pub root_cfvs: [Vec<f32>; 2],
}

/// Persistent GPU solver for batched subgame solving.
pub struct GpuBatchSolver {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel: HandParallelKernel,
    
    // Pre-uploaded topology (constant across all batches)
    d_edge_parent: CudaSlice<i16>,
    d_edge_child: CudaSlice<i16>,
    d_edge_player: CudaSlice<i8>,
    d_actions_per_node: CudaSlice<f32>,
    d_level_starts: CudaSlice<i32>,
    d_level_counts: CudaSlice<i32>,
    d_node_depth: CudaSlice<i32>,
    
    // Terminal structure (constant — node IDs, depths, card indices, fold payoffs)
    d_fold_node_ids: CudaSlice<i32>,
    d_fold_payoffs_p0: CudaSlice<f32>,
    d_fold_payoffs_p1: CudaSlice<f32>,
    d_fold_depths: CudaSlice<i32>,
    d_showdown_node_ids: CudaSlice<i32>,
    d_showdown_num_player: CudaSlice<i32>,
    d_showdown_depths: CudaSlice<i32>,
    d_player_card1: CudaSlice<i32>,
    d_player_card2: CudaSlice<i32>,
    d_opp_card1: CudaSlice<i32>,
    d_opp_card2: CudaSlice<i32>,
    d_same_hand_idx: CudaSlice<i32>,
    
    // Leaf injection (empty for river)
    d_leaf_cfv_p0: CudaSlice<f32>,
    d_leaf_cfv_p1: CudaSlice<f32>,
    d_leaf_node_ids: CudaSlice<i32>,
    d_leaf_depths: CudaSlice<i32>,
    
    // Pre-allocated solver state (reused per batch, sized for max_batch)
    d_regrets: CudaSlice<f32>,
    d_strategy_sum: CudaSlice<f32>,
    d_reach: CudaSlice<f32>,
    d_cfv: CudaSlice<f32>,
    
    // Dimensions
    max_batch: usize,
    num_nodes: usize,
    num_edges: usize,
    num_hands: usize,
    max_depth: usize,
    max_iterations: u32,
    num_folds: usize,
    num_showdowns: usize,
    num_hands_p0: usize,
    num_hands_p1: usize,
    shared_mem_bytes: u32,
    
    // Level info for root strategy extraction
    root_num_actions: usize,
}

impl GpuBatchSolver {
    pub fn new(
        topo: &TreeTopology,
        max_batch: usize,
        num_hands: usize,
        max_iterations: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let kernel = HandParallelKernel::compile(&ctx)?;
        
        // Build sorted topology (reuse existing helper)
        // Upload topology arrays
        // Allocate solver state for max_batch × E × H
        // ... (same pattern as gpu_solve_hand_parallel but split into init + per-batch)
        
        todo!() // Detailed implementation follows the pattern in solver.rs
    }
    
    /// Solve a batch of subgames. `specs.len()` must be ≤ `max_batch`.
    pub fn solve_batch(
        &mut self,
        specs: &[SubgameSpec],
    ) -> Result<Vec<SubgameResult>, Box<dyn std::error::Error>> {
        let batch_size = specs.len();
        assert!(batch_size <= self.max_batch);
        
        // 1. Build batched initial_weights [B × 2 × H] and showdown outcomes [B × ...]
        let initial_weights_flat = self.build_batched_weights(specs);
        let (showdown_p0, showdown_p1) = self.build_batched_showdowns(specs);
        
        // 2. Upload per-batch data
        let d_initial_weights = self.stream.clone_htod(&initial_weights_flat)?;
        let d_showdown_p0 = self.stream.clone_htod(&showdown_p0)?;
        let d_showdown_p1 = self.stream.clone_htod(&showdown_p1)?;
        
        // 3. Zero solver state for this batch
        let beh = batch_size * self.num_edges * self.num_hands;
        let bnh = batch_size * self.num_nodes * self.num_hands;
        self.d_regrets = self.stream.alloc_zeros::<f32>(beh.max(1))?;
        self.d_strategy_sum = self.stream.alloc_zeros::<f32>(beh.max(1))?;
        self.d_reach = self.stream.alloc_zeros::<f32>(bnh.max(1))?;
        self.d_cfv = self.stream.alloc_zeros::<f32>(bnh.max(1))?;
        
        // 4. Launch kernel: <<<batch_size, num_hands, shared_mem>>>
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (self.num_hands as u32, 1, 1),
            shared_mem_bytes: self.shared_mem_bytes,
        };
        
        unsafe {
            let mut b = self.stream.launch_builder(&self.kernel.cfr_solve);
            // ... args identical to gpu_solve_hand_parallel launch ...
            b.launch(cfg)?;
        }
        self.stream.synchronize()?;
        
        // 5. Download CFVs and extract per-subgame results
        let cfv_host: Vec<f32> = self.stream.clone_dtoh(&self.d_cfv)?;
        let strategy_sum_host: Vec<f32> = self.stream.clone_dtoh(&self.d_strategy_sum)?;
        
        let results = (0..batch_size).map(|b| {
            self.extract_result(b, &cfv_host, &strategy_sum_host)
        }).collect();
        
        Ok(results)
    }
    
    fn extract_result(
        &self, batch_idx: usize,
        cfv_host: &[f32],
        _strategy_sum_host: &[f32],
    ) -> SubgameResult {
        let nh = self.num_nodes * self.num_hands;
        let base = batch_idx * nh;
        
        // Root CFVs are at node 0 for each player
        // The kernel's last iteration leaves CFVs for player 1 in the cfv array
        // We need both players' CFVs — extract from the cfv array after the final backward pass
        // 
        // Actually: we need to run the kernel's final iteration for each player
        // and capture root CFVs. The current kernel alternates players and the
        // final cfv state is for the last player (player 1).
        //
        // For datagen, we need strategy_sum (to compute average strategy) and then
        // compute CFVs via a CPU best-response or directly from the solved game.
        //
        // Simplest: download strategy_sum, normalize to avg strategy on CPU,
        // then compute per-hand EVs using range_solver's compute_current_ev.
        
        // For now, use root CFVs from the last backward pass
        let mut cfvs = [vec![0.0f32; self.num_hands], vec![0.0f32; self.num_hands]];
        for h in 0..self.num_hands {
            cfvs[1][h] = cfv_host[base + h]; // player 1's CFVs from last iteration
        }
        // TODO: need player 0's CFVs too — requires strategy_sum normalization
        
        SubgameResult { root_cfvs: cfvs }
    }
    
    fn build_batched_weights(&self, specs: &[SubgameSpec]) -> Vec<f32> {
        let mut flat = vec![0.0f32; specs.len() * 2 * self.num_hands];
        for (b, spec) in specs.iter().enumerate() {
            for p in 0..2 {
                for (h, &w) in spec.initial_weights[p].iter().enumerate() {
                    if h < self.num_hands {
                        flat[b * 2 * self.num_hands + p * self.num_hands + h] = w;
                    }
                }
            }
        }
        flat
    }
    
    fn build_batched_showdowns(&self, specs: &[SubgameSpec]) -> (Vec<f32>, Vec<f32>) {
        let mut p0 = Vec::new();
        let mut p1 = Vec::new();
        for spec in specs {
            p0.extend_from_slice(&spec.showdown_outcomes_p0);
            p1.extend_from_slice(&spec.showdown_outcomes_p1);
        }
        (p0, p1)
    }
}

impl SubgameSpec {
    /// Build a SubgameSpec from extracted terminal data (for testing).
    pub fn from_terminal_data(term: &TerminalData, num_hands: usize) -> Self {
        // ... build initial_weights and showdown outcomes from term ...
        todo!()
    }
}
```

**Key implementation note:** The `new()` constructor must follow the EXACT same pattern as `gpu_solve_hand_parallel` in solver.rs for uploading topology, building sorted edges, computing card data, etc. — but split into one-time init (topology, fold payoffs, card indices) vs per-batch (showdown outcomes, initial weights).

**Step 4: Run tests**

Run: `cargo test -p gpu-range-solver -- batch_solver`
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat(gpu-range-solver): GpuBatchSolver for batched subgame solving"
```

---

## Task 2: Config + Pipeline GPU Path

Add `backend` field to DatagenConfig and the `run_gpu()` path to the domain pipeline.

**Files:**
- Modify: `crates/cfvnet/src/config.rs` (add `backend` and `gpu_batch_size` fields)
- Modify: `crates/cfvnet/Cargo.toml` (add gpu-range-solver dependency)
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs` (add `run_gpu()`)

**Step 1: Add config fields**

In `crates/cfvnet/src/config.rs`, add to `DatagenConfig`:

```rust
    /// Solver backend: "cpu" (default) or "gpu".
    /// GPU mode uses batched CUDA kernel for river datagen.
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Number of subgames per GPU batch. Default: 142 (RTX 6000 Ada SM count).
    #[serde(default)]
    pub gpu_batch_size: Option<usize>,
```

Add default function:
```rust
fn default_backend() -> String { "cpu".to_string() }
```

Update `Default` impl to include `backend: default_backend()` and `gpu_batch_size: None`.

**Step 2: Add gpu-range-solver dependency**

In `crates/cfvnet/Cargo.toml`, add:
```toml
gpu-range-solver = { path = "../gpu-range-solver", optional = true }
```

Add feature:
```toml
[features]
gpu-datagen = ["gpu-range-solver"]
```

**Step 3: Add run_gpu() to pipeline.rs**

```rust
impl DomainPipeline {
    pub fn run(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
        // ... existing setup ...
        
        #[cfg(feature = "gpu-datagen")]
        if config.datagen.backend == "gpu" && board_size >= 5 {
            return Self::run_gpu(config, output_path, sit_gen, builder, solver_config, writer, pb);
        }
        
        // ... existing CPU path ...
    }
    
    #[cfg(feature = "gpu-datagen")]
    fn run_gpu(
        config: &CfvnetConfig,
        output_path: &Path,
        sit_gen: SituationGenerator,
        builder: GameBuilder,
        solver_config: SolverConfig,
        writer: Arc<Mutex<RecordWriter>>,
        pb: Arc<ProgressBar>,
    ) -> Result<(), String> {
        use gpu_range_solver::batch::{GpuBatchSolver, SubgameSpec};
        
        let batch_size = config.datagen.gpu_batch_size.unwrap_or(142);
        let num_hands_max = 1024; // max hands supported by kernel
        
        // Build a representative game for topology extraction
        let mut sit_gen = sit_gen;
        let first_sit = sit_gen.next().ok_or("no situations to generate")?;
        let first_game = builder.build(&first_sit, &mut rng)
            .ok_or("failed to build representative game")?;
        
        let topo = gpu_range_solver::extract::extract_topology(&first_game.inner());
        let num_hands = first_game.num_hands_max();
        
        // Initialize GPU batch solver (CUDA context + kernel compile + topology upload)
        eprintln!("[gpu] Initializing CUDA context and compiling kernel...");
        let mut gpu = GpuBatchSolver::new(
            &topo, batch_size, num_hands, solver_config.max_iterations,
        ).map_err(|e| format!("GPU init failed: {e}"))?;
        eprintln!("[gpu] Ready. Batch size: {batch_size}, hands: {num_hands}");
        
        // Process first situation (already consumed from generator)
        let mut pending = vec![first_sit];
        
        loop {
            // Collect up to batch_size situations
            while pending.len() < batch_size {
                match sit_gen.next() {
                    Some(sit) => pending.push(sit),
                    None => break,
                }
            }
            if pending.is_empty() { break; }
            
            // Build SubgameSpecs for this batch
            let specs: Vec<SubgameSpec> = pending.iter()
                .filter_map(|sit| build_subgame_spec(sit, &topo, &first_game, num_hands))
                .collect();
            
            if specs.is_empty() {
                pending.clear();
                continue;
            }
            
            // Solve batch on GPU
            let results = gpu.solve_batch(&specs)
                .map_err(|e| format!("GPU solve failed: {e}"))?;
            
            // Extract and write records
            for (sit, result) in pending.iter().zip(results) {
                let records = build_training_records(sit, &result, num_hands);
                let mut w = writer.lock().unwrap();
                w.write(&records).map_err(|e| format!("write failed: {e}"))?;
            }
            
            pb.inc(pending.len() as u64);
            pending.clear();
        }
        
        Ok(())
    }
}

/// Build a SubgameSpec from a Situation for one river board.
fn build_subgame_spec(
    sit: &Situation,
    topo: &TreeTopology,
    representative_game: &Game,
    num_hands: usize,
) -> Option<SubgameSpec> {
    // Build initial weights from sit.ranges, applying board card blocking
    // Build showdown outcomes for this specific board
    // Uses the same logic as extract_terminal_data but for this board's cards
    todo!()
}

/// Build TrainingRecords from a SubgameResult.
fn build_training_records(
    sit: &Situation,
    result: &SubgameResult,
    num_hands: usize,
) -> Vec<TrainingRecord> {
    let mut records = Vec::with_capacity(2);
    for player in 0..2 {
        records.push(TrainingRecord {
            board: sit.board[..sit.board_size].to_vec(),
            pot: sit.pot as f32,
            effective_stack: sit.effective_stack as f32,
            player: player as u8,
            game_value: compute_game_value(&sit.ranges[player], &result.root_cfvs[player]),
            oop_range: sit.ranges[0],
            ip_range: sit.ranges[1],
            cfvs: pad_to_1326(&result.root_cfvs[player]),
            valid_mask: compute_valid_mask(&sit.board[..sit.board_size]),
        });
    }
    records
}
```

**Step 4: Run integration test**

Add test in pipeline.rs:

```rust
#[test]
#[cfg(feature = "gpu-datagen")]
fn pipeline_gpu_produces_records_for_river() {
    let tmp = NamedTempFile::new().unwrap();
    let mut config = test_config(5, 5);
    config.datagen.backend = "gpu".into();
    config.datagen.gpu_batch_size = Some(2);
    DomainPipeline::run(&config, tmp.path()).unwrap();
    
    let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
    let r0 = read_record(&mut reader).unwrap();
    assert_eq!(r0.board.len(), 5);
    assert!(r0.pot > 0.0);
}
```

**Step 5: Commit**

```bash
git commit -m "feat(cfvnet): GPU datagen pipeline for batched river solving"
```

---

## Task 3: SubgameSpec Builder — Per-Board Terminal Data

Implement `build_subgame_spec()` which constructs showdown outcomes and initial weights for a specific river board. This is the bridge between the datagen Situation and the GPU kernel's expected data format.

**Files:**
- Modify: `crates/gpu-range-solver/src/batch.rs` (add `SubgameSpec::for_river_board()`)
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs` (implement `build_subgame_spec`)

**The core challenge:** Given a `Situation` with 5 board cards and two player ranges, construct the showdown outcome matrices and initial weights matching the topology extracted from a representative game.

The showdown outcome computation reuses the logic from `extract_terminal_data` (in extract.rs) and `build_mega_terminal_data` / `build_batched_river_terminal_data` (in solver.rs). For each showdown node, build a `[H × H]` matrix of pre-scaled payoffs (+amount_win, -amount_lose, 0.0) based on the board's hand rankings.

**Key function:**

```rust
impl SubgameSpec {
    /// Build a spec for a specific river board.
    pub fn for_river_board(
        board: &[u8; 5],
        ranges: &[[f32; 1326]; 2],
        topo: &TreeTopology,
        hand_cards: &[Vec<(u8, u8)>; 2],
        hand_strength_data: &[/* per-board strength data */],
        num_hands: usize,
    ) -> Self {
        // 1. Build initial_weights with board card blocking
        // 2. Build showdown outcome matrices per showdown node
        // 3. Return SubgameSpec
    }
}
```

**Step 1: Implement and test**

Test that `SubgameSpec::for_river_board()` produces non-zero showdown outcomes and correctly-blocked initial weights.

**Step 2: Commit**

```bash
git commit -m "feat(gpu-range-solver): SubgameSpec builder for river boards"
```

---

## Task 4: End-to-End Test + Benchmark

Test the full pipeline: config → GPU datagen → record file → verify records.

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs` (integration test)

**Step 1: CLI test**

```bash
# Create a test config for GPU river datagen
cat > /tmp/gpu_river_test.yaml << 'EOF'
game:
  initial_stack: 200
  bet_sizes: ["50%", "100%", "a"]
  board_size: 5

datagen:
  street: "river"
  backend: "gpu"
  gpu_batch_size: 142
  num_samples: 1000
  solver_iterations: 500
  seed: 42
  pot_intervals: [[4, 50], [50, 200]]
EOF

time cargo run -p cfvnet --release --features gpu-datagen -- generate \
  -c /tmp/gpu_river_test.yaml -o /tmp/gpu_river_test.bin
```

**Step 2: Compare output with CPU datagen**

```bash
# Same config but CPU backend
cat > /tmp/cpu_river_test.yaml << 'EOF'
game:
  initial_stack: 200
  bet_sizes: ["50%", "100%", "a"]
  board_size: 5

datagen:
  street: "river"
  backend: "cpu"
  num_samples: 1000
  solver_iterations: 500
  seed: 42
  pot_intervals: [[4, 50], [50, 200]]
  threads: 18
EOF

time cargo run -p cfvnet --release -- generate \
  -c /tmp/cpu_river_test.yaml -o /tmp/cpu_river_test.bin
```

Compare: GPU should produce valid records at higher throughput.

**Step 3: Commit**

```bash
git commit -m "test(cfvnet): GPU datagen end-to-end test + benchmark"
```

---

## Summary

| Task | What | Crate |
|------|------|-------|
| 1 | GpuBatchSolver with persistent CUDA context | gpu-range-solver |
| 2 | Config fields + GPU pipeline path | cfvnet |
| 3 | Per-board SubgameSpec builder | gpu-range-solver + cfvnet |
| 4 | End-to-end test + benchmark | cfvnet |

**Critical path:** 1 → 3 → 2 → 4

**Expected throughput:** ~10,000-15,000 river subgames/second (GPU) vs ~600/second (CPU, 18 threads). ~20x speedup for river datagen.
