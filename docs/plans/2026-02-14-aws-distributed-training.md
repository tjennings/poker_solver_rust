# AWS Distributed MCCFR Training

**Date:** 2026-02-14
**Status:** Draft
**Prerequisite:** Vertical scaling plan (for AWS deployment workflow)

## Overview

A parameter-server architecture for distributed MCCFR training across multiple EC2 instances. Workers perform independent CFR traversals and send sparse regret/strategy deltas to a central coordinator that applies merges and DCFR discounting.

## Architecture

```
                    ┌─────────────────────┐
                    │   Coordinator (1x)  │
                    │                     │
                    │  regret_sum         │
                    │  strategy_sum       │
                    │  iteration counter  │
                    │  DCFR discount      │
                    │  checkpoint I/O     │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │ broadcast    │ broadcast     │ broadcast
              │ regret snap  │ regret snap   │ regret snap
              ▼              ▼               ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │  Worker 1  │ │  Worker 2  │ │  Worker K  │
       │            │ │            │ │            │
       │  M/K samps │ │  M/K samps │ │  M/K samps │
       │  local acc  │ │  local acc  │ │  local acc  │
       │  Rayon par  │ │  Rayon par  │ │  Rayon par  │
       └────────────┘ └────────────┘ └────────────┘
              │              │               │
              │ send deltas  │ send deltas   │ send deltas
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────────┐
                    │   Coordinator       │
                    │   merge + discount  │
                    │   → next iteration  │
                    └─────────────────────┘
```

## Iteration Protocol

### Synchronous mode (simpler, recommended first)

```
Iteration T:
  1. Coordinator broadcasts IterationStart {
       iteration: T,
       regret_snapshot: FxHashMap<u64, Vec<f64>>,  // ~10-100 MB serialized
       strategy_discount: f64,
       traversing_player: Player,
       base_seed: u64,
       sample_range: (start_idx, end_idx),         // per worker
     }

  2. Each Worker:
     - Receives regret_snapshot (immutable for this iteration)
     - Runs assigned sample range using cfr_traverse_pure()
     - Each sample uses per_sample_seed(base_seed, T, idx) — deterministic
     - Accumulates into local TraversalAccumulator
     - Sends IterationResult {
         regret_deltas: FxHashMap<u64, Vec<f64>>,   // sparse, ~1-10 MB
         strategy_deltas: FxHashMap<u64, Vec<f64>>,
         pruned_count: u64,
         total_count: u64,
       }

  3. Coordinator:
     - Waits for all K workers
     - Merges all deltas into regret_sum / strategy_sum (same as merge_accumulator)
     - Applies discount_regrets()
     - Advances iteration counter
     - Broadcasts next iteration
```

### Async mode (advanced, better throughput)

Workers don't wait for the latest snapshot — they use a slightly stale one (delay of 1-2 iterations). This hides network latency but trades off convergence rate per iteration. Literature shows this is safe for MCCFR convergence.

```
Coordinator maintains a version counter V.
Workers request the latest snapshot asynchronously.
Deltas tagged with the snapshot version they were computed against.
Coordinator accepts deltas from version V-d to V (staleness window d).
```

**Recommendation:** Start with synchronous mode. Only add async if network latency becomes a bottleneck (unlikely on same-AZ EC2 instances).

## Crate Structure

```
crates/distributed/
├── Cargo.toml
└── src/
    ├── lib.rs              # Re-exports, DistributedError
    ├── protocol.rs         # Message types (IterationStart, IterationResult, etc.)
    ├── codec.rs            # Serialization for regret maps (bincode + optional compression)
    ├── coordinator.rs      # Coordinator server (tokio + tonic gRPC)
    ├── worker.rs           # Worker client (connects to coordinator, runs traversals)
    └── config.rs           # DistributedConfig (YAML-driven)
```

### Dependencies

```toml
[dependencies]
poker-solver-core = { path = "../core" }
tokio = { version = "1", features = ["full"] }
tonic = "0.12"                   # gRPC framework
prost = "0.13"                   # protobuf serialization
bincode = "1"                    # fast regret map serialization
lz4_flex = "0.11"                # optional compression for large snapshots
rustc-hash = "2"
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"
```

## Protocol Messages (protobuf)

```protobuf
syntax = "proto3";
package poker_solver.distributed;

service MccfrCoordinator {
  // Worker registers and receives its sample range assignment
  rpc Register(RegisterRequest) returns (RegisterResponse);

  // Worker requests the current regret snapshot for an iteration
  rpc GetSnapshot(SnapshotRequest) returns (stream SnapshotChunk);

  // Worker submits traversal results
  rpc SubmitDeltas(DeltaSubmission) returns (SubmitResponse);

  // Worker polls for iteration readiness
  rpc WaitForIteration(WaitRequest) returns (IterationReady);
}

message RegisterRequest {
  string worker_id = 1;
  uint32 num_threads = 2;    // Worker reports its core count
}

message RegisterResponse {
  uint32 worker_index = 1;
  uint32 total_workers = 2;
  uint64 samples_start = 3;  // Assigned sample range
  uint64 samples_end = 4;
}

message SnapshotRequest {
  uint64 iteration = 1;
}

message SnapshotChunk {
  bytes data = 1;             // Bincode-serialized regret map chunk
  bool is_last = 2;
}

message DeltaSubmission {
  string worker_id = 1;
  uint64 iteration = 2;
  bytes regret_deltas = 3;    // Bincode-serialized FxHashMap<u64, Vec<f64>>
  bytes strategy_deltas = 4;
  uint64 pruned_count = 5;
  uint64 total_count = 6;
}

message SubmitResponse {
  bool accepted = 1;
}

message WaitRequest {
  uint64 after_iteration = 1;
}

message IterationReady {
  uint64 iteration = 1;
  uint64 base_seed = 2;
  double strategy_discount = 3;
  uint32 traversing_player = 4;  // 0 = P1, 1 = P2
}
```

## Coordinator Implementation

```rust
/// Central coordinator that owns the regret/strategy tables.
pub struct Coordinator {
    /// Game definition (shared, immutable after init)
    game_config: PostflopConfig,

    /// The authoritative regret and strategy tables
    regret_sum: FxHashMap<u64, Vec<f64>>,
    strategy_sum: FxHashMap<u64, Vec<f64>>,

    /// Global iteration counter
    iteration: u64,

    /// DCFR parameters
    dcfr_alpha: f64,
    dcfr_beta: f64,
    dcfr_gamma: f64,

    /// Pruning config
    pruning: PruningCtx,

    /// RNG state (same sequence as single-machine)
    rng_state: u64,

    /// Connected workers
    workers: Vec<WorkerInfo>,

    /// Pending deltas for current iteration
    pending_deltas: Vec<TraversalAccumulator>,
}
```

### Key coordinator operations

1. **Snapshot broadcast:** Serialize `regret_sum` with bincode, optionally compress with LZ4. Stream to workers in chunks (gRPC streaming) to avoid message size limits.

2. **Delta merge:** Same logic as `MccfrSolver::merge_accumulator()` — extracted into a shared function usable by both local and distributed paths.

3. **Discount:** Same `discount_regrets()` logic, runs single-threaded on coordinator after merge.

4. **Checkpoint:** Same `save_checkpoint()` flow, writes to local disk or S3.

### Extracting shared logic from MccfrSolver

The key refactor: extract `merge_accumulator()`, `discount_regrets()`, and `cfr_traverse_pure()` into standalone functions that both the local `MccfrSolver` and the distributed coordinator/worker can call. These functions are already nearly standalone — they just need the `&mut self` dependency removed.

```rust
// crates/core/src/cfr/mccfr_ops.rs (new file, extracted from mccfr.rs)

/// Merge a TraversalAccumulator into regret/strategy tables.
pub fn merge_accumulator(
    regret_sum: &mut FxHashMap<u64, Vec<f64>>,
    strategy_sum: &mut FxHashMap<u64, Vec<f64>>,
    acc: TraversalAccumulator,
) -> (u64, u64) { ... }

/// Apply DCFR discounting to regret table.
pub fn discount_regrets(
    regret_sum: &mut FxHashMap<u64, Vec<f64>>,
    iteration: u64,
    alpha: f64,
    beta: f64,
) { ... }

// cfr_traverse_pure() is already a standalone function — just make it pub
```

## Worker Implementation

```rust
/// Worker that connects to coordinator and runs CFR traversals.
pub struct Worker {
    worker_id: String,
    coordinator_url: String,

    /// Local copy of the game (constructed from config)
    game: HunlPostflop,

    /// Cached deal pool (same as single-machine)
    deals: Vec<PostflopState>,

    /// Sample range assigned by coordinator
    sample_range: (usize, usize),
}
```

### Worker loop

```rust
impl Worker {
    async fn run(&mut self) -> Result<()> {
        let mut client = CoordinatorClient::connect(&self.coordinator_url).await?;

        // Register and get sample assignment
        let reg = client.register(RegisterRequest {
            worker_id: self.worker_id.clone(),
            num_threads: rayon::current_num_threads() as u32,
        }).await?;
        self.sample_range = (reg.samples_start as usize, reg.samples_end as usize);

        loop {
            // Wait for iteration to be ready
            let iter_info = client.wait_for_iteration(WaitRequest {
                after_iteration: current_iter,
            }).await?;

            // Download regret snapshot
            let regret_snapshot = self.download_snapshot(&mut client, iter_info.iteration).await?;

            // Run local traversals (reuses existing Rayon parallel code)
            let acc = self.run_traversals(
                &regret_snapshot,
                iter_info.base_seed,
                iter_info.traversing_player,
                iter_info.strategy_discount,
            );

            // Submit deltas
            client.submit_deltas(DeltaSubmission {
                worker_id: self.worker_id.clone(),
                iteration: iter_info.iteration,
                regret_deltas: bincode::serialize(&acc.regret_deltas)?,
                strategy_deltas: bincode::serialize(&acc.strategy_deltas)?,
                pruned_count: acc.pruned_count,
                total_count: acc.total_count,
            }).await?;

            current_iter = iter_info.iteration;
        }
    }

    /// Run traversals for assigned sample range using Rayon.
    fn run_traversals(
        &self,
        regret_snapshot: &FxHashMap<u64, Vec<f64>>,
        base_seed: u64,
        player: Player,
        strategy_discount: f64,
    ) -> TraversalAccumulator {
        let (start, end) = self.sample_range;
        let num_states = self.deals.len();
        let sample_weight = num_states as f64 / total_samples as f64;

        (start..end)
            .into_par_iter()
            .fold(
                TraversalAccumulator::new,
                |mut acc, idx| {
                    acc.rng_state = per_sample_seed(base_seed, iteration, idx);
                    let state_idx = (per_sample_seed(base_seed, iteration, idx)
                        % num_states as u64) as usize;
                    cfr_traverse_pure(
                        &self.game, regret_snapshot, &mut acc,
                        &self.deals[state_idx], player,
                        1.0, 1.0, sample_weight, strategy_discount, pruning,
                    );
                    acc
                },
            )
            .reduce(TraversalAccumulator::new, TraversalAccumulator::merge)
    }
}
```

## Network Bandwidth Analysis

### Per-iteration data transfer

| Direction | Data | Size (100K info sets) | Size (1M info sets) |
|-----------|------|----------------------|---------------------|
| Coordinator → Workers | Regret snapshot | ~6 MB (bincode) | ~60 MB |
| Coordinator → Workers | + LZ4 compressed | ~2 MB | ~20 MB |
| Workers → Coordinator | Regret deltas (sparse) | ~0.5-2 MB each | ~2-10 MB each |
| Workers → Coordinator | Strategy deltas | ~0.5-2 MB each | ~2-10 MB each |

### Bandwidth requirements

With K=8 workers, 100K info sets, 1 iteration/second target:
- Broadcast: 8 × 2 MB = 16 MB/s
- Collect: 8 × 2 MB = 16 MB/s
- Total: ~32 MB/s = ~256 Mbps

EC2 instances in the same placement group provide 25-100 Gbps — bandwidth is not a bottleneck.

### Latency budget

| Phase | Est. time (96-core worker) | Notes |
|-------|---------------------------|-------|
| Snapshot download | 5-50 ms | LZ4-compressed, same AZ |
| Traversal (500 samples) | 200-2000 ms | Depends on tree depth |
| Delta upload | 2-10 ms | Sparse, small |
| Coordinator merge | 1-5 ms | Single-threaded hash merge |
| Discount | 1-5 ms | Linear scan of regret table |
| **Total per iteration** | **~250-2100 ms** | |

Network overhead is <5% of iteration time for typical configs. Communication is not the bottleneck.

## Serialization Format

### Regret snapshot (broadcast)

Use bincode for the full `FxHashMap<u64, Vec<f64>>`. This is the simplest approach and benchmarks show bincode serialization of hash maps is fast (~10 MB/s on a single core for 100K entries).

For larger tables (>50 MB), add LZ4 compression. Regret tables compress well (~3:1) because many entries cluster near zero.

### Sparse deltas (worker → coordinator)

Only transmit info sets that were actually visited during traversal. The `TraversalAccumulator` already tracks only visited info sets, so serializing it directly gives sparse representation for free.

```rust
/// Serialize only non-zero delta entries for network transmission.
pub fn serialize_deltas(acc: &TraversalAccumulator) -> Vec<u8> {
    bincode::serialize(&(&acc.regret_deltas, &acc.strategy_deltas))
        .expect("serialization should not fail")
}
```

## Deal Pool Consistency

All workers must use the same deal pool to maintain semantic equivalence with single-machine training. Two approaches:

### Option A: Seed-based regeneration (recommended)

Each worker generates its own deal pool from the same seed. Since `initial_states()` is deterministic given the same game config, all workers will have identical deal pools.

```rust
// Worker initialization
let game = HunlPostflop::new(config);
let deals = game.initial_states();  // Same seed → same deals everywhere
```

### Option B: Coordinator distributes deals

Coordinator generates deals once and sends to workers during registration. More network traffic but guarantees consistency even if `initial_states()` has non-deterministic behavior.

**Recommendation:** Option A. The game's `initial_states()` is already deterministic (seeded RNG).

## Configuration

```yaml
# distributed_training.yaml
game:
  stack_depth: 100
  bet_sizes: [0.33, 0.5, 0.75, 1.0]

abstraction:
  flop_buckets: 200
  turn_buckets: 200
  river_buckets: 500
  samples_per_street: 5000

training:
  iterations: 10000
  seed: 42
  output_dir: "s3://my-bucket/runs/distributed-001"
  mccfr_samples: 4000       # Total across all workers
  deal_count: 100000

distributed:
  coordinator_addr: "0.0.0.0:50051"
  num_workers: 8             # Expected worker count
  worker_threads: 16         # Rayon threads per worker
  snapshot_compression: lz4  # none | lz4
  checkpoint_interval: 100   # Iterations between checkpoints
  checkpoint_to_s3: true
  s3_bucket: "my-poker-solver"
  s3_prefix: "runs/distributed-001"
```

## AWS Infrastructure

### Placement group (critical for low latency)

```bash
aws ec2 create-placement-group \
  --group-name mccfr-cluster \
  --strategy cluster
```

All instances (coordinator + workers) should be in the same placement group for minimal network latency.

### Instance recommendations

| Role | Instance | vCPUs | Count | Cost/hr |
|------|----------|-------|-------|---------|
| Coordinator | `c7i.4xlarge` | 16 | 1 | $0.68 |
| Worker | `c7i.8xlarge` | 32 | 8 | $1.36 each |
| **Total** | | 272 | 9 | **$11.56** |

The coordinator doesn't need many cores (merge is single-threaded). Workers benefit from more cores for Rayon parallelism.

### Terraform sketch

```hcl
resource "aws_placement_group" "mccfr" {
  name     = "mccfr-cluster"
  strategy = "cluster"
}

resource "aws_instance" "coordinator" {
  ami                    = var.trainer_ami_id
  instance_type          = "c7i.4xlarge"
  placement_group        = aws_placement_group.mccfr.id
  vpc_security_group_ids = [aws_security_group.mccfr.id]

  tags = { Name = "mccfr-coordinator" }
}

resource "aws_instance" "worker" {
  count                  = var.num_workers
  ami                    = var.trainer_ami_id
  instance_type          = "c7i.8xlarge"
  placement_group        = aws_placement_group.mccfr.id
  vpc_security_group_ids = [aws_security_group.mccfr.id]

  tags = { Name = "mccfr-worker-${count.index}" }
}

resource "aws_security_group" "mccfr" {
  name = "mccfr-cluster"

  # gRPC between coordinator and workers
  ingress {
    from_port   = 50051
    to_port     = 50051
    protocol    = "tcp"
    self        = true
  }

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

## Implementation Plan

### Phase 1: Extract shared MCCFR operations (~1 day)

Extract the following from `MccfrSolver` into standalone public functions in a new `mccfr_ops.rs`:

1. `merge_accumulator()` — takes `&mut regret_sum`, `&mut strategy_sum`, `acc`
2. `discount_regrets()` — takes `&mut regret_sum`, iteration, alpha, beta
3. Make `TraversalAccumulator` and `cfr_traverse_pure()` public
4. Make `per_sample_seed()` public

Refactor `MccfrSolver` to call these extracted functions. All existing tests must pass.

### Phase 2: Protocol definition (~1 day)

1. Define protobuf messages
2. Implement bincode serialization for regret maps
3. Add LZ4 compression layer
4. Write round-trip tests for serialization

### Phase 3: Coordinator server (~2-3 days)

1. gRPC server with tonic
2. Worker registration and sample range assignment
3. Snapshot streaming (chunked for large maps)
4. Delta collection and merge
5. Iteration lifecycle management
6. Checkpoint integration (local + S3)

### Phase 4: Worker client (~1-2 days)

1. gRPC client connecting to coordinator
2. Snapshot download and deserialization
3. Local traversal execution (reuses existing Rayon code)
4. Delta submission

### Phase 5: CLI integration (~1 day)

1. Add `Distributed` variant to `SolverMode` enum
2. Parse distributed config from YAML
3. Coordinator and worker subcommands:
   ```bash
   poker-solver-trainer distributed coordinator -c config.yaml
   poker-solver-trainer distributed worker -c config.yaml --coordinator-addr 10.0.0.1:50051
   ```

### Phase 6: Testing (~2-3 days)

1. Unit tests for extracted operations
2. Integration test: coordinator + 2 workers on localhost
3. Equivalence test: verify distributed training produces same convergence rate as single-machine (not bit-identical due to float ordering, but statistically equivalent)
4. Stress test: large regret tables, many iterations, worker crash/restart

### Phase 7: AWS deployment (~1-2 days)

1. Docker images for coordinator and worker
2. Terraform configuration
3. S3 checkpoint integration
4. Monitoring and logging (CloudWatch)

**Total estimated effort: 2-3 weeks**

## Scaling Analysis

### Linear speedup region

With K workers and M total samples per iteration:

```
T_iteration = T_broadcast + T_compute(M/K) + T_collect + T_merge + T_discount
```

Speedup is approximately linear when `T_compute(M/K)` dominates. This holds when:
- M/K is large enough that each worker does meaningful work (M/K > 50)
- Network transfer time is small relative to compute (~5% threshold)
- Merge time is small relative to compute (linear in info set count)

### Diminishing returns

Beyond K=16-32 workers, expect diminishing returns because:
1. Broadcast time grows linearly with K (unless using multicast)
2. Merge time grows linearly with K (sum of K delta maps)
3. Staleness effects reduce convergence rate per iteration (async mode)

### Theoretical maximum

For 100K info sets, 2000 samples/iteration:
- Compute per sample: ~1-5 ms
- Total compute: 2-10 seconds
- Network + merge overhead: ~50-200 ms
- Maximum useful workers: ~20-40 (before overhead > compute)

## Fault Tolerance

### Worker crash

Coordinator detects missing delta submission after timeout. Two options:
1. **Retry:** Reassign failed worker's sample range to surviving workers
2. **Skip:** Accept slightly fewer samples for this iteration (safe — MCCFR converges with any sample count)

**Recommendation:** Skip. MCCFR is robust to sample count variation. Retrying adds complexity.

### Coordinator crash

Coordinator is a single point of failure. Mitigate with:
1. Periodic checkpoints to S3 (already planned)
2. Coordinator restart from latest checkpoint
3. Workers reconnect automatically after coordinator restart

### Network partition

Workers that can't reach coordinator should pause and retry with exponential backoff. No data corruption risk since workers never modify shared state.

## Comparison with Alternatives

### Why not MPI?

MPI (via `rsmpi` crate) would be more efficient for the allreduce pattern but:
- Harder to deploy on AWS (requires MPI runtime configuration)
- Less fault-tolerant (MPI typically kills all processes on single failure)
- gRPC is more debuggable and familiar

### Why not Ray/Dask?

Python-based distributed frameworks don't integrate naturally with Rust. The FFI overhead and serialization costs would negate the performance benefits.

### Why not AWS Lambda?

Lambda has a 15-minute timeout and limited compute. The per-invocation overhead of downloading the regret snapshot would dominate for short iterations.

### Why not S3-based coordination?

Using S3 as a coordination layer (workers write deltas to S3, coordinator polls) is simpler but adds ~100ms latency per S3 operation. Viable for iterations that take >10 seconds but too slow for sub-second iterations.

## Key References

- Lanctot et al., "Monte Carlo Sampling for Regret Minimization in Extensive Games" (NIPS 2009)
- Brown & Sandholm, "Solving Imperfect-Information Games via Discounted Regret Minimization" (AAAI 2019)
- Dean et al., "Large Scale Distributed Deep Networks" (NIPS 2012) — parameter server architecture
