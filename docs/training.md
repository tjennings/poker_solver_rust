# Training Reference

All commands are run via the `poker-solver-trainer` crate:

```bash
cargo run -p poker-solver-trainer --release -- <subcommand> [options]
```

Always use `--release` for training and diagnostics.

## Commands

### train-blueprint

Train a blueprint strategy using MCCFR. See `sample_configurations/blueprint_v2_with_tui.yaml` for a complete config example.

```bash
cargo run -p poker-solver-trainer --release -- train-blueprint \
  -c sample_configurations/blueprint_v2_with_tui.yaml
```

### train-blueprint-mp

Train a multiplayer (2-8 player) blueprint strategy using external-sampling MCCFR.

```bash
cargo run -p poker-solver-trainer --release -- train-blueprint-mp <config.yaml>
```

#### Config Format

The N-player config uses a different format from the 2-player `train-blueprint` command:

```yaml
game:
  name: "6-max 100bb BB-ante"
  num_players: 6
  stack_depth: 200        # chips (1 BB = 2 chips)
  blinds:
    - seat: 0
      type: small_blind
      amount: 1
    - seat: 1
      type: big_blind
      amount: 2
    - seat: 1
      type: bb_ante
      amount: 2

action_abstraction:
  preflop:
    lead: ["5bb", "6bb"]        # opening raise sizes
    raise:
      - ["3.0x"]                 # 3-bet sizes (first raise depth)
      - ["2.5x"]                 # 4-bet+ sizes (repeats for deeper)
  flop:
    lead: [0.33, 0.67, 1.0]     # pot fractions for opening bets
    raise:
      - [0.5, 1.0, 2.0]         # raise sizes (first raise)
  turn:
    lead: [0.5, 1.0]
    raise:
      - [0.67, 1.0]
  river:
    lead: [0.5, 1.0]
    raise:
      - [1.0]

clustering:
  preflop:
    buckets: 169
  flop:
    buckets: 200
  turn:
    buckets: 200
  river:
    buckets: 200

training:
  iterations: 100000
  batch_size: 200
  dcfr_alpha: 1.5
  dcfr_beta: 0.0
  dcfr_gamma: 2.0
  lcfr_warmup_iterations: 5000000

snapshots:
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: "/data/blueprint_mp_6p"
```

#### Key Differences from `train-blueprint`

| Feature | `train-blueprint` (v2) | `train-blueprint-mp` |
|---------|----------------------|---------------------|
| Players | 2 only | 2-8 |
| Blind structure | `small_blind` + `big_blind` fields | Per-seat `blinds` list with types |
| Bet sizing | Per-street, indexed by raise depth | Lead/raise split per street |
| Info key | 64-bit, 6 action slots | 128-bit, 22 action slots |
| Side pots | N/A (2 players) | Full multi-way resolution |

#### Sample Configs

- `sample_configurations/blueprint_mp_3player.yaml` -- 3-player 50bb test
- `sample_configurations/blueprint_mp_6player_ante.yaml` -- 6-player 100bb with BB-ante

---

### range-solve

Solve a postflop spot with exact (no abstraction) Discounted CFR. Uses the `range-solver` crate -- a self-contained reimplementation of b-inary/postflop-solver producing identical output.

Solves a **single spot** with full hand granularity (1326 hole card combos, no bucketing) and suit isomorphism reduction.

```bash
# River spot with specific ranges
cargo run -p poker-solver-trainer --release -- range-solve \
  --oop-range "QQ+,AKs,AKo" \
  --ip-range "22+,A2s+,KQs" \
  --flop "Qs Jh 2c" --turn "8d" --river "3s" \
  --pot 100 --effective-stack 200 \
  --iterations 1000

# Flop spot (turn + river solved internally via chance nodes)
cargo run -p poker-solver-trainer --release -- range-solve \
  --oop-range "AA,KK,QQ,AKs" \
  --ip-range "TT-66,AQs-ATs,KQs,QJs" \
  --flop "Qs Jh 2c" \
  --pot 100 --effective-stack 300 \
  --iterations 500

# Custom bet sizing
cargo run -p poker-solver-trainer --release -- range-solve \
  --oop-range "QQ+,AKs" --ip-range "22+,A2s+" \
  --flop "Ah Kd 7c" --turn "2s" \
  --pot 80 --effective-stack 160 \
  --oop-bet-sizes "33%,67%,a" --oop-raise-sizes "2.5x" \
  --ip-bet-sizes "33%,67%,a" --ip-raise-sizes "2.5x" \
  --iterations 1000 --target-exploitability 0.3
```

Options:
- `--oop-range <RANGE>` -- OOP player's range in PioSOLVER format (required)
- `--ip-range <RANGE>` -- IP player's range (required)
- `--flop <CARDS>` -- Flop cards, e.g. `"Qs Jh 2c"` (required)
- `--turn <CARD>` -- Turn card, e.g. `"8d"` (optional)
- `--river <CARD>` -- River card, e.g. `"3s"` (optional; requires `--turn`)
- `--pot <N>` -- Starting pot size (default: 100)
- `--effective-stack <N>` -- Effective stack size (default: 100)
- `--iterations <N>` -- Maximum DCFR iterations (default: 1000)
- `--target-exploitability <F>` -- Stop early when exploitability drops below this (default: 0.5)
- `--oop-bet-sizes <SIZES>` -- OOP bet sizes, comma-separated (default: `"50%,100%"`)
- `--oop-raise-sizes <SIZES>` -- OOP raise sizes (default: `"60%,100%"`)
- `--ip-bet-sizes <SIZES>` -- IP bet sizes (default: `"50%,100%"`)
- `--ip-raise-sizes <SIZES>` -- IP raise sizes (default: `"60%,100%"`)
- `--compressed` -- Use 16-bit compressed storage (less memory, slightly less precision)

**Bet size syntax:**
| Format | Meaning | Example |
|-|-|-|
| `N%` | Pot-relative | `50%` = half pot |
| `Nx` | Previous-bet-relative (raises only) | `2.5x` = 2.5x previous bet |
| `Ne` | Geometric over N streets | `2e` = geometric over 2 streets |
| `Nc` | Additive (chips) | `100c` = 100 chips |
| `a` | All-in | |

**Output:** Per-iteration exploitability, then a per-hand strategy table at the root node showing action probabilities for each hole card combo.

**Street determination:** Automatically set from which cards are provided:
- Flop only -> solves from flop (turn + river as chance nodes)
- Flop + turn -> solves from turn (river as chance node)
- Flop + turn + river -> solves river only (fastest)

**Algorithm:** Discounted CFR with a=1.5, b=0.5, g=3.0. Strategy resets at power-of-4 iterations. Multithreaded via rayon.

---

### gpu-range-solve

GPU-accelerated version of `range-solve` using custom CUDA kernels via the `gpu-range-solver` crate. Same inputs and output format as `range-solve`. Requires an NVIDIA GPU with CUDA 12.1+.

```bash
cargo run -p poker-solver-trainer --release -- gpu-range-solve \
  --oop-range "QQ+,AKs" --ip-range "JJ-99,AQs" \
  --flop "Qs Jh 2c" --turn "8d" --river "3s" \
  --pot 100 --effective-stack 100 --iterations 500
```

Options are identical to `range-solve` except `--compressed` is not supported.

**Architecture:** Hand-parallel CUDA kernel — one thread block per subgame, up to 1024 threads handling 1024 hands in parallel. Tree traversal is sequential within the block; `__syncthreads` is used only for fold terminal evaluation (card-blocking reduction). No cooperative groups required.

**Performance characteristics:**
- CUDA context initialization: ~280ms one-time cost per invocation
- Per-iteration: ~0.6-1.2ms (vs ~0.02-0.08ms CPU) for single river subgames
- GPU advantage is in **throughput**: 142 independent subgames solved simultaneously when batched (one per SM on RTX 6000 Ada)
- Best for: batched datagen (many subgames), not single-spot analysis (use `range-solve` for that)

---

### cluster

Run the potential-aware clustering pipeline to build bucket assignments for all four streets. Uses Pluribus-style bottom-up abstraction: river (equity k-means) → turn (EMD over river buckets) → flop (EMD over turn buckets) → preflop (EMD over flop buckets).

```bash
cargo run -p poker-solver-trainer --release -- cluster \
  -c sample_configurations/blueprint_v2_with_tui.yaml \
  -o output/buckets
```

Options:
- `-c <CONFIG>` -- YAML config file (uses the `clustering` section)
- `-o <DIR>` -- Output directory for `.buckets` files

Produces four files: `river.buckets`, `turn.buckets`, `flop.buckets`, `preflop.buckets`. If bucket files already exist in the output directory, clustering is skipped.

Clustering config parameters (in the `clustering` section of the YAML):

| Parameter | Default | Description |
|-|-|-|
| `algorithm` | `potential_aware_emd` | Clustering algorithm |
| `river.buckets` | -- | Number of river buckets |
| `turn.buckets` | -- | Number of turn buckets |
| `flop.buckets` | -- | Number of flop buckets |
| `preflop.buckets` | -- | Number of preflop buckets |
| `kmeans_iterations` | 100 | K-means iterations per street |
| `seed` | 42 | Random seed for board sampling |

### diag-clusters

Diagnostics for pre-computed cluster bucket files.

```bash
# Basic bucket distribution report
cargo run -p poker-solver-trainer --release -- diag-clusters \
  -d output/buckets

# Equity audit (sample boards and check intra-bucket equity consistency)
cargo run -p poker-solver-trainer --release -- diag-clusters \
  -d output/buckets --audit --audit-boards 100

# Cross-street transition matrices (verify potential-aware linkage)
cargo run -p poker-solver-trainer --release -- diag-clusters \
  -d output/buckets --transitions

# Sample hands from a specific bucket
cargo run -p poker-solver-trainer --release -- diag-clusters \
  -d output/buckets --sample-bucket river 5
```

Options:
- `-d <DIR>` -- Directory containing `.buckets` files (required)
- `--audit` -- Run equity audit by sampling boards
- `--audit-boards <N>` -- Number of boards to sample for audit (default: 50)
- `--transitions` -- Print cross-street transition matrices for adjacent street pairs (preflop→flop, flop→turn, turn→river)
- `--sample-bucket <STREET> <BUCKET_ID>` -- Show 10 sample hands from the given bucket
- `--centroid-emd <STREET>` -- Placeholder; centroid EMD requires feature vectors not stored in bucket files

### diff-clusters

Compare two sets of bucket files to measure quality improvement and clustering similarity.

```bash
cargo run -p poker-solver-trainer --release -- diff-clusters \
  --dir-a /path/to/old/clusters \
  --dir-b /path/to/new/clusters \
  --sample-boards 200

# Verbose mode with equity histogram
cargo run -p poker-solver-trainer --release -- diff-clusters \
  --dir-a /path/to/old/clusters \
  --dir-b /path/to/new/clusters \
  --sample-boards 200 \
  --verbose
```

- `--dir-a`, `--dir-b` -- directories containing `.buckets` files to compare
- `--sample-boards` -- boards to sample for equity audit (default 200, 0 = skip)
- `--verbose` -- show per-equity-bin bucket histogram

Reports per-street: bucket size stats, intra-bucket equity std (lower = better), and Adjusted Rand Index (1.0 = identical groupings, 0.0 = random agreement).

---

## Training TUI Dashboard

When `tui.enabled: true` in the config, `train-blueprint` launches a full-screen terminal dashboard instead of text output.

**Parallel Training:** Blueprint V2 automatically uses all available CPU cores. Each batch of `batch_size` deals (default: 200) is processed in parallel using Rayon's thread pool. LCFR discount and snapshots run between batches. Set `RAYON_NUM_THREADS=N` to limit core usage.

**Strategy Delta Stopping:** Set `target_strategy_delta` in the training config to auto-stop when the average strategy stabilises. The delta is the mean max absolute probability change across all (node, bucket) information sets between metric checks. Checked every `print_every_minutes`. Example: `target_strategy_delta: 0.001` stops when the strategy is changing by less than 0.1% on average.

**Resume Training:** Set `resume: true` under `snapshots:` to continue from the latest snapshot in `output_dir`. The trainer loads regrets and iteration count from the highest-numbered `snapshot_NNNN/` directory (or `final/` if present).

**Snapshot Retention:** Set `max_snapshots: N` under `snapshots:` to keep only the N most recent snapshots. After each save, older `snapshot_NNNN/` directories are deleted. The `final/` directory is never pruned. Omit or set to `null` for unlimited retention.

**Left panel:** iteration progress, throughput sparkline, exploitability chart
**Right panel:** tabbed 13x13 strategy grids for configured scenarios

**Hotkeys:**
- `p` -- pause/resume training
- `s` -- trigger immediate snapshot
- `e` -- trigger exploitability calculation
- left/right arrows -- switch scenario tabs
- `q` -- quit gracefully

**Convergence indicators:** Cells where strategy has stabilized (delta < 0.01) show a bright green border. As training progresses, more cells "light up" -- giving visual feedback on convergence.

Use `--no-tui` to disable the dashboard and use text output instead.

See `sample_configurations/blueprint_v2_with_tui.yaml` for a complete example.

## Blueprint Training Configuration

All `game:` section values are in **chips** (1 BB = 2 chips). Example: `stack_depth: 200` = 100 BB, `small_blind: 1`, `big_blind: 2`. Preflop action sizes use chip amounts with a `bb` suffix: `"5bb"` = raise to 5 chips (2.5 BB). Display converts to BB at the UI/CLI boundary only (dividing by 2). See `docs/architecture.md` for full unit convention.

The `training:` section of the blueprint YAML config controls the MCCFR training loop. Key parameters:

### Optimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer` | `"dcfr"` | CFR variant: `"dcfr"`, `"sapcfr+"`, `"brcfr+"`, `"lcfr"`, `"cfr+"` |
| `dcfr_alpha` | `1.5` | Positive regret discount exponent. Higher = retain positive regrets longer |
| `dcfr_beta` | `0.0` | Negative regret discount exponent. Used by DCFR only (SAPCFR+ floors to 0) |
| `dcfr_gamma` | `2.0` | Strategy sum discount exponent. Higher = weight recent strategies more |
| `dcfr_epoch_cap` | `null` | Optional cap on discount epoch counter. Prevents discount from converging to 1.0 |
| `sapcfr_eta` | `0.5` | SAPCFR+ prediction step size. 0 = no prediction (DCFR+RM+), 1 = full PCFR+, 0.5 = dampened |
| `brcfr_eta` | `0.6` | BRCFR+ BR prediction weight. Scales the best-response signal in strategy computation |
| `brcfr_warmup_iterations` | `0` | Iterations of pure DCFR+ before the first BR prediction pass |
| `brcfr_interval` | `100000000` | Iterations between BR prediction passes (after warmup) |

**DCFR** (default): Discounted CFR with polynomial decay. Positive regrets multiplied by `t^α/(t^α+1)`, negative by `t^β/(t^β+1)`, strategy sums by `(t/(t+1))^γ`. Standard choice from Brown & Sandholm 2019.

**SAPCFR+**: Simplified Asymmetric Predictive CFR+. Combines DCFR discount with RM+ (negative regret flooring) and predictive strategy computation. Stores previous iteration's instantaneous regret as a prediction, then computes strategy from `R + eta * prediction` instead of raw cumulative regrets. Based on Xu et al. 2025. Requires extra ~1.1 GB for prediction buffer. Since negative regrets are floored to 0, `dcfr_beta` is ignored and `prune_threshold` should be 0 or negative.

**LCFR**: Linear CFR (α=β=γ=1). Used by Pluribus. Simplest discounting scheme.

**CFR+**: Regret matching+ with negative regret flooring. No discounting.

**BRCFR+**: Best-Response augmented DCFR+. Layers periodic best-response prediction passes on top of the standard DCFR+ optimizer. During the warmup phase (`brcfr_warmup_iterations`), behaves identically to DCFR+. After warmup, a full BR traversal runs every `brcfr_interval` iterations for both players. The BR-derived per-infoset regrets are stored in the prediction buffer and used in strategy computation as `R_tilde = max(0, R + eta * decay * v_br)`. The decay factor starts at 1.0 after each BR pass and decreases linearly to 0.0 over the refresh interval, so stale predictions fade naturally. When decay reaches 0, behavior is pure DCFR+. Exploitability is measured for free during each BR pass (no separate exploitability calculation needed). Requires the same prediction buffer as SAPCFR+ (~1.1 GB extra). Based on ideas from CFR-BR (Johanson 2012) with decay scheduling.

### Example: BRCFR+ Configuration

```yaml
training:
  cluster_path: "./local_data/buckets/200_v1"
  time_limit_minutes: 7200
  optimizer: "brcfr+"
  brcfr_eta: 0.6
  brcfr_warmup_iterations: 300000000
  brcfr_interval: 100000000
  dcfr_alpha: 1.5
  dcfr_gamma: 2.0
  dcfr_epoch_cap: 40
  batch_size: 4000
```

### Variance Reduction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_baselines` | `false` | Enable VR-MCCFR variance-reducing baselines (Schmid et al. 2019) |
| `baseline_alpha` | `0.01` | Baseline EMA learning rate. Lower = smoother estimates, slower adaptation |

When enabled, the opponent traversal uses learned baselines to reduce sampling variance by up to 1000×. Each (node, bucket, action) gets an exponential moving average of observed counterfactual values. The baseline-corrected formula is unbiased and degenerates to standard sampling when baselines are zero. Requires extra ~1.1 GB for the baseline buffer (same size as regret buffer).

### Schedule & Pruning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lcfr_warmup_iterations` | `0` | Iterations before discounting starts |
| `lcfr_discount_interval` | `1` | Iterations between discount applications |
| `prune_after_iterations` | `0` | Iterations before action pruning starts |
| `prune_threshold` | `-300` | Cumulative regret threshold for pruning. Actions below this are skipped |
| `batch_size` | `200` | Deals per parallel batch |
| `time_limit_minutes` | `0` | Stop after this many minutes (0 = unlimited) |
| `purify_threshold` | `0.0` | Purify strategies with probability below this threshold (0 = disabled) |

**Important for SAPCFR+**: Since RM+ floors negative regrets to 0, they can't accumulate below the prune threshold. Set `prune_threshold: 0` to effectively disable pruning, or use a small negative value as a safety margin.

**Regret overflow**: Regrets are stored as `i32` (×1000 scaling, max ~2.1M). If `lcfr_discount_interval` is too large, regrets overflow and the trainer panics with a clear message. For SAPCFR+ (which only accumulates positive regrets), keep the discount interval reasonable (e.g., 1M-10M).

### Example: SAPCFR+ with Baselines

```yaml
training:
  cluster_path: "./local_data/buckets/1k_v3"
  time_limit_minutes: 7200
  optimizer: "sapcfr+"
  sapcfr_eta: 0.5
  dcfr_alpha: 1.5
  dcfr_gamma: 2.0
  dcfr_epoch_cap: 80
  lcfr_warmup_iterations: 10000000
  lcfr_discount_interval: 10000000
  prune_after_iterations: 10000000
  prune_threshold: 0
  batch_size: 2000
  use_baselines: true
  baseline_alpha: 0.01
```

See `sample_configurations/blueprint_v2_1kbkt_sapcfr.yaml` for the full config.

---

## CFVnet Training Pipeline

The `cfvnet` crate trains Deep Counterfactual Value Networks following the Supremus/DeepStack approach: solve random subgames, extract per-combo counterfactual values, and train a neural network to predict them. Networks are trained bottom-up: river first, then turn (using the river network as leaf evaluator).

### River Network

#### Generate River Training Data

**CPU backend (default):**

```bash
cargo run -p cfvnet --release -- generate \
  --config sample_configurations/river_cfvnet.yaml \
  --output data/river_training.bin \
  --num-samples 1000000 \
  --threads 8
```

**GPU backend (NVIDIA GPU required):**

```bash
cargo run -p cfvnet --release --features gpu-datagen -- generate \
  --config sample_configurations/river_cfvnet.yaml \
  --output data/river_training.bin \
  --num-samples 1000000
```

Set `datagen.backend: "gpu"` in the YAML config to use GPU solving. The GPU backend solves batches of river subgames simultaneously using the hand-parallel CUDA kernel. Each batch launches up to `gpu_batch_size` (default: 142) subgames in a single kernel launch.

```yaml
datagen:
  street: "river"
  backend: "gpu"          # "cpu" (default) or "gpu"
  gpu_batch_size: 142     # subgames per GPU launch (default: 142)
  num_samples: 1000000
  solver_iterations: 500
```

**GPU requirements:**
- NVIDIA GPU with CUDA 12.1+ and compute capability ≥ 6.0
- Build with `--features gpu-datagen` to enable the GPU dependency
- Ranges must produce ≤ 1024 hands per player (games exceeding this fall back to CPU)
- True batching (142 games per launch) requires matching hand counts across games — use blueprint-derived ranges (`blueprint_path`) for best GPU utilization; random RSP ranges may have varying counts

#### Train the River Network

```bash
cargo run -p cfvnet --release -- train \
  --config sample_configurations/river_cfvnet.yaml \
  --data data/river_training.bin \
  --output models/river_v1
```

#### Training Configuration

The `training` section of the YAML config controls the network architecture and training loop. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_layers` | 7 | Number of hidden layers |
| `hidden_size` | 500 | Width of each hidden layer |
| `batch_size` | 2048 | Mini-batch size |
| `epochs` | 2 | Number of training epochs |
| `learning_rate` | 0.001 | Initial learning rate (cosine annealed) |
| `lr_min` | 0.00001 | Minimum learning rate at end of schedule |
| `huber_delta` | 1.0 | Huber loss delta threshold |
| `aux_loss_weight` | 1.0 | Weight for auxiliary game-value loss |
| `validation_split` | 0.05 | Fraction of data reserved for validation |
| `checkpoint_every_n_epochs` | 1000 | Save checkpoint every N epochs (0 = disabled) |
| `shuffle_buffer_size` | 262144 | Streaming shuffle buffer capacity (records). Larger = better shuffle quality, more RAM |
| `prefetch_depth` | 4 | Number of pre-encoded batches buffered in the channel ahead of the training loop |

The training loop uses a **streaming dataloader** with an eviction-based shuffle buffer. A background reader fills a buffer of `shuffle_buffer_size` records, then continuously reads one record at a time — each new record randomly replaces a buffer slot and the evicted record flows into the next batch. This keeps disk reads continuous and eliminates pipeline stalls. A second encoder thread encodes batches in parallel via rayon and sends them through a bounded channel with `prefetch_depth` slots. Every record is seen exactly once per epoch. Increase `shuffle_buffer_size` for better randomization; increase `prefetch_depth` to keep the GPU fed when encoding is slow.

#### Evaluate on Held-Out Data

```bash
cargo run -p cfvnet --release -- evaluate \
  --model models/river_v1 \
  --data data/river_validation.bin
```

#### Compare Against Exact Solves

```bash
cargo run -p cfvnet --release -- compare \
  --model models/river_v1 \
  --num-spots 100
```

### Turn Network

Turn training requires a trained river network. The turn datagen solves random 4-card board situations using DCFR with the river CFV network as leaf evaluator (instead of solving all 46 river runouts exactly).

#### Generate Turn Training Data

Set `datagen.street: "turn"` and `game.river_model_path` in the config:

```yaml
game:
  initial_stack: 200
  bet_sizes: ["25%", "50%", "100%", "a"]
  river_model_path: "models/river_v1/model"
datagen:
  street: "turn"
  num_samples: 1000000
  solver_iterations: 1000
```

```bash
cargo run -p cfvnet --release -- generate \
  --config sample_configurations/turn_cfvnet.yaml \
  --output data/turn_training.bin \
  --num-samples 1000000
```

#### Train the Turn Network

```bash
cargo run -p cfvnet --release -- train \
  --config sample_configurations/turn_cfvnet.yaml \
  --data data/turn_training.bin \
  --output models/turn_v1
```

#### Compare Turn Model Against River Net Evaluator

Validates the turn model by comparing its predictions against fresh PostFlopGame solves using the river network as leaf evaluator:

```bash
cargo run -p cfvnet --release -- compare-net \
  --model models/turn_v1 \
  --river-model models/river_v1 \
  --num-spots 100
```

#### Compare Turn Model Against Exact River Solves

Validates the turn model against PostFlopGame with exact river evaluation (solves all 46 runouts). Slow but provides ground-truth comparison:

```bash
cargo run -p cfvnet --release -- compare-exact \
  --model models/turn_v1 \
  --num-spots 20
```

### BoundaryNet (Normalized EV Model)

BoundaryNet is a sibling model to CfvNet that outputs **normalized EVs** (`chip_ev / (pot + effective_stack)`) instead of pot-relative CFVs. It uses the same training data but with different input encoding (pot/stack as fractions of total stake) and normalized targets.

BoundaryNet is designed as a depth-boundary evaluator for the range-solver, enabling turn solving with neural network leaf values at river boundaries.

#### Train a BoundaryNet

Uses the same training data as CfvNet (generated with `generate`):

```bash
cargo run -p cfvnet --release -- train-boundary \
  --config sample_configurations/river_cfvnet.yaml \
  --data data/river_training.bin \
  --output models/boundary_v1
```

#### Evaluate BoundaryNet

Reports normalized MAE with per-SPR bucket breakdown (<1, 1-3, 3-10, 10+):

```bash
cargo run -p cfvnet --release -- eval-boundary \
  --model models/boundary_v1 \
  --data data/river_validation.bin
```

#### Compare BoundaryNet Against Ground Truth

Compares model predictions against datagen ground truth, reporting per-SPR MAE and worst-case error:

```bash
cargo run -p cfvnet --release -- compare-boundary \
  --model models/boundary_v1 \
  --data data/river_validation.bin \
  --num-positions 100
```

#### Using BoundaryNet in the Explorer

To enable neural boundary evaluation for turn solving in the explorer, configure the model path in the Tauri app's postflop settings. When set, turn subgame solving uses BoundaryNet at river boundaries instead of full-depth or rollout evaluation.

### Inspect Training Data Distribution

Print frequency histograms (stack size and pot size, 20 equal-width buckets) for generated training data:

```bash
cargo run -p cfvnet --release -- datagen-eval \
  --data data/river_training.bin

# Also works with a directory of split files
cargo run -p cfvnet --release -- datagen-eval \
  --data data/river_chunks/
```

### Compare Output

All compare commands (`compare`, `compare-net`, `compare-exact`) print:
- Summary statistics (mean/worst MAE and mBB)
- Best and worst spots by mBB
- mBB error histograms by stack size and pot size (20 equal-width buckets)
- Frequency histograms by stack size and pot size

### Configuration

See `sample_configurations/river_cfvnet.yaml` for all options. Key parameters:

| Parameter | Default | Description |
|-|-|-|
| `datagen.street` | `"river"` | Street to generate data for (`"river"` or `"turn"`) |
| `datagen.backend` | `"cpu"` | Solver backend: `"cpu"` or `"gpu"` (GPU requires `--features gpu-datagen`) |
| `datagen.gpu_batch_size` | 142 | Subgames per GPU kernel launch (only with `backend: "gpu"`) |
| `datagen.num_samples` | 1,000,000 | Training situations to generate |
| `datagen.solver_iterations` | 1000 | DCFR iterations per situation |
| `game.river_model_path` | none | Path to trained river model (required for turn) |
| `training.hidden_layers` | 7 | MLP depth |
| `training.hidden_size` | 500 | Hidden layer width |
| `training.batch_size` | 2048 | Training batch size |
| `training.epochs` | 2 | Training epochs |

---

## Convergence Testing

Test CFR algorithm convergence against an exact baseline using the `convergence-harness` crate. Defines a small tractable game ("Flop Poker") via YAML config, solves it exactly with range-solver DCFR, then compares MCCFR with bucketing against that baseline.

### Game Config

Define the game in a YAML file (see `sample_configurations/convergence_test.yaml`):

```yaml
game:
  flops:
    - "QhJdTh"   # draw-heavy, connected
    - "Ks7d2c"   # dry, rainbow
    - "8c8d3h"   # paired
  starting_pot: 2
  effective_stack: 20
  bet_sizes: "50%,100%,a"
  raise_sizes: "50%,100%,a"

baseline:
  max_iterations: 1000
  target_exploitability: 0.001

mccfr:
  iterations: 1000000
  buckets:
    preflop: 169
    flop: 169
    turn: 200
    river: 200
  checkpoints: [1000, 10000, 100000, 500000, 1000000]
```

### Generate Exact Baseline

Solves each flop exactly with range-solver DCFR (no abstraction). One-time cost — may take minutes to hours depending on game size.

```bash
cargo run -p convergence-harness --release -- generate-baseline \
  --config sample_configurations/convergence_test.yaml \
  --output-dir baselines/convergence
```

Produces: `summary.json`, `convergence.csv`, `strategy.bin`, `combo_ev.bin` in the output directory. Also prints a colored 13x13 SB strategy matrix on exit.

### Run MCCFR Comparison

Runs MCCFR with potential-aware bucketing on the same game, clusters each flop at startup (~2-10s per flop), then compares against the baseline via head-to-head EV (mbb/hand).

```bash
cargo run -p convergence-harness --release -- run-solver \
  --config sample_configurations/convergence_test.yaml \
  --baseline-dir baselines/convergence \
  --output-dir results/mccfr_run
```

At each checkpoint, prints: `h2h mbb/hand = -X.XX (OOP -X.XX, IP -X.XX)`. Negative = MCCFR loses to the exact solution. Final summary:

```
=== Result ===
solver:     MCCFR (200t/200r buckets)
iterations: 1000000
time:       4.4s
mbb/hand:   -230.90
output:     results/mccfr_run
```

### Compare Saved Results

Re-compare any two saved solver results without re-solving:

```bash
cargo run -p convergence-harness --release -- compare \
  --baseline-dir baselines/convergence \
  --result-dir results/mccfr_run
```

### Key Metrics

| Metric | Meaning |
|-|-|
| **mbb/hand** | Milli-big-blinds per hand lost vs exact strategy (negative = losing). 1000 mbb = 1 bb. |
| **L1 distance** | Average total variation distance between strategies per info set. < 0.05 = excellent. |
| **Combo EV diff** | Per-hand EV difference at each decision node. |

### Bucket Sweep

Test different bucket counts to find the optimal abstraction granularity:

```bash
for bkt in 10 50 100 200 500; do
  cargo run -p convergence-harness --release -- run-solver \
    --config sample_configurations/convergence_test.yaml \
    --baseline-dir baselines/convergence \
    --output-dir /tmp/sweep_${bkt} \
    2>&1 | grep "mbb/hand"
done
```

---

## Cloud Training (AWS)

See [`docs/cloud.md`](cloud.md) for running training jobs on AWS EC2 instances via the `solver-cloud` CLI.
