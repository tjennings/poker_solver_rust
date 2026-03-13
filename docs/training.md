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

---

## CFVnet Training Pipeline

The `cfvnet` crate trains Deep Counterfactual Value Networks following the Supremus/DeepStack approach: solve random subgames, extract per-combo counterfactual values, and train a neural network to predict them. Networks are trained bottom-up: river first, then turn (using the river network as leaf evaluator).

### River Network

#### Generate River Training Data

```bash
cargo run -p cfvnet --release -- generate \
  --config sample_configurations/river_cfvnet.yaml \
  --output data/river_training.bin \
  --num-samples 1000000 \
  --threads 8
```

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
| `reservoir_size` | 1500000 | Number of training records held in the GPU-resident reservoir |
| `reservoir_turnover` | 1.0 | Fraction of reservoir replaced per epoch by background refresh (0 = no refresh) |

The training loop uses a **GPU-resident reservoir**: `reservoir_size` records are loaded into GPU memory as tensors, and mini-batches are sampled directly on-device (zero PCIe transfer). A background thread continuously reads fresh records from disk and scatters them into random reservoir positions. Set `reservoir_turnover` to 0 to disable refresh (useful for small datasets that fit entirely in the reservoir). Values above 1.0 mean the full reservoir is replaced more than once per epoch.

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

Validates the turn model by comparing its predictions against fresh CfvSubgameSolver solves using the river network as leaf evaluator:

```bash
cargo run -p cfvnet --release -- compare-net \
  --model models/turn_v1 \
  --river-model models/river_v1 \
  --num-spots 100
```

#### Compare Turn Model Against Exact River Solves

Validates the turn model against CfvSubgameSolver with exact river evaluation (solves all 46 runouts). Slow but provides ground-truth comparison:

```bash
cargo run -p cfvnet --release -- compare-exact \
  --model models/turn_v1 \
  --num-spots 20
```

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
| `datagen.num_samples` | 1,000,000 | Training situations to generate |
| `datagen.solver_iterations` | 1000 | DCFR iterations per situation |
| `game.river_model_path` | none | Path to trained river model (required for turn) |
| `training.hidden_layers` | 7 | MLP depth |
| `training.hidden_size` | 500 | Hidden layer width |
| `training.batch_size` | 2048 | Training batch size |
| `training.epochs` | 2 | Training epochs |

---

## Cloud Training (AWS)

See [`docs/cloud.md`](cloud.md) for running training jobs on AWS EC2 instances via the `solver-cloud` CLI.
