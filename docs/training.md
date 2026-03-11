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

The `cfvnet` crate provides tools for training Deep Counterfactual Value Networks.

### Generate Training Data

```bash
cargo run -p cfvnet --release -- generate \
  --config sample_configurations/river_cfvnet.yaml \
  --output data/river_training.bin \
  --num-samples 1000000 \
  --threads 8
```

### Train the Network

```bash
cargo run -p cfvnet --release -- train \
  --config sample_configurations/river_cfvnet.yaml \
  --data data/river_training.bin \
  --output models/river_v1
```

### Evaluate on Held-Out Data

```bash
cargo run -p cfvnet --release -- evaluate \
  --model models/river_v1 \
  --data data/river_validation.bin
```

### Compare Against Exact Solves

```bash
cargo run -p cfvnet --release -- compare \
  --model models/river_v1 \
  --num-spots 100
```

### Configuration

See `sample_configurations/river_cfvnet.yaml` for all options. Key parameters:

| Parameter | Default | Description |
|-|-|-|
| `datagen.num_samples` | 1,000,000 | Training situations to generate |
| `datagen.solver_iterations` | 1000 | DCFR iterations per situation |
| `training.hidden_layers` | 7 | MLP depth |
| `training.hidden_size` | 500 | Hidden layer width |
| `training.batch_size` | 2048 | Training batch size |
| `training.epochs` | 2 | Training epochs |

---

## Cloud Training (AWS)

See [`docs/cloud.md`](cloud.md) for running training jobs on AWS EC2 instances via the `solver-cloud` CLI.
