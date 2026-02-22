# Training Reference

All commands are run via the `poker-solver-trainer` crate:

```bash
cargo run -p poker-solver-trainer --release -- <subcommand> [options]
```

Always use `--release` for training and diagnostics.

## Commands

### solve-preflop

Solve preflop strategy using Linear CFR with optional postflop model.

```bash
# From a config file
cargo run -p poker-solver-trainer --release -- solve-preflop \
  -c sample_configurations/preflop_medium.yaml -o preflop_hu_25bb

# With postflop model preset override
cargo run -p poker-solver-trainer --release -- solve-preflop \
  -c config.yaml -o output/ --postflop-model medium

# CLI overrides (supplement or replace config file)
cargo run -p poker-solver-trainer --release -- solve-preflop \
  -o output/ --stack-depth 25 --iterations 5000 --equity-samples 10000
```

Options:
- `-c, --config <FILE>` -- YAML config file (optional; CLI flags override config values)
- `-o, --output <DIR>` -- Output directory for preflop bundle
- `--stack-depth <N>` -- Stack depth in BB
- `--players <N>` -- Number of players (2=HU, 6=six-max)
- `--iterations <N>` -- LCFR iterations
- `--equity-samples <N>` -- Monte Carlo samples per hand matchup for equity table (0=uniform)
- `--postflop-model <PRESET>` -- Postflop model preset: fast, medium, standard, accurate
- `--print-every <N>` -- Print strategy matrices every N iterations (0=only at end)

### train

Run MCCFR/sequence/GPU training for the postflop HUNL game.

```bash
cargo run -p poker-solver-trainer --release -- train -c config.yaml
cargo run -p poker-solver-trainer --release -- train -c config.yaml --solver sequence
cargo run -p poker-solver-trainer --features gpu --release -- train -c config.yaml --solver gpu
cargo run -p poker-solver-trainer --release -- train -c config.yaml -t 4  # limit threads
```

Options:
- `-c, --config <FILE>` -- Training config YAML
- `-t, --threads <N>` -- Thread count (default: all cores)
- `--solver <MODE>` -- Backend: `mccfr` (default), `sequence`, `gpu`, `sd-cfr`

### tree

Inspect trained strategy bundles. Two modes: deal tree and key describe.

**Deal tree** -- walk a random deal showing strategy probabilities:

```bash
cargo run -p poker-solver-trainer --release -- tree -b ./my_strategy
cargo run -p poker-solver-trainer --release -- tree -b ./my_strategy --hand AKs
```

**Key describe** -- translate an info set key to human-readable format:

```bash
cargo run -p poker-solver-trainer --release -- tree -b ./my_strategy --key 0xa800000c02000000
```

Options:
- `-b, --bundle <DIR>` -- Strategy bundle directory
- `-d, --depth <N>` -- Max tree depth (default: 4)
- `-m, --min-prob <P>` -- Prune branches below this probability (default: 0.01)
- `-s, --seed <N>` -- RNG seed (default: 42)
- `--hand <HAND>` -- Filter to canonical hand (e.g. "AKs", "QQ", "T9o")
- `--key <HEX>` -- Info set key for key describe mode

### tree-stats

Estimate game tree size without training:

```bash
cargo run -p poker-solver-trainer --release -- tree-stats -c config.yaml --sample-deals 100
```

### diag-buckets

Run EHS bucket diagnostics on a postflop abstraction:

```bash
cargo run -p poker-solver-trainer --release -- diag-buckets -c config.yaml
cargo run -p poker-solver-trainer --release -- diag-buckets -c config.yaml --json
```

Options:
- `-c, --config <FILE>` -- YAML config (same format as solve-preflop)
- `--cache-dir <DIR>` -- Abstraction cache directory (default: `cache/postflop`)
- `--json` -- Output as JSON

### trace-hand

Trace all 169 hands through the full postflop pipeline (EHS -> buckets -> EV):

```bash
cargo run -p poker-solver-trainer --release -- trace-hand -c config.yaml
```

### generate-deals

Pre-generate exhaustive abstract deals for sequence/GPU solvers:

```bash
# Estimate size
cargo run -p poker-solver-trainer --release -- generate-deals -c config.yaml -o ./deals/ --dry-run

# Generate
cargo run -p poker-solver-trainer --release -- generate-deals -c config.yaml -o ./deals/
```

Options:
- `-o, --output <DIR>` -- Output directory
- `--dry-run` -- Estimate only
- `-t, --threads <N>` -- Thread count
- `--batch-size <N>` -- Canonical flops per batch (default: 20, 0=in-memory)

### merge-deals

Merge batch files from a batched generate-deals run:

```bash
cargo run -p poker-solver-trainer --release -- merge-deals -i ./batches/ -o ./deals/
```

### inspect-deals

Display summary statistics for pre-generated deals:

```bash
cargo run -p poker-solver-trainer --release -- inspect-deals -i ./deals/
cargo run -p poker-solver-trainer --release -- inspect-deals -i ./deals/ --limit 20 --sort equity
cargo run -p poker-solver-trainer --release -- inspect-deals -i ./deals/ --csv deals.csv
```

### flops

List all 1,755 canonical (suit-isomorphic) flops:

```bash
cargo run -p poker-solver-trainer --release -- flops --format json --output flops.json
cargo run -p poker-solver-trainer --release -- flops --format csv
```

### eval-lhe

Evaluate a trained LHE SD-CFR model (exploitability + strategy visualization):

```bash
cargo run -p poker-solver-trainer --release -- eval-lhe -d ./lhe_sdcfr
cargo run -p poker-solver-trainer --release -- eval-lhe -d ./lhe_sdcfr --checkpoint lhe_checkpoint_100
```

### trace-lhe

Trace strategy evolution across SD-CFR checkpoints:

```bash
cargo run -p poker-solver-trainer --release -- trace-lhe -d ./lhe_sdcfr \
  --spot "SB AA" --spot "BB.R AKs"
```

Spot notation: `SB AA` (SB root), `BB.R AKs` (BB facing raise), `BB.C JTs` (BB after limp).

---

## Preflop Config (solve-preflop)

The preflop solver uses a YAML config with game structure and DCFR parameters. See `sample_configurations/preflop_medium.yaml` for a complete example.

```yaml
# Training
iterations: 10000
equity_samples: 10000
print_every: 1000

# Game structure (internal units: SB=1, BB=2)
positions:
  - { name: Small Blind, short_name: SB }
  - { name: Big Blind, short_name: BB }
blinds:
  - [0, 1]   # SB posts 1
  - [1, 2]   # BB posts 2
stacks: [50, 50]  # 25BB x 2

raise_sizes:
  - [2.5]    # open raise
  - [3.0]    # 3-bet
raise_cap: 4

# DCFR discounting
dcfr_alpha: 1.5
dcfr_beta: 0.5
dcfr_gamma: 2.0
exploration: 0.05

# Postflop model (optional; omit for raw equity)
postflop_model:
  num_hand_buckets_flop: 200
  num_hand_buckets_turn: 200
  num_hand_buckets_river: 200
  bet_sizes: [0.5, 1.0, 2.0]
  raises_per_street: 1
  canonical_sprs: [0.5, 1.0, 1.5, 3.0, 5.0, 10.0, 20.0, 50.0]
  postflop_solve_iterations: 1000
  postflop_solve_samples: 100000
```

### Postflop Model Presets

| Preset | Buckets (flop/turn/river) | Use case |
|-|-|-|
| `fast` | 50/50/50 | Quick testing (~30s build) |
| `medium` | 200/200/200 | Development iteration |
| `standard` | 500/500/500 | Production training (Pluribus-like) |
| `accurate` | 1000/1000/1000 | High-fidelity analysis |

---

## HUNL Training Config (train)

Training configs have `game` and `training` sections.

### Game Settings

```yaml
game:
  stack_depth: 100
  bet_sizes: [0.33, 0.67, 1.0, 2.0, 3.0]
  max_raises_per_street: 3
```

### Training Settings

```yaml
training:
  iterations: 5000
  seed: 42
  output_dir: "./my_strategy"
  mccfr_samples: 5000
  deal_count: 50000
  abstraction_mode: hand_class_v2
  strength_bits: 4
  equity_bits: 4
```

### Abstraction Modes

**`ehs2`** -- EHS2 bucketing with Monte Carlo equity estimation:

```yaml
training:
  abstraction_mode: ehs2
abstraction:
  flop_buckets: 200
  turn_buckets: 200
  river_buckets: 500
  samples_per_street: 5000
```

**`hand_class`** -- Categorical 19-class hand classification (O(1), no extra config).

**`hand_class_v2`** -- Hand class + intra-class strength + equity + draw flags:

```yaml
training:
  abstraction_mode: hand_class_v2
  strength_bits: 4   # 0-4 bits (16 levels at 4)
  equity_bits: 4     # 0-4 bits
```

### Solver Backends

| Solver | Best for | Deal handling |
|-|-|-|
| `mccfr` | Large games, production training | Samples per iteration |
| `sequence` | Small-medium games, exact convergence | Full traversal |
| `gpu` | Same as sequence, faster on GPU | Full traversal on GPU |
| `sd-cfr` | Neural network advantage estimation | SD-CFR checkpoints |

### Advanced Options

**Stratified deals** (hand_class modes):
```yaml
training:
  min_deals_per_class: 50
  max_rejections_per_class: 500000
```

**Regret-based pruning:**
```yaml
training:
  pruning: true
  pruning_threshold: -5.0
  pruning_warmup_fraction: 0.30
  pruning_probe_interval: 20
```

**Convergence-based stopping:**
```yaml
training:
  convergence_threshold: 0.001
  convergence_check_interval: 100
```

**Exhaustive abstract deals** (low-bit hand_class_v2 only):
```yaml
training:
  exhaustive: true
  # or pre-generated:
  abstract_deals_dir: ./my_deals/
```

### Training Output

```
my_strategy/
├── config.yaml       # Game and abstraction settings
├── blueprint.bin     # Trained strategy (bincode, FxHashMap<u64, Vec<f32>>)
└── boundaries.bin    # EHS2 bucket boundaries (ehs2 mode only)
```

---

## Sample Configs

| Config | Purpose |
|-|-|
| `sample_configurations/preflop_medium.yaml` | HU 25BB preflop solve with medium postflop model |
| `sample_configurations/fast_buckets.yaml` | Quick postflop bucketing test |
