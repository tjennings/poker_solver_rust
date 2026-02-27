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
  -c config.yaml -o output/ --postflop-model standard

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
- `--postflop-model <PRESET>` -- Postflop model preset: fast, standard, exhaustive_fast, exhaustive_standard
- `--print-every <N>` -- Print strategy matrices every N iterations (0=only at end)

When a postflop model is used, the postflop solve data is automatically saved into a `postflop/` subdirectory of the output:

```
output/
├── config.yaml       # PreflopConfig
├── strategy.bin      # PreflopStrategy
└── postflop/         # PostflopBundle (auto-saved when postflop model is used)
    ├── config.yaml   # PostflopModelConfig
    └── solve.bin     # Values, hand-averaged EVs, flops, SPR
```

The Explorer loads this postflop data automatically and displays per-hand average postflop equity when a cell is selected. Old bundles without a `postflop/` subdirectory continue to work -- the equity panel simply doesn't appear.

### solve-postflop

Build a postflop abstraction and save it as a reusable bundle. Use `postflop_model_path` in preflop training configs to load a pre-built bundle instead of rebuilding each time.

```bash
# Build a postflop bundle from a training config
cargo run -p poker-solver-trainer --release -- solve-postflop \
  -c sample_configurations/preflop_medium.yaml -o postflop_models/medium

# Then reference it in preflop training:
# postflop_model_path: postflop_models/medium
```

Options:
- `-c, --config <FILE>` -- YAML config file (reads the `postflop_model` section)
- `-o, --output <DIR>` -- Output directory for the postflop bundle

The bundle directory contains:
- `config.yaml` -- Human-readable `PostflopModelConfig`
- `solve.bin` -- Bincode-serialized values, flops, and SPR

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

Run diagnostics on a postflop abstraction:

```bash
cargo run -p poker-solver-trainer --release -- diag-buckets -c config.yaml
cargo run -p poker-solver-trainer --release -- diag-buckets -c config.yaml --json
```

Options:
- `-c, --config <FILE>` -- YAML config (same format as solve-preflop)
- `--cache-dir <DIR>` -- Abstraction cache directory (default: `cache/postflop`)
- `--json` -- Output as JSON

### trace-hand

Trace all 169 hands through the full postflop pipeline:

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
iterations: 10000                   # number of LCFR iterations to run
equity_samples: 10000               # Monte Carlo samples per hand matchup for equity table (0=uniform)
print_every: 1000                   # print strategy matrices every N iterations (0=only at end)
preflop_exploitability_threshold_mbb: 25  # early-stop when exploitability drops below this (mBB/hand)
# checkpoint_every: 5000           # save intermediate bundles every N iterations

# Game structure (internal units: SB=1, BB=2)
positions:                          # ordered list of seats; index determines blind/stack mapping
  - { name: Small Blind, short_name: SB }   # position 0
  - { name: Big Blind, short_name: BB }      # position 1
blinds:                             # [position_index, amount] pairs
  - [0, 1]                         # position 0 (SB) posts 1 unit
  - [1, 2]                         # position 1 (BB) posts 2 units
stacks: [50, 50]                    # per-position starting stacks in SB units; index matches positions above
                                    # here: both SB and BB start with 50 units = 25 BB each

raise_sizes:                        # raise sizes indexed by depth ("Xbb" = to X BB, "Xp" = by X × pot)
  - ["2.5bb"]                      # depth 0 (open raise): raise to 2.5 BB
  - ["3.0bb"]                      # depth 1 (3-bet): raise to 3 BB
raise_cap: 4                        # max number of raises allowed per round

# DCFR discounting (ignored when cfr_variant is vanilla or cfrplus)
dcfr_alpha: 1.5                     # positive regret discount exponent
dcfr_beta: 0.5                      # negative regret discount exponent
dcfr_gamma: 2.0                     # strategy sum discount exponent
exploration: 0.05                   # epsilon-greedy exploration rate

# Postflop model: either inline config or pre-built bundle path
# Option A: reference a pre-built bundle (skips postflop build)
# postflop_model_path: postflop_models/medium
# Option B: build inline (existing behavior)
postflop_model:
  solve_type: mccfr                 # backend: "mccfr" (sampled) or "exhaustive" (full traversal)

  max_flop_boards: 0                # canonical flops to solve; 0 = all ~1,755; lower = faster
  # fixed_flops: ['AhKsQd']        # explicit flop boards (overrides max_flop_boards)

  bet_sizes: [0.5, 1.0, 2.0]       # pot-fraction bet sizes for the postflop tree
  max_raises_per_street: 1          # raise cap per postflop street
  postflop_sprs: [3.5]             # stack-to-pot ratio(s) for the postflop tree; scalar also accepted

  postflop_solve_iterations: 1000   # CFR/MCCFR iterations per flop
  cfr_convergence_threshold: 0.01   # per-flop early-stop threshold (strategy delta or exploitability)

  # MCCFR-specific (only when solve_type: mccfr)
  mccfr_sample_pct: 0.01           # fraction of deal space sampled per iteration
  value_extraction_samples: 10000   # Monte Carlo samples for post-convergence EV extraction
  ev_convergence_threshold: 0.001   # weighted-avg delta threshold for early-stop EV estimation
```

### Postflop Model Presets

| Preset | Backend | Max boards | Sample % | Iterations | Use case |
|-|-|-|-|-|-|
| `fast` | mccfr | 10 | 5% | 100 | Quick testing (~1 min) |
| `standard` | mccfr | all | 1% | 500 | Balanced accuracy and speed |
| `exhaustive_fast` | exhaustive | 10 | -- | 200 | Quick exhaustive CFR testing |
| `exhaustive_standard` | exhaustive | all | -- | 1000 | Full exhaustive CFR |

### Convergence Metrics

Both metrics are printed every `print_every` iterations:

- **Strategy delta**: mean L1 distance between consecutive strategies -- how much the strategy changed since the last checkpoint.
- **Exploitability** (`preflop_exploitability_threshold_mbb`): gold-standard metric. Computes the best-response value for both players -- how much an optimal opponent could exploit the current strategy. Reported in mBB/hand. At Nash equilibrium, exploitability is 0. When it drops below the threshold, training stops early.

Set `checkpoint_every` to save intermediate strategy bundles at regular intervals (e.g. every 5000 iterations). Each checkpoint is saved to `{output}/checkpoint_{iteration}/` in the same format as the final bundle.

The `max_flop_boards` parameter controls how many canonical flop textures are solved. Lower values dramatically speed up training. Set to `0` (or omit) to use all ~1,755 canonical flops.

The `postflop_sprs` field accepts a scalar or list of SPR values for the shared postflop tree (replaces `postflop_spr`; scalar values are auto-wrapped for backward compatibility).

### Postflop Model Parameters

| Parameter | Default | Type | Description |
|-|-|-|-|
| `solve_type` | `mccfr` | string | Postflop backend: `mccfr` (sampled concrete hands) or `exhaustive` (pre-computed equity tables + vanilla CFR) |
| `bet_sizes` | [0.5, 1.0] | [f32] | Pot-fraction bet sizes for postflop tree |
| `max_raises_per_street` | 1 | u8 | Raise cap per postflop street |
| `postflop_sprs` | [3.5] | [f64] | SPR(s) for shared postflop tree (scalar or list) |
| `postflop_solve_iterations` | 500 | u32 | CFR/MCCFR iterations per flop |
| `max_flop_boards` | 0 | usize | Max canonical flops to solve; 0 = all ~1,755 |
| `fixed_flops` | none | [string] | Explicit flop boards (overrides `max_flop_boards`) |
| `cfr_convergence_threshold` | 0.01 | f64 | Early per-flop CFR stopping threshold (mccfr: strategy delta; exhaustive: exploitability in pot fractions) |
| `mccfr_sample_pct` | 0.01 | f64 | Fraction of deal space per MCCFR iteration (MCCFR only) |
| `value_extraction_samples` | 10,000 | u32 | Monte Carlo samples for post-convergence EV extraction (MCCFR only) |
| `ev_convergence_threshold` | 0.001 | f64 | Weighted-avg delta threshold for early-stop EV estimation (MCCFR only) |

### MCCFR Backend

The MCCFR backend (`solve_type: mccfr`, default) uses direct 169-hand indexing per flop. For each canonical flop, it builds a combo map expanding canonical hands to concrete card pairs, samples random deals (hero_hand, opp_hand, turn, river) per iteration, and evaluates showdowns using `rank_hand()` on the actual 5-card board.

```yaml
postflop_model:
  solve_type: mccfr
  mccfr_sample_pct: 0.01          # 1% of deal space per iteration
  value_extraction_samples: 10000  # Monte Carlo samples for EV extraction
  ev_convergence_threshold: 0.001 # early-stop when weighted-avg delta drops below this
  postflop_solve_iterations: 500
  postflop_spr: 3.5
  bet_sizes: [0.5, 1.0]
```

Key characteristics:
- **Direct 169-hand indexing**: each canonical hand maps to concrete card pairs per flop via a combo map
- **No bucket abstraction**: hands are not clustered -- each of 169 canonical hands is a separate info set
- **Real showdown eval**: uses 7-card hand ranking instead of equity tables
- **`mccfr_sample_pct`**: controls sampling density per iteration (higher = slower but more accurate)
- **`value_extraction_samples`**: Monte Carlo samples for post-convergence EV extraction
- **`ev_convergence_threshold`**: weighted-average delta threshold for early-stopping EV estimation (default 0.001)

### Exhaustive Backend

The exhaustive backend (`solve_type: exhaustive`) uses pre-computed pairwise equity tables and vanilla CFR over all 169x169 hand pairs per flop. No sampling -- full traversal every iteration.

```yaml
postflop_model:
  solve_type: exhaustive
  postflop_solve_iterations: 1000
  postflop_spr: 3.5
  bet_sizes: [0.5, 1.0]
```

Key characteristics:
- **Pre-computed equity**: pairwise equity tables for all hand pairs per street
- **Vanilla CFR**: deterministic full traversal, no sampling variance
- **Slower per iteration** but converges with fewer iterations than MCCFR

Presets: `exhaustive_fast` (10 boards, 200 iterations) and `exhaustive_standard` (all boards, 1000 iterations).

---

## HUNL Training Config (train)

Training configs have `game` and `training` sections.

### Game Settings

```yaml
game:
  stack_depth: 100                  # effective stack in BB (stacks = stack_depth × 2 SB units)
  bet_sizes: [0.33, 0.67, 1.0, 2.0, 3.0]  # pot-fraction bet sizes available postflop
  max_raises_per_street: 3          # raise cap per postflop street
```

### Training Settings

```yaml
training:
  iterations: 5000                  # number of CFR iterations to run
  seed: 42                          # RNG seed for reproducibility
  output_dir: "./my_strategy"       # where to save the trained strategy bundle
  mccfr_samples: 5000              # deals sampled per MCCFR iteration
  deal_count: 50000                 # total pre-generated deals in the deal pool
  abstraction_mode: hand_class_v2   # info set abstraction: hand_class, hand_class_v2, or ehs2
  strength_bits: 4                  # intra-class strength resolution (0-4 bits; hand_class_v2 only)
  equity_bits: 4                    # equity bucket resolution (0-4 bits; hand_class_v2 only)
```

### Abstraction Modes

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

**Convergence-based stopping (exploitability):**
```yaml
training:
  preflop_exploitability_threshold_mbb: 25  # mBB/hand
  convergence_check_interval: 100
```

**Exhaustive abstract deals** (low-bit hand_class_v2 only):
```yaml
training:
  exhaustive: true
  # or pre-generated:
  abstract_deals_dir: ./my_deals/
```

**CFR variant selection** (sequence/GPU solvers):
```yaml
training:
  cfr_variant: dcfr       # dcfr (default), linear, vanilla, cfrplus
  dcfr_warmup: 100         # iterations before DCFR discounting starts
  seq_exploration: 0.05    # ε-greedy exploration (0.0 = none)
```

| Variant | Regret weighting | Strategy weighting | Discounting |
|-|-|-|-|
| `dcfr` | linear (LCFR) | linear | α/β/γ DCFR |
| `linear` | linear (LCFR) | linear | α=β=γ=1.0 |
| `vanilla` | uniform | uniform | none |
| `cfrplus` | uniform | linear | regrets floored to 0 |

**CFR variant selection** (sequence/GPU solvers):
```yaml
training:
  cfr_variant: dcfr       # dcfr (default), linear, vanilla, cfrplus
  dcfr_warmup: 100         # iterations before DCFR discounting starts
  seq_exploration: 0.05    # ε-greedy exploration (0.0 = none)
```

| Variant | Regret weighting | Strategy weighting | Discounting |
|-|-|-|-|
| `dcfr` | linear (LCFR) | linear | α/β/γ DCFR |
| `linear` | linear (LCFR) | linear | α=β=γ=1.0 |
| `vanilla` | uniform | uniform | none |
| `cfrplus` | uniform | linear | regrets floored to 0 |

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
| `sample_configurations/preflop_medium.yaml` | HU 25BB preflop solve with standard MCCFR postflop model |
| `sample_configurations/fast_buckets.yaml` | Quick MCCFR postflop test |
| `sample_configurations/mccfr_smoke.yaml` | MCCFR smoke test (single flop, fast) |
| `sample_configurations/smoke.yaml` | Minimal smoke test (single flop, 10 iterations) |
| `sample_configurations/ultra_fast.yaml` | Fast MCCFR with extended raise sizes |
| `sample_configurations/AKQr_vs_234r.yaml` | Multi-flop texture comparison (AKQr vs 234r) |

---

## Cloud Training (AWS)

See [`docs/cloud.md`](cloud.md) for running training jobs on AWS EC2 instances via the `solver-cloud` CLI.
