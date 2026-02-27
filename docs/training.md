# Training Reference

All commands are run via the `poker-solver-trainer` crate:

```bash
cargo run -p poker-solver-trainer --release -- <subcommand> [options]
```

Always use `--release` for training and diagnostics.

## Commands

### solve-preflop

Solve preflop strategy using Linear CFR with a pre-built postflop bundle.

**Workflow:** first build a postflop bundle with `solve-postflop`, then reference it via `postflop_model_path` in the preflop config.

```bash
# 1. Build the postflop bundle
cargo run -p poker-solver-trainer --release -- solve-postflop \
  -c sample_configurations/minimal_postflop.yaml -o postflop_models/medium

# 2. Solve preflop (config must include postflop_model_path)
cargo run -p poker-solver-trainer --release -- solve-preflop \
  -c sample_configurations/minimal_preflop.yaml -o preflop_hu_25bb

# CLI overrides
cargo run -p poker-solver-trainer --release -- solve-preflop \
  -c config.yaml -o output/ --iterations 5000 --equity-samples 10000
```

Options:
- `-c, --config <FILE>` -- YAML config file (required; must include `postflop_model_path`)
- `-o, --output <DIR>` -- Output directory for preflop bundle
- `--iterations <N>` -- LCFR iterations (overrides config)
- `--equity-samples <N>` -- Monte Carlo samples per hand matchup for equity table (0=uniform; overrides config)
- `--print-every <N>` -- Print strategy matrices every N iterations (0=only at end; overrides config)

```
output/
├── config.yaml       # PreflopConfig
├── strategy.bin      # PreflopStrategy
```

The Explorer loads postflop data from the referenced bundle automatically and displays per-hand average postflop equity when a cell is selected.

### solve-postflop

Build a postflop abstraction and save it as a reusable bundle. The resulting bundle is referenced by `postflop_model_path` in preflop training configs.

```bash
# Build a postflop bundle from a postflop config
cargo run -p poker-solver-trainer --release -- solve-postflop \
  -c sample_configurations/minimal_postflop.yaml -o postflop_models/medium

# Then reference it in preflop training:
# postflop_model_path: postflop_models/medium
```

Options:
- `-c, --config <FILE>` -- YAML config file with a top-level `postflop_model` section (see `minimal_postflop.yaml`)
- `-o, --output <DIR>` -- Output directory for the postflop bundle

The bundle directory contains:
- `config.yaml` -- Human-readable `PostflopModelConfig`
- `spr_X.Y/solve.bin` -- Bincode-serialized values, flops, and SPR (one per configured SPR)

### train

Run MCCFR/sequence training for the postflop HUNL game.

```bash
cargo run -p poker-solver-trainer --release -- train -c config.yaml
cargo run -p poker-solver-trainer --release -- train -c config.yaml --solver sequence
cargo run -p poker-solver-trainer --release -- train -c config.yaml -t 4  # limit threads
```

Options:
- `-c, --config <FILE>` -- Training config YAML
- `-t, --threads <N>` -- Thread count (default: all cores)
- `--solver <MODE>` -- Backend: `mccfr` (default), `sequence`

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
- `-c, --config <FILE>` -- YAML config with a `postflop_model` section (same format as solve-postflop)
- `--cache-dir <DIR>` -- Abstraction cache directory (default: `cache/postflop`)
- `--json` -- Output as JSON

### trace-hand

Trace all 169 hands through the full postflop pipeline. Config uses the same format as `solve-postflop` (top-level `postflop_model` section):

```bash
cargo run -p poker-solver-trainer --release -- trace-hand -c sample_configurations/minimal_postflop.yaml
```

### generate-deals

Pre-generate exhaustive abstract deals for the sequence solver:

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

---

## Preflop Config (solve-preflop)

The preflop solver uses a YAML config with game structure, DCFR parameters, and a reference to a pre-built postflop bundle. See `sample_configurations/minimal_preflop.yaml` for a minimal example.

```yaml
# Path to a pre-built postflop bundle (required — build with solve-postflop)
postflop_model_path: postflop_models/medium

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
```

### Convergence Metrics

Both metrics are printed every `print_every` iterations:

- **Strategy delta**: mean L1 distance between consecutive strategies -- how much the strategy changed since the last checkpoint.
- **Exploitability** (`preflop_exploitability_threshold_mbb`): gold-standard metric. Computes the best-response value for both players -- how much an optimal opponent could exploit the current strategy. Reported in mBB/hand. At Nash equilibrium, exploitability is 0. When it drops below the threshold, training stops early.

Set `checkpoint_every` to save intermediate strategy bundles at regular intervals (e.g. every 5000 iterations). Each checkpoint is saved to `{output}/checkpoint_{iteration}/` in the same format as the final bundle.

---

## Postflop Config (solve-postflop)

The `solve-postflop` command reads a standalone YAML config with a top-level `postflop_model` section. See `sample_configurations/minimal_postflop.yaml` for a minimal example.

```yaml
postflop_model:
  solve_type: exhaustive
  postflop_sprs: [2, 6, 20]
  fixed_flops: ['AhKsQd', '2c3h4d']
  bet_sizes: [0.5, 1.0]
  max_raises_per_street: 2
  postflop_solve_iterations: 500
  value_extraction_samples: 5
  cfr_convergence_threshold: 200
  ev_convergence_threshold: 0.5
```

The `postflop_sprs` field accepts a scalar or list of SPR values. Each SPR builds an independent postflop tree and value table. At runtime, the preflop solver selects the closest SPR model for each showdown terminal. N SPRs = ~Nx postflop build time. Scalar values are auto-wrapped for backward compatibility.

Multi-SPR bundles store one subdirectory per SPR (e.g. `spr_2.0/`, `spr_6.0/`, `spr_20.0/`). Legacy single-SPR bundles (flat `solve.bin`) are loaded with backward compatibility.

### Postflop Model Parameters

| Parameter | Default | Type | Description |
|-|-|-|-|
| `solve_type` | `mccfr` | string | Postflop backend: `mccfr` (sampled concrete hands) or `exhaustive` (pre-computed equity tables + vanilla CFR) |
| `bet_sizes` | [0.5, 1.0] | [f32] | Pot-fraction bet sizes for postflop tree |
| `max_raises_per_street` | 1 | u8 | Raise cap per postflop street |
| `postflop_sprs` | [3.5] | [f64] | SPR(s) for postflop tree; one model built per SPR; closest selected at runtime |
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
  postflop_sprs: [2, 6, 20]       # one model per SPR; closest selected at runtime
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
  postflop_sprs: [3.5]
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

**CFR variant selection** (sequence solver):
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

**CFR variant selection** (sequence solver):
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
| `sample_configurations/minimal_postflop.yaml` | Minimal postflop-only config (2 flops, fast smoke test) |
| `sample_configurations/minimal_preflop.yaml` | Minimal preflop config referencing a postflop bundle |
| `sample_configurations/preflop_medium.yaml` | HU 25BB preflop solve with standard MCCFR postflop model |
| `sample_configurations/fast_buckets.yaml` | Quick MCCFR postflop test |
| `sample_configurations/mccfr_smoke.yaml` | MCCFR smoke test (single flop, fast) |
| `sample_configurations/smoke.yaml` | Minimal smoke test (single flop, 10 iterations) |
| `sample_configurations/ultra_fast.yaml` | Fast MCCFR with extended raise sizes |
| `sample_configurations/AKQr_vs_234r.yaml` | Multi-flop texture comparison (AKQr vs 234r) |
| `sample_configurations/full.yaml` | Full multi-SPR exhaustive postflop (SPR 2/6/20) with DCFR |

---

## Cloud Training (AWS)

See [`docs/cloud.md`](cloud.md) for running training jobs on AWS EC2 instances via the `solver-cloud` CLI.
