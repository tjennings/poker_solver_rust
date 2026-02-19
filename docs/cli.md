# CLI Reference

All commands are run via the `poker-solver-trainer` crate:

```bash
cargo run -p poker-solver-trainer --release -- <subcommand> [options]
```

## train

Run MCCFR/sequence/GPU training. See [Training Reference](training.md) for config options.

```bash
cargo run -p poker-solver-trainer --release -- train -c config.yaml
cargo run -p poker-solver-trainer --release -- train -c config.yaml --solver sequence
cargo run -p poker-solver-trainer --features gpu --release -- train -c config.yaml --solver gpu
cargo run -p poker-solver-trainer --release -- train -c config.yaml -t 4  # limit threads
```

## tree

Inspect trained strategy bundles. Two modes: **deal tree** and **key describe**.

### Deal tree

Walk a random deal showing strategy probabilities and path-weighted EV:

```bash
cargo run -p poker-solver-trainer --release -- tree -b ./my_strategy
cargo run -p poker-solver-trainer --release -- tree -b ./my_strategy --hand AKs
```

Options:
- `-b, --bundle <DIR>` — Strategy bundle directory
- `-d, --depth <N>` — Max tree depth (default: 4)
- `-m, --min-prob <P>` — Prune branches below this probability (default: 0.01)
- `-s, --seed <N>` — RNG seed for deal selection (default: 42)
- `--hand <HAND>` — Filter to a canonical hand (e.g. "AKs", "QQ", "T9o")

Output:

```
Preflop (P1 to act) — AcKh
├── Fold (2%)
├── Call (35%)
├── Bet 33% (48%) ── ...
└── Bet All-In (5%) ── ...

─── EV Summary (P1 perspective) ──────────────
  Path                                            Reach%     EV (BB)
  ──────────────────────────────────────────────────────────────────
  P1:Fold                                           2.0%      -0.50
  P1:Call → P2:Check → ...                         18.2%      +1.23
```

### Key describe

Translate an info set key to human-readable format:

```bash
cargo run -p poker-solver-trainer --release -- tree -b ./my_strategy --key 0xa800000c02000000
```

Output:

```
Key:     0xA800000C02000000
Compose: river:TopSet:spr0:d1:check,bet33

  Street:  River
  Hand:    TopSet (bits: 0x00000015)
  SPR:     0    Depth: 1
  Actions: [Check, Bet 33%]

  Strategy: Check 0.15 | Bet 33% 0.42 | Bet 67% 0.28 | Bet All-In 0.15
```

Training checkpoints print highest/lowest regret keys with a ready-to-copy `tree --key` command.

## tree-stats

Inspect materialized tree size for a config without training:

```bash
cargo run -p poker-solver-trainer --release -- tree-stats -c config.yaml --sample-deals 100
```

Prints node counts, info set estimates, and memory usage for planning larger runs.

## generate-deals

Pre-generate exhaustive abstract deals for `sequence` or `gpu` solvers:

```bash
# Estimate size without generating
cargo run -p poker-solver-trainer --release -- generate-deals -c config.yaml -o ./my_deals/ --dry-run

# Generate and save
cargo run -p poker-solver-trainer --release -- generate-deals -c config.yaml -o ./my_deals/
```

Options:
- `-c, --config <FILE>` — Training config (must use `hand_class_v2`)
- `-o, --output <DIR>` — Output directory
- `--dry-run` — Estimate deal count and memory without generating
- `-t, --threads <N>` — Thread count (default: all cores)

Output:
```
my_deals/
├── abstract_deals.bin   # Bincode-serialized deal data
└── manifest.yaml        # Config snapshot and statistics
```

Reference in training config with `abstract_deals_dir: ./my_deals/`.

## inspect-deals

Load pre-generated deal files and display summary statistics:

```bash
cargo run -p poker-solver-trainer --release -- inspect-deals -i ./my_deals/
cargo run -p poker-solver-trainer --release -- inspect-deals -i ./my_deals/ --limit 20 --sort equity
cargo run -p poker-solver-trainer --release -- inspect-deals -i ./my_deals/ --limit 0        # summary only
cargo run -p poker-solver-trainer --release -- inspect-deals -i ./my_deals/ --csv deals.csv  # export
```

Options:
- `-i, --input <DIR>` — Directory containing `abstract_deals.bin` and `manifest.yaml`
- `-l, --limit <N>` — Sample deals to display (default: 10, 0 = summary only)
- `--sort <ORDER>` — Sort by `weight`, `equity`, or `none` (default: weight)
- `--csv <FILE>` — Export all deals to CSV

Output includes manifest, equity/weight distributions, hand class histogram, and sample deal trajectories.

## flops

Generate all 1,755 strategically distinct (suit-isomorphic) flops with metadata:

```bash
cargo run -p poker-solver-trainer --release -- flops --format json --output datasets/flops.json
cargo run -p poker-solver-trainer --release -- flops --format csv --output datasets/flops.csv
```

Or use the convenience script:

```bash
./scripts/generate_flops.sh
```

Omit `--output` to print to stdout.

## trace-lhe

Trace strategy evolution across SD-CFR checkpoints for specific spots. Useful for verifying training convergence.

```bash
cargo run -p poker-solver-trainer --release -- trace-lhe -d ./lhe_sdcfr \
  --spot "SB AA" --spot "SB 72o" --spot "BB.R AKs"

cargo run -p poker-solver-trainer --release -- trace-lhe -d ./lhe_sdcfr \
  --spot "SB AA" --every 5 --board-samples 200

cargo run -p poker-solver-trainer --release -- trace-lhe -d ./lhe_sdcfr \
  --spots-file spots.yaml
```

### Spot notation

| Notation | Meaning |
|-|-|
| `SB AA` | SB at preflop root with AA |
| `SB AKs` | SB at preflop root with AKs |
| `BB.R AKs` | BB facing SB raise with AKs |
| `BB.C JTs` | BB after SB limp with JTs |

### Spots file format

```yaml
- player: SB
  hand: AA
- player: BB.R
  hand: AKs
```

### Output

```
=== SB AA (Fold/Call/Raise) ===
 Iter |   Fold |   Call |  Raise | Nets | Raw Adv (latest net)
------|--------|--------|--------|------|---------------------
    1 |  0.330 |  0.340 |  0.330 |    1 | [ -0.01,   0.02,   0.01]
    5 |  0.250 |  0.300 |  0.450 |    5 | [ -0.34,  -0.12,   0.89]
   10 |  0.100 |  0.150 |  0.750 |   10 | [ -1.20,  -0.45,   2.10]
   37 |  0.020 |  0.080 |  0.900 |   37 | [ -2.50,  -0.90,   3.80]
```

If strategies remain near-uniform across many checkpoints and raw advantages stay near zero, the network is not learning.

Options:
- `-d, --dir <DIR>` — Training output directory with `lhe_checkpoint_N` subdirs
- `--spot <SPOT>` — Spot to trace (repeatable)
- `--spots-file <FILE>` — YAML file with spot definitions
- `--every <N>` — Sample every Nth checkpoint (default: 1)
- `--board-samples <N>` — Board samples per hand per checkpoint (default: 50)
- `--num-actions`, `--hidden-dim`, `--seed`, `--stack-depth`, `--num-streets` — Override parameters (auto-detected from `training_config.yaml`)

## Benchmarks

Compare solver backends:

```bash
# Quick benchmark (20 iterations, 1000 deals)
cargo run --release -p poker-solver-gpu-cfr --example bench_solvers

# Longer benchmark (100 iterations, 5000 deals)
cargo run --release -p poker-solver-gpu-cfr --example bench_solvers -- long
```

MCCFR-only benchmark (no GPU dependency):

```bash
cargo run --release -p poker-solver-core --example bench_mccfr
```
