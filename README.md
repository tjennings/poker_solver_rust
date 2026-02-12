# Poker Solver

A Rust-based poker solver using Counterfactual Regret Minimization (CFR) for Heads-Up No-Limit Texas Hold'em, with a Tauri desktop UI for strategy exploration.

## Features

- **HUNL Postflop solver** - Full preflop-through-river training with card abstraction
- **MCCFR training** - Monte Carlo CFR with chance sampling for scalable blueprint computation
- **Sequence-form CFR** - Full-traversal CFR on materialized game tree (no sampling variance)
- **GPU-accelerated CFR** - wgpu compute shaders for sequence-form CFR on Metal/Vulkan/DX12
- **Three abstraction modes** - EHS2 bucketing, hand-class, or hand-class v2 (class + strength + equity + draws)
- **DCFR** - Discounted CFR with configurable regret-based pruning
- **Parallel training** - Rayon-based parallel MCCFR with frozen-snapshot accumulation
- **Strategy explorer** - Interactive 13x13 hand matrix for browsing trained strategies
- **Agent simulation** - Pit trained blueprints and rule-based agents against each other
- **Kuhn Poker** - 3-card toy game for algorithm testing and validation

## Prerequisites

- [Rust](https://rustup.rs/) (edition 2024)
- [Node.js](https://nodejs.org/) (v18+)
- [Tauri CLI](https://tauri.app/start/prerequisites/)

## Setup

```bash
git clone <repo-url>
cd poker_solver_rust

# Install frontend dependencies
cd frontend && npm install && cd ..
```

## Training a Strategy

Training uses the `poker-solver-trainer` CLI to run MCCFR iterations and save a strategy bundle.

MCCFR (Monte Carlo CFR) samples random card deals and performs full CFR traversal on the betting tree for each deal. This avoids materializing the entire game tree, making it practical for full HUNL. The implementation uses Discounted CFR (DCFR):
- DCFR discounting with α=1.5, β=0.5, γ=2.0 (positive regrets weighted up, negative regrets decay)
- Configurable regret-based pruning with negative threshold support
- Parallel training via Rayon (frozen-snapshot accumulation pattern)
- Average strategy skips first 50% of iterations

### 1. Create a training config

Create a YAML file. The config has two top-level sections: `game` and `training`.

#### Game settings

```yaml
game:
  stack_depth: 100                        # Effective stack in big blinds
  bet_sizes: [0.33, 0.67, 1.0, 2.0, 3.0] # Pot-fraction bet sizes (all-in always included)
  max_raises_per_street: 3                # Cap on bets/raises per street (default: 3)
```

- `bet_sizes` are pot fractions. `[0.5, 1.0, 1.5]` means half-pot, pot, and 1.5x pot. All-in is always available as an additional action regardless of this list.
- `max_raises_per_street` keeps the game tree tractable. After this many bets on a street, only fold/call/check remain.

#### Training settings

```yaml
training:
  iterations: 5000          # Total MCCFR iterations (or use convergence_threshold)
  seed: 42                  # RNG seed for reproducibility
  output_dir: "./my_strategy"
  mccfr_samples: 5000       # Deals sampled per iteration (default: 500)
  deal_count: 50000          # Pre-generated deal pool size (default: 50000)
```

#### Abstraction mode

Choose one of three abstraction modes via `abstraction_mode`:

**`ehs2`** (default) — EHS2 bucketing with Monte Carlo equity estimation. Fine-grained but expensive to compute. Requires an `abstraction` section:

```yaml
training:
  abstraction_mode: ehs2

abstraction:
  flop_buckets: 200          # EHS2 buckets on flop (production: 5000)
  turn_buckets: 200          # EHS2 buckets on turn (production: 5000)
  river_buckets: 500         # EHS2 buckets on river (production: 20000)
  samples_per_street: 5000   # Monte Carlo samples for bucket boundaries
```

**`hand_class`** — Categorical hand classification (28 classes: NutFlush, TopSet, Overpair, etc.). O(1) per hand, interpretable, no `abstraction` section needed:

```yaml
training:
  abstraction_mode: hand_class
```

**`hand_class_v2`** — Hand class + intra-class strength + showdown equity + draw flags. Finer resolution than `hand_class` while remaining interpretable:

```yaml
training:
  abstraction_mode: hand_class_v2
  strength_bits: 4           # Intra-class strength resolution, 0-4 bits (default: 4)
  equity_bits: 4             # Showdown equity bin resolution, 0-4 bits (default: 4)
```

The `strength_bits` and `equity_bits` control how finely hands within the same class are distinguished. 4 bits = 16 levels, 0 = dimension omitted. Higher values produce more info sets (larger strategy tables) but capture more nuance.

#### Stratified deal generation

For `hand_class` and `hand_class_v2` modes, you can ensure rare hand classes (e.g., quads, straight flushes) appear in the deal pool:

```yaml
training:
  min_deals_per_class: 50       # Minimum deals per hand class (default: 0 = disabled)
  max_rejections_per_class: 500000  # Max rejection-sample attempts per deficit class
```

#### Regret-based pruning

Pruning skips actions with low cumulative regret, speeding up training by avoiding clearly bad lines. Probe iterations periodically explore all actions to prevent permanent pruning.

```yaml
training:
  pruning: true                  # Enable pruning (default: false)
  pruning_threshold: -5.0        # Regret threshold for pruning (default: 0.0)
  pruning_warmup_fraction: 0.30  # Fraction of iterations before pruning starts (default: 0.2)
  pruning_probe_interval: 20     # Full exploration every N iterations (default: 20)
```

- `pruning_threshold` controls how negative a regret must be before the action is pruned. With DCFR, negative regrets decay asymptotically but never reach zero — a threshold of 0 would prune actions permanently. A negative threshold (e.g., -5.0) allows DCFR's decay to bring regrets back above the threshold between probes, so actions can recover if the strategy shifts.
- `pruning_warmup_fraction` delays pruning until the strategy has partially converged. At 0.30, the first 30% of iterations explore everything.
- `pruning_probe_interval` runs a full un-pruned iteration every N iterations to discover if pruned actions have become viable.

#### Convergence-based stopping

Instead of a fixed iteration count, you can train until the strategy stabilizes. Set `convergence_threshold` to stop when the mean L1 strategy delta between consecutive checkpoints drops below the threshold:

```yaml
training:
  convergence_threshold: 0.001   # Stop when mean L1 delta < this value
  convergence_check_interval: 100  # Check every N iterations (default: 100)
```

When `convergence_threshold` is set, `iterations` is ignored. The trainer runs in a loop: train for `convergence_check_interval` iterations, snapshot the strategy, compute the mean L1 distance from the previous snapshot, and stop if it falls below the threshold. Each check also saves a numbered checkpoint bundle.

The strategy delta is the average per-info-set sum of absolute differences in action probabilities between consecutive snapshots. A value of 0.001 means action probabilities are changing by less than 0.1% on average per info set.

If both `convergence_threshold` and `iterations` are set, a warning is printed and `convergence_threshold` takes precedence. At least one of the two must be specified.

#### Exhaustive abstract deals

For `hand_class_v2` mode, you can enumerate **all** abstract deal trajectories instead of sampling randomly. This eliminates Monte Carlo variance entirely — every iteration is a complete traversal of the finite abstract game.

The enumerator walks all hole card pairs × 1,755 canonical flops × all turn/river completions, encodes per-street hand bits, determines showdown winners, and deduplicates into weighted abstract deals. With `strength_bits=0, equity_bits=0`, billions of concrete deals compress to ~1M abstract deals (~100x compression).

```yaml
training:
  abstraction_mode: hand_class_v2
  strength_bits: 0
  equity_bits: 0
  exhaustive: true            # generate abstract deals in-memory
```

Or pre-generate deals to disk and reference them:

```yaml
training:
  abstract_deals_dir: ./my_deals/   # load pre-generated deals
```

Pre-generate with the `generate-deals` command (see below). Use `exhaustive` with the `sequence` or `gpu` solver — MCCFR ignores it.

Higher bit configs produce more unique trajectories and less compression. At 4/4 bits there is essentially no compression, so `exhaustive` is only practical for low-bit configs (0/0 or 1/1).

#### Complete example

```yaml
game:
  stack_depth: 25
  bet_sizes: [0.5, 1.0, 1.5]

training:
  iterations: 5000                    # or use convergence_threshold instead
  seed: 42
  output_dir: "./handclass_25bb_v2"
  mccfr_samples: 5000
  deal_count: 50000
  abstraction_mode: hand_class_v2
  strength_bits: 4
  equity_bits: 4
  min_deals_per_class: 50
  max_rejections_per_class: 500000
  pruning: true
  pruning_threshold: -5.0
  pruning_warmup_fraction: 0.30
  # convergence_threshold: 0.001     # uncomment to train until converged
  # convergence_check_interval: 100
  # exhaustive: true                 # uncomment for exhaustive abstract deals (low-bit configs)
```

### 2. Run training

The trainer supports three solver backends, selected with the `--solver` flag:

#### MCCFR (default)

Samples random deals per iteration. Best for large games with many info sets. Low memory, handles any abstraction mode.

```bash
cargo run -p poker-solver-trainer --release -- train -c config.yaml
```

Use `-t` to control thread count (defaults to all cores):

```bash
cargo run -p poker-solver-trainer --release -- train -c config.yaml -t 4
```

#### Sequence-form CFR (full traversal, CPU)

Materializes the game tree as a flat graph and runs level-by-level CFR over all deals every iteration. No sampling variance — each iteration is a complete traversal. Best for `hand_class` mode where the tree is small enough to materialize (~200-300K info sets).

```bash
cargo run -p poker-solver-trainer --release -- train -c config.yaml --solver sequence
```

**Trade-offs vs MCCFR:**
- Each iteration is more expensive (traverses all deals, not a sample)
- But each iteration makes more progress (no sampling noise)
- No `mccfr_samples` parameter needed — all deals are processed every iteration
- Memory scales with tree size × deal count

#### GPU-accelerated CFR (wgpu)

Same algorithm as sequence-form, but the inner loop (regret matching, reach propagation, utility computation) runs on GPU compute shaders via [wgpu](https://wgpu.rs/). Cross-platform: Metal (macOS), Vulkan (Linux/Windows), DX12 (Windows).

Requires building with the `gpu` feature flag:

```bash
cargo run -p poker-solver-trainer --features gpu --release -- train -c config.yaml --solver gpu
```

**Performance:** ~7.7x faster than CPU sequence-form on Apple Silicon (M-series) for 25BB hand_class configs. The speedup increases with larger deal counts.

**When to use each solver:**

| Solver | Best for | Deal handling |
|--------|----------|---------------|
| `mccfr` | Large games, `ehs2`/`hand_class_v2`, production training | Samples per iteration |
| `sequence` | Small-medium games, exact convergence, `exhaustive` mode | Full traversal (all deals every iteration) |
| `gpu` | Same as `sequence` but faster, when GPU is available | Full traversal on GPU |

With `exhaustive: true` in the config, `sequence` and `gpu` solvers use weighted abstract deals instead of random concrete deals. This gives exact convergence (no sampling variance) for hand-class abstractions.

Release mode is essential for performance. Training prints progress at 10 checkpoints with exploitability, sample hand strategies, and ETA:

```
=== Checkpoint 3/10 (300/1000 iterations) ===
Exploitability: 2.4531 (↓ was 3.1204)
Time: 12.3s elapsed, ~28.7s remaining

SB Opening Strategy (preflop, facing BB):
Hand  |  Fold  Call   R50  R9950
------|------------------------
AA    |  0.00  0.12  0.72  0.16
AKs   |  0.02  0.35  0.58  0.05
72o   |  0.85  0.10  0.05  0.00
```

### 3. Strategy bundle output

Training saves to the output directory:

```
my_strategy/
├── config.yaml       # Game and abstraction settings (human-readable)
├── blueprint.bin     # Trained strategy (bincode, FxHashMap<u64, Vec<f32>>)
└── boundaries.bin    # EHS2 bucket boundaries (only for ehs2 mode)
```

## Exploring a Strategy

The desktop app lets you browse trained strategies interactively.

### 1. Start the app

```bash
cd frontend && npm install && cd ..
cd crates/tauri-app
cargo tauri dev
```

### 2. Load a strategy bundle

Click **Load Strategy Bundle** and select the output directory from training (e.g. `my_strategy/`). The app displays the bundle metadata: stack depth, bet sizes, info set count, and training iterations.

### 3. Browse the game tree

**Preflop:** The 13x13 hand matrix shows action probabilities for every starting hand. Each cell displays a color-coded bar:
- Blue = fold, Green = call/check, Red = bet/raise, Purple = all-in

Click an action button (fold, call, raise) to advance down the game tree.

**Postflop:** When the game reaches the flop, enter board cards (e.g. `AcTh4d`). The app computes EHS2 buckets for all 169 canonical hands (progress bar shown), then displays the strategy matrix for that board.

Continue navigating through turn and river by entering additional cards.

**History:** The action strip at the top shows the full history. Click any point to rewind.

## Generating Flop Data

Generate a reference dataset of all 1,755 strategically distinct (suit-isomorphic) flops with metadata (suit texture, rank texture, high card class, connectedness, weight):

```bash
./scripts/generate_flops.sh
```

This builds the trainer in release mode and writes both formats to `datasets/`:

```
datasets/
├── flops.json    # Array of objects with full metadata
└── flops.csv     # One row per canonical flop
```

You can also run the underlying command directly for a single format:

```bash
cargo run -p poker-solver-trainer --release -- flops --format json --output datasets/flops.json
cargo run -p poker-solver-trainer --release -- flops --format csv --output datasets/flops.csv
```

Omit `--output` to print to stdout.

## Generating Abstract Deals

Pre-generate exhaustive abstract deals for use with the `sequence` or `gpu` solver. This is useful when the generation step is expensive and you want to reuse the same deal set across multiple training runs.

```bash
# Estimate size without generating
cargo run -p poker-solver-trainer --release -- generate-deals -c config.yaml -o ./my_deals/ --dry-run

# Generate and save to disk
cargo run -p poker-solver-trainer --release -- generate-deals -c config.yaml -o ./my_deals/
```

Options:
- `-c, --config <FILE>` — Training config YAML (must use `hand_class_v2` abstraction)
- `-o, --output <DIR>` — Output directory for deal files
- `--dry-run` — Estimate deal count and memory without generating
- `-t, --threads <N>` — Thread count for parallel generation (default: all cores)

Output:
```
my_deals/
├── abstract_deals.bin   # Bincode-serialized deal data
└── manifest.yaml        # Config snapshot and statistics
```

Then reference the directory in your training config:

```yaml
training:
  abstract_deals_dir: ./my_deals/
```

## Inspecting Abstract Deals

The `inspect-deals` subcommand loads pre-generated abstract deal files and displays summary statistics, sample deals, and optionally exports to CSV. Useful for debugging deal generation and verifying abstraction quality.

```bash
# Summary statistics + 10 sample deals (sorted by weight)
cargo run -p poker-solver-trainer --release -- inspect-deals -i ./my_deals/

# Show 20 deals sorted by equity
cargo run -p poker-solver-trainer --release -- inspect-deals -i ./my_deals/ --limit 20 --sort equity

# Summary only (no sample deals)
cargo run -p poker-solver-trainer --release -- inspect-deals -i ./my_deals/ --limit 0

# Export all deals to CSV
cargo run -p poker-solver-trainer --release -- inspect-deals -i ./my_deals/ --csv deals.csv
```

Options:
- `-i, --input <DIR>` — Directory containing `abstract_deals.bin` and `manifest.yaml`
- `-l, --limit <N>` — Number of sample deals to display (default: 10, 0 = summary only)
- `--sort <ORDER>` — Sort sample deals by `weight`, `equity`, or `none` (default: weight)
- `--csv <FILE>` — Export all deals to a CSV file

Output includes:
- **Manifest** — Config snapshot (stack depth, strength/equity bits, compression ratio)
- **Equity distribution** — min, max, mean, median, stddev
- **Weight distribution** — min, max, mean
- **Hand class histogram** — P1 river hand class frequency table sorted by count
- **Sample deals** — Per-street hand classification trajectories for both players

## Analyzing the Game Tree

The `tree` subcommand inspects trained strategy bundles. It has two modes: **deal tree** (walk a concrete deal showing strategies at each node) and **key describe** (translate info set keys).

### Deal tree

Walk a random deal from a trained bundle, showing strategy probabilities and path-weighted EV:

```bash
cargo run -p poker-solver-trainer --release -- tree -b ./my_strategy
```

Filter to a specific starting hand:

```bash
cargo run -p poker-solver-trainer --release -- tree -b ./my_strategy --hand AKs
```

Options:
- `-d, --depth <N>` — Max tree depth (default: 4)
- `-m, --min-prob <P>` — Prune branches below this probability (default: 0.01)
- `-s, --seed <N>` — RNG seed for deal selection (default: 42)
- `--hand <HAND>` — Filter to a canonical hand (e.g. "AKs", "QQ", "T9o")

Output includes an ASCII tree with action probabilities and an EV summary:

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

Translate an info set key (from training diagnostics) to human-readable format:

```bash
cargo run -p poker-solver-trainer --release -- tree -b ./my_strategy --key 0xa800000c02000000
```

Output shows the decoded fields and the blueprint strategy for that key:

```
Key:     0xA800000C02000000
Compose: river:TopSet:spr0:d1:check,bet33

  Street:  River
  Hand:    TopSet (bits: 0x00000015)
  SPR:     0    Depth: 1
  Actions: [Check, Bet 33%]

  Strategy: Check 0.15 | Bet 33% 0.42 | Bet 67% 0.28 | Bet All-In 0.15
```

This is useful for investigating convergence outliers — training checkpoints print the highest/lowest regret keys with a ready-to-copy `tree --key` command.

## Benchmarking Solvers

Compare all three solver backends on the same config:

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

### Game tree statistics

Inspect the materialized tree size for a training config without actually training:

```bash
cargo run -p poker-solver-trainer --release -- tree-stats -c config.yaml --sample-deals 100
```

This prints node counts, info set estimates, and memory usage for planning larger training runs.

## Running Tests

```bash
# Fast tests only (~3 seconds)
cargo test -p poker-solver-core

# Include slow tests (convergence, integration)
cargo test -p poker-solver-core -- --include-ignored

# With timing output
cargo test -p poker-solver-core -- --nocapture
```

All tests use the `#[timed_test]` proc macro which prints elapsed time and enforces a timeout (default 10s, configurable via `#[timed_test(300)]`).

## Project Structure

```
poker_solver_rust/
├── crates/
│   ├── core/              # CFR solver, game trees, card abstraction, blueprint storage
│   ├── gpu-cfr/           # GPU-accelerated CFR via wgpu compute shaders (optional)
│   ├── trainer/           # CLI for running training
│   ├── tauri-app/         # Tauri desktop app (strategy explorer backend)
│   └── test-macros/       # #[timed_test] proc macro
├── scripts/               # Data generation and utility scripts
├── frontend/              # React + TypeScript explorer UI
└── docs/plans/            # Design documents
```

## License

MIT
