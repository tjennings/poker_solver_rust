# Poker Solver

A Rust-based poker solver using Counterfactual Regret Minimization (CFR) for Heads-Up No-Limit Texas Hold'em, with a Tauri desktop UI for strategy exploration.

## Features

- **HUNL Postflop solver** - Full preflop-through-river training with card abstraction
- **MCCFR training** - Monte Carlo CFR with chance sampling for scalable blueprint computation
- **EHS2 card abstraction** - Equity and potential-aware hand bucketing for tractable game trees
- **Strategy explorer** - Interactive 13x13 hand matrix for browsing trained strategies
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
- DCFR discounting with α=1.5, β=0.5, γ=2.0 (positive regrets retained, negative regrets decay)
- Regret-based pruning: actions with non-positive cumulative regret are skipped (with probe iterations)
- Average strategy skips first 50% of iterations

### 1. Create a training config

Create a YAML file (e.g. `training_mccfr.yaml`):

```yaml
game:
  stack_depth: 100         # Effective stack in big blinds
  bet_sizes: [0.33, 0.67, 1.0, 2.0, 3.0]  # Pot-fraction bet sizes (all-in always included)
  samples_per_iteration: 50000  # Deal pool size (used internally by initial_states)

abstraction:
  flop_buckets: 200        # EHS2 buckets on flop (production: 5000)
  turn_buckets: 200        # EHS2 buckets on turn (production: 5000)
  river_buckets: 500       # EHS2 buckets on river (production: 20000)
  samples_per_street: 5000 # Monte Carlo samples for computing bucket boundaries

training:
  iterations: 1000         # Total MCCFR iterations
  seed: 42                 # RNG seed for reproducibility
  output_dir: "./mccfr_100bb"
  mccfr_samples: 500       # Deals sampled per MCCFR iteration (default: 500)
  deal_count: 50000        # Random deal pool size (default: 50000)
```

**Tuning notes:**
- `deal_count` controls how many random deals (hole cards + 5-card board) are pre-generated. Larger pools give better coverage of the card space.
- `mccfr_samples` controls how many deals are sampled per iteration. Higher = more node visits per iteration but slower.
- More buckets = finer hand distinction but larger strategy table.
- More `samples_per_street` = more accurate EHS2 bucket boundaries.
- More iterations = better convergence (watch exploitability trend).

### 2. Run training

```bash
cargo run -p poker-solver-trainer --release -- train -c training_mccfr.yaml
```

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

Training saves three files to the output directory:

```
mccfr_100bb/
├── config.yaml       # Game and abstraction settings (human-readable)
├── blueprint.bin     # Trained strategy (bincode)
└── boundaries.bin    # EHS2 bucket boundaries (bincode)
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
│   ├── trainer/           # CLI for running training
│   ├── tauri-app/         # Tauri desktop app (strategy explorer backend)
│   └── test-macros/       # #[timed_test] proc macro
├── scripts/               # Data generation and utility scripts
├── frontend/              # React + TypeScript explorer UI
└── docs/plans/            # Design documents
```

## License

MIT
