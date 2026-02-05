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

MCCFR (Monte Carlo CFR) samples random card deals and performs full CFR traversal on the betting tree for each deal. This avoids materializing the entire game tree, making it practical for full HUNL. The implementation follows Brown et al.'s depth-limited solving paper:
- CFR+ regret flooring
- Early iteration discounting: first 30 iterations weighted by `sqrt(T)/(sqrt(T)+1)`
- Average strategy skips first 50% of iterations

### 1. Create a training config

Create a YAML file (e.g. `training_mccfr.yaml`):

```yaml
game:
  stack_depth: 100         # Effective stack in big blinds
  bet_sizes: [0.33, 0.5, 0.75, 1.0]  # Pot-fraction bet sizes (all-in always included)
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
├── frontend/              # React + TypeScript explorer UI
└── docs/plans/            # Design documents
```

## License

MIT
