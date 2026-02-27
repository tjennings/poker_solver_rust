# Poker Solver

A Rust-based poker solver using Counterfactual Regret Minimization (CFR) for Heads-Up No-Limit Texas Hold'em, with a Tauri desktop UI for strategy exploration.

## Features

- **MCCFR training** — Monte Carlo CFR with chance sampling, DCFR discounting, parallel via Rayon
- **Sequence-form CFR** — Full-traversal CFR on materialized game tree (no sampling variance)
- **GPU-accelerated CFR** — wgpu compute shaders for sequence-form on Metal/Vulkan/DX12
- **Three abstraction modes** — EHS2 bucketing, hand-class, hand-class v2 (class + strength + equity + draws)
- **Strategy explorer** — Interactive 13x13 hand matrix desktop app for browsing trained strategies
- **Agent simulation** — Pit blueprints and rule-based agents against each other

## Getting Started

### 1. Install dependencies

```bash
# Rust (https://rustup.rs)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Node.js v18+ (https://nodejs.org)
brew install node        # macOS — or download from nodejs.org

# Tauri system dependencies (macOS)
xcode-select --install
```

### 2. Clone and build

```bash
git clone <repo-url>
cd poker_solver_rust
cd frontend && npm install && cd ..
cargo test -p poker-solver-core    # builds and runs ~500 tests
```

### 3. Create a training config

Save as `quickstart.yaml`:

```yaml
game:
  stack_depth: 25
  bet_sizes: [0.5, 1.0, 1.5]

training:
  iterations: 1000
  seed: 42
  output_dir: "./quickstart_strategy"
  mccfr_samples: 5000
  deal_count: 50000
  abstraction_mode: hand_class_v2
  strength_bits: 4
  equity_bits: 4
```

### 4. Train

```bash
cargo run -p poker-solver-trainer --release -- train -c quickstart.yaml
```

Training prints progress with exploitability and sample strategies at each checkpoint. This config trains ~90 seconds on a 10-core machine (first build adds a few minutes for compilation).

### 5. Explore the strategy

```bash
cd crates/tauri-app && cargo tauri dev
```

Click **Load Strategy Bundle** and select the `quickstart_strategy/` directory. The 13x13 hand matrix shows action probabilities for every starting hand — click actions to walk the game tree, enter board cards for postflop play.

## Project Structure

```
poker_solver_rust/
├── crates/
│   ├── core/              # CFR solver, game trees, card abstraction, blueprint storage
│   ├── gpu-cfr/           # GPU-accelerated CFR via wgpu compute shaders (optional)
│   ├── trainer/           # CLI for training and analysis
│   ├── tauri-app/         # Tauri desktop app backend
│   └── test-macros/       # #[timed_test] proc macro
├── frontend/              # React + TypeScript explorer UI
├── scripts/               # Data generation utilities
└── docs/                  # Detailed documentation
```

## Documentation

- **[Training Reference](docs/training.md)** — Config options, abstraction modes, solver backends, pruning, convergence
- **[Strategy Explorer](docs/explorer.md)** — Desktop app usage guide
- **[CLI Reference](docs/cli.md)** — All trainer subcommands: `tree`, `inspect-deals`, `generate-deals`, `flops`, benchmarks

## License

MIT
