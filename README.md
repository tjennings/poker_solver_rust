# Poker Solver

A Rust-based poker solver using Counterfactual Regret Minimization (CFR) with a Tauri desktop UI.

## Features

- **Kuhn Poker solver** - 3-card toy game for testing CFR algorithms
- **HUNL Preflop solver** - Heads-up No-Limit Texas Hold'em preflop training
- **MCCFR** - Monte Carlo CFR with external sampling for large game trees
- **Convergence visualization** - Real-time exploitability chart showing training progress

## Prerequisites

- [Rust](https://rustup.rs/) (edition 2024)
- [Node.js](https://nodejs.org/) (v18+)
- [Tauri CLI](https://tauri.app/start/prerequisites/)

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd poker_solver_rust

# Install frontend dependencies
cd frontend
npm install
cd ..
```

## Running the App

```bash
# Development mode (with hot reload)
cd crates/tauri-app
cargo tauri dev
```

The app will open with:
- Iterations input (default: 10,000)
- Checkpoints input (default: 20)
- Convergence chart showing exploitability over training
- Strategy table with Nash equilibrium probabilities

## Running Tests

```bash
# Run all tests
cargo test

# Run core library tests only
cargo test -p poker-solver-core

# Run with output
cargo test -- --nocapture
```

## Project Structure

```
poker_solver_rust/
├── crates/
│   ├── core/           # CFR algorithms, game implementations
│   └── tauri-app/      # Tauri desktop app backend
├── frontend/           # React + TypeScript UI
└── docs/plans/         # Design documents
```

## License

MIT
