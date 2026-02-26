# Solver Architecture

## Overview

The solver finds a Nash equilibrium for heads-up no-limit Texas Hold'em by splitting the game into two phases: a **preflop solver** that uses Linear CFR (LCFR) over 169 canonical hand matchups, and an **abstracted postflop model** that provides expected values at preflop showdown boundaries. The postflop model uses direct 169-hand indexing per flop -- each canonical hand is its own info set, with no bucket clustering. Two backends are available: **MCCFR** (sampled concrete hands with real showdown eval) and **Exhaustive** (pre-computed equity tables with vanilla CFR).

```
PreflopConfig -> PreflopTree -> PreflopSolver
                                      |
                                      v
                              PostflopAbstraction
                              +----------------------------+
                              | ComboMap (per flop)         |  <-- 169 hands -> concrete card pairs
                              | PostflopTree (shared)       |  <-- single tree at fixed SPR
                              | PostflopValues (per-flop)   |  <-- solved EV table per flop
                              +----------------------------+

Per canonical flop (in parallel):
  combo map -> MCCFR solve (or exhaustive CFR) -> extract values -> DROP intermediate data
```

## Project Structure

```
poker_solver_rust/
├── crates/
│   ├── core/                  # Core solver library
│   │   └── src/
│   │       ├── game/          # Game implementations (HUNL, Kuhn, LHE)
│   │       ├── cfr/           # CFR variants (vanilla, MCCFR, sequence-form)
│   │       ├── blueprint/     # Blueprint strategy, subgame solving, bundling
│   │       ├── preflop/       # Preflop LCFR solver & postflop abstraction pipeline
│   │       ├── abstraction/   # Card abstraction (isomorphism)
│   │       ├── hand_class.rs  # 19-variant hand classification system
│   │       ├── info_key.rs    # Info set key encoding (64-bit packed)
│   │       ├── agent.rs       # Agent system (TOML-configured play styles)
│   │       ├── simulation.rs  # Arena-based agent simulation
│   │       └── abstract_game.rs  # Exhaustive deal enumeration
│   ├── trainer/               # CLI for training, diagnostics, deal generation
│   ├── deep-cfr/              # Neural network SD-CFR (candle backend)
│   ├── gpu-cfr/               # GPU sequence-form CFR (wgpu)
│   │   └── src/shaders/       # WGSL compute shaders
│   ├── tauri-app/             # Desktop strategy explorer (Tauri)
│   ├── devserver/             # HTTP mirror of Tauri API for browser debugging
│   └── test-macros/           # #[timed_test] proc macro
├── frontend/                  # React/TypeScript explorer UI
│   └── src/
│       ├── Explorer.tsx       # Strategy browsing interface
│       ├── Simulator.tsx      # Agent simulation interface
│       └── invoke.ts          # Tauri/fetch abstraction layer
├── agents/                    # Agent personality configs (TOML)
├── sample_configurations/     # Training config presets (YAML)
├── docs/                      # Architecture, training, explorer, cloud docs
```

### Component Summary

**`core`** -- The heart of the solver. Contains the `Game` trait and its implementations (heads-up no-limit, Kuhn poker, limit hold'em), multiple CFR solver backends (vanilla, MCCFR, sequence-form), the preflop LCFR solver with its postflop abstraction pipeline (169-hand direct indexing, combo maps, tree building, solving), the blueprint strategy system with subgame refinement, card abstractions, hand classification, and the agent simulation framework.

**`trainer`** -- CLI entry point that orchestrates training runs, diagnostics, hand tracing, and deal generation. Parses YAML configs and drives the core library.

**`deep-cfr`** -- Neural network approach to CFR using the candle ML framework. Encodes game states as card features, trains advantage networks via reservoir sampling, and supports both HUNL and limit hold'em.

**`gpu-cfr`** -- GPU-accelerated sequence-form CFR using wgpu. Includes tabular and tiled solver variants with WGSL compute shaders for forward pass, backward pass, regret matching, and DCFR discounting.

**`tauri-app`** -- Desktop application for exploring solved strategies. Loads blueprint bundles, navigates the game tree, queries strategy/EV at any node, and runs agent simulations.

**`devserver`** -- Lightweight HTTP server that mirrors all Tauri exploration commands as POST endpoints, enabling browser-based UI development without the full Tauri build cycle.

**`frontend`** -- React/TypeScript UI shared between the Tauri app and dev server. Provides the strategy explorer (tree browsing, action frequencies, EV display) and agent simulator. Auto-detects Tauri vs browser environment.

**`agents/`** -- TOML files defining agent play styles (tight-aggressive, loose-aggressive, etc.) that map hand classes to action frequency distributions for simulation.

## Preflop Solver

**Algorithm:** Linear CFR (LCFR) with DCFR discounting. Both regret and strategy sums are weighted by iteration number. Simultaneous updates -- both players traverse every iteration.

**Key types:**
- `PreflopConfig` -- game structure: blinds, stacks, raise sizes, DCFR params (alpha/beta/gamma), exploration epsilon
- `PreflopTree` -- arena-allocated game tree; nodes are Decision or Terminal (Fold/Showdown)
- `PreflopSolver` -- flat buffer layout: `regret_sum[node_idx * 169 + hand_idx]`, `strategy_sum[...]`

**Flow:**
1. Build `PreflopTree` from config (raise sizes, raise cap)
2. Optionally build `PostflopAbstraction` (see below)
3. Run LCFR iterations with epsilon-greedy exploration
4. At showdown terminals: if postflop model exists, query it for EV; otherwise use raw preflop equity

**Exploration:** epsilon-greedy -- `intended` strategy (for strategy sums) vs `traversal` strategy (explored) ensure sufficient visits to all actions.

**Files:**
- Config: `crates/core/src/preflop/config.rs`
- Tree: `crates/core/src/preflop/tree.rs`
- Solver: `crates/core/src/preflop/solver.rs`
- Equity: `crates/core/src/preflop/equity.rs`

## Postflop Abstraction Pipeline

The postflop model replaces raw equity evaluation at preflop showdown terminals with solved postflop EVs. It uses direct 169-hand indexing -- each canonical hand is its own info set per flop, with no bucket clustering.

### 169-Hand Direct Indexing

For each canonical flop, a **combo map** expands the 169 canonical hands into concrete (hero, opponent) card pairs that are compatible with the flop. Each hand index (0-168) corresponds to a canonical hand (AA, AKs, AKo, ..., 32s, 32o). The combo map handles card removal (dead cards blocked by the flop) and tracks the count of concrete combos per canonical hand for proper weighting.

This replaces the prior EHS bucket-based approach. Instead of clustering hands into buckets via k-means on equity histograms, each of the 169 hands is its own "bucket". This eliminates information loss from clustering and simplifies the pipeline.

### Streaming Architecture

The pipeline streams per-flop: for each canonical flop in parallel, it builds the combo map, runs CFR to convergence, extracts the value row, then drops all intermediate data. Only the per-flop value rows persist.

```
For each canonical flop (parallel):
  1. Build combo map: 169 hands -> concrete card pairs
  2. Build shared postflop tree at fixed SPR
  3. CFR solve (MCCFR or Exhaustive) using 169-hand indexing
  4. Extract value row: values[hero_pos][hero_hand][opp_hand] -> EV
  5. DROP: regret/strategy buffers, combo map
```

**Output retained at runtime:** `PostflopAbstraction` contains the shared tree and `PostflopValues`.

### MCCFR Backend

The default backend (`solve_type: mccfr`). For each canonical flop:
- Builds a combo map expanding 169 canonical hands to concrete card pairs
- Samples random (hero_hand, opp_hand, turn, river) deals per iteration
- Evaluates showdowns using `rank_hand()` on the actual 5-card board
- Uses `mccfr_sample_pct` to control sampling density (default 1% of deal space)
- Post-convergence: extracts per-hand EVs via Monte Carlo sampling (`value_extraction_samples`)

Embarrassingly parallel: one independent solve per canonical flop.

### Exhaustive Backend

Alternative backend (`solve_type: exhaustive`). For each canonical flop:
- Pre-computes pairwise equity tables for all 169x169 hand pairs per street
- Runs vanilla CFR with full traversal every iteration (no sampling)
- Slower per iteration but converges with fewer iterations and no sampling variance

### Postflop Tree

A single shared tree template at a fixed SPR (`postflop_spr`, default 3.5). All per-flop solves share this tree structure.

Each tree has three streets of Decision nodes (OOP=0, IP=1), Chance nodes at street transitions, and Terminals (Fold/Showdown). Bet sizes are pot fractions.

`PostflopLayout` maps `(node_idx, hand_idx)` to flat buffer offsets for regret/strategy storage.

### Runtime Integration

At each preflop showdown terminal:
1. Loop over all canonical flops
2. Per flop: look up hero and opponent hand indices (0-168)
3. Lookup: `PostflopValues::get_by_flop(flop_idx, hero_pos, hero_hand, opp_hand) -> EV fraction`
4. Average across flops, then scale: `EV_fraction * actual_pot + (pot/2 - hero_investment)`

**File:** `crates/core/src/preflop/postflop_abstraction.rs`

## Key Control Parameters

### Preflop (`PreflopConfig`)

| Parameter | Default | Description |
|-|-|-|
| `stack_depth` | 100 | Stack size in BB |
| `open_raise` | 2.5 | Open raise multiplier |
| `three_bet` | 3.0 | 3-bet multiplier |
| `raise_cap` | 4 | Max raises per street |
| `alpha` | 1.5 | DCFR positive regret weight |
| `beta` | 0.5 | DCFR negative regret weight |
| `gamma` | 2.0 | DCFR strategy weight |
| `exploration` | 0.05 | Epsilon-greedy exploration rate |
| `iterations` | 10000 | LCFR iterations |

### Postflop (`PostflopModelConfig`)

| Parameter | Default | Description |
|-|-|-|
| `solve_type` | mccfr | Backend: `mccfr` (sampled concrete hands) or `exhaustive` (equity tables + vanilla CFR) |
| `bet_sizes` | [0.5, 1.0] | Pot-fraction bet sizes |
| `max_raises_per_street` | 1 | Raise cap per postflop street |
| `postflop_sprs` | [3.5] | SPR(s) for shared postflop tree (scalar or list) |
| `postflop_solve_iterations` | 500 | CFR/MCCFR iterations per flop |
| `max_flop_boards` | 0 | Max canonical flops; 0 = all ~1,755 |
| `fixed_flops` | none | Explicit flop boards (overrides `max_flop_boards`) |
| `cfr_convergence_threshold` | 0.01 | Per-flop early stopping threshold |
| `mccfr_sample_pct` | 0.01 | Fraction of deal space per MCCFR iteration (MCCFR only) |
| `value_extraction_samples` | 10,000 | Monte Carlo samples for EV extraction (MCCFR only) |
| `ev_convergence_threshold` | 0.001 | Early-stop EV estimation threshold (MCCFR only) |

## Caching & Bundles

| Cache | Key | Stores | Status |
|-|-|-|-|
| `solve_cache` | config hash + equity flag | `PostflopValues` | Active |
| `PostflopBundle` | directory path | `PostflopModelConfig` + `PostflopValues` + flops + SPR | Active |

**Files:**
- `crates/core/src/preflop/solve_cache.rs`
- `crates/core/src/preflop/postflop_bundle.rs`

**Postflop bundles:** Build a postflop abstraction once with `solve-postflop`, then reference the bundle directory via `postflop_model_path` in training configs to skip the expensive rebuild.

## Known Limitations

- **Preflop-only model limitation:** The preflop solver with postflop equity reference finds a limp-trap equilibrium (AA ~30% raise) rather than full-game GTO (AA 100% raise). This is inherent to the preflop-only model, not a bug.
- **No real-time subgame solving yet:** The postflop model is a static blueprint. Pluribus-style real-time search is planned but not implemented.
